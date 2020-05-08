#!/usr/bin/env python

# ==============================================================================
# MIT License
#
# Copyright 2020 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import os
import argparse
import yaml
import tqdm
import multiprocessing
import numpy as np
import cv2
import skimage.draw


BLOCKING_LABELS = ["building", "wall", "car", "truck", "bus", "caravan", "trailer", "train"]
TALL_NON_BLOCKING_LABELS = ["vegetation"] # will be visible behind small blocking objects (e.g. cars)
COLORS = {
    "occluded"     : (150, 150, 150),
    "static"       : (  0,   0,   0),
    "dynamic"      : (111,  74,   0),
    "ground"       : ( 81,   0,  81),
    "road"         : (128,  64, 128),
    "sidewalk"     : (244,  35, 232),
    "parking"      : (250, 170, 160),
    "rail track"   : (230, 150, 140),
    "building"     : ( 70,  70,  70),
    "wall"         : (102, 102, 156),
    "fence"        : (190, 153, 153),
    "guard rail"   : (180, 165, 180),
    "bridge"       : (150, 100, 100),
    "tunnel"       : (150, 120,  90),
    "pole"         : (153, 153, 153),
    "polegroup"    : (153, 153, 153),
    "traffic light": (250, 170,  30),
    "traffic sign" : (220, 220,   0),
    "vegetation"   : (107, 142,  35),
    "terrain"      : (152, 251, 152),
    "sky"          : ( 70, 130, 180),
    "person"       : (255,   0,   0),
    "rider"        : (220,  20,  60),
    "car"          : (  0,   0, 142),
    "truck"        : (  0,   0,  70),
    "bus"          : (  0,  60, 100),
    "caravan"      : (  0,   0,  90),
    "trailer"      : (  0,   0, 110),
    "train"        : (  0,  80, 100),
    "motorcycle"   : (  0,   0, 230),
    "bicycle"      : (119,  11,  32),
    "roadmark"     : (255, 255, 255)
}

DUMMY_COLOR = tuple(np.random.randint(0, 256, 3))
while DUMMY_COLOR in COLORS.values():
    DUMMY_COLOR = tuple(np.random.randint(0, 256, 3))


class Camera:

    def __init__(self, config, frame, pxPerM):

        self.origin = (frame[0] + config["XCam"] * pxPerM[0], frame[1] - config["YCam"] * pxPerM[1])
        self.yaw = -config["yaw"]
        self.fov = 2.0 * np.arctan(config["px"] / config["fx"]) * 180.0 / np.pi
        thetaMin = self.yaw - self.fov / 2.0
        thetaMax = (self.yaw + self.fov / 2.0)
        thetaMin = thetaMin % 180 if thetaMin < -180 else thetaMin
        thetaMax = thetaMax % -180 if thetaMax > 180 else thetaMax
        self.fovBounds = (thetaMin, thetaMax)

    def canSee(self, x, y):

        dx, dy = x - self.origin[0], y - self.origin[1]
        theta = np.arctan2(dy, dx) * 180.0 / np.pi
        if self.fovBounds[0] > self.fovBounds[1]:
            return (self.fovBounds[0] <= theta) or (theta <= self.fovBounds[1])
        else:
            return (self.fovBounds[0] <= theta) and (theta <= self.fovBounds[1])


def floodFill(px, color, inputImg, outputImg):

    mask = np.zeros((inputImg.shape[0]+2, inputImg.shape[1]+2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(image=inputImg, mask=mask, seedPoint=(px[0], px[1]), newVal=(255, 255, 255), loDiff=(1,1,1), upDiff=(1,1,1), flags=flags)
    outputImg[np.where(mask[1:-1, 1:-1] == 255)] = color


def castRay(fromPoint, toPoint, inputImg, outputImg):

    # loop over all pixels along the ray, moving outwards
    ray = list(zip(*skimage.draw.line(*(int(fromPoint[0]), int(fromPoint[1])), *(int(toPoint[0]), int(toPoint[1])))))
    stopRay = stopTransfer = False
    for px in ray:

        # out-of-bounds check
        if not (0 <= px[0] and px[0] < inputImg.shape[1] and 0 <= px[1] and px[1] < inputImg.shape[0]):
            continue

        # check if ray hit a blocking object class
        for label in BLOCKING_LABELS:
            if (inputImg[px[1], px[0], :] == COLORS[label]).all():

                # if car, continue ray to look for more blocking objects, else stop ray
                if label == "car":
                    if stopTransfer: # if car behind another car, skip
                        continue
                    else: # if first car in line of ray
                        stopTransfer = True
                else:
                    stopRay = True

                # transfer blocking object to output image
                if not (outputImg[px[1], px[0], :] == COLORS[label]).all():
                    floodFill(px, COLORS[label], inputImg, outputImg)
                break

        if stopRay: # stop ray if blocked
            break
        if stopTransfer: # if transfer is stopped, still look for tall non-blocking labels to transfer
            for label in TALL_NON_BLOCKING_LABELS:
                if (inputImg[px[1], px[0], :] == COLORS[label]).all():
                    outputImg[px[1], px[0], :] = inputImg[px[1], px[0], :]
                    break
        else: # transfer px to output image
            outputImg[px[1], px[0], :] = inputImg[px[1], px[0], :]

# ==============================================================================

# parse command line arguments and read image
parser = argparse.ArgumentParser(description="Determines the areas not visible from vehicle cameras and removes them from drone camera footage.")
parser.add_argument("img", help="segmented drone image")
parser.add_argument("drone", help="drone camera config file")
parser.add_argument("cam", nargs="+", help="camera config file")
parser.add_argument("--batch", help="process folders of images instead of single images", action="store_true")
parser.add_argument("--output", help="output directory to write output  images to")
args = parser.parse_args()

# load image paths
imagePaths = []
if not args.batch:
    imagePaths.append(os.path.abspath(args.img))
else:
    path = os.path.abspath(args.img)
    imagePaths = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f[0] != "."]

# parse camera configs
with open(os.path.abspath(args.drone)) as stream:
    droneConfig = yaml.safe_load(stream)
cameraConfigs = []
for cameraConfig in args.cam:
  with open(os.path.abspath(cameraConfig)) as stream:
    cameraConfigs.append(yaml.safe_load(stream))

# create output directories
if args.output:
  outputDir = os.path.abspath(args.output)
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# determine image dimensions in (m)
inputImg = cv2.imread(imagePaths[0])
dxm = inputImg.shape[1] / droneConfig["fx"] * droneConfig["ZCam"]
dym = inputImg.shape[0] / droneConfig["fy"] * droneConfig["ZCam"]
pxPerM = (inputImg.shape[1] / dxm, inputImg.shape[0] / dym)
base_link = (int(inputImg.shape[1] / 2.0 - droneConfig["XCam"] * pxPerM[0]), int(inputImg.shape[0] / 2.0 + droneConfig["YCam"] * pxPerM[0]))

# create cameras
cameras = []
for cameraConfig in cameraConfigs:
    cam = Camera(cameraConfig, base_link, pxPerM)
    cameras.append(cam)


# define processing of a single image
def processImage(imagePath):

    filename = os.path.basename(imagePath)

    # read input image and create blank output image
    inputImg = cv2.imread(imagePath)
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
    outputImg = np.zeros(inputImg.shape, dtype=np.uint8) + np.array(COLORS["occluded"], dtype=np.uint8)

    # temporarily recolor ego vehicle (if in image), s.t. it does not block
    if base_link[0] > 0 and base_link[1] > 0:
        floodFill(base_link, DUMMY_COLOR, inputImg, inputImg)

    # loop over all border pixels to determine if ray is visible
    rays = []
    for cam in cameras:
        for x in range(inputImg.shape[1]):
            if cam.canSee(x, 0):
                rays.append((cam.origin, (x, 0)))
        for x in range(inputImg.shape[1]):
            if cam.canSee(x, inputImg.shape[0]):
                rays.append((cam.origin, (x, inputImg.shape[0])))
        for y in range(inputImg.shape[0]):
            if cam.canSee(0, y):
                rays.append((cam.origin, (0, y)))
        for y in range(inputImg.shape[0]):
            if cam.canSee(inputImg.shape[1], y):
                rays.append((cam.origin, (inputImg.shape[1], y)))

    # cast rays
    for ray in rays:
        castRay(ray[0], ray[1], inputImg, outputImg)

    # recolor ego vehicle as car and transfer to output
    if base_link[0] > 0 and base_link[1] > 0:
        floodFill(base_link, COLORS["car"], inputImg, outputImg)
        floodFill(base_link, COLORS["car"], inputImg, outputImg)

    # display or export output image
    outputImg = cv2.cvtColor(outputImg, cv2.COLOR_RGB2BGR)
    if args.output:
        cv2.imwrite(os.path.join(outputDir, filename), outputImg)
    else:
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
        cv2.imshow(filename, outputImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# process images in parallel
if args.batch:
    print("Warning: This might take an extremely long time, are you sure you need to (re)generate the occluded labels?")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm.tqdm(pool.imap(processImage, imagePaths), desc="Processing images", total=len(imagePaths), smoothing=0):
        pass
    pool.close()
    pool.join()
else:
    processImage(imagePaths[0])
