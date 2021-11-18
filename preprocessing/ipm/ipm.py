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
import sys
import yaml
import numpy as np
import cv2
import argparse
from tqdm import tqdm


class Camera:

  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self, y, p, r):

    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
    Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) # switch axes (x = -y, y = -z, z = x)
    self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    self.setK(config["fx"], config["fy"], config["px"], config["py"])
    self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
    self.setT(config["XCam"], config["YCam"], config["ZCam"])
    self.updateP()


# parse command line arguments
parser = argparse.ArgumentParser(description="Warps camera images to the plane z=0 in the world frame.")
parser.add_argument("camera_img_pair", metavar="CAM IMG", nargs='*', help="camera config file and image file")
parser.add_argument("-wm", type=float, help="output image width in [m]", default=20)
parser.add_argument("-hm", type=float, help="output image height in [m]", default=40)
parser.add_argument("-r", type=float, help="output image resolution in [px/m]", default=20)
parser.add_argument("--drone", type=str, help="camera config file of drone to map to")
parser.add_argument("--batch", help="process folders of images instead of single images", action="store_true")
parser.add_argument("--output", help="output directory to write transformed images to")
parser.add_argument("--cc", help="use with color-coded images to enable NN-interpolation", action="store_true")
parser.add_argument("-v", help="only print homography matrices", action="store_true")
args = parser.parse_args()


# load camera configurations and image paths
cameraConfigs = []
imagePathArgs = []
for aIdx in range(int(len(args.camera_img_pair) / 2.0)):
  with open(os.path.abspath(args.camera_img_pair[2*aIdx])) as stream:
    cameraConfigs.append(yaml.safe_load(stream))
  imagePathArgs.append(args.camera_img_pair[2*aIdx+1])
toDrone = False
if args.drone:
  toDrone = True
  with open(os.path.abspath(args.drone)) as stream:
    droneConfig = yaml.safe_load(stream)

# load image paths
imagePaths = []
if not args.batch:
  imagePaths.append(imagePathArgs)
else:
  for path in imagePathArgs:
    imagePaths.append([os.path.join(path, f) for f in sorted(os.listdir(path)) if f[0] != "."])
  imagePaths = list(map(list, zip(*imagePaths))) # transpose ([[f1,f2],[r1,r2]] -> [[f1,r1],[f2,r2]])
outputFilenames = [os.path.basename(imageTuple[0]) for imageTuple in imagePaths]

# create output directories
export = False
if args.output:
  export = True
  outputDir = os.path.abspath(args.output)
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# initialize camera objects
cams = []
for config in cameraConfigs:
  cams.append(Camera(config))
if toDrone:
  drone = Camera(droneConfig)

# calculate output shape; adjust to match drone image, if specified
if not toDrone:
  pxPerM = (args.r, args.r)
  outputRes = (int(args.wm * pxPerM[0]), int(args.hm * pxPerM[1]))
else:
  outputRes = (int(2 * droneConfig["py"]), int(2 * droneConfig["px"]))
  dx = outputRes[1] / droneConfig["fx"] * droneConfig["ZCam"]
  dy = outputRes[0] / droneConfig["fy"] * droneConfig["ZCam"]
  pxPerM = (outputRes[0] / dy, outputRes[1] / dx)

# setup mapping from street/top-image plane to world coords
shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
if toDrone:
  shift = shift[0] + droneConfig["YCam"] * pxPerM[0], shift[1] - droneConfig["XCam"] * pxPerM[1]
M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]], [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# find IPM as inverse of P*M
IPMs = []
for cam in cams:
  IPMs.append(np.linalg.inv(cam.P.dot(M)))

# print homographies
if args.v:
  for idx, ipm in enumerate(IPMs):
    print(f"OpenCV homography for {args.camera_img_pair[2*idx+1]}:")
    print(ipm.tolist())
  exit(0)

# setup masks to later clip invalid parts from transformed images (inefficient, but constant runtime)
masks = []
for config in cameraConfigs:
  mask = np.zeros((outputRes[0], outputRes[1], 3), dtype=bool)
  for i in range(outputRes[1]):
    for j in range(outputRes[0]):
      theta = np.rad2deg(np.arctan2(-j + outputRes[0] / 2 - droneConfig["YCam"] * pxPerM[0], i - outputRes[1] / 2 + droneConfig["XCam"] * pxPerM[1]))
      if abs(theta - config["yaw"]) > 90 and abs(theta - config["yaw"]) < 270:
        mask[j,i,:] = True
  masks.append(mask)

# process images
progBarWrapper = tqdm(imagePaths)
for imageTuple in progBarWrapper:

  filename = os.path.basename(imageTuple[0])
  progBarWrapper.set_postfix_str(filename)

  # load images
  images = []
  for imgPath in imageTuple:
    images.append(cv2.imread(imgPath))

  # warp input images
  interpMode = cv2.INTER_NEAREST if args.cc else cv2.INTER_LINEAR
  warpedImages = []
  for img, IPM in zip(images, IPMs):
    warpedImages.append(cv2.warpPerspective(img, IPM, (outputRes[1], outputRes[0]), flags=interpMode))

  # remove invalid areas (behind the camera) from warped images
  for warpedImg, mask in zip(warpedImages, masks):
    warpedImg[mask] = 0

  # stitch separate images to total bird's-eye-view
  birdsEyeView = np.zeros(warpedImages[0].shape, dtype=np.uint8)
  for warpedImg in warpedImages:
    mask = np.any(warpedImg != (0,0,0), axis=-1)
    birdsEyeView[mask] = warpedImg[mask]

  # display or export bird's-eye-view
  if export:
    cv2.imwrite(os.path.join(outputDir, filename), birdsEyeView)
  else:
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    cv2.imshow(filename, birdsEyeView)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
