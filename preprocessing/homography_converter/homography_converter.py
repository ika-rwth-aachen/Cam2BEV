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
import numpy as np
import argparse


# parse command line arguments
parser = argparse.ArgumentParser(description="Converts a given homography matrix to work on images of different resolution. Also converts OpenCV homography matrices for use with SpatialTransformer units.")
parser.add_argument("homography", type=str, help="homography to convert (string representation)")
parser.add_argument("-roi", type=int, nargs=2, required=True, metavar=("H", "W"), help="original input resolution (HxW)")
parser.add_argument("-roo", type=int, nargs=2, default=None, metavar=("H", "W"), help="original output resolution (HxW)")
parser.add_argument("-rni", type=int, nargs=2, required=True, metavar=("H", "W"), help="new input resolution (HxW)")
parser.add_argument("-rno", type=int, nargs=2, default=None, metavar=("H", "W"), help="new output resolution (HxW)")
args = parser.parse_args()

oldInputResolution = args.roi
oldOutputResolution = args.roo if args.roo is not None else oldInputResolution
newInputResolution = args.rni
newOutputResolution = args.rno if args.rno is not None else newInputResolution

# read original homography
cvH = np.array(eval(args.homography))

# calculate intermediate shapes
newInputAspectRatio = newInputResolution[0] / newInputResolution[1]
newOutputAspectRatio = newOutputResolution[0] / newOutputResolution[1]
isNewInputWide = newInputAspectRatio <= 1
isNewOutputWide = newOutputAspectRatio <= 1
if isNewInputWide:
  newInputResolutionAtOldInputAspectRatio = np.array((newInputResolution[1] / oldInputResolution[1] * oldInputResolution[0], newInputResolution[1]))
else:
  newInputResolutionAtOldInputAspectRatio = np.array((newInputResolution[0], newInputResolution[0] / oldInputResolution[0] * oldInputResolution[1]))
if isNewOutputWide:
  oldOutputResolutionAtNewOutputAspectRatio = np.array((newOutputAspectRatio * oldOutputResolution[1], oldOutputResolution[1]))
else:
  oldOutputResolutionAtNewOutputAspectRatio = np.array((oldOutputResolution[0], oldOutputResolution[0] / newOutputAspectRatio))

#=== introduce additional transformation matrices to correct for different aspect ratio

# shift input to simulate padding to original aspect ratio
px = (newInputResolutionAtOldInputAspectRatio[1] - newInputResolution[1]) / 2 if not isNewInputWide else 0
py = (newInputResolutionAtOldInputAspectRatio[0] - newInputResolution[0]) / 2 if isNewInputWide else 0
Ti = np.array([[ 1,  0, px],
               [ 0,  1, py],
               [ 0,  0,  1]], dtype=np.float32)

# scale input to original resolution
fx = oldInputResolution[1] / newInputResolutionAtOldInputAspectRatio[1]
fy = oldInputResolution[0] / newInputResolutionAtOldInputAspectRatio[0]
Ki = np.array([[fx,  0, 0],
               [ 0, fy, 0],
               [ 0,  0, 1]], dtype=np.float32)

# crop away part of size 'oldOutputResolutionAtNewOutputAspectRatio' from original output resolution
px = -(oldOutputResolution[1] - oldOutputResolutionAtNewOutputAspectRatio[1]) / 2
py = -(oldOutputResolution[0] - oldOutputResolutionAtNewOutputAspectRatio[0]) / 2
To = np.array([[ 1,  0, px],
               [ 0,  1, py],
               [ 0,  0,  1]], dtype=np.float32)

# scale output to new resolution
fx = newOutputResolution[1] / oldOutputResolutionAtNewOutputAspectRatio[1]
fy = newOutputResolution[0] / oldOutputResolutionAtNewOutputAspectRatio[0]
Ko = np.array([[fx,  0, 0],
               [ 0, fy, 0],
               [ 0,  0, 1]], dtype=np.float32)

# assemble adjusted homography
cvHr = Ko.dot(To.dot(cvH.dot(Ki.dot(Ti))))


#=== introduce additional transformation matrices to correct for implementation differences
#    between cv2.warpPerspective() and STN's BilinearInterpolation

# scale from unit grid (-1,1)^2 to new input resolution
fx = newInputResolution[1] / 2
fy = newInputResolution[0] / 2
px = newInputResolution[1] / 2
py = newInputResolution[0] / 2
Si = np.array([[fx,  0, px],
               [ 0, fy, py],
               [ 0,  0,  1]], dtype=np.float32)

# scale from output resolution back to unit grid (-1,1)^2
fx = 2 / newOutputResolution[1]
fy = 2 / newOutputResolution[0]
px = -1
py = -1
So = np.array([[fx,  0, px],
               [ 0, fy, py],
               [ 0,  0,  1]], dtype=np.float32)

# assemble adjusted homography
stnHr = np.linalg.inv(So.dot(cvHr.dot(Si)))


#=== print transformation matrices

print(f"\nOriginal OpenCV homography used for resolution {oldInputResolution[0]}x{oldInputResolution[1]} -> {oldOutputResolution[0]}x{oldOutputResolution[1]}:")
print(cvH.tolist())

print(f"\nAdjusted OpenCV homography usable for resolution {newInputResolution[0]}x{newInputResolution[1]} -> {newOutputResolution[0]}x{newOutputResolution[1]}:")
print(cvHr.tolist())

print(f"\nAdjusted SpatialTransformer homography usable for resolution {newInputResolution[0]}x{newInputResolution[1]} -> {newOutputResolution[0]}x{newOutputResolution[1]}:")
print(stnHr.tolist())
