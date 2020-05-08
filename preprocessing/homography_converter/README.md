## Homography Converter

This script can be used to convert a given homography for usage on different input/output resolutions and convert a given OpenCV homography for use with SpatialTransformer units.

When preprocessing the dataset and creating homography images by running `ipm.py`, all images are processed at their native resolution. The actual neural network training can however be performed at a decreased resolution and different aspect ratio. Additionally, the SpatialTransformer units in *uNetXST* work slightly differently than OpenCV's warping method. In order to configure _uNetXST_ with correct homographies for in-network transformation, this script needs to be used.

1. Use `ipm.py` with `-v` flag to only print the computed homographies.
1. Run this script with the homographies from `ipm.py` as input to convert them for usage with SpatialTransformer units.
1. Create a file similar to [preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py](preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py) and paste the converted SpatialTransformer homography there, if _uNetXST_ is chosen as the neural network architecture. Don't forget to set the `unetxst-homographies` parameter in the training config file.

Note that for our datasets we already provide the correct homographies to be used within *uNetXST*.

### Usage

```
usage: homography_converter.py [-h] -roi H W [-roo H W] -rni H W [-rno H W]
                               homography

Converts a given homography matrix to work on images of different resolution.
Also converts OpenCV homography matrices for use with SpatialTransformer
units.

positional arguments:
  homography  homography to convert (string representation)

optional arguments:
  -h, --help  show this help message and exit
  -roi H W    original input resolution (HxW)
  -roo H W    original output resolution (HxW)
  -rni H W    new input resolution (HxW)
  -rno H W    new output resolution (HxW)
```

### Example

#### Convert homography from Inverse Perspective Mapping (`ipm.py`) for 604x964 front images (dataset `1_FRLR`) in order to be used on 256x512 images

```bash
./ipm.py -v --drone ../camera_configs/1_FRLR/drone.yaml ../camera_configs/1_FRLR/front.yaml front ../camera_configs/1_FRLR/rear.yaml rear ../camera_configs/1_FRLR/left.yaml left ../camera_configs/1_FRLR/right.yaml right
# OpenCV homography for front:
# [[0.0, 0.8841865353311344, -253.37277367000263], [0.049056392233805146, 0.5285437237795494, -183.265385638118], [-0.0, 0.001750144780726984, -0.5285437237795492]]
# OpenCV homography for rear:
# [[6.288911300436434e-18, 0.8292344604207404, -264.08036704706365], [-0.04905639223380515, 0.5285437237795513, -135.9750235247304], [-0.0, 0.0017501447807269904, -0.5285437237795512]]
# OpenCV homography for left:
# [[0.04905639223380514, 0.7984814950483465, -264.7865925612947], [3.0038376863423275e-18, 0.4821577791689496, -159.26320930902278], [-0.0, 0.0016334684620118568, -0.49330747552758086]]
# OpenCV homography for right:
# [[-0.04905639223380516, 0.7984814950483448, -217.49623044790604], [3.0038376863423283e-18, 0.5044571718862112, -138.69450590963578], [-0.0, 0.0016334684620118542, -0.49330747552758]]
```

```bash
./homography_converter.py '[[0.0, 0.8841865353311344, -253.37277367000263], [0.049056392233805146, 0.5285437237795494, -183.265385638118], [-0.0, 0.001750144780726984, -0.5285437237795492]]' -roi 604 964 -rni 256 512
```

#### Convert homography from Inverse Perspective Mapping (`ipm.py`) for 1936x1216 front images and 1936x1216 drone images (dataset `2_F`) in order to be used on 256x512 images

```bash
./ipm.py -v --drone ../camera_configs/2_F/drone.yaml ../camera_configs/2_F/front.yaml front
# OpenCV homography for front:
# [[-8.869343394420501e-18, -0.031686570310719885, 58.219888879244245], [0.017875377577360244, 0.1995620198169141, -140.13793664741047], [-0.0, 0.0004091757529793379, -0.25180641631255507]]
```

```bash
./homography_converter.py '[[-8.869343394420501e-18, -0.031686570310719885, 58.219888879244245], [0.017875377577360244, 0.1995620198169141, -140.13793664741047], [-0.0, 0.0004091757529793379, -0.25180641631255507]]' -roi 1216 1936 -roo 968 1936 -rni 256 512
```
