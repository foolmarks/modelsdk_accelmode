'''
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                

**************************************************************************
'''

'''
Author: Mark Harvey
'''



'''
Common configuration parameters
'''

import cv2
import numpy as np
DIVIDER = '-'*50


# input dimensions
height=1024
width=2048
channels=3
image_shape=(height,width,3)
mask_shape=(height,width,1)

# dataset mean & std deviation (per color channel, assumed to be RGB order)
scale = np.array([255.0,255.0,255.0], dtype=np.float32)
means = np.array([0.485,0.456,0.406], dtype=np.float32)
std_dev = np.array([0.229,0.224,0.225],dtype=np.float32)


ignore_class=255

'''
name                   id    trainId
--------------------------------------
unlabeled              0       255
ego vehicle            1       255
rectification border   2       255
out of roi             3       255
static                 4       255
dynamic                5       255
ground                 6       255
road                   7         0
sidewalk               8         1
parking                9       255
rail track            10       255
building              11         2
wall                  12         3
fence                 13         4
guard rail            14       255
bridge                15       255
tunnel                16       255
pole                  17         5
polegroup             18       255
traffic light         19         6
traffic sign          20         7
vegetation            21         8
terrain               22         9
sky                   23        10
person                24        11
rider                 25        12
car                   26        13
truck                 27        14
bus                   28        15
caravan               29       255
trailer               30       255
train                 31        16
motorcycle            32        17
bicycle               33        18
license plate         -1       255
'''

'''
Remapping of original classes to classes based on trainIDs
'''
class_map = {
  -1: ignore_class,
  0 : ignore_class,
  1 : ignore_class,
  2 : ignore_class,
  3 : ignore_class,
  4 : ignore_class,
  5 : ignore_class,
  6 : ignore_class,
  7 :   0,
  8 :   1,
  9 : ignore_class,
  10: ignore_class,
  11:   2,
  12:   3,
  13:   4,
  14: ignore_class,
  15: ignore_class,
  16: ignore_class,
  17:   5,
  18: ignore_class,
  19:   6,
  20:   7,
  21:   8,
  22:   9,
  23:  10,
  24:  11,
  25:  12,
  26:  13,
  27:  14,
  28:  15,
  29: ignore_class,
  30: ignore_class,
  31:  16,
  32:  17,
  33:  18
  }


'''
Map trainIDs to colors
'''
colors = np.array(
 [[128, 64,128],
  [244, 35,232],
  [ 70, 70, 70],
  [102,102,156],
  [190,153,153],
  [153,153,153],
  [250,170, 30],
  [220,220,  0],
  [107,142, 35],
  [152,251,152],
  [ 70,130,180],
  [220, 20, 60],
  [255,  0,  0],
  [  0,  0,142],
  [  0,  0, 70],
  [  0, 60,100],
  [  0, 80,100],
  [  0,  0,230],
  [119, 11, 32]], dtype=np.uint8)
zero_map=np.zeros((237,3), dtype=np.uint8)
color_map=np.concatenate((colors,zero_map),axis=0)


def mask_to_rgb(mask):
  '''
  Color the predicted pixels
  '''
  mask=np.squeeze(mask)
  return color_map[mask]



#def preprocess(image=None, transpose=False):
#  '''
#  Image pre-processing
#  Optional transpose to NCHW format
#  '''
#  image = np.divide(image,scale)
#  image = np.subtract(image,means)
#  image = np.divide(image,std_dev)
#  if (transpose):
#    image = np.transpose(image, axes=[0, 3, 1, 2])
#  return image


def preprocess(image=None, transpose=False):
    '''
    Image pre-processing
    Optional transpose to NCHW format.
    Performs operations in-place on a single working copy to avoid temporaries.
    Expects globals: scale (scalar), means (scalar or 1D per-channel), std_dev (scalar or 1D per-channel).
    '''
    if image is None:
        raise ValueError("image must not be None")

    # Make exactly one working copy with a float dtype; all ops reuse this buffer.
    x = np.array(image, dtype=np.float32, copy=True)  # single copy

    # x = (x / scale - means) / std_dev, but done stepwise with ufuncs + out= to avoid temporaries
    np.divide(x, scale, out=x)  # x /= scale

    # Broadcast-safe subtract/divide; supports scalar or per-channel arrays (length C on last axis)
    # Using in-place ops keeps memory use flat.
    x -= np.asarray(means, dtype=x.dtype)
    np.divide(x, np.asarray(std_dev, dtype=x.dtype), out=x)

    if transpose:
        # NHWC -> NCHW (returns a view when possible; no data copy)
        x = np.transpose(x, (0, 3, 1, 2))

    return x


def pixel_match_count(prediction, label, ignore_class):
  '''
  Simple acc check - replace with a standard metric like mIoU.
  Count number of matching pixels between prediction & label.
  '''
  ignore_pixels=np.count_nonzero(label==ignore_class)
  matching_pixels=np.count_nonzero(label==prediction)
  return matching_pixels, ignore_pixels


def write_image(mask,label,dest_folder,index,ignore_class):
  '''
  create output image
  '''
  overlay = mask_to_rgb(mask) 
  filepath = f'{dest_folder}/pred_{str(index)}.png'
  overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
  return (cv2.imwrite(filepath, overlay))



