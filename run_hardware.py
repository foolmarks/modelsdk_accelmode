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


import os, sys, shutil
import argparse
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

# Palette-specific imports
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted
from afe.apis.model import Model

import config as cfg

height = cfg.height
width = cfg.width
channels= cfg.channels
ignore_class = cfg.ignore_class
DIVIDER = cfg.DIVIDER


# pre-processing for quantizing and test
def _preprocessing(image):
  '''
  Image preprocess, add batchsize dimension
  '''
  image = cfg.preprocess(image)
  return image.reshape([1,height,width,channels])


def implement(args):

  enable_verbose_error_messages()


  '''
  load quantized model
  '''
  model_path = f'{args.build_dir}/{args.model_name}'
  print(f'Loading {args.model_name} quantized model from {model_path}')
  quant_model = Model.load(f'{args.model_name}.sima', model_path)


  '''
  Execute quantized model in accel mode
  Note that input data is always NHWC
  '''
  print('Executing quantized model in accelerator mode...')


  '''
  Prepare test data
    - create list of dictionaries
    - Each dictionary key is an input name, value is a preprocessed data sample
  '''
  
  dataset_path = './validation_dataset.npz'
  assert (os.path.exists(dataset_path)), f'Did not find {dataset_path}'
  dataset_f = np.load(dataset_path)
  data = dataset_f['x']
  labels = dataset_f['y']
  
  test_images = min(args.num_test_images, data.shape[0])
  
  
  input_data = [{'input_': _preprocessing(data[i])} for i in range(test_images)]


  total_matching_pixels=0
  total_ignore_pixels=0
  dest_folder = f'{args.build_dir}/accel_pred'
  if (os.path.exists(dest_folder)):
    shutil.rmtree(dest_folder, ignore_errors=False)
  os.makedirs(dest_folder)


  # returns a list of lists of np arrays
  outputs = quant_model.execute_in_accelerator_mode(input_data=length_hinted(test_images, input_data),
                                                    devkit=args.hostname,
                                                    username=args.username,
                                                    password=args.password)

  print("Model is executed in accelerator mode.")


  '''
  Evaluate results
  '''
  for i in range(test_images):

    prediction=outputs[i][0]
    if (prediction.shape[-1] > 1):
      prediction = np.argmax(prediction,axis=-1,keepdims=True)

    '''
    Simple acc check - replace with a standard metric like mIoU.
    Count number of matching pixels between prediction & label, ignore_class is not counted
    '''
    matching_pixels,ignore_pixels=cfg.pixel_match_count(prediction, labels[i], ignore_class)
    total_matching_pixels+=matching_pixels
    total_ignore_pixels+=ignore_pixels

    # prediction as image and write to PNG file
    _ = cfg.write_image(prediction,labels[i],dest_folder,i,ignore_class)


  total_pixels=(test_images*height*width) - total_ignore_pixels
  accuracy = (total_matching_pixels/total_pixels)*100
  print(f'Pixel matching accuracy: {accuracy:.2f}%')
    
  return



def run_main():
  
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir',       type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-m',  '--model_name',      type=str, default='segmenter', help='quantized model name')
  ap.add_argument('-ti', '--num_test_images', type=int, default=10, help='Number of test images. Default is 10')
  ap.add_argument('-u',  '--username',        type=str, default='root', help='Target device user name. Default is root')
  ap.add_argument('-p',  '--password',        type=str, default='commitanddeliver', help='Target device password. Default is commitanddeliver')
  ap.add_argument('-hn', '--hostname',        type=str, default='192.168.8.20', help='Target device IP address. Default is 192.168.8.20')
  args = ap.parse_args()

  print('\n'+DIVIDER,flush=True)
  print('Model SDK version',get_model_sdk_version())
  print(sys.version,flush=True)
  print(DIVIDER,flush=True)


  implement(args)


if __name__ == '__main__':
    run_main()


