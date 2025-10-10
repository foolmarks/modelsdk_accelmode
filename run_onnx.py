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
Run ONNX semantic-segmentation inference on a small validation set and
report pixel-wise accuracy (ignoring a specified class). Predictions and
label overlays are written to disk.

Inputs:
- ./validation_dataset.npz : contains arrays 'x' (images) and 'y' (labels)
- ONNX model: path provided via --model_path

Outputs:
- <build_dir>/onnx_pred/ : PNGs of predictions (and any visualizations from cfg)

Example Usage:
    python run_onnx.py -m ./segmenter.onnx -ti 10 -bd build
'''


'''
Author: Mark Harvey
'''


import onnx
import onnxruntime as ort
import os, sys, shutil
import argparse
import numpy as np



import config as cfg
 

height = cfg.height
width = cfg.width
channels= cfg.channels
ignore_class = cfg.ignore_class
DIVIDER = cfg.DIVIDER



def implement(args):

  '''
  Prepare test data
  '''
  dataset_path = './validation_dataset.npz'
  assert (os.path.exists(dataset_path)), f'Did not find {dataset_path}'
  dataset_f = np.load(dataset_path)
  data = dataset_f['x']
  labels = dataset_f['y']
  test_images = min(args.num_test_images, data.shape[0])


  # Load & validate ONNX model
  onnx_model = onnx.load(args.model_path)
  onnx.checker.check_model(onnx_model)

# Create an ONNX Runtime inference session
  ort_sess = ort.InferenceSession(args.model_path)

  total_matching_pixels=0
  total_ignore_pixels=0

  # Prepare output directory
  # Remove any existing predictions to avoid mixing runs
  dest_folder = f'{args.build_dir}/onnx_pred'
  if (os.path.exists(dest_folder)):
    shutil.rmtree(dest_folder, ignore_errors=False)
  os.makedirs(dest_folder)
  
  '''
  ONNX inference
  Loop over test images
  '''
  for i in range(test_images):

    # preprocess image, transpose to NCHW
    image = cfg.preprocess(data[i],transpose=True)
    
    # run inference - outputs NCHW format
    pred = ort_sess.run(None, {'input_': image })
    prediction = pred[0]

    # post-processing - change to NHWC & reduce with argmax
    prediction = np.transpose(prediction, axes=[0, 2, 3, 1])
    if prediction.shape[-1] > 1:
      prediction = np.argmax(prediction,axis=-1,keepdims=True)

    # count number of matching pixels
    matching_pixels,ignore_pixels=cfg.pixel_match_count(prediction, labels[i], ignore_class)
    total_matching_pixels+=matching_pixels
    total_ignore_pixels+=ignore_pixels

    # write prediction to PNG file
    _ = cfg.write_image(prediction,dest_folder,i)

  # pixel accuracy is number of pixels that match between prediction and ground truth mask.
  # This pixel-wise accuracy is calculated across all test images 
  accuracy = (total_matching_pixels/((test_images*height*width)-total_ignore_pixels))*100
  print(f'Pixel matching accuracy: {accuracy:.2f}%')


  return



def run_main():
  
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir',         type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-m',  '--model_path',        default='./segmenter.onnx', type=str, help='path to ONNX model')
  ap.add_argument('-ti', '--num_test_images',   type=int, default=10, help='Number of test images. Default is 10')
  args = ap.parse_args()

  # print Python version
  print('\n'+DIVIDER,flush=True)
  print(sys.version,flush=True)
  print(DIVIDER,flush=True)


  implement(args)


if __name__ == '__main__':
    run_main()
