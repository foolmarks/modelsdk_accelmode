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
Quantize, evaluate and compile model
'''


'''
Author: Mark Harvey
'''


import onnx
import os, sys, shutil
import argparse
import numpy as np
import tarfile 
import logging

# Palette-specific imports
from afe.load.importers.general_importer import ImporterParams, onnx_source
from afe.apis.defines import default_quantization, gen1_target, gen2_target
from afe.ir.tensor_type import ScalarType
from afe.apis.loaded_net import load_model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted


import config as cfg
 
height = cfg.height
width = cfg.width
channels= cfg.channels
ignore_class = cfg.ignore_class
DIVIDER = cfg.DIVIDER


def get_onnx_input_shapes_dtypes(model_path):
    """
    Load an ONNX model and return two dictionaries describing its *true* inputs,
    ignoring any graph initializers (weights/biases).

    Returns:
        shapes_by_input:
            { input_name: (d0, d1, ...) } where each dimension (dn) is:
              - int for fixed sizes,
              - str for symbolic dimensions (e.g., "batch", "N"),
              - None if the dimension is present but unknown,
              - or the entire value can be None if the tensor is rank-unknown.
        dtypes_by_input:
            { input_name: dtype } where:
              - if the ONNX dtype is float32 -> the value is the symbol ScalarType.float32
              - otherwise -> the original NumPy-style dtype string (e.g., 'float16', 'int64')
              - or None if it could not be determined.
    """
    # Parse and sanity-check the model graph structure.
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    # Filter out parameters that appear as graph inputs.
    initializer_names = {init.name for init in model.graph.initializer}

    # Plain dictionaries
    shapes_by_input = {}
    dtypes_by_input = {}

    # Iterate over declared graph inputs
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue  # not a real runtime input

        # Only handle tensor inputs
        if not vi.type.HasField("tensor_type"):
            continue

        ttype = vi.type.tensor_type

        # ----- dtype -----
        elem_type = ttype.elem_type
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(elem_type, None)

        if np_dtype is None:
            dtypes_by_input[vi.name] = None
        else:
            dtype_name = np_dtype.name  # e.g., 'float32', 'int64'
            if dtype_name == 'float32':
                # Use the symbol (not a string) as requested
                dtypes_by_input[vi.name] = ScalarType.float32
            else:
                dtypes_by_input[vi.name] = dtype_name

        # ----- shape -----
        if not ttype.HasField("shape"):
            shapes_by_input[vi.name] = None  # rank-unknown
            continue

        dims_list = []
        for d in ttype.shape.dim:
            if d.HasField("dim_value"):
                dims_list.append(int(d.dim_value))       # fixed dimension
            elif d.HasField("dim_param"):
                dims_list.append(d.dim_param)            # symbolic dimension
            else:
                dims_list.append(None)                   # unknown dimension

        # Store as immutable tuple
        shapes_by_input[vi.name] = tuple(dims_list)

    return shapes_by_input, dtypes_by_input


# pre-processing for quantizing and test
def _preprocessing(image):
  '''
  Image preprocess, add batchsize dimension
  '''
  image = cfg.preprocess(image)
  return image.reshape([1,height,width,channels])


def implement(args):

  # Uncomment the following line to enable verbose error messages.
  enable_verbose_error_messages()

  # make destination folder
  base_name = os.path.basename(args.model_path)
  output_model_name, _ = os.path.splitext(base_name)

  output_path = os.path.join(args.build_dir,output_model_name)
  os.makedirs(output_path,exist_ok=True)
  print('Results will be written to',output_path,flush=True)


  
  '''
  Load the floating-point ONNX model
  input types & shapes are dictionaries
  input types dictionary: each key,value pair is an input name (string) and a type
  input shapes dictionary: each key,value pair is an input name (string) and a shape (tuple)
  '''
  input_shapes_dict, input_types_dict = get_onnx_input_shapes_dtypes(args.model_path)
  print(input_shapes_dict)
  print(input_types_dict)

     
  # importer parameters
  importer_params: ImporterParams = onnx_source(model_path=args.model_path,
                                                shape_dict=input_shapes_dict,
                                                dtype_dict=input_types_dict)
  
  # load ONNX floating-point model into SiMa's LoadedNet format
  target = gen2_target if args.generation == 2 else gen1_target
  loaded_net = load_model(importer_params,target=target)
  print(f'Loaded model from {args.model_path}')

  '''
  Prepare calibration data
    - create list of dictionaries
    - Each dictionary key is an input name, value is a preprocessed data sample
  '''
  # unpack the numpy file
  dataset_path = './calib_dataset.npz'
  assert (os.path.exists(dataset_path)), f'Did not find {dataset_path}'
  dataset_f = np.load(dataset_path)
  data = dataset_f['x']

  calib_data=[]
  calib_images = min(args.num_calib_images, data.shape[0])

  # make a list of dictionaries
  # key = input name, value = pre-processed calibration data
  for input_name in input_names_list:
    inputs = dict()
    for i in range(calib_images):
      inputs[input_name] = _preprocessing(data[i])
      calib_data.append(inputs)


  '''
  Quantize
  '''
  print(f'Quantizing with {calib_images} calibration samples')


  quant_model = loaded_net.quantize(calibration_data=length_hinted(calib_images,calib_data),
                                    quantization_config=default_quantization,
                                    model_name=output_model_name,
                                    log_level=logging.ERROR)

  # optional save of quantized model - saved model can be opened with Netron
  quant_model.save(model_name=output_model_name, output_directory=output_path)
  print(f'Quantized model saved to {output_path}/{output_model_name}.sima.json')



  '''
  Execute, evaluate quantized model
  '''
  if (args.evaluate):

    # unpack validation data
    dataset_path = './validation_dataset.npz'
    assert (os.path.exists(dataset_path)), f'Did not find {dataset_path}'    
    dataset_f = np.load(dataset_path)
    data = dataset_f['x']
    labels = dataset_f['y']

    # number of test images
    test_images = min(args.num_test_images, data.shape[0])

    total_matching_pixels=0
    total_ignore_pixels=0
    
    # make folder for output images, delete any previous results
    dest_folder = f'{args.build_dir}/quant_pred'
    if (os.path.exists(dest_folder)):
      shutil.rmtree(dest_folder, ignore_errors=False)
    os.makedirs(dest_folder)

    for i in range(test_images):

      inputs = dict() 
      inputs[input_name] = _preprocessing(data[i])

      quantized_net_output = quant_model.execute(inputs, fast_mode=True)
      if (quantized_net_output[0].shape[-1] > 1):
        quantized_net_output = np.argmax(quantized_net_output[0],axis=-1,keepdims=True)
      else:
        quantized_net_output = quantized_net_output[0]  

      '''
      Simple pixelwise accuracy check - could be replaced with a standard metric like mIoU.
      Count number of matching pixels between prediction & label, ignore_class is not counted
      '''
      matching_pixels,ignore_pixels=cfg.pixel_match_count(quantized_net_output, labels[i], ignore_class)
      total_matching_pixels+=matching_pixels
      total_ignore_pixels+=ignore_pixels

      # prediction as image and write to PNG file
      _ = cfg.write_image(quantized_net_output,dest_folder,i)

    total_pixels=(test_images*height*width) - total_ignore_pixels
    accuracy = (total_matching_pixels/total_pixels)*100
    print(f'Pixel matching accuracy: {accuracy:.2f}%')


  '''
  Compile
  '''
  print('Compiling with batch size set to',args.batch_size,flush=True)
  quant_model.compile(output_path=output_path,
                      batch_size=args.batch_size,
                      log_level=logging.INFO)  

  print(f'Wrote compiled model to {output_path}/{output_model_name}_mpk.tar.gz')

  model = tarfile.open(f'{output_path}/{output_model_name}_mpk.tar.gz')
  model.extractall(f'{output_path}')
  model.close() 


  return



def run_main():
  
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir',         type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-m',  '--model_path',        type=str, default='segmenter.onnx', help='path to FP model')
  ap.add_argument('-b',  '--batch_size',        type=int, default=1, help='requested batch size. Default is 1')
  ap.add_argument('-om', '--output_model_name', type=str, default='segmenter', help="Output model name. Default is segmenter")
  ap.add_argument('-ci', '--num_calib_images',  type=int, default=50, help='Number of calibration images. Default is 50')
  ap.add_argument('-ti', '--num_test_images',   type=int, default=10, help='Number of test images. Default is 10')
  ap.add_argument('-g',  '--generation',        type=int, default=2, choices=[1,2], help='Target device: 1 = DaVinci, 2 = Modalix. Default is 2')
  ap.add_argument('-e',  '--evaluate',          action="store_true", default=False, help="If set, evaluate the quantized model") 
  args = ap.parse_args()

  print('\n'+DIVIDER,flush=True)
  print('Model SDK version',get_model_sdk_version())
  print(sys.version,flush=True)
  print(DIVIDER,flush=True)


  implement(args)


if __name__ == '__main__':
    run_main()
