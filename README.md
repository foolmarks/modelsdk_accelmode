# Running a Model on a Hardware Platform #

This tutorial shows how to run a model on a hardware platform using the .execute_in_accelerator_mode() API.



## Numpy files ##

For convenience, two numpy files are provided - train_dataset.npz and validation_dataset.npz.

Each file contains two numpy arrays, the 'x' array contains the images and the 'y' array contains the segmentation masks (i.e. class labels for each pixel).




## Execute Floating-Point ONNX model ##

ONNXRuntime is included in the SDK docker, so we can run the floating-point model. The run_onnx.py script includes pre- and postprocessing.  The ONNX model is the modified version as described in ONNX model modifications. 


```shell
python run_onnx.py
```

*Note: the image preprocessing is assumed to be division by 255, means subtraction followed by division using the standard deviation with these values:*

```python
scale = np.array([255.0,255.0,255.0], dtype=np.float32)
means = np.array([0.485,0.456,0.406], dtype=np.float32)
std_dev = np.array([0.229,0.224,0.225],dtype=np.float32)
```

The images are written into build/onnx_pred folder:


<img src="./images/onnx_pred_0.png" alt="" style="height: 250px; width:500px;"/>


The expected console output is like this:

```shell
--------------------------------------------------
3.10.12 (main, May 15 2025, 05:38:06) [GCC 11.4.0]
--------------------------------------------------
Pixel matching accuracy: 95.33%
```


## Quantize & Compile ##

The run_modelsdk.py script will do the following:

* unpack the numpy file containing training data and build a list of preprocessed calibration data images.
* load the floating-point ONNX model.
* quantize using calibration data and quantization parameters set using command line arguments.
* unpack the numpy file containing validation data and then test the quantized model accuracy using pre-processed images.
* compile and then untar to extract the .lm and .json files (for use in benchmarking on the target board)

*Note: the quantization is done using default configuration, better results may be obtained with a different configuration.*


```shell
python run_modelsdk.py
```

The images are written into build/quant_pred folder:


<img src="./images/quant_pred_0.png" alt="" style="height: 250px; width:500px;"/>


The expected console output is like this:

```shell
--------------------------------------------------
Model SDK version 1.6.0
3.10.12 (main, May 15 2025, 05:38:06) [GCC 11.4.0]
--------------------------------------------------
Results will be written to build/segmenter
Model inputs:
 input_  (1, 3, 1024, 2048)
Loaded model from segmenter.onnx
Quantizing with 5 calibration samples
Calibration Progress: |██████████████████████████████| 100.0% 5|5 Complete.  5/5
Running Calibration ...DONE
Running quantization ...DONE
Pixel matching accuracy: 95.24%
Compiling with batch size set to 1
Wrote compiled model to build/segmenter/segmenter_mpk.tar.gz
```


## Test model on hardware ##

Run the model directly on the target board. This requires the target board to be reachable via ssh. Make sure to set the IP address, password and user name of the target board:


```shell
python run_hardware.py -p edgeai -u sima -hn 192.168.8.20
```


The images are written into build/accel_pred folder:

<img src="./images/hw_pred_0.png" alt="" style="height: 250px; width:500px;"/>


The output in the console will be soemthing like this:


```shell
--------------------------------------------------
Model SDK version 1.6.0
3.10.12 (main, May 15 2025, 05:38:06) [GCC 11.4.0]
--------------------------------------------------
Loading segmenter quantized model from build/segmenter
Executing quantized model in accelerator mode...
Compiling model segmenter to .elf file
Creating the Forwarding from host
Copying the model files to DevKit
Creating the Forwarding from host
ZMQ Connection successful.
Executing model graph in accelerator mode:
Progress: |██████████████████████████████| 100.0% 5|5 Complete.  5/5
Model is executed in accelerator mode.
Pixel matching accuracy: 95.24%
```




### Files Used ###

* .gitignore - list of files to exclude from versioning.
* start.py, stop.py - start & stop SDk docker.
* commands.txt - list of commands used during the flow (just for easy copy & paste)
* config.py - configurations and common functions.
* run_onnx.py - execute the floating-point ONNX model, write predictions to PNG files.
* run_modelsdk.py - quantize, evaluate the quantized model and write predictions to PNG files, compile.
* run_hardware.py - run model on devkit and write predictions to PNG files.
* train_dataset.npz - images and masks from the training dataset
* validation_dataset.npz - images and masks from the validation dataset



### Contact ###

* Mark Harvey (mark.harvey@sima.ai)

