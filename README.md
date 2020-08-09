# 4-bit-quantization-with-tensorflow-1.15.2
Provides a custom `quantization_utils.h` and `quantization_utils.cc` file with modifications to the file from [https://github.com/tensorflow/tensorflow/releases/tag/v1.15.3] source code at location
`tensorflow-1.15.2/tensorflow/lite/tools/optimize/`

#### Refer the publication at the link [https://arxiv.org/abs/1810.05723] for all the concepts behind modifications made.
1. Per-channel bit allocation\
2. Bias-Correction

#### Requirements
1. bazel version 0.24 or 0.25\
2. Tensorflow 1.15.x source code 

#### Steps
1.Download both the files.\
2.Replace original files in the tensorflow source code with custom ones.\
3.Build tensorflow from source, directions are provided here [https://www.tensorflow.org/install/source].\
4.Perform `Full-integer (DEFAULT) quantization` by providing representative dataset.\ 

#### Full integer quantization
`Frozen.pb` is the representation of the trained DNN/CNN model (Programme to create frozen.pb file from checkpoints is provided in [https://github.com/AkashB23/Quantization-of-DNNs-with-Tensorflow/blob/master/create_model.py])
`path to representation dataset with atleast 100 images`
Output will be the .TFlite file with weights and activations in INT8 datatype but with precision of INT4.
Run `python IntegerQuantization.py >> file.txt` to get layer and channel wise characteristics along with allocated Bit value for each channel.  
