import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

input_node_names = "input node name"
output_node_names = "output node name"

input_array = [input_node_names]
output_array = [output_node_names]

folder = "path to represntative dataset/*.JPEG"
data = []
files = glob.glob(folder)
for f1 in files:
  img = cv2.imread(f1,cv2.IMREAD_COLOR)
  img = cv2.resize(img,(224,224)) #input image shape
imgnet = np.asarray(data,dtype=np.float32)/255.0



def representative_data_gen():
  for input_value in imgnet:
    input_value = np.reshape(input_value,(1,224,224,3))
    yield [input_value]


converter = tf.lite.TFLiteConverter.from_frozen_graph('tf_model_pb.pb',input_arrays=input_array,output_arrays=output_array) #Frozen graph file
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_quant_model = converter.convert()
open("model-INT8.tflite", "wb").write(tflite_quant_model)#output file



