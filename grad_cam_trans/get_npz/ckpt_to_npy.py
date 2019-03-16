import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path='checkpoints/flowers.ckpt'#your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

'''
reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
dict001 = {} # 字典
for key in sorted(var_to_shape_map):
    dict001[key] = reader.get_tensor(key) # numpy.ndarray

'''



vgg={}
vgg_layer = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']
add_info = ['weights','biases']

vgg={'conv1_1':[[],[]],'conv1_2':[[],[]],'conv2_1':[[],[]],'conv2_2':[[],[]],'conv3_1':[[],[]],'conv3_2':[[],[]],'conv3_3':[[],[]],'conv4_1':[[],[]],'conv4_2':[[],[]],'conv4_3':[[],[]],'conv5_1':[[],[]],'conv5_2':[[],[]],'conv5_3':[[],[]],'fc6':[[],[]],'fc7':[[],[]],'fc8':[[],[]]}


for key in var_to_shape_map:
 #print ("tensor_name",key)

 str_name = key
 # 因为模型使用Adam算法优化的，在生成的ckpt中，有Adam后缀的tensor
 if str_name.find('Adam') > -1:
  continue

 print('tensor_name:' , str_name)
'''
 if str_name.find('/') > -1:
  names = str_name.split('/')
  # first layer name and weight, bias
  layer_name = names[0]
  layer_add_info = names[1]
 else:
  layer_name = str_name
  layer_add_info = None

 if layer_add_info == 'weights':
  vgg[layer_name][0]=reader.get_tensor(key)
 elif layer_add_info == 'biases':
  vgg[layer_name][1] = reader.get_tensor(key)
 else:
  vgg[layer_name] = reader.get_tensor(key)

# save npy
np.save('vgg.npy',vgg)
print('save npy over...')
#print(alexnet['conv1'][0].shape)
#print(alexnet['conv1'][1].shape)
'''
