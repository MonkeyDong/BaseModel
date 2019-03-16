import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path='checkpoints/flowers.ckpt'#your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#var_to_shape_map=reader.get_variable_to_shape_map()

ss=[]
ll = ['fully_connected/biases','fully_connected/weights','fully_connected_1/biases','fully_connected_1/weights']
for key in ll:
  aa=reader.get_tensor(key)
  ss.append(aa)

np.save('last.npy',ss)  