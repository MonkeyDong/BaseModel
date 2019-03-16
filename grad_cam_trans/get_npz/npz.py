import numpy as np

aa = np.load('vgg16_weights.npz')
keys = sorted(aa.keys())
bb = np.load('last.npy')
#aa = np.load('vgg16.npy',encoding='latin1').item()

'''
keys
['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 
'conv2_2_W', 'conv2_2_b', 'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 
'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 
'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 
'conv5_3_W', 'conv5_3_b', 
'fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b']
'''
np.savez('vgg.npz',conv1_1_W=aa['conv1_1_W'],conv1_1_b=aa['conv1_1_b'],conv1_2_W=aa['conv1_2_W'],
  conv1_2_b=aa['conv1_2_b'],conv2_1_W=aa['conv2_1_W'],conv2_1_b=aa['conv2_1_b'],
  conv2_2_W=aa['conv2_2_W'],conv2_2_b=aa['conv2_2_b'],conv3_1_W=aa['conv3_1_W'],
  conv3_1_b=aa['conv3_1_b'],conv3_2_W=aa['conv3_2_W'],conv3_2_b=aa['conv3_2_b'],
  conv3_3_W=aa['conv3_3_W'],conv3_3_b=aa['conv3_3_b'],conv4_1_W=aa['conv4_1_W'],
  conv4_1_b=aa['conv4_1_b'],conv4_2_W=aa['conv4_2_W'],conv4_2_b=aa['conv4_2_b'],
  conv4_3_W=aa['conv4_3_W'],conv4_3_b=aa['conv4_3_b'],conv5_1_W=aa['conv5_1_W'],
  conv5_1_b=aa['conv5_1_b'],conv5_2_W=aa['conv5_2_W'],conv5_2_b=aa['conv5_2_b'],
  conv5_3_W=aa['conv5_3_W'],conv5_3_b=aa['conv5_3_b'],fc6_W=aa['fc6_W'],fc6_b=aa['fc6_b'],
  fc7_W=bb[1],fc7_b=bb[0],fc8_W=bb[3],fc8_b=bb[2])


