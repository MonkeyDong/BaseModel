1. 先使用vgg16.npy模型进行微调<br>
trans_tune中需要有data文件夹和npy模型
2. 将ckpt保存的模型的后几层和标准vgg16的 npy模型进行拼接，最终转换成npy模型<br>
get_npz中要拷贝之前训练好的cpkt和标准vgg16.npy
3. 生成带有cam的图片<br>
Cam_Tensorflow中要有输入图片和合成好的vgg.npy模型