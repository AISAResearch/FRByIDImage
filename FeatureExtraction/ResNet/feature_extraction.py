#encoding=utf-8
import numpy as np
import resnet_model
import tensorflow as tf
import os
import re
import sys

pre_path = os.path.abspath("..")
print pre_path
sys.path.append(pre_path)
from face_input import get_Trains_and_Lables


# 这一级目录路径
this_path = os.path.dirname(os.path.abspath(__file__))
model_base_path = os.path.join(this_path,'all_model')
model_names = [model_name for model_name in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path,model_name))]
model_paths = [(os.path.join(model_base_path,model_name),int([_ for _ in re.split('\\D',model_name) if len(_)>0][0])) for model_name in model_names]

print model_names

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
# 模式：训练、测试
tf.app.flags.DEFINE_string('mode', 
                           'eval', 
                           'train or eval.')
# 图片尺寸
tf.app.flags.DEFINE_integer('image_size', 
                            224, 
                            'Image side length.')
# 一次性测试
tf.app.flags.DEFINE_bool('eval_once', 
                         False,
                         'Whether evaluate the model only once.')


def get_sess_model(batch_size=1,model_type=0):
  model_path,num_classes = model_paths[model_type]
  print model_path,num_classes
  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             is_gray=False,
                             num_feature=10,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')
  model = resnet_model.ResNet(hps,FLAGS.image_size, FLAGS.mode,is_gray=hps.is_gray)
  model.build_graph()
  # 模型变量存储器
  saver = tf.train.Saver()
  
  # 执行Session
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  try:
    ckpt_state = tf.train.get_checkpoint_state(model_path)
  except tf.errors.OutOfRangeError as e:
    tf.logging.error('Cannot restore checkpoint: %s', e)
    # continue
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('No model to eval yet at %s', model_path)
    # continue
    # 读取模型数据(训练期间生成)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  saver.restore(sess, ckpt_state.model_checkpoint_path)
  return sess,model

def get_feature_by_sess_model(sess,model,input_img,nums_gpus=0):
  if nums_gpus == 0:
    dev = '/cpu:0'
  elif nums_gpus == 1:
    dev = '/gpu:0'
  with tf.device(dev):
    vec = sess.run(model.x_feature,feed_dict={model._images:input_img})
  return vec

"""
传入参数:
images=[batch_size,img_size,img_size,nums_channel]
model_type 参数为0,和1,分别为为不同的模型
nums_gpus GPU的个数,默认0

返回:
直接返回 [batch_size,nums_feature]
"""
def get_feature_from_images(images,model_type=2,nums_gpus=0):
  tf.reset_default_graph()
  sess,model = get_sess_model(batch_size=images.shape[0],model_type=model_type)
  features = get_feature_by_sess_model(sess,model,images,nums_gpus)
  print features.shape
  return features

"""
img_path为图片的总路径,该路劲下每个人都有一个文件夹,文件夹下面每个人有多张图片(这里是12张)

返回值:
features 特征
labels 每个人的onehot编码
names 没人的标签值
"""
def get_feature_from_imgPath(img_path,model_type=2,nums_gpus=0):
  images, labels , names = get_Trains_and_Lables(img_path)
  features = get_feature_from_images(images)
  return features,labels,names

import cv2
if __name__ == '__main__':
  # print model_names
  # images, labels , names = get_Trains_and_Lables("../../FaceAugment/tempImages")
  get_sess_model(model_type=2)
  # batch_size = images.shape[0]

  # # 输出总的结果
  # res = [index for index,name in enumerate(names) if name == '430821199901100030']
  # faetures = get_feature_from_images(images)
  # for i in res:
  #   print faetures[i]

  # #输出单独预测结果
  # img_path = "../../FaceAugment/tempImages/430821199901100030/1.jpg"
  # img_data = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)/255.
  # img_data = img_data.reshape(1,img_data.shape[0],img_data.shape[1],1)
  # print get_feature_from_images(img_data)
  pass

