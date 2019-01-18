#encoding=utf-8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys
import numpy as np
import resnet_model
import tensorflow as tf
from face_input import get_Trains_and_Lables

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
# 数据集类型
tf.app.flags.DEFINE_string('dataset', 
                           'crop_face', 
                           'this dataset on')
# 模式：训练、测试
tf.app.flags.DEFINE_string('mode', 
                           'eval', 
                           'train or eval.')
# 训练数据路径
tf.app.flags.DEFINE_string('train_data_path', 
                           '../datas/train_data',
                           'Filepattern for training data.')
# 测试数据路劲
tf.app.flags.DEFINE_string('eval_data_path', 
                           '../datas/test_data',
                           'Filepattern for eval data')
# 图片尺寸
tf.app.flags.DEFINE_integer('image_size', 
                            112, 
                            'Image side length.')
# 训练过程数据的存放路劲
tf.app.flags.DEFINE_string('train_dir', 
                           '../train',
                           'Directory to keep training outputs.')
# 测试过程数据的存放路劲
tf.app.flags.DEFINE_string('eval_dir', 
                           '../eval',
                           'Directory to keep eval outputs.')
# 测试数据的Batch数量
tf.app.flags.DEFINE_integer('eval_batch_count', 
                            50,
                            'Number of batches to eval.')
# 一次性测试
tf.app.flags.DEFINE_bool('eval_once', 
                         False,
                         'Whether evaluate the model only once.')
# 模型存储路劲
tf.app.flags.DEFINE_string('log_root', 
                           '../model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
# GPU设备数量（0代表CPU）
tf.app.flags.DEFINE_integer('num_gpus', 
                            0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_string('f', '', 'kernel')

def train(hps):
  # 构建输入数据(读取队列执行器）
  images, labels = get_Trains_and_Lables(
      FLAGS.train_data_path,hps.num_classes)
  # 构建残差网络模型
  model = resnet_model.ResNet(hps,FLAGS.image_size, FLAGS.mode)
  model.build_graph()

  # 计算预测准确率
  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  # 建立总结存储器，每100步存储一次
  summary_hook = tf.train.SummarySaverHook(
              save_steps=1,
              output_dir=FLAGS.train_dir,
              summary_op=tf.summary.merge(
                              [model.summaries,
                               tf.summary.scalar('Precision', precision)]))
  # 建立日志打印器，每100步打印一次
  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=1)

  # 学习率更新器，基于全局Step
  class _LearningRateSetterHook(tf.train.SessionRunHook):

    def begin(self):
      #初始学习率
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
                      # 获取全局Step
                      model.global_step,
                      # 设置学习率
                      feed_dict={model.lrn_rate: self._lrn_rate})  

    def after_run(self, run_context, run_values):
      # 动态更新学习率
      train_step = run_values.results
      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 60000:
        self._lrn_rate = 0.01
      elif train_step < 80000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001

  # 建立监控Session
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # 禁用默认的SummarySaverHook，save_summaries_steps设置为0
      save_summaries_steps=0, 
      config=tf.ConfigProto(allow_soft_placement=True,gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))) as mon_sess:
    while not mon_sess.should_stop():
#         mon_sess.run(model.train_op)
    # 执行优化训练操作
        for i in range(images.shape[0]//hps.batch_size):
            mon_sess.run(model.train_op,feed_dict={model._images:images[i*hps.batch_size:(i+1)*hps.batch_size],model.labels:labels[i*hps.batch_size:(i+1)*hps.batch_size]})


def evaluate(hps):
  # 构建输入数据
  images, labels = get_Trains_and_Lables(
      FLAGS.eval_data_path,hps.num_classes)
  # 构建残差网络模型
  model = resnet_model.ResNet(hps,FLAGS.image_size, FLAGS.mode)
  model.build_graph()
  # 模型变量存储器
  saver = tf.train.Saver()
  # 总结文件 生成器
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  
  # 执行Session
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  
  # 启动所有队列执行器
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    # 检查checkpoint文件
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
  
    # 读取模型数据(训练期间生成)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    features = sess.run(model.x_feature,feed_dict={model._images:images[0:hps.batch_size]})
    print(features.shape)

    # 逐Batch执行测试
#     total_prediction, correct_prediction = 0, 0
#     for _ in six.moves.range(FLAGS.eval_batch_count):
#       # 执行预测
#       (loss, predictions, truth, train_step) = sess.run(
#           [model.cost, model.predictions,
#            model.labels, model.global_step])
#       # 计算预测结果
#       truth = np.argmax(truth, axis=1)
#       predictions = np.argmax(predictions, axis=1)
#       correct_prediction += np.sum(truth == predictions)
#       total_prediction += predictions.shape[0]

    # 计算准确率
#     precision = 1.0 * correct_prediction / total_prediction
#     best_precision = max(precision, best_precision)
#     print()
#     # 添加准确率总结
#     precision_summ = tf.Summary()
#     precision_summ.value.add(
#         tag='Precision', simple_value=precision)
#     summary_writer.add_summary(precision_summ, train_step)
    
#     # 添加最佳准确总结
#     best_precision_summ = tf.Summary()
#     best_precision_summ.value.add(
#         tag='Best Precision', simple_value=best_precision)
#     summary_writer.add_summary(best_precision_summ, train_step)
    
#     # 添加测试总结
#     #summary_writer.add_summary(summaries, train_step)
    
#     # 打印日志
#     tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
#                     (loss, precision, best_precision))
    
#     # 执行写文件
#     summary_writer.flush()

#     if FLAGS.eval_once:
#       break

#     time.sleep(60)

def get_images_feature(images,nums_gpus=0):
  if len(images.shape)==3:images = images.reshape(1,images.shape[0],images.shape[1],images.shape[2])
  tf.logging.set_verbosity(tf.logging.INFO)
  if nums_gpus == 0:
    dev = '/cpu:0'
  elif nums_gpuss == 1:
    dev = '/gpu:0' 
  hps = resnet_model.HParams(batch_size=images.shape[0],
                             num_classes=50,
                             num_feature=10,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')
  model = resnet_model.ResNet(hps,FLAGS.image_size, FLAGS.mode)
  model.build_graph()
  # 模型变量存储器
  saver = tf.train.Saver()
  # 总结文件 生成器
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  
  # 执行Session
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  try:
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
  except tf.errors.OutOfRangeError as e:
    tf.logging.error('Cannot restore checkpoint: %s', e)
    # continue
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
    # continue
    # 读取模型数据(训练期间生成)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  all_length = images.shape[0]
  i=0
  # while True:
  #   if i>=(all_length/12-1):
  #     break
  #   i+=1
  with tf.device(dev):
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    features = sess.run(model.x_feature,feed_dict={model._images:images[i*hps.batch_size:(i+1)*hps.batch_size]})
  print features.shape
  return features

def get_feature(img_path,nums_gpus=0):
  images, labels = get_Trains_and_Lables(img_path)
  features = get_images_feature(images)
  return features,labels



def main(_):
  # 设备选择
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')
    
  # 执行模式
  # ----- 需要改动的地方 batch_size
  if FLAGS.mode == 'train':
    batch_size = 12*5
  elif FLAGS.mode == 'eval':
    batch_size = 12*1

  # 数据集类别数量
  num_classes = 50

  # 残差网络模型参数
  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             is_gray=True,
                             num_feature=10,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')
  # 执行训练或测试
  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)

if __name__ == '__main__':
  # tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run()
  get_feature('../datas/test_data')



