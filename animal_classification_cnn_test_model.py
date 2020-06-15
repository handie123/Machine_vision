import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from skimage import io, transform
import tensorflow as tf
import numpy as np
import glob

flower_dict = {0: 'cat', 1: 'dog'}

w = 100
h = 100
c = 3

with tf.compat.v1.Session() as sess:
    data = []
    for i in glob.glob("F:/CV/animal_classification_cnn/animal_photos/test_dataSet/*.jpg"):  # 测试集路径
            # print('reading the images:%s'%(i))
            img = io.imread(i)
            img = transform.resize(img, (w, h))
            data.append(img)

    saver = tf.compat.v1.train.import_meta_graph('F:/CV/animal_classification_cnn/animal_model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('F:/CV/animal_classification_cnn/animal_model/'))

    graph = tf.compat.v1.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    # print(x)
    feed_dict = {x: data}
    # print(feed_dict)

    logits = graph.get_tensor_by_name("logits_eval:0")
    # print(logits)

    classification_result = sess.run(logits,feed_dict)

    # 打印出预测矩阵
    print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("第", i+1, "图预测:"+flower_dict[output[i]])

with tf.Session() as sess:
    # 网络结构写入
    summary_writer = tf.summary.FileWriter('./log/', sess.graph)
    # summary_writer = tf.summary.FileWriter('./log/', tf.get_default_graph())


