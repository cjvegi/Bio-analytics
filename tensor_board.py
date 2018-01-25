# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:13:17 2018

@author: Khan
"""

# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PATH = os.getcwd()

LOG_DIR = PATH + '/mnist-tensorboard/log-1/'
metadata = os.path.join(LOG_DIR, 'df_labels.tsv')

df = pd.read_csv("scaled.csv",index_col =0)
df = df.drop(['cluster_2','cluster_3'], axis = 1)
pca = PCA(n_components=300,
         random_state = 123,
         svd_solver = 'auto'
         )

df_pca = pd.DataFrame(pca.fit_transform(df))

mnist = df_pca.values #sabaaa can i run the code?


#Feature Scaling
'''
sc_X = StandardScaler()
mnist = sc_X.fit_transform(mnist)
'''
#mnist = np.asarray((mnist))
images = tf.Variable(mnist)
#def save_metadata(file):
'''
with open(metadata, 'w') as metadata_file:
    for row in range(10000):
        c = np.nonzero(mnist.test.labels[::1])[1:][0][row]
        metadata_file.write('{}\n'.format(c))
'''


with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config) 


#  tensorboard --logdir=C:\Users\Khan\Desktop\Files\mnist-tensorboard/log-1 --port=6006
