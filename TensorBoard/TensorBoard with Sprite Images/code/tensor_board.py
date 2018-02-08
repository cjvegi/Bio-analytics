import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


PATH = os.getcwd()

LOG_DIR = PATH + '/bio-tensorboard/log-1/'
metadata = os.path.join(LOG_DIR, 'df_labels.tsv')

df = pd.read_csv("scaled.csv",index_col =0)
df = df.drop(['cluster_2','cluster_3'], axis = 1)
pca = PCA(n_components=300,
         random_state = 123,
         svd_solver = 'auto'
         )

df_pca = pd.DataFrame(pca.fit_transform(df))

bio = df_pca.values 

## Tensorflow Variable
images = tf.Variable(bio)


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
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_32classes.png')
    embedding.sprite.single_image_dim.extend([32, 32]) 
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config) 


#  tensorboard --logdir=C:\Users\Name\Desktop\Files\bio-tensorboard/log-1 --port=6006
