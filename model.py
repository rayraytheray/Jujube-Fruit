import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Simple MLP regression model with single output to be used on the generated embeddings and other data
class FundingModel(tf.keras.Model):

    def __init__(self, embedding_dim, additional_dim):
        super(FundingModel, self).__init__()
        #self.optimizer = tf.keras.optimizers.Adam()

        self.dense_1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(32, activation='relu')
        #single output within range 0-1 (normalized) so we dont't get huge values
        self.out_layer = tf.keras.layers.Dense(1, activation='sigmoid') 

    def call(self, inputs):
        transformer_embeddings, additional_data = inputs
        x = tf.concat([transformer_embeddings, additional_data], axis=1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        output = self.out_layer(x)
        return output

    
    #should not need custom train/test functions -- should be okay to compile and fit when using
    #additionally need to be able to normalize the output
    