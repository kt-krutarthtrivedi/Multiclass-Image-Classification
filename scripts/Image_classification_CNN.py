"""
| **@created on:** 02/25/22
| **@author:** Krutarth Trivedi, ktrivedi@wpi.edu 
|
| **Description:**
|   Create a fully-connected Convolution Neural Network to classify images of the fashion items from the 
|   Fashion MNIST Dataset 
| **Tuned and Trained on Google Colab**
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

numberofRandomImagePick = 5

#################################################################
# Fashion MNIST Dataset - Create training, validation, and testing data
#################################################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#################################################################
# Insert TensorFlow code here to complete the tutorial in part 1.
#################################################################

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),  padding='valid', strides=(2,2)))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Softmax())

model.summary()

#################################################################
# Insert TensorFlow code here to *train* the CNN for part 2.
#################################################################

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=5,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

yhat1 = model.predict(x_train[numberofRandomImagePick:numberofRandomImagePick+1,:,:,:])[0]  # Save model's output

#################################################################
# Write a method to extract the weights from the trained
# TensorFlow model. In particular, be *careful* of the fact that
# TensorFlow packs the convolution kernels as KxKx1xF, where
# K is the width of the filter and F is the number of filters.
#################################################################

def convertWeights (model):
    # Extract W1, b1, W2, b2, W3, b3 from model and convert them to HW4 - type.
        
    W1_model, b1_model = (model.layers[0].trainable_variables)
    W2_model, b2_model = (model.layers[4].trainable_variables)
    W3_model, b3_model = (model.layers[6].trainable_variables)

    img_size = 28
    kernel_size = W1_model.shape[0]

    w = np.reshape(W1_model, (W1_model.shape[0], W1_model.shape[1], W1_model.shape[3]))
    W_1 = np.zeros(((img_size - W1_model.shape[0]+1)**2, img_size**2, W1_model.shape[3]))
    
    temp = np.zeros((w.shape[0], img_size - w.shape[1], w.shape[2]))
    w_t = np.concatenate((w, temp), axis=1)
    w_t = w_t.reshape((w_t.shape[0]*w_t.shape[1], w_t.shape[2]))
    w_t = w_t[:w_t.shape[0] - (img_size - kernel_size) , :]
    w_t = w_t.T

    l = w_t.shape[1]
    
    for i in range(w_t.shape[0]):
      k = 0
      for j in range(W_1.shape[0]):
        if ((j>0) and (j%(img_size-kernel_size+1) == 0)):
             k = k + kernel_size - 1
        else:
          k

        W_1[j, k:k+l, i] = w_t[i]
        k +=1

    b_1 = np.reshape(b1_model, (1,b1_model.shape[0]))
    final_b_1 = np.atleast_3d(b_1)
    final_b_1 = np.reshape(final_b_1, (1,1,final_b_1.shape[1]))

    b_2 = b2_model
    b_2 = np.reshape(b_2, (1, b_2.shape[0]))

    b_3 = b3_model
    b_3 = np.reshape(b_3, (1, b_3.shape[0]))

    W1_ = W_1
    b1_ = final_b_1
    W2_ = W2_model
    b2_ = b_2
    W3_ = W3_model
    b3_ = b_3
    
    return W1_, b1_, W2_, b2_, W3_, b3_

#################################################################
# Below here, use numpy code ONLY (i.e., no TensorFlow) to use the
# extracted weights to replicate the output of the TensorFlow model.
#################################################################

#Implement a 2D convolution function,to be used for convolution layer.

def conv2d(W, b, x):
    z_temp = np.zeros((x.shape[0], W.shape[0], x.shape[2]))

    for i in range(x.shape[2]):
      x_temp = x[:, :, i]
      w_temp = W[:, :, i]
      z_temp[:, :, i] = np.dot(x_temp, w_temp.T)
    
    final_z = z_temp + b
    result = np.reshape(final_z, (int(np.sqrt(final_z.shape[1])), int(np.sqrt(final_z.shape[1])), final_z.shape[2]))
    return result

# Implement a fully-connected layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def fullyConnected (W, b, x):
    z = np.dot(x, W) + b
    return z

# Implement a max-pooling layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def maxPool (x, poolingWidth):
    m, n, d = x.shape
    padding_size = 0 if (m%poolingWidth == 0) else (poolingWidth - m%poolingWidth)
    left_pad = right_pad = padding_size // 2
    if (padding_size % 2 != 0) :
        right_pad += 1
    pad_x = np.pad(x, (left_pad, right_pad), 'constant', constant_values=(0, 0))

    maxpool = np.zeros((int(pad_x.shape[0]/poolingWidth), int(pad_x.shape[1]/poolingWidth), d))
    
    for k in range(d):
        for i in range(maxpool.shape[0]):
            for j in range(maxpool.shape[1]):
                maxpool[i][j][k] = np.max(x[i*poolingWidth:poolingWidth*(i+1), j*poolingWidth:poolingWidth*(j+1), k])
    return maxpool

# Implement a softmax function.
def softmax (x):
    return np.exp(x) / np.sum(np.exp(x))

# Implement a ReLU activation function
def relu (x):
    return np.maximum(0, x)

# Load weights from TensorFlow-trained model.
W1, b1, W2, b2, W3, b3 = convertWeights(model)

# Implement the CNN with the same architecture and weights
# as the TensorFlow-trained model but using only numpy.

stride_maxPool = 2

x = x_train[numberofRandomImagePick]          #random image taken from x_train
input_image = x.flatten()
input_image = input_image.reshape(1,input_image.shape[0])
input_image_3d = np.atleast_3d(input_image)
input_image_3d = np.repeat(input_image_3d, 64, axis = 2)

z1 = conv2d(W1, b1, input_image_3d)

maxpool_image = maxPool(z1, stride_maxPool)

h1 = relu(maxpool_image) 

flatten_image = h1.flatten()
flatten_image = np.reshape(flatten_image, (1,flatten_image.shape[0]))

z2 = fullyConnected (W2, b2, flatten_image)
h2 = relu(z2)

z3 = fullyConnected (W3, b3, h2)
yhat2 = softmax(z3)

print(f'''\n\n-----------The Predictive Distribution by TensorFlow\'s Softmax:--------------
{yhat1}\n''')
print(f'''--------------The Predictive Distribution by Our Softmax:-----------------------
{yhat2}''')