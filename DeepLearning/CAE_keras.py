from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, ZeroPadding2D, MaxPooling2D, Cropping2D
from keras.models import Model 
from keras.datasets import mnist
import numpy as np 

# This is a convolutional auto-encoder

# data setup
(x_train,_), (x_test,_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# from (60000,28,28）（10000,28,28) reshape into （60000,784）(10000,784)
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# the dimension of the codes
code_dim = 32

# placeholder for input
input_ = Input(shape = (28,28,1))

# Convolution
input_padding = ZeroPadding2D((2,2))(input_)
h_1 = Conv2D(16, (3,3), padding = 'same', activation = 'relu')(input_padding)
p_1 = MaxPooling2D((2,2))(h_1)
h_2 = Conv2D(8, (3,3), padding = 'same', activation = 'relu')(p_1)
p_2 = MaxPooling2D((2,2))(h_2)
h_3 = Conv2D(8, (3,3), padding = 'same', activation = 'relu')(p_2)
encode = MaxPooling2D((2,2))(h_3)
h_4 = Conv2DTranspose(8, (3,3), padding = 'same', strides = (2,2), activation = 'relu')(encode)
h_5 = Conv2DTranspose(16 , (3,3), padding = 'same', strides = (2,2), activation = 'relu')(h_4)
decode = Conv2DTranspose(1, (3,3), padding = 'same', strides = (2,2), activation = 'sigmoid')(h_5)
decode = Cropping2D((2,2))(decode)

# construct auto-encoder
autoencoder = Model(input_, decode)

# training
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(x_train, x_train,
				epochs = 10,
				batch_size = 256,
				shuffle = True,
				validation_data = (x_test, x_test))

# reconstruction
img_decode = autoencoder.predict(x_test)

# display
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20,4))
for i in range(n):
	# original
	ax = plt.subplot(2, n, i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	#reconstruction
	ax = plt.subplot(2, n, i+1+n)
	plt.imshow(img_decode[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

autoencoder.summary()
plt.show()



#loss: 0.1373 - val_loss: 0.1359

