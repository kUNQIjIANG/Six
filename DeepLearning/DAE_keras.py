from keras.layers import Input, Dense
from keras.models import Model 
from keras.datasets import mnist
import numpy as np 

# This is a deep auto-encoder

# data setup
(x_train,_), (x_test,_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# from (60000,28,28）（10000,28,28) reshape into （60000,784）(10000,784)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_train.shape[1:])))

# the dimension of the codes
code_dim = 32

# placeholder for input
input_ = Input(shape = (784,))

# encode layer
encode = Dense(128, activation = 'relu')(input_)
encode = Dense(64, activation = 'relu')(encode)
encode = Dense(code_dim, activation = 'relu')(encode)

# encode layer
decode = Dense(64, activation = 'relu')(encode)
decode = Dense(128, activation = 'relu')(decode)
decode = Dense(784, activation = 'sigmoid')(decode)

# construct auto-encoder
autoencoder = Model(input_, decode)

# training
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(x_train, x_train,
				epochs = 50,
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
	#plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	#reconstruction
	ax = plt.subplot(2, n, i+1+n)
	plt.imshow(img_decode[i].reshape(28,28))
	#plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()

#loss: 0.1142 - val_loss: 0.1143

