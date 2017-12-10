import numpy as  np
import random
import matplotlib.pyplot as plt

class sigmoid:
    def act(self,z):
        return 1 / (1 + np.exp(-z))
    def prime(self,z):
        return self.act(z) * (1 - self.act(z))

class relu:
    def act(self,z):
        return np.maximum(z,0)
    def prime(self,z):
        return np.greater(z, 0).astype(int)

class softmax:
    def act(self,z):
        # z is output vector
        return np.exp(z) / np.sum(np.exp(z))

class Network(object):
    def __init__(self,sizes,activator):
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.normal(0,1,(y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
        self.bias = [np.random.normal(0,1,(y,1)) for y in sizes[1:]]
        self.activator = activator
    
    def feedForward(self,a):
        for w, b, actor in zip(self.weights,self.bias,self.activator):
            a = actor.act(np.dot(w,a)+b)
        return a

    def vectorize(self,j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    def cross_entropy(self, a, y):
        return np.sum(np.nan_to_num( -y*np.log(a) - (1-y)*np.log(1-a))) / len(y)

    def cal_loss(self, data, lamda):
        cross_entropy_loss = 0
        cross_entropy_loss += np.sum((self.cross_entropy(self.feedForward(x),self.vectorize(y)) for x,y in data))
        cross_entropy_loss /= len(data) 
        #cross_entropy_loss += 1/2 * lamda / len(data) * np.sum(np.linalg.norm(w)**2 for w in self.weights)
        return cross_entropy_loss
    
    def cal_train_loss(self, data, lamda):
        cross_entropy_loss = 0
        cross_entropy_loss += np.sum((self.cross_entropy(self.feedForward(x),y) for x,y in data))
        cross_entropy_loss /= len(data) 
        #cross_entropy_loss += 1/2 * lamda / len(data) * np.sum(np.linalg.norm(w)**2 for w in self.weights)
        return cross_entropy_loss

    def SGD(self,training_data,epochs,eta,lamda,mini_batch_size,test_data = None):
        if test_data:
            n_test = len(test_data)
        n_training = len(training_data)
        test_loss = []
        train_loss = []
        ep = []
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[n:n+mini_batch_size] for n in range(0,n_training,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta,lamda, n_training)
                if test_data:
                    print("epoch{0}: test_data: {1}/{2}, training_data: {3}/{4}".format(i,self.evaluate(test_data),n_test,self.evaluate_train(training_data),n_training))

                else:
                    print("epoch{0}".format(i))
            test_loss.append(self.cal_loss(test_data, lamda))
            train_loss.append(self.cal_train_loss(training_data, lamda))
            ep.append(i)
        plt.plot(ep,test_loss,'r',label = 'test_loss')
        plt.plot(ep,train_loss,'b',label = 'train_loss')
        plt.legend()
        plt.show()

    def update_mini_batch(self,mini_batch,eta,lamda,n_data):
        weights_batch = [np.zeros(w.shape) for w in self.weights]
        bias_batch = [np.zeros(b.shape) for b in self.bias]

        for x,y in mini_batch:
            delta_weight_batch, delta_bias_batch = self.backpropagate(x,y)
            weights_batch  = [wb + dwb for wb, dwb in zip(weights_batch,delta_weight_batch)]
            bias_batch = [bb + dbb for bb, dbb in zip(bias_batch,delta_bias_batch)]
        self.weights = [w - eta/len(mini_batch)* wb for w, wb in zip(self.weights,weights_batch)]
        self.bias = [b - eta/len(mini_batch)* bb for b, bb in zip(self.bias,bias_batch)]

    def softmax_cross_entropy_gradient(self,output, y):
        grad = np.zeros_like(y)
        for k in range(len(y)):
            g = 0
            for i in range(len(output)):
                if (i == k):
                    g += -(y[i] * (1-output[i]))
                else:
                    g += (y[i] * output[k])
            grad[k] = g
        return grad

    def backpropagate(self,x,y):
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_bias = [np.zeros(b.shape) for b in self.bias]

        activation = x
        zs = []
        activations = [x]

        #feedforward
        for w, b, actor in zip(self.weights, self.bias, self.activator):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = actor.act(z)
            activations.append(activation)

        #propaback 

        # the output layer error
        #error = activations[-1] - y
        error = self.softmax_cross_entropy_gradient(activations[-1],y) 
        delta_bias[-1] = error
        delta_weights[-1] = np.dot(error,activations[-2].transpose())

        for i in range(2,self.num_layer):
            sp = self.activator[-i].prime(zs[-i])
            error = np.dot(self.weights[-i+1].transpose(),error) * sp
            delta_bias[-i] = error
            delta_weights[-i] = np.dot(error,activations[-i-1].transpose())

        return delta_weights, delta_bias

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_train(self, train_data):
        test_results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for (x, y) in train_data]
        return sum(int(x == y) for (x, y) in test_results)

import loader

training_data, validatioin_data, test_data = loader.load_data_wrapper()

sigmoid = sigmoid()
relu = relu()
softmax = softmax()

net = Network([784,100,50,10],[sigmoid,relu,softmax])
net.SGD(training_data,20,1.0,0.1,500,test_data=test_data)

