import numpy as np 

sample = [1,1,1,-1,-1,-1,1,1,1,-1]



class Adaboost(object):
	def __init__(self):
		self.classifiers = [self.classify_1,self.classify_2,self.classify_3]
		self.weights = [0] * len(self.classifiers)

	def classify_1(self,input):
		if input < 2.5:
			return 1
		else:
			return -1

	def classify_2(self,input):
		if input < 8.5:
			return 1
		else:
			return -1
	def classify_3(self,input):
		if input < 5.5:
			return 1
		else:
			return -1

	def adaboost(self,sample):
		error_weight = [1/len(sample)] * 10
		for classify in self.classifiers:
			result = [classify(i) for i in range(10)]
			error = self.error_rate(sample,result,error_weight)
			alpha = np.log((1-error)/error) / 2
			error_weight = self.updata_error_weight(error_weight,sample,result,alpha)
			self.weights[self.classifiers.index(classify)] = alpha
			print("correct{0}".format(self.evaluate(sample)))

	def predict(self,input):
		r = 0
		for i in range(len(self.classifiers)):
			r += self.classifiers[i](input) * self.weights[i]
		return self.sigmoid(r)


	def evaluate(self,sample):
		test_result = [(self.predict(x),y) for x,y in zip([x for x in range(10)], sample)]
		return sum(int(x==y) for (x,y) in test_result)


	def sigmoid(self,input):
		
		if input < 0:
			return -1
		else:
			return 1
		
	def updata_error_weight(self,error_weight,sample,result,alpha):
		z = 0 
		for i in range(len(error_weight)):
			z += error_weight[i] * np.exp(-alpha*sample[i]*result[i])
		for i in range(len(error_weight)):
			error_weight[i] = error_weight[i] / z * np.exp(-alpha*sample[i]*result[i])
		return error_weight



	def error_rate(self,y,y_,error_weight): 
		e = 0
		for i in range(len(y)):
			if y[i] != y_[i]:
				e += error_weight[i]
		return e 
	

ada = Adaboost()
ada.adaboost(sample)
