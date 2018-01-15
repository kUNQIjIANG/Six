import tensorflow as tf 
import numpy as np 
from sklearn.utils import shuffle
from collections import Counter
import math
import pickle
import os
import re

class skipGram:
	def __init__(self,text_file,test_word,window,embed_size,batch_size,num_sampled,epochs,top_simi,subsample_thred,analogies):
		self.epochs = epochs
		self.subsamp_thred = subsample_thred
		self.batch_size = batch_size
		self.win_size = window
		self.embed_size = embed_size
		self.num_sampled = num_sampled
		self.top_simi = top_simi
		self.analogies = analogies 
		self.process_data(text_file, test_word, analogies)
		self.get_source_target()
		self.build_graph()
		self.saver = tf.train.Saver()
		self.save_path = './saved_skipGram/sg.ckpt'

	def process_data(self,file_path, test_word, analogies):
		print("data processing ...")
		if os.path.isfile('save_words.pickel'):
			print("loading words...")
			self.words = pickle.load(open('save_words.pickel','rb'))
		else:
			text = open(file_path).read()
			text = text.replace("/n", " ")
			self.text = re.sub('\s+', ' ', text).strip().lower()
			words = self.text.split()
			word_freq = Counter(words)
			subsample = self.subsampling(words)
			self.words = [word for word in words if word_freq[word] > 100 and subsample[word] < self.subsamp_thred]
			pickle.dump(self.words, open("sav_words.pickel", "wb"))
		self.vocab = set(self.words)
		self.vocab_size = len(self.vocab)
		print("vocab_size", self.vocab_size)
		self.w2ind = {w:i for i, w in enumerate(self.vocab)}
		self.ind2w = {i:w for i, w in enumerate(self.vocab)}
		self.test_w_id = [self.w2ind[word] for word in test_word]
		# analogies array:
		# [[1,2,3]
		#  [4,5,6]]
		self.analogies_id = np.array([[self.w2ind[word] for word in group] for group in analogies])
		self.ana_a = self.analogies_id[:,0]
		self.ana_b = self.analogies_id[:,1]
		self.ana_c = self.analogies_id[:,2]
		self.word_ind = [self.w2ind[word] for word in self.words]
		print("data processed completed")

	def subsampling(self, word_list):
		t = 1e-5
		word_freq = Counter(word_list)
		totol_num = len(word_list)
		word_unigram = {w : c / totol_num for w, c in word_freq.items()}
		subsample = {w : 1 - np.sqrt(t / word_unigram[w]) for w in set(word_list)}
		return subsample

	def get_source_target(self):
		print("getting source and target...")
		if os.path.isfile('save_word_source.pickel') and os.path.isfile('save_word_target.pickel'):
			print("loading data...")
			self.source = pickle.load(open("save_word_source.pickel", "rb"))
			self.target = pickle.load(open("save_word_target.pickel", "rb"))
			print("source", len(self.source))
			print("target", len(self.target))
			print("finish loading")
		else:
			print("building records...")
			self.source, self.target = [], []
			for i in range(self.win_size,len(self.word_ind) - self.win_size):
				left = i - self.win_size
				right = i + self.win_size
				self.source.extend([self.word_ind[i]] * (2 * self.win_size))
				tar_ = self.word_ind[left:i] + self.word_ind[i+1:right+1]
				self.target.extend(tar_)
			pickle.dump(self.source, open("sav_word_source.pickel", "wb"))
			pickle.dump(self.target, open("sav_word_target.pickel", "wb"))
		print("source and target getted")
		
	def build_graph(self):
		print("building graph...")
		self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))
		self.output_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
													stddev = 1.0 / math.sqrt(self.embed_size)))
		self.output_bias = tf.Variable(tf.zeros([self.vocab_size]))
		self.train_sources = tf.placeholder(tf.int32, shape = [None])
		self.train_labels = tf.placeholder(tf.int32, shape = [None,1])
		self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_sources)
		self.loss = tf.reduce_mean(
					tf.nn.nce_loss(weights = self.output_weights,
									biases = self.output_bias,
									labels = self.train_labels,
									inputs = self.embed,
									num_sampled = self.num_sampled,
									num_classes = self.vocab_size))
		self.optimizer = tf.train.AdamOptimizer(0.03).minimize(self.loss)
		print("graph built")

	def train(self,sess):
		for i in range(self.epochs):
			print("start training...")
			print("epoch %d" % i)
			print("shuffle...")
			sources_words, targets_words = shuffle(self.source, self.target)
			for num, (input_sources, input_targets) in enumerate(zip(self.next_batch(sources_words, self.batch_size),
																	 self.next_batch(targets_words, self.batch_size))):

				input_targets = np.array(input_targets)[:,np.newaxis]
				_, loss = sess.run([self.optimizer, self.loss], {self.train_sources : input_sources,
								 	 				   			 self.train_labels : input_targets})
				if num % 1000 == 0:
					#print("input source shape", input_sources[:10])
					#print("input source shape", len(input_sources))
					#print("input target shape", input_targets.shape)
					#print("input target shape", input_targets[:10])
					print('Epoch: %d, Step: %d, loss: %.4f' % (i, num, loss))
					#similarity, 
					#test_word = sess.run(self.word_distance())
					#print("test_word", test_word)
					
														
					distant_, analogies_word_id = sess.run(self.analogy_test(self.ana_a, self.ana_b, self.ana_c))
															
					print("distant ", distant_)
					for k in range(len(self.analogies)):
						candidate = analogies_word_id[k]
						neighb = [self.ind2w[ind] for ind in candidate]
						print(self.analogies[k][0] + '-' + self.analogies[k][1] + '+' + self.analogies[k][2] + "-->")
						print(str(neighb))
					"""
					for j in range(len(self.test_w_id)):
						word = self.ind2w[self.test_w_id[j]]
						sim_array = -similarity[j]
						closet = sim_array.argsort()[:self.top_simi]
						neighb = [self.ind2w[ind] for ind in closet]
						print('Closest to %s : %s' % (word, str(neighb)))
					"""
			save_path = self.saver.save(sess,self.save_path)
			print("model saved in %s" % save_path)

	def word_distance(self):
		#norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings),1, keep_dims = True))
		#normed_embeddings = self.embeddings / norm # for cosine similarity
		#norm_emb = tf.nn.l2_normalize(self.embeddings,1)
		unnorm_test_word = tf.nn.embedding_lookup(self.embeddings, self.test_w_id)
		#test_w_embed = tf.nn.embedding_lookup(norm_emb, self.test_w_id)
		#test_w_neibor = tf.matmul(test_w_embed, normed_embeddings, transpose_b = True)
		#test_w_neibor = 0
		return unnorm_test_word

	def analogy_test(self,ana_a, ana_b, ana_c):
		#ana_a = tf.placeholder(dtype=tf.int32) # first word id col
		#ana_b = tf.placeholder(dtype=tf.int32) # second word id col
		#ana_c = tf.placeholder(dtype=tf.int32) # third word id col

		norm_emb = tf.nn.l2_normalize(self.embeddings,1)
		emb_a = tf.gather(norm_emb, ana_a) # with dim: [N,emb_dim] N is num of group
		emb_b = tf.gather(norm_emb, ana_b)
		emb_c = tf.gather(norm_emb, ana_c)

		target = emb_a - emb_b + emb_c

		dist = tf.matmul(target, norm_emb, transpose_b = True) # with dim: [N, vocab_size]

		dist_value, target_id = tf.nn.top_k(dist,5)

		return dist_value, target_id

	def next_batch(self, arr, batch_size):
		for i in  range(0, len(arr), batch_size):
			yield arr[i: i+batch_size]

