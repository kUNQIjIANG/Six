import tensorflow as tf 
import os
from skipGram import skipGram

with tf.Session() as sess:
	txt_path = 'pg3207.txt'
	test_words = ["china","beijing","mother","technology","love"]
	test_ana = [["china","usa","washington"],["uk","italy","roma"]]
	sG = skipGram(text_file = txt_path,
				  test_word = test_words,
				  window = 5,
				  embed_size = 300, 
				  batch_size = 1000,
				  num_sampled = 500,
				  epochs = 3,
				  top_simi = 5,
				  subsample_thred = 0.8,
				  analogies = test_ana)

	if os.path.isdir('saved_skipGram'):
		print("loading the model")
		sG.saver.restore(sess, sG.save_path)
		sG.train(sess)
	else:
		print("initialing...")
		sess.run(tf.global_variables_initializer())
		sG.train(sess)
