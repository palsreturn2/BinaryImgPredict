import tensorflow as tf
import numpy as np
import indices as INDICES
from sklearn.preprocessing import OneHotEncoder

#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class MLP:
	def __init__(self,shape,labels,bs=100,reg_param = 0.05, use_saved_model = False, save_model = True):
		self.input_shape=shape
		self.out_labels=labels
		self.batch_size=bs
				
				
		self.X = tf.placeholder(tf.float32, [None, self.input_shape])
		w = tf.Variable(tf.random_normal([self.input_shape, 100]))
		w1 = tf.Variable(tf.random_normal([100, 50]))
		w2 = tf.Variable(tf.random_normal([50, 10]))
		wf = tf.Variable(tf.random_normal([10,self.out_labels]))
		b = tf.Variable(tf.random_normal([self.out_labels]))
		
		
		l1=tf.tanh(tf.matmul(self.X,w))
		l2=tf.tanh(tf.matmul(l1,w1))
		l3=tf.tanh(tf.matmul(l2,w2))		
		py = tf.nn.softmax(tf.matmul(l3, wf) + b)
		self.Y = tf.placeholder(tf.float32, [None, self.out_labels])
		
		reg = reg_param * (tf.nn.l2_loss(w)+tf.nn.l2_loss(wf))
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(py), reduction_indices=[1])) + reg
		self.train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(cross_entropy)
		self.predict_step = tf.argmax(py,1)
		
		correct_prediction = tf.equal(self.predict_step, tf.argmax(self.Y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		initialize = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(initialize)
		'''if(not use_saved_model):
			saver = tf.train.saver()
			with tf.session() as self.sess:
				saver.restore(self.sess,'/tmp/model.ckpt')
				print 'Model Restored'''
	
	def fit(self,trX,trY):
		trX=np.array(trX)
		val_size=int(0.1*trX.shape[0])
		
		#vY=np.array(trY[tr_size:trX.shape[0]])-1
		enc=OneHotEncoder()
		y=np.array([i+1 for i in range(self.out_labels)]).reshape((self.out_labels,1))
		
		E=enc.fit_transform(y).toarray()
		trY=np.array([E[int(l)] for l in trY])	
		
		for i in range(500):
			#print 'Epoch ',i
			for j in range(0,trX.shape[0],self.batch_size):
				batch_xs = trX[j:min(j+self.batch_size,trX.shape[0])]
				batch_ys = trY[j:min(j+self.batch_size,trX.shape[0])]
				self.sess.run(self.train_step, feed_dict = {self.X : batch_xs, self.Y : batch_ys})		
			print self.sess.run(self.accuracy, feed_dict = {self.X : trX[0:val_size], self.Y : trY[0:val_size]})
		'''if(save_model):
			save_path = saver.save(self.sess, "/tmp/model.ckpt")
			print 'Model saved at ',save_path'''
		return
	
	def predict(self,teX):
		teX=np.array(teX)
		y_pred=self.sess.run(self.predict_step, feed_dict = {self.X : teX})
		return y_pred

class ConvNet:
	def __init__(self,shape,labels,bs = 100, reg_param = 0):
		self.input_shape=shape
		self.out_labels=labels
		self.batch_size=bs		
		
		self.X = tf.placeholder(tf.float32, self.input_shape)
		self.Y = tf.placeholder(tf.float32, [None,self.out_labels])
		
		filter_height=3
		filter_width=3
		maxpool_height = 2
		maxpool_width = 2
		w1=tf.Variable(tf.random_normal([filter_height,filter_width,shape[3],10]))

		out_height = self.input_shape[1] - filter_height + 1
		out_width  = self.input_shape[2] - filter_width + 1

		w2=tf.Variable(tf.random_normal([filter_height,filter_width,10,20]))
		
		out_height = out_height - filter_height + 1
		out_width = out_width - filter_width + 1
		
		wf=tf.Variable(tf.random_normal([20*out_height*out_width,self.out_labels]))
		b=tf.Variable(tf.random_normal([self.out_labels]))
		
		l1 = tf.nn.sigmoid(tf.nn.conv2d(self.X,w1,strides=[1,1,1,1],padding='VALID'))
		
		l2 = tf.nn.sigmoid(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='VALID'))
		
		py = tf.matmul(tf.reshape(l2,shape=[-1,out_height*out_width*20]),wf)+b
		
		#self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(py+0.0001), reduction_indices=[1]))
		reg = reg_param * (tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(wf))
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py,self.Y)) + reg
		self.train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(self.cross_entropy)
		self.predict_step = tf.argmax(py,1)
		
		correct_prediction = tf.equal(self.predict_step, tf.argmax(self.Y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		initialize = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(initialize)
	
	def fit(self,trX,trY):
		trX=np.array(trX)
		trY=np.array(trY)
		val_size=int(0.1*trX.shape[0])
		trX=trX.reshape((trX.shape[0],self.input_shape[3],self.input_shape[1],self.input_shape[2]))
		trX=DATASET.change_dataformat(trX)
		enc=OneHotEncoder()
		y=np.array([i+1 for i in range(self.out_labels)]).reshape((self.out_labels,1))
		E=enc.fit_transform(y).toarray()
		trY=np.array([E[int(y)-1] for y in trY])
		for i in range(100):
			#print 'Epoch ',i
			for j in range(val_size,trX.shape[0],self.batch_size):
				batch_xs = trX[j:min(j+self.batch_size,trX.shape[0])]
				batch_ys = trY[j:min(j+self.batch_size,trX.shape[0])]
				#print self.sess.run(self.cross_entropy, feed_dict={self.X:batch_xs,self.Y: batch_ys})
				self.sess.run(self.train_step, feed_dict = {self.X : batch_xs, self.Y : batch_ys})		
			print self.sess.run(self.accuracy, feed_dict = {self.X : trX[0:val_size], self.Y : trY[0:val_size]})
		return
	
	def predict(self,teX):
		teX=np.array(teX)
		teX=teX.reshape((teX.shape[0],self.input_shape[3],self.input_shape[1],self.input_shape[2]))
		teX=DATASET.change_dataformat(teX)
		y_pred=self.sess.run(self.predict_step, feed_dict = {self.X : teX})
		return y_pred+1

class DeepConvNet:
	def __init__(self,shape,labels,bs=100,reg_param=10):
		self.input_shape=shape
		self.out_labels=labels
		self.batch_size=bs		
		
		self.X = tf.placeholder(tf.float32, self.input_shape)
		self.Y = tf.placeholder(tf.float32, [None,self.out_labels])
		
		filter_height=3
		filter_width=3
		maxpool_height = 2
		maxpool_width = 2
		w1=tf.Variable(tf.random_normal([filter_height,filter_width,shape[3],10]))

		out_height = self.input_shape[1] - filter_height + 1
		out_width  = self.input_shape[2] - filter_width + 1

		w2=tf.Variable(tf.random_normal([filter_height,filter_width,10,20]))
		
		out_height = out_height - filter_height + 1
		out_width = out_width - filter_width + 1
		
		w3=tf.Variable(tf.random_normal([filter_height,filter_width,20,30]))
		
		out_height = out_height - filter_height + 1
		out_width = out_width - filter_width + 1
		
		w4=tf.Variable(tf.random_normal([filter_height,filter_width,30,40]))
		
		out_height = out_height - filter_height + 1
		out_width = out_width - filter_width + 1
		
		wf=tf.Variable(tf.random_normal([40*out_height*out_width,self.out_labels]))
		b=tf.Variable(tf.random_normal([self.out_labels]))
		
		l1 = tf.nn.sigmoid(tf.nn.conv2d(self.X,w1,strides=[1,1,1,1],padding='VALID'))
		
		l2 = tf.nn.sigmoid(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='VALID'))
		
		l3 = tf.nn.sigmoid(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='VALID'))
		
		l4 = tf.nn.sigmoid(tf.nn.conv2d(l3,w4,strides=[1,1,1,1],padding='VALID'))
		
		py = tf.matmul(tf.reshape(l4,shape=[-1,out_height*out_width*40]),wf)+b
		
		#self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(py+0.0001), reduction_indices=[1]))
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py,self.Y))
		self.train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(self.cross_entropy)
		self.predict_step = tf.argmax(py,1)
		
		correct_prediction = tf.equal(self.predict_step, tf.argmax(self.Y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		initialize = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(initialize)
	
	def fit(self,trX,trY):
		trX=np.array(trX)
		trY=np.array(trY)
		val_size=int(0.1*trX.shape[0])
		trX=trX.reshape((trX.shape[0],self.input_shape[3],self.input_shape[1],self.input_shape[2]))
		trX=DATASET.change_dataformat(trX)
		enc=OneHotEncoder()
		y=np.array([i+1 for i in range(self.out_labels)]).reshape((self.out_labels,1))
		E=enc.fit_transform(y).toarray()
		trY=np.array([E[int(y)-1] for y in trY])
		for i in range(100):
			#print 'Epoch ',i
			for j in range(val_size,trX.shape[0],self.batch_size):
				batch_xs = trX[j:min(j+self.batch_size,trX.shape[0])]
				batch_ys = trY[j:min(j+self.batch_size,trX.shape[0])]
				#print self.sess.run(self.cross_entropy, feed_dict={self.X:batch_xs,self.Y: batch_ys})
				self.sess.run(self.train_step, feed_dict = {self.X : batch_xs, self.Y : batch_ys})		
			print self.sess.run(self.accuracy, feed_dict = {self.X : trX[0:val_size], self.Y : trY[0:val_size]})
		return
	
	def predict(self,teX):
		teX=np.array(teX)
		teX=teX.reshape((teX.shape[0],self.input_shape[3],self.input_shape[1],self.input_shape[2]))
		teX=DATASET.change_dataformat(teX)
		y_pred=self.sess.run(self.predict_step, feed_dict = {self.X : teX})
		return y_pred+1

'''	
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))'''

