import numpy as np
import scipy
from scipy import misc
from sklearn.svm import SVC
import createdataset as DATASET
from tfmodel import MLP
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm
from sklearn.externals import joblib
import pickle


def tpredictModel(mnist,nsize):
	X=np.load('trainX.npy')
	Y=np.load('trainY.npy')
	#model=MLP(shape = len(X[0]), labels=2)
	model = SVC(C=1000)
	model.fit(X,Y)
	return model

def tpredict(model,I,radius):
	shp=I.shape
	X=[]
	Y=[]
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			N=DATASET.getNBHn(I,i,j,radius)
			x=np.concatenate([np.array([I[i,j]]),N])
			X.append(x)
	Y=model.predict(X)
	It=Y.reshape([64,64])
	scipy.misc.imsave('It.png',It)
	images=[I,It]
	f = pylab.figure()
	for n, fname in enumerate(('1.png', '2.png')):
		arr=np.asarray(It)
		f.add_subplot(1, 2, n+1)  # this line outputs images on top of each other
		# f.add_subplot(1, 2, n)  # this line outputs images side-by-side
		pylab.imshow(images[n],cmap=cm.Greys_r)
		
	pylab.show()

mnist = np.load('/home/ubuntu/workplace/saptarshi/Data/moving_mnist/mnist_binary_ver.npy')


#print DATASET.findRadiusDataSet(mnist[:,0:2,:,:])
#for i in range(0,20):
#	scipy.misc.imsave('./mnist/mnist'+str(i)+'.png',mnist[i,2])
	
model = tpredictModel(mnist[:,0:2,:,:],11)

#joblib.dump(model, 'mlp_model.pkl') 

#model = pickle.load('./tmp/mlp_model.pkl') 


tpredict(model,mnist[0][10],11)

