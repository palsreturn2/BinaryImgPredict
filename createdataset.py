import numpy as np
import scipy
from scipy import signal
from scipy import misc


def getTopoRing(I,x,y,n):
	N=np.zeros((2*n+1)**2 - (2*n-1)**2)
	shp=I.shape
	c=0
	tx=x-n
	bx=x+n
	ly=y-n
	ry=y+n
	for j in range(ly,ry+1):
		if tx>=0 and tx<shp[0] and j>=0 and j<shp[1]:
			N[c]=I[tx][j]
		c=c+1
		if bx>=0 and bx<shp[0] and j>=0 and j<shp[1]:
			N[c]=I[bx][j]			
		c=c+1
	for i in range(tx+1,bx):
		if ly>=0 and ly<shp[1] and i>=0 and i<shp[0]:
			N[c]=I[i][ly]
		c=c+1
		if ry>=0 and ry<shp[1] and i>=0 and i<shp[0]:
			N[c]=I[i][ry]
		c=c+1
	return N

def findRadius(I,It):
	sizeI=I.shape
	radius=1
	
	change_pts = []
	for i in range(0,sizeI[0]):
		for j in range(0,sizeI[1]):
			if(I[i][j]!=It[i][j]):
				change_pts.append([i,j])
	
	while((2*radius+1)<min(sizeI)):
		inp=False
		print 'Starting with ',radius
		for p in change_pts:
			for q in change_pts:
				if(p!=q):
					Rp=getNBHn(I,p[0],p[1],2*radius+1)
					Rq=getNBHn(I,q[0],q[1],2*radius+1)
					if((Rp==Rq).all()):
						inp=True
						radius=radius+1
						break
			if(inp):
				break
		if(not inp):
			return radius		
	if(inp):
		print 'Not possible to predict'
		exit(1)

def findRadiusDataSet(dset):
	shp=dset.shape
	rad=1
	for j in range(0,shp[1]):
		for i in range(0,shp[0]-1):
			print 'Frames', j, i
			I=dset[i][j]
			It=dset[i+1][j]
			rad = max(rad,findRadius(I,It))
	return rad

def getNBHn(I,x,y,n):
	N=np.zeros(n*n-1)
	shp=I.shape
	c=0
	for i in range(x-(n/2),x+(n/2)+1):
		for j in range(y-(n/2),y+(n/2)+1):
			if i!=x or j!=y:
				if i>=0 and j>=0 and i<shp[0] and j<shp[1]:
					N[c]=I[i,j]
				c=c+1
	return N

def gendataset(I,It,radius,L,Y):
	ne=0
	sizeI=I.shape
	for i in range(0,sizeI[0]):
		for j in range(0,sizeI[1]):
			N=getNBHn(I,i,j,radius)
			D=np.concatenate([np.array([I[i,j]]),N])			
			flag=True
			for k in range(0,len(L)):
				if((L[k]==D).all()):
					flag=False 
					break				
			if(flag):
				L.append(list(D))				
				Y.append(It[i,j])
	return L,Y

def genFullDataset(dset,radius):
	L=list()
	Y=list()
	shp=dset.shape
	print 'Generating Dataset'
	for j in range(0,shp[1]):
		for i in range(0,shp[0]-1):
			print j,i
			I=dset[i,j]
			It=dset[i+1,j]
			L,Y = gendataset(I,It,radius,L,Y)
			print 'Dataset size: ', len(Y)
	print 'Dataset Generation Complete'
	print 'Dataset size: ',len(Y)
	print 'Feature Vector Dimension: ',len(L[0])
	return L,Y
