import numpy as np
import matplotlib.pyplot as plt
import sklearn
from skimage import io

def scale_to_unit_interval(ndar, eps=1e-8):
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0 / (ndar.max() + eps)
	return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),scale_rows_to_unit_interval=True,output_pixel_vals=True):
	assert len(img_shape) == 2
	assert len(tile_shape) == 2
	assert len(tile_spacing) == 2

	out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

	if isinstance(X, tuple):
		assert len(X) == 4

		if output_pixel_vals:
			out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
		else:
			out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

		if output_pixel_vals:
			channel_defaults = [0, 0, 0, 255]
		else:
			channel_defaults = [0., 0., 0., 1.]

		for i in range(4):
			if X[i] is None:
				out_array[:, :, i] = np.zeros(out_shape,dtype='uint8' if output_pixel_vals else out_array.dtype) + channel_defaults[i]
			else:
				out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
		return out_array
	else:
		H, W = img_shape
		Hs, Ws = tile_spacing

      # generate a matrix to store the output
		out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


		for tile_row in range(tile_shape[0]):
			for tile_col in range(tile_shape[1]):
				if tile_row * tile_shape[1] + tile_col < X.shape[0]:
					if scale_rows_to_unit_interval:
						this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
					else:
						this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
						out_array[tile_row * (H+Hs): tile_row * (H + Hs) + H,tile_col * (W+Ws): tile_col * (W + Ws) + W] = this_img * (255 if output_pixel_vals else 1)
		return out_array

def plot_data(X,Y):
	vX=X
	if(len(X[0])>50):
		pca=sklearn.decomposition.PCA(n_components=15)
		pca=pca.fit(X)
		vX=pca.transform(X)
	tsne=sklearn.manifold.TSNE(n_components=2,random_state=0)
	vX=tsne.fit_transform(np.array(vX))

	plt.scatter(vX[:,0],vX[:,1],marker='o',c=Y)
	plt.show()      
	  
def overlay(R,L):
	shp=R.shape
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			flag=True
			if(L[i][j]>0):
				for m in range(-1,2):
					for n in range(-1,2):
						if(i+m>=0 and i+m<shp[0] and j+n>=0 and j+n<shp[1]):
							if(L[i+m][j+n]==0):
								R[i,j,:]=255
								flag=False
								break
	io.imshow(R)
	io.show()


	      
      
      
