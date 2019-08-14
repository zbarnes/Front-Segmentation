#!/usr/local/bin/python

import matplotlib.pyplot as plt
import helper
import numpy as np
from skimage import morphology, img_as_bool
from tqdm import tqdm
import cv2
from medpy.filter.smoothing import anisotropic_diffusion
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import warnings


f = open("pics_to_seg.txt", "r")

# initialize image array
img_arr = []

#create array of images
for i, imgs in enumerate(f):
	# remove newline char
	imgs = imgs[:-1]
	# append images to list
	img_arr.append(imgs)

print('Starting run')
for i in tqdm(range(len(img_arr))):
	img = plt.imread(img_arr[i])
	plt.imshow(img)
	plt.savefig('orig_img_iter_' + str(i) + '.png')

	flt_img = helper.anisotropic(img,niter=10)
	plt.imshow(flt_img)
	plt.savefig('anisotropic_iter_' + str(i) + '.png')
	print('\nFilter complete')
		

	# Group similar grey levels using 3 clusters
	values, labels = helper.km_clust(flt_img, 3)

	# Create the segmented array from labels and values
	img_segm = np.choose(labels, values)
	# Reshape the array as the original image
	img_segm.shape = img.shape
	print('\nK-means complete')
	plt.imshow(img_segm)
	plt.savefig('k-means_iter_' + str(i) + '.png')
	img_segm = img_segm.astype('uint8')

	adap_thresh = cv2.adaptiveThreshold(img_segm,np.max(img_segm)/2,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,0)
	plt.imshow(adap_thresh)
	plt.savefig('adap_seg_iter_' + str(i) + '.png')
	print('\nAdaptive Thresh complete')

	adap_thresh = img_as_bool(adap_thresh)
		  
	small_objs_rem = morphology.remove_small_objects(adap_thresh, 75000)
	print('\nRemove Small Objects Complete')
	plt.imshow(small_objs_rem)
	plt.savefig('small_objs_removed_iter_' + str(i) + '.png')

plt.show()

	
