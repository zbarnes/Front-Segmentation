#!/usr/local/bin/python

import matplotlib.pyplot as plt
import helper as h
import numpy as np
from skimage import morphology, img_as_bool
import cv2
from PIL import Image
import warnings
import time as t

#--------Set Max Pixels to None-----------------
Image.MAX_IMAGE_PIXELS = None
#-----------------------------------------------

print('Starting run')

#img = plt.imread('/home/zbarnes/Pictures/20170108_20170311_01_T2_B8.TIF')
#img = plt.imread('/home/zbarnes/Pictures/20170124_20170311_01_T2_B8.TIF')
img = plt.imread('my.jpeg')


start = t.time()
flt_img = h.anisotropic(img,niter=10)
plt.gray()
plt.imshow(flt_img)
plt.savefig('filtered_img')
print('\nFilter complete')

# Group similar grey levels using 3 clusters
values, labels = h.km_clust(flt_img, 3)

# Create the segmented array from labels and values
img_segm = np.choose(labels, values)
# Reshape the array as the original image
img_segm.shape = img.shape
print('K-means complete')
plt.gray()
plt.imshow(img_segm)
plt.savefig('k-means_img')

img_segm = img_segm.astype('uint8')

# adaptiveThreshold(image,thresh val,threshold algorithm,threshold type, blocksize, constant subtracted from mean)

show = cv2.adaptiveThreshold(img_segm,np.max(img_segm)/2,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,0)
plt.gray()
plt.imshow(show)
plt.savefig('thresh_img.png')
print('\nAdaptive Thresh complete')

#remove_small_objects() does not seem to work if input array is not boolean
small = img_as_bool(show)	

small = morphology.remove_small_objects(small, 75000)
print('\nRemove Small Objects Complete')
plt.gray()
plt.imshow(small)
plt.savefig('img_work.png')
end = t.time()
total = end-start
print('total time was: ' + str(round((total/60),2) + ' minutes'))
plt.show
