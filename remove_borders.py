import matplotlib.pyplot as plt
import cv2
import pickle
import cython
import skimage
import numpy as np
from scipy.spatial import distance
from skimage import morphology, img_as_bool, img_as_int
from PIL import Image
import get_ones_cy

def get_points(img):
    """
    Gets the four points of the four corner points 
    of the border edges and uses the fact that the 
    image is a closed convex set and thus extrema 
    will occur at corner points.
    
    Parameters
    ----------
    input: image represented as np array
    
    output: the coordinates of the 4 corner points
    """
    
    
    im = img_as_int(img).astype('int16')
    actv_lst = get_ones_cy.get_ones_fast(im)
    
    p2 = min(actv_lst)
    p4 = max(actv_lst)
    
    
    img_to_flip  = Image.fromarray(im)
    transposed  = img_to_flip.transpose(Image.ROTATE_90)
    rotated = np.array(transposed)
    rotated = img_as_int(rotated)
    other_lst = get_ones_cy.get_ones_fast(rotated)
    
    p1 = max(other_lst)
    p1 = switch_pair(p1,img)
    
    p3 = min(other_lst)
    p3 = switch_pair(p3,img)

    return p1,p2,p3,p4

def getEquidistantPoints(p1, p2, step_size):
    return zip(np.linspace(p1[0], p2[0], step_size+1), np.linspace(p1[1], p2[1], step_size+1))


def remove_stuff(img, pts_arr):
   """
   This function will set the pixels with coordinates in the pts_arr to 0.
   
   Params
   -----------------
   input: img - np array representing an image
          pts_arr - an array of coordinate points to be set to 0
          
   output: img with specified pixels set to 0
   """

    for i, pts in enumerate(pts_arr):
        
        # a bunch of try statements in case we go out of bounds no exceptions are thrown
        
        try: img[pts[0]][pts[1]] = 0
        except: continue
        try: img[pts[0]-1][pts[1]-1] = 0
        except: continue                
        try:img[pts[0]-2][pts[1]-2] = 0
        except: continue
        try: img[pts[0]+1][pts[1]+1] = 0
        except: continue
        try:img[pts[0]+2][pts[1]+2] = 0
        except: continue
            
                
    return img

def switch_pair(point, img):
    """
    Translates the observed coordinates in the rotated image
    back to the basis of the non-rotated image
    
    Params
    -----------------
    input: point - the ordered pair to be switched
    
           img - a np array representing an image
 
    """
    
    return (point[1],(img.shape[1]-1)- point[0])


def get_line(p1,p2,step_size):
    pts = getEquidistantPoints(p1, p2, step_size)
    pts_lst = list(pts)
    pts_arr = np.asarray(pts_lst, dtype='int16')
    return pts_arr 

def get_distance(img):
    p1, p2, p3, p4 = get_points(img.astype('int16'))

    # Get distance rounded down to the nearest int
    dist_1 = distance.euclidean(p1, p2)
    dist_1 = np.floor(dist_1).astype('int')

    dist_2 = distance.euclidean(p2, p3)
    dist_2 = np.floor(dist_2).astype('int')

    dist_3 = distance.euclidean(p3, p4)
    dist_3 = np.floor(dist_3).astype('int')

    dist_4 = distance.euclidean(p4, p1)
    dist_4 = np.floor(dist_4).astype('int')

    return (dist_1, dist_2,dist_3, dist_4)

def remove_borders(img, Lines):
    for line in Lines:
        img = remove_stuff(img, line)
    inverse = img_as_bool(rem_im)
    morph = morphology.remove_small_objects(inverse, min_size=75000)
    
    return morph