cpdef list get_ones_fast(short [:, :] image):
    # declare variables
    cdef int x, y, w, h
    cdef list ones
    
    # initialize variables
    h = image.shape[0]
    w = image.shape[1]
    ones = []
    
    # loop over the image
    for y in range(h): 
        for x in range(w):
            # if pixel is 'activated' save it to list
            if(image[y,x] > 0):
                ones.append((y,x))
                
    
    # return the list of activated pixels image
    return ones
