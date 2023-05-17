# import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

colors = [[0,0,0], [255,0,0], [255,165,0], [255,255,0], [0,0,255]]
cv2.setRNGSeed(0)

def watershed(src,src_kmeans,m,kernel_size,morph,morph_i,dilate_i,dist_thd,
              open_i,bg_dilate,color_i):
    """Calculates class outlines and draws them on the RGB image

    Args:
        src: Source image to draw on
        src_kmeans: Segmented src image for calculating markers with watershed
        m: Mask of a class
        kernel_size: Size of morphology kernel
        morph: Type of morphology
        morhp_i: Number of times morph should be applied
        dilate_i: Number of times dilation should be applied
        dist_thd: Threshold of distance function
        open_i: Number of times open should be applied
        bg_dilate: Number of times dilate should be applied to get background
        color_i: Chosen outline color
    
    Returns:
        None
    """
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    
    #Using morphologies to remove outliers from class mask
    morphed = cv2.morphologyEx(m,morph,kernel, iterations = morph_i)
    morphed = cv2.morphologyEx(morphed,cv2.MORPH_DILATE,kernel,
                               iterations = dilate_i)
    dist_tf = cv2.distanceTransform(morphed,cv2.DIST_L2,5)
    _, foreground = cv2.threshold(dist_tf,dist_thd*dist_tf.max(),255,0)
    foreground = np.uint8(foreground)
    
    #Foreground contains the pixels that are most likely inclass pixels
    foreground = cv2.morphologyEx(foreground,cv2.MORPH_OPEN,kernel,
                                  iterations = open_i)
    
    #Unknown region
    background = cv2.dilate(foreground,kernel,iterations=bg_dilate)
    unknown = cv2.subtract(background,foreground)

    #Labelling
    _, markers = cv2.connectedComponents(foreground)
    
    #Add to all labels, so background is 1 instead of 0
    markers = markers+1

    #Unknown is labelled 0
    markers[unknown==255] = 0

    #Use watershed to get contour, draw it
    markers = cv2.watershed(src_kmeans,markers)
    src[markers == -1] = colors[color_i]

def filter_bg(src):
    """Background filtering with 3. channel

    Args:
        src: Source image (BGR) to filter
    
    Returns:
        result_v: Blurred and filtered image (HSV)
    """
    #Median blur to make image more homogenous
    src = cv2.medianBlur(src,15)

    #Filter white background from HSV image
    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    v = src[:,:,2]
    mask_v = cv2.inRange(v, 0, 205)
    result_v = cv2.bitwise_and(src_hsv, src_hsv, mask=mask_v)
    return result_v

def change_masks(src,masks,m):
    """Change mask

    Args:
        src: Source image (BGR)
        masks: List of class masks
        m: Index of mask to change

    Returns:
        masks: List of class masks after change
    """
    #Filter out noise from mask
    src_m = cv2.bitwise_and(src, src, mask=masks[m])
    b = src_m[:,:,0]
    masks[m] = cv2.inRange(b, 1, 95)
    return masks

def kmeans(src_filt):
    """Segmenting with kmeans clustering
    
    Args:
        src_filt: Image to segment

    Returns:
        src_kmeans: Segmented image
        masks: List of class masks
    """
    #Convert to useable format
    data = src_filt.reshape((-1,3))
    data = np.float32(data)

    #Parameters for kmeans
    K = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,label,center=cv2.kmeans(data,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)

    #Rebuild image
    clustered = center[label.flatten()]
    src_kmeans = clustered.reshape((src_filt.shape))
    
    #Use sorted V values for masks
    value = src_kmeans[:,:,2]
    center = center[center[:, 2].argsort()]
    masks = []
    for c in center:
        mask = cv2.inRange(value, int(c[2]), int(c[2]))
        masks.append(mask)
    return src_kmeans,masks

def segment(in_dir,out_dir):
    """Control function for the segmenting process

    Args:
        in_dir: Path to input image
        out_dir: Path to output image
    
    """
    #Read in image (BGR)
    src = cv2.imread(in_dir)

    #Background filtered image (HSV)
    src_filt = filter_bg(src)

    #Segmented image (HSV), binary masks
    src_kmeans,masks = kmeans(src_filt)
    masks = change_masks(src,masks,2)

    #Change input to RGB to draw on
    cv2.cvtColor(src =src, code =cv2.COLOR_BGR2RGB, dst =src)
    
    #Call watershed on classes
    watershed(src,src_kmeans,masks[1],5,cv2.MORPH_OPEN,4,0,0.1,0,5,1)
    watershed(src,src_kmeans,masks[2],5,cv2.MORPH_OPEN,4,0,0.05,1,5,2)
    watershed(src,src_kmeans,masks[3],3,cv2.MORPH_OPEN,2,10,0.1,0,7,3)
    watershed(src,src_kmeans,masks[4],3,cv2.MORPH_OPEN,5,0,0.15,0,7,4)
    
    #Write to output
    cv2.imwrite(out_dir, cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    
    #Show result
    plt.imshow(src)
    plt.show()

if __name__ == "__main__":
    in_dir = "path/to/input"
    out_dir = "path/to/output"
    segment(in_dir,out_dir)
