#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline
#plt.ion()

# Dashboard class
class Dashboard:
    """ A place holder for the telemetry of the vehicle.
    Currently this contains a set of lane lines."""
    lm = 0.0
    lb = 0.0
    rm = 0.0
    rb = 0.0
    minly = 0.0
    minry = 0.0
    rtic = 0
    ltic = 0

# Global static dashboard
dash = Dashboard()


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_fitted_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #right_lines = []
    #left_lines = []
    rx = []
    ry = []
    lx = []
    ly = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1) > 0):
                #right_lines.append(line)
                rx.append(x1)
                ry.append(y1)
                rx.append(x2)
                ry.append(y2)
            else:
                #left_lines.append(line)
                lx.append(x1)
                ly.append(y1)
                lx.append(x2)
                ly.append(y2)
            cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
    
    #right_lines = np.array(right_lines)
    #left_lines = np.array(left_lines)

    # Fit and draw right lane
    if len(rx) > 2:
        rz = np.polyfit(rx,ry,1)
        m = rz[0]
        b = rz[1]
        yy1 = img.shape[0]
        yy2 = min(ry)
        xx1 = int((yy1-b)/m)
        xx2 = int((yy2-b)/m)
        #if (xx1 > 0) and (xx1 < img.shape[1]):
        cv2.line(img, (xx1, yy1), (xx2, yy2), color, thickness)

    # Fit and draw left lane
    if len(lx) > 2:
        lz = np.polyfit(lx,ly,1)
        m = lz[0]
        b = lz[1]
        yy1 = img.shape[0]
        yy2 = min(ly)
        xx1 = int((yy1-b)/m)
        xx2 = int((yy2-b)/m)
        #if (xx1 > 0) and (xx1 < img.shape[1]):
        cv2.line(img, (xx1, yy1), (xx2, yy2), color, thickness)

def dist_point_line(x1, y1, x2, y2, px, py):
    return abs((y2-y1)*px-(x2-x1)*py+x2*y1-y2*x1)/pow((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1), 0.5)

def line_sse(x1, y1, x2, y2, points_x, points_y):
    zipped = zip(points_x, points_y)
    sse = 0.0
    for x, y in zipped:
        sse += pow(dist_point_line(x1, y1, x2, y2, x, y), 2.0)
    return sse

def best_line(points_x, points_y):
    zipped = zip(points_x, points_y)
    best_sse = 1e6
    best_line = []
    list(zipped)
    for x1,y1,x2,y2 in zipped:
        sse = line_sse(x1, y1, x2, y2, points_x, points_y)
        if (sse < best_sse):
            best_sse = sse
            best_line = [x1, y1, x2, y2]
    return best_line

def get_best_line(m, b, miny, tic, points_x, points_y):
    if len(points_x) >= 2:
        x1,y1,x2,y2 = best_line(points_x, points_y)
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        miny = min(points_y)
    elif (tic == 0):
        return (0.0, 0.0, 0.0)
    return (m, b, miny)

def fit_filter_line(m, b, miny, tic, rate, points_x, points_y):
    if len(points_x) >= 2:
        z = np.polyfit(points_x,points_y,1)
        if (tic == 0):
            m = z[0]
            b = z[1]
            miny = min(points_y)
        else:
            m = m * (1.0-rate) + z[0] * rate
            b = b * (1.0-rate) + z[1] * rate
            miny = min(points_y)#miny * (1.0-rate) +  min(points_y) * rate
    elif (tic == 0):
        return (0.0, 0.0, 0.0)
    return (m, b, miny)


def draw_filtered_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    rate = 0.1 # The filter rate
    rx = []
    ry = []
    lx = []
    ly = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1) > 0):
                rx.append(x1)
                ry.append(y1)
                rx.append(x2)
                ry.append(y2)
            else:
                lx.append(x1)
                ly.append(y1)
                lx.append(x2)
                ly.append(y2)
            # draw the detected hough lines green
            cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
    
    # Fit, Filter and draw right lane
    #(dash.rm, dash.rb, dash.minry) = fit_filter_line(dash.rm, dash.rb, dash.minry, dash.rtic, rate, rx, ry)
    (dash.rm, dash.rb, dash.minry) = get_best_line(dash.rm, dash.rb, dash.minry, dash.rtic, rx, ry)
    if (dash.rm != 0.0):
        m = dash.rm
        b = dash.rb
        yy1 = img.shape[0]
        yy2 = int(dash.minry)
        xx1 = int((yy1-b)/m)
        xx2 = int((yy2-b)/m)
        #if (xx1 > 0) and (xx1 < img.shape[1]):
        cv2.line(img, (xx1, yy1), (xx2, yy2), color, thickness)
        dash.rtic += 1

    # Fit, Filter and draw left lane
    (dash.lm, dash.lb, dash.minly) = fit_filter_line(dash.lm, dash.lb, dash.minly, dash.ltic, rate, lx, ly)
    print(lx,ly)
    cv2.line
    if (dash.lm != 0.0):
        m = dash.lm
        b = dash.lb
        minly = dash.minly
        yy1 = img.shape[0]
        yy2 = int(minly)
        xx1 = int((yy1-b)/m)
        xx2 = int((yy2-b)/m)
        #if (xx1 > 0) and (xx1 < img.shape[1]):
        cv2.line(img, (xx1, yy1), (xx2, yy2), color, thickness)
        dash.ltic += 1

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_filtered_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


import os
dirs = os.listdir("test_images_1/")
#explicitly force the printing of dirs when running from command line
print(dirs)

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

# Image pipeline
for filename in dirs:

    # Read in and grayscale the image
    image = mpimg.imread("test_images_1/" + filename)

    gray = grayscale(image)
    dash = Dashboard()

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create a masked edges image
    imshape = edges.shape
    #ylow = 0.625
    #xlow1 = 0.48
    #xlow2 = 0.53 
    ylow = 0.6
    xlow1 = 0.45
    xlow2 = 0.58
    vertices = np.array([[(0,imshape[0]),(imshape[1]*xlow1, imshape[0]*ylow), (imshape[1]*xlow2, imshape[0]*ylow), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Detect Hough lines
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #30 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Save weighted lane lines image
    final_img = weighted_img(line_img, image)

    cv2.line(final_img, (0, imshape[0]), (int(imshape[1]*xlow1), int(imshape[0]*ylow)), [0, 0, 255], 2)
    cv2.line(final_img, (int(imshape[1]*xlow2), int(imshape[0]*ylow)), (imshape[1],imshape[0]), [0, 0, 255], 2)

    mpimg.imsave("test_images_output/" + filename, final_img, format='jpg')
    plt.imshow(final_img)
    #plt.show(block=True)


#don't close the plot when running from command line
#plt.show(block=True)

#Part 2

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):

    gray = grayscale(image)
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create a masked edges image
    imshape = edges.shape
    #ylow = 0.625
    #xlow1 = 0.48
    #xlow2 = 0.53 
    ylow = 0.6
    xlow1 = 0.43
    xlow2 = 0.58
    vertices = np.array([[(0,imshape[0]),(imshape[1]*xlow1, imshape[0]*ylow), (imshape[1]*xlow2, imshape[0]*ylow), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Detect Hough lines
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Return weighted lane lines image
    final_img = weighted_img(line_img, image)
    cv2.line(final_img, (0, imshape[0]), (int(imshape[1]*xlow1), int(imshape[0]*ylow)), [0, 0, 255], 2)
    cv2.line(final_img, (int(imshape[1]*xlow2), int(imshape[0]*ylow)), (imshape[1],imshape[0]), [0, 0, 255], 2)
    return final_img

if 0:
    dash = Dashboard()
    white_output = 'test_videos_output/challenge.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/challenge.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    #%time white_clip.write_videofile(white_output, audio=False)
    white_clip.write_videofile(white_output, audio=False)

if 0:
    dash = Dashboard()
    white_output = 'test_videos_output/solidYellowLeft.mp4'
    clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

if 0:
    dash = Dashboard()
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)