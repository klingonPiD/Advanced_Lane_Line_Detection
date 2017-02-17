import matplotlib.image as mpimg
from visualize_images import *

debug = 0

def grad_thresh(img, thresh_x=(0,255), thresh_y=(0,255), thresh_mag=(0,255), thresh_dir=(0,np.pi/2)):
    """Mthod to perform edge detection"""
    # 1) Convert to grayscale or desired color channel
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    ch = hls[:, :, 2]
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(ch, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(ch, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    scaled_sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= thresh_x[0]) & (scaled_sobelx <= thresh_x[1])] = 1
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_y[0]) & (scaled_sobely <= thresh_y[1])] = 1
    magbinary = np.zeros_like(scaled_sobel_mag)
    magbinary[(scaled_sobel_mag >= thresh_mag[0]) & (scaled_sobel_mag <= thresh_mag[1])] = 1
    dirbinary = np.zeros_like(dir_sobel)
    dirbinary[(dir_sobel >= thresh_dir[0]) & (dir_sobel <= thresh_dir[1])] = 1
    return sxbinary, sybinary, magbinary, dirbinary

# note - hue values get divided by 2 - for  eg. yellow is 30-80 but maps to 15-40
def color_threshold(img, h_thresh = (15,40), s_thresh=(170, 255), l_thresh=(215,255)):
    """Method to threshold images in HLS color space"""
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Threshold color channel
    # create yellow mask
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    hs_binary = np.zeros_like(s_binary)
    hs_binary[(h_binary > 0) & (s_binary > 0)] = 1
    #create white mask
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    hls_binary = np.zeros_like(l_binary)
    hls_binary[(hs_binary > 0) | (l_binary > 0)] = 1
    #color_binary = np.dstack((np.zeros_like(hls_binary), grad_binary, hls_binary))
    return hls_binary

def combine_grad_color_thresh_images(grad_binary, color_binary):
    """Method to combine edge detected and color thresholded images"""
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    stack_binary = np.dstack((np.zeros_like(color_binary), grad_binary, color_binary))
    comb_binary = np.zeros_like(grad_binary)
    comb_binary[(color_binary == 1) | (grad_binary == 1)] = 1
    return comb_binary, stack_binary

def get_perspective_transform(img):
    """Method to warp/perspective transform an image"""
    x_offset = 300  # offset for dst points
    y_offset = 100
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    # Source points - chosen by visually inspecting the image
    src = np.float32([[567, 466], [718, 466], [1008, 651], [303, 651]])
    # Destination points - must be a rectilinear region
    dst = np.float32([[x_offset, y_offset], [img_size[0] - x_offset, y_offset],
                      [img_size[0] - x_offset, img_size[1] - y_offset],
                      [x_offset, img_size[1] - y_offset]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv

if debug:
    # read image
    # Read in an image and grayscale it
    #image = mpimg.imread('signs_vehicles_xygrad.png')
    #image = mpimg.imread('./test_images/straight_lines1.jpg')
    image = mpimg.imread('./test_images/test6.jpg')

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    thresh_x, thresh_y, thresh_mag, thresh_dir = (20, 100), (20, 100), (30, 100), (0.7, 1.3)
    # Apply sobel ops
    gradx, grady, mag_binary, dir_binary = grad_thresh(image, thresh_x, thresh_y, thresh_mag, thresh_dir)
    show_image(image, gradx, titles=['orig', 'sobel_x'], disp_flag=True)
    show_image(image, grady, titles=['orig', 'sobel_y'], disp_flag=True)
    show_image(image, mag_binary, titles=['orig', 'sobel_mag'], disp_flag=True)
    show_image(image, dir_binary, titles=['orig', 'sobel_dir'], disp_flag=True)

    comb_grad = np.zeros_like(dir_binary)
    comb_grad[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    show_image(image, comb_grad, titles=['orig', 'sobel_combined'], disp_flag=True)

    comb_color = color_threshold(image, h_thresh = (15,40), s_thresh=[200,255], l_thresh=(215,255))
    #show_image(image, comb_color, titles=['orig', 'color thresh'], disp_flag=True)

    comb_grad_color, _ = combine_grad_color_thresh_images(comb_grad, comb_color)
    show_image(image, comb_grad_color, titles=['orig', 'comb grad + color'], disp_flag=True)



