
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

debug = 0
# compute and pickle object points and image points
# # prepare object points
def compute_camera_calib_distortion_params():
    """Method to compute and save camera calibration params"""
    nx = 9#number of inside corners in x
    ny = 6#number of inside corners in y
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Read a particular image just to get image size (all images in the directory are same size)
    img = cv2.imread('./camera_cal/calibration3.jpg')
    img_size = (img.shape[1], img.shape[0])
    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # #write_name = 'corners_found'+str(idx)+'.jpg'
            # #cv2.imwrite(write_name, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "data/cam_calib_pickle.p", "wb" ) )
    print("Pickling done")

# Note- this method is not required for this project
def corners_unwarp(undist_img, nx, ny):
    """Method to undistort calibration images and draw the chessboard corners"""
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    print("ret is", ret)
    print("num corners", len(corners))

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist_img, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist_img, M, img_size)

    # Return the resulting image and matrix
    return warped, M

if debug:
    compute_camera_calib_distortion_params()
    # Test undistortion on an image
    img = cv2.imread('./camera_cal/calibration3.jpg')
    calib_pickle = pickle.load( open( "data/cam_calib_pickle.p", "rb" ) )
    mtx = calib_pickle["mtx"]
    dist = calib_pickle["dist"]
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imwrite('submission/test_undist.jpg',dst)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig('submission/test_undist.png')
    plt.show()