
from preprocess_lane_lines import *
from visualize_images import *
import ProcessFrame
import cv2
import pickle

# Create a ProcessFrame object
process_frame = ProcessFrame.ProcessFrame()
def lane_detect_pipeline(image, mtx, dist):
    """Lane detection pipeline"""
    #undistort the image
    undist_img = cv2.undistort(image, mtx, dist, None, mtx)
    thresh_x, thresh_y, thresh_mag, thresh_dir = (20, 100), (20, 100), (30, 100), (0.7, 1.3)
    # Apply sobel ops
    gradx, grady, mag_binary, dir_binary = grad_thresh(image, thresh_x, thresh_y, thresh_mag, thresh_dir)
    combined_grad_binary = np.zeros_like(dir_binary)
    combined_grad_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #Apply color ops
    comb_color = color_threshold(image, h_thresh = (15,40), s_thresh=[200,255], l_thresh=(215,255))
    # Combine grad and color thresh binary images
    comb_grad_color, _ = combine_grad_color_thresh_images(combined_grad_binary, comb_color)
    # Apply perspective transform
    perspective_img, M, Minv = get_perspective_transform(comb_grad_color)
    # Apply sliding windows
    if process_frame.apply_sliding_window_flag:
        process_frame.get_best_fit_using_sliding_window(perspective_img)
    else:
        process_frame.get_best_fit(perspective_img)
    # Return the average best fit dict over last n frames
    process_frame.update_average_best_fit()
    # Draw poly
    result = draw_poly(undist_img, perspective_img, Minv, process_frame.avg_fit_dict)
    left_curve_rad = process_frame.avg_fit_dict['left_curve_rad']
    right_curve_rad = process_frame.avg_fit_dict['right_curve_rad']
    vehicle_offset = process_frame.avg_fit_dict['vehicle_offset']
    # Display lane statistics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of curvature (Left)  = %.2f m' % (left_curve_rad), (10, 40), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature (Right) = %.2f m' % (right_curve_rad), (10, 70), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Vehicle position  = %.2f m off center' % (vehicle_offset),
           (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return result


# load the camera matrix and distortion co-efficients
calib_pickle = pickle.load(open("data/cam_calib_pickle.p", "rb"))
mtx = calib_pickle["mtx"]
dist = calib_pickle["dist"]
process_frame.apply_sliding_window_flag = True

def process_image(frame):
    """Method that is invoked by movie py for every frame in a video"""
    result = lane_detect_pipeline(frame, mtx, dist)# first_frame_flag)
    return result

from moviepy.editor import VideoFileClip
myclip = VideoFileClip('project_video.mp4')#('challenge_video.mp4')
mod_clip = myclip.fl_image(process_image)
mod_clip.write_videofile('project_video_output.mp4', audio=False)





