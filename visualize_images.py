
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_image(orig_img, grad_img, titles=['Original Image', 'Transformed Image'], disp_flag = False):
    """Method to display pairs of images"""
    # Plot the result
    if disp_flag:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        f.tight_layout()
        ax1.imshow(orig_img)
        ax1.set_title(titles[0], fontsize=25)
        ax2.imshow(grad_img, cmap='gray')
        ax2.set_title(titles[1], fontsize=25)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()

def draw_poly(undist, warped, Minv, fit_dict, show_fig = False):
    """Method to color the region between detected lanes"""
    left_fit, right_fit = fit_dict['left_fit'], fit_dict['right_fit']
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (206, 57, 174))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    if show_fig:
        plt.imshow(result)
        plt.show()
    return result

def show_line_fit(binary_warped, fit_dict):
    """Method to show the fitted lane lines"""
    left_fit, right_fit = fit_dict['left_fit'], fit_dict['right_fit']
    left_lane_inds, right_lane_inds = fit_dict['left_lane_inds'], fit_dict['right_lane_inds']
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #out_img *= 1/np.max(out_img)
    cv_rgb = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    #plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()