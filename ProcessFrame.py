import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

class ProcessFrame():
    def __init__(self):
        self.fit_dict = {}
        self.avg_fit_dict = {}
        # ring buffer to keep track of params for the last 10 frames
        self.result_buffer = deque(maxlen=10)
        self.apply_sliding_window_flag = False
        self.debug = False

    def get_best_fit_using_sliding_window(self, binary_warped):
        """Method that calls histogram computation / sliding windows and returns the best polyfit"""
        #print('sliding window fit')
        left_peaks, right_peaks = self.compute_histogram(binary_warped)
        leftx_base, rightx_base = left_peaks[0], right_peaks[0]
        leftx, lefty, rightx, righty = self.apply_sliding_windows(binary_warped, leftx_base, rightx_base)
        if not self.check_valid_params(leftx, lefty, rightx, righty):
            # Fallback mode - Try alternate peaks in histogram
            leftx_base, rightx_base = left_peaks[1], right_peaks[1]
            leftx, lefty, rightx, righty = self.apply_sliding_windows(binary_warped, leftx_base, rightx_base)
            if not self.check_valid_params(leftx, lefty, rightx, righty):
                leftx_base, rightx_base = left_peaks[2], right_peaks[2]
                leftx, lefty, rightx, righty = self.apply_sliding_windows(binary_warped, leftx_base, rightx_base)
                if not self.check_valid_params(leftx, lefty, rightx, righty):
                    self.apply_sliding_window_flag = True
                    return
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.fit_dict['left_fit'], self.fit_dict['right_fit'] = left_fit, right_fit
        left_curve_rad, right_curve_rad, vehicle_offset = self.compute_lane_statistics(binary_warped)
        self.fit_dict['left_curve_rad'], self.fit_dict['right_curve_rad'] = left_curve_rad, right_curve_rad
        self.fit_dict['vehicle_offset'] = vehicle_offset
        self.result_buffer.append(self.fit_dict.copy())
        self.apply_sliding_window_flag = False
        return

    def check_valid_params(self, leftx, lefty, rightx, righty):
        if leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0:
            return False
        else:
            return True

    def compute_histogram(self, binary_warped):
        """Method that computes the histogram and returns the top 3 peaks"""
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        # Keep track of the top 3 peaks
        left_peaks = np.argsort(histogram[:midpoint])[-3:][::-1]
        right_peaks = np.argsort(histogram[midpoint:])[-3:][::-1] + midpoint
        return left_peaks, right_peaks

    def apply_sliding_windows(self, binary_warped, leftx_base, rightx_base):
        """Method that applies sliding windows on the warped image"""
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current, rightx_current = leftx_base, rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds, right_lane_inds = [], []

        # Step through the windows one by one
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            # Draw the windows on the visualization image
            if self.debug:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if self.debug:
            cv_rgb = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            plt.imshow(cv_rgb)
            #cv2.imshow('Sliding window computation',out_img)
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        if self.debug:
            self.fit_dict['left_lane_inds'] = left_lane_inds
            self.fit_dict['right_lane_inds'] = right_lane_inds

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        return leftx, lefty, rightx, righty

    def get_best_fit(self, binary_warped):
        """Method that computes best fit without applying sliding windows"""
        # print('smart fit')
        left_fit, right_fit = self.fit_dict['left_fit'], self.fit_dict['right_fit']
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        minpix = 5000
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        if len(left_lane_inds.nonzero()[0]) < minpix or len(right_lane_inds.nonzero()[0]) < minpix:
            self.apply_sliding_window_flag = True
            return self.get_best_fit_using_sliding_window(binary_warped)
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if not self.check_valid_params(leftx, lefty, rightx, righty):
            self.apply_sliding_window_flag = True
            return self.get_best_fit_using_sliding_window(binary_warped)
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.fit_dict['left_fit'], self.fit_dict['right_fit'] = left_fit, right_fit
        left_curve_rad, right_curve_rad, vehicle_offset = self.compute_lane_statistics(binary_warped)
        self.fit_dict['left_curve_rad'], self.fit_dict['right_curve_rad'] = left_curve_rad, right_curve_rad
        self.fit_dict['vehicle_offset'] = vehicle_offset
        self.result_buffer.append(self.fit_dict.copy())
        return

    def update_average_best_fit(self):
        """Mthod to compute the running average of the values in the ring buffer"""
        # print(result_buffer)
        total = len(self.result_buffer)
        left_fit, right_fit = np.empty((0, 3)), np.empty((0, 3))
        left_curve_rad, right_curve_rad = [], []
        vehicle_offset = []
        for i in range(total):
            calc_fit_dict = self.result_buffer[i]
            left_fit = np.append(left_fit, [calc_fit_dict['left_fit']], axis=0)
            right_fit = np.append(right_fit, [calc_fit_dict['right_fit']], axis=0)
            left_curve_rad.append(calc_fit_dict['left_curve_rad'])
            right_curve_rad.append(calc_fit_dict['right_curve_rad'])
            vehicle_offset.append(calc_fit_dict['vehicle_offset'])
        self.avg_fit_dict['left_fit'] = np.mean(left_fit, axis=0)
        self.avg_fit_dict['right_fit'] = np.mean(right_fit, axis=0)
        self.avg_fit_dict['left_curve_rad'] = np.mean(left_curve_rad)
        self.avg_fit_dict['right_curve_rad'] = np.mean(right_curve_rad)
        self.avg_fit_dict['vehicle_offset'] = np.mean(vehicle_offset)
        # print(avg_fit_dict)
        return

    def compute_lane_statistics(self, warped):
        """Method that computes radius of curvature for left and right lanes and vehicle offset from lane center"""
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fit, right_fit = self.fit_dict['left_fit'], self.fit_dict['right_fit']
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # compute vehicle offset from center
        image_center_x = (warped.shape[1]/2) * xm_per_pix
        y_eval_m = y_eval * ym_per_pix
        left_fitx = left_fit_cr[0] * y_eval_m ** 2 + left_fit_cr[1] * y_eval_m + left_fit_cr[2]
        right_fitx = right_fit_cr[0] * y_eval_m ** 2 + right_fit_cr[1] * y_eval_m + right_fit_cr[2]
        lane_center_x = (left_fitx + right_fitx)/2.
        vehicle_offset = image_center_x - lane_center_x

        return left_curverad, right_curverad, vehicle_offset


