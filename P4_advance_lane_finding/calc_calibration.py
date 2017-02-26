import numpy as np
import cv2
import glob
import pickle

camera_cal_dir_glob = 'camera_cal/calibration*.jpg'
test_images_dir_glob = 'test_images/test*.jpg'
calibration_mtx_dist_filename = 'dist_pickle.p'
chessboard_size = (9,6)

def calibrate_camera_and_pickle_mtx_dist():
    print("Starting calibration process....")
    # Make a list of calibration images
    images = glob.glob(camera_cal_dir_glob)
    nx,ny  = chessboard_size

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.


    # Step through the list and search for chessboard corner
    for index, filename in enumerate(images):
        image  = cv2.imread(filename)
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
            # plt.imshow(img)

    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size , None, None)

    # Save Distortion matrix and coefficient
    with open(calibration_mtx_dist_filename, 'wb') as f:
        saved_obj = {"mtx": mtx, "dist" : dist}
        pickle.dump(saved_obj, f)

    print("Calibration process complete! [pickled file saved to 'dist_pickle.p']")


if __name__ == '__main__':
    calibrate_camera_and_pickle_mtx_dist()
