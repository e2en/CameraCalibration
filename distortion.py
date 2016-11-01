import numpy as np
import cv2
import glob
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

os.chdir("/Users/eren/Desktop/ComputerVision/CameraCalibration")
f = open('dir.txt', 'r')
lines = f.readlines()
os.chdir(lines[0])

types = ('*.jpg', '*.JPG')
images = []
for files in types:
    images.extend(glob.glob(files))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners,ret)
        
        print fname
        cv2.imshow('img',img)
        cv2.waitKey(1000)
               
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1] ,None,None)

img = cv2.imread(fname)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
cv2.imshow('final', dst)

# write parameters
parameterlist=[mtx[0,0], mtx[0,1], mtx[0,2], mtx[1,1], mtx[1,2], dist[0,0], dist[0,1], dist[0,2], dist[0,3], dist[0,4]]
f = open('data.txt','w')
for item in parameterlist:
  f.write("%s\n" % item)
f.close()

cv2.destroyAllWindows()