import cv2
import cv2.aruco as aruco
import numpy as np

dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
squaresX = 5
squaresY = 7
squareLength = 100
markerLength = 80

board = aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dict)

dImg = cv2.imread('/Users/wangyajun/Downloads/ca.png')

corners, ids, rejectedImgPoints = aruco.detectMarkers(dImg, dict)
cameraMatrix = np.zeros([3, 3])
distCoeffs = np.zeros([5])
_c = np.array([corners,corners])
_i = np.array([ids, ids])

retval, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharucoExtended(_c, _i, board,
                                                                              (500, 700), cameraMatrix, distCoeffs)
