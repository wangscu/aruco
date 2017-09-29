import cv2
import cv2.aruco as aruco
import yaml
import numpy as np
import transformations as tf

cornerPoints = np.array([[[-87.5, 87.5, 0.0], [-57.5, 87.5, 0.0], [-57.5, 57.5, 0.0], [-87.5, 57.5, 0.0]],
                         [[57.5, 87.5, 0.0], [87.5, 87.5, 0.0], [87.5, 57.5, 0.0], [57.5, 57.5, 0.0]],
                         [[-87.5, -57.5, 0.0], [-57.5, -57.5, 0.0], [-57.5, -87.5, 0.0], [-87.5, -87.5, 0.0]],
                         [[57.5, -57.5, 0.0], [87.5, -57.5, 0.0], [87.5, -87.5, 0.0], [57.5, -87.5, 0.0]]])

dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
dImg = cv2.imread('aruco.jpg')
# dImg = cv2.GaussianBlur(dImg, (5, 5), 0)
# _, dImg = cv2.threshold(dImg, 200, 255, cv2.THRESH_BINARY)

corners, ids, rejectedImgPoints = aruco.detectMarkers(dImg, dict)
dImgd = aruco.drawDetectedMarkers(cv2.imread('aruco.jpg'), corners, ids)
dImgr = aruco.drawDetectedMarkers(cv2.imread('aruco.jpg'), rejectedImgPoints)

# You can use the following 4 lines of code to load the data in file "calibration.yaml"
with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
camera_matrix = np.array(loadeddict.get('camera_matrix'))
dist_coeff = np.array(loadeddict.get('dist_coeff'))

rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, 30, camera_matrix, dist_coeff)

for i in range(0, len(corners)):
    aruco.drawAxis(dImgd, camera_matrix, dist_coeff, rvecs[i], tvecs[i], 40)

    data = {'rvecs': np.asarray(rvecs[i]).tolist(), 'tvecs': np.asarray(tvecs[i]).tolist()}
    with open("rt{}.yaml".format(i + 1), "w") as f:
        yaml.dump(data, f)

""""
imgPoints = np.zeros(shape=(0, 2))
objectPoints = np.zeros(shape=(0, 3))
for i in range(0, len(corners)):
    id = ids[i]
    imgPoints = np.append(imgPoints, corners[i][0], axis=0)
    objectPoints = np.append(objectPoints, cornerPoints[id - 1][0], axis=0)


_retval, _rvec, _tvec = cv2.solvePnP(objectPoints, imgPoints, camera_matrix, dist_coeff)
aruco.drawAxis(dImgd, camera_matrix, dist_coeff, _rvec, _tvec, 10)
data = {'rvec': np.asarray(_rvec).tolist(), 'tvec': np.asarray(_tvec).tolist()}
with open("rt_center.yaml", "w") as f:
    yaml.dump(data, f)
"""

cv2.imwrite('aruco_d.jpg', dImgd)
cv2.imwrite('aruco_r.jpg', dImgr)
