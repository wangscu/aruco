import cv2
import cv2.aruco as aruco

dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

squaresX = 5
squaresY = 7
squareLength = 100
markerLength = 80

retval = aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dict)

width = squaresX * squareLength
height = squaresY * squareLength
board = aruco.drawPlanarBoard(retval, (width, height))

cv2.imwrite('/Users/wangyajun/Downloads/charuco_board.png', board)