import cv2 as cv

filename = r'F:\test\images\season.jpg'
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('source image', img)
cv.imshow('gray', gray)
cv.waitKey()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow("Hue", hsv[:, :, 0])
cv.imshow("Saturation", hsv[:, :, 1])
cv.imshow("Value", hsv[:, :, 2])
cv.waitKey()

cv.imshow("Blue", img[:, :, 0])
cv.imshow("Green", img[:, :, 1])
cv.imshow("Red", img[:, :, 2])

cv.waitKey()
cv.destroyAllWindows()