import cv2 as cv
import copy
import matplotlib.pyplot as plt

# 打开图像
filename = r'f:/test/images/rice.jpg'
image = cv.imread(filename)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# 大津算法灰度阈值化
thr, bw = cv.threshold(gray, 0, 0xff, cv.THRESH_OTSU)
print('Threshold is :', thr)

# 画出灰度直方图
plt.hist(gray.ravel(), 256, [0, 256])
plt.show()

element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
bw = cv.morphologyEx(bw, cv.MORPH_OPEN, element)

seg = copy.deepcopy(bw)
# 计算轮廓
cnts, hier = cv.findContours(seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

count = 0
# 遍历所有区域，并去除面积过小的
for i in range(len(cnts), 0, -1):
    c = cnts[i-1]
    area = cv.contourArea(c)
    if area < 10:
        continue
    count = count + 1
    print("blob", i, " : ", area)

    # 区域画框并标记
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0xff), 1)
    cv.putText(image, str(count), (x, y), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0xff, 0))

print("米粒数量： ", count)
cv.imshow("源图", image)
cv.imshow("阈值化图", bw)

cv.waitKey()
cv.destroyAllWindows()