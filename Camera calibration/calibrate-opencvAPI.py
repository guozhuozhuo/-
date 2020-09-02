import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
print(criteria)
# 获取标定板角点的位置-标定的选取
objp = np.zeros((6 * 7, 3), np.float32)

objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y(张正友标定法)
                                                   # np.mgrid : 返回多维结构，常见的如2D图形，3D图形
obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("left/*.jpg") #进行相机标定的图片位置(文件和内容必须在同一路径)---要求：标定图片需要使用标定板在不同位置、
                                 #不同角度、不同姿态下拍摄，最少需要3张，以10~20张为宜。标定板需要是黑白相间的矩形构成的棋盘图，
                                 #制作精度要求较高，
for fname in images:
    img = cv2.imread(fname)
    cv2.imshow('img',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #颜色空间转换，(转换的图片，转换成何种格式)

    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None) # 提取角点的信息
                                                                 # 注：需要使用findChessboardCorners函数提取角点，这里的角点专指的是标定板上的内角点，这些角点与标定板的边缘不接触。
    #print("寻找结果：",ret) #ret表示的是是否查询到，corners表示的是提取到的角点信息
    #print("角点信息：",corners.shape)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (7, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.imshow('img', img) #输出图像
        cv2.waitKey(2000) #显示2秒

print(len(img_points))
cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

#输出标定结果
print("ret:", ret)
print("-----------------------------------------------------")
print("mtx（内参数矩阵）:\n", mtx) # 内参数矩阵
print("-----------------------------------------------------")
print("dist（畸变系数）:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("-----------------------------------------------------")
print("rvecs（旋转向量）:\n", rvecs)  # 旋转向量（为什么这么多组？）  # 外参数
print("-----------------------------------------------------")
print("tvecs:（平移向量）\n", tvecs ) # 平移向量（为什么这么多组？）   # 外参数


#畸变校正

img = cv2.imread('left/left12.jpg')
print("---------------------------")
h,w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print(roi)
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.jpg',dst)

img = cv2.imread('calibresult.jpg')#保存矫正之后的图像

#我们可以利用反向投影误差对我们找到的参数的准确性进行估计。
#得到的结果越接近 0 越好。有了内部参数，畸变参数和旋转变换矩阵，我们就可以使用 cv2.projectPoints() 将对象点转换到图像点。
#然后就可以计算变换得到图像与角点检测算法的绝对差了。然后我们计算所有标定图像的误差平均值。
tot_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
    tot_error += error

print("mean error: ", tot_error/len(obj_points))

#reference https://www.cnblogs.com/wenbozhu/p/10697374.html
#reference http://www.woshicver.com/
#reference https://www.cnblogs.com/Undo-self-blog/p/8448500.html !important
#reference https://blog.csdn.net/weixin_43842653/article/details/89288565