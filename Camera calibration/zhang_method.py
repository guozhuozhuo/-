import cv2
import numpy as np
import os
from zhang.homography import get_homography
from zhang.intrinsics import get_intrinsics_param
from zhang.extrinsics import get_extrinsics_param
from zhang.distortion import get_distortion
from zhang.refine_all import refinall_all_param

if __name__ == "__main__":

    file_dir = r'./left'
    pic_name = os.listdir(file_dir)

    cross_corners = [9, 6]
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)
    real_coor[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    real_points = []
    real_points_x_y = []
    pic_points = []

    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv2.imread(pic_path)

        # 寻找到棋盘角点
        ret, pic_coor = cv2.findChessboardCorners(pic_data, (cross_corners[0], cross_corners[1]), None)

        if ret:
            # 添加每幅图的对应3D-2D坐标
            pic_coor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)

            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])

    # 求单应矩阵
    H = get_homography(pic_points, real_points_x_y)

    # 求内参
    intrinsics_param = get_intrinsics_param(H)

    # 求对应每幅图外参
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)

    # 畸变矫正
    k = get_distortion(intrinsics_param, extrinsics_param, pic_points, real_points_x_y)

    # 微调所有参数
    [new_intrinsics_param, new_k, new_extrinsics_param] = refinall_all_param(intrinsics_param,
                                                                             k, extrinsics_param, real_points,
                                                                             pic_points)

    print("new_intrinsics_param:\n", new_intrinsics_param)
    print("new_distortion:\n", new_k)
    print("new_extrinsics_param:\n", new_extrinsics_param)