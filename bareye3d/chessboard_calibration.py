import cv2
import numpy as np
import os
import glob

# ==============================
# 配置参数
# ==============================
chessboard_size = (7, 7)          # 内角点
square_size = 20.0                # 每格20mm（按你实际打印尺寸修改）
image_dir = "../mediapipe_usage/calibration_images"  # 图像保存目录
os.makedirs(image_dir, exist_ok=True)

# 生成世界坐标（Z=0 平面）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
objp *= square_size


def calculate_fov(camera_matrix, image_size):
    """
    根据相机内参矩阵计算水平和垂直视场角（FOV）

    参数:
        camera_matrix (np.ndarray): 相机内参矩阵 (3x3)
        image_size (tuple): 图像尺寸 (width, height)

    返回:
        fov_h (float): 水平视场角（弧度）
        fov_v (float): 垂直视场角（弧度）
    """
    fx = camera_matrix[0, 0]  # x方向焦距
    fy = camera_matrix[1, 1]  # y方向焦距
    w, h = image_size  # 图像宽度和高度

    # 计算水平和垂直视场角（弧度）
    fov_h = 2 * np.arctan(w / (2 * fx))
    fov_v = 2 * np.arctan(h / (2 * fy))

    return fov_h, fov_v

def capture_images(width=1280, height=720):
    """从摄像头采集并保存带有效角点的图像"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    saved_count = 0
    print("实时采集模式：")
    print("  - 对准棋盘格，确保清晰且角度多样")
    print("  - 按 's' 保存当前帧（仅当检测到角点时）")
    print("  - 按 'q' 退出采集")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 绘制角点（如果找到）
        display_frame = frame.copy()
        if ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, chessboard_size, corners_sub, ret_corners)

        cv2.imshow('Capture - Press s to save (when corners detected), q to quit', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and ret_corners:
            filename = os.path.join(image_dir, f"calib_{saved_count:03d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"保存图像: {filename}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"共保存 {saved_count} 张图像到 '{image_dir}' 目录。")


def calibrate_from_images():
    """从已保存的图像中进行标定"""
    image_paths = sorted(glob.glob(os.path.join(image_dir, "calib_*.png")))
    if not image_paths:
        print(f"'{image_dir}' 目录中没有图像，请先采集！")
        return

    objpoints = []
    imgpoints = []

    print(f"从 {len(image_paths)} 张图像中检测角点...")
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_sub)
            print(f"{os.path.basename(path)}: 角点检测成功")
        else:
            print(f"{os.path.basename(path)}: 未检测到角点（跳过）")

    if len(objpoints) < 5:
        print("有效图像不足（至少需要5张），无法标定！")
        return

    h, w = gray.shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    print("\n相机标定完成！")
    print("相机内参矩阵 K:\n", camera_matrix)
    print("畸变系数 D:", dist_coeffs.ravel())
    fov_h_rad, fov_v_rad = calculate_fov(camera_matrix, (w, h))
    fov_h_deg = np.degrees(fov_h_rad)
    fov_v_deg = np.degrees(fov_v_rad)

    print(f"水平视场角 (FOV_h): {fov_h_deg:.2f}°")
    print(f"垂直视场角 (FOV_v): {fov_v_deg:.2f}°")

    # 保存标定结果
    np.savez('../mediapipe_usage/camera_calibration.npz',
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             chessboard_size=chessboard_size,
             square_size=square_size,
             fov_h_deg=fov_h_deg,
             fov_v_deg=fov_v_deg)

    print("\n逐张图像的重投影误差：")
    errors = []  # 存储每张图像的误差
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
        print(f"图像 {i + 1} ({os.path.basename(image_paths[i])}): 重投影误差 = {error:.4f} 像素")

    average_error = sum(errors) / len(errors)
    print(f"\n平均重投影误差: {average_error:.4f} 像素")

    # 可选：显示一张去畸变效果
    test_img = cv2.imread(image_paths[0])
    h, w = test_img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    cv2.imshow("Original", test_img)
    cv2.imshow("Undistorted", undistorted)
    print("显示第一张图像的去畸变效果（按任意键关闭）")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================
# 主程序：用户选择模式
# ==============================
if __name__ == "__main__":
    print("相机标定工具")
    print("1. 采集新图像（覆盖现有图像）")
    print("2. 使用已有图像进行标定")
    choice = input("请选择 (1/2): ").strip()

    if choice == "1":
        capture_images()
        # 采集完后自动进入标定（可选）
        auto_calib = input("是否立即使用新图像标定？(y/n): ").strip().lower()
        if auto_calib == 'y':
            calibrate_from_images()
    elif choice == "2":
        calibrate_from_images()
    else:
        print("无效输入")

