# main.py
import time
import numpy as np
import pygame
import cv2

from eye_tracker import EyeTracker
from perspective import OffAxisProjection, ScreenConfig
from renderer import Renderer
from calibration import Calibrator


def main():
    # 固定配置（替代命令行参数）
    FULLSCREEN = False
    SCREEN_WIDTH = 0.258  # 屏幕宽度（米）
    SCREEN_HEIGHT = 0.145  # 屏幕高度（米）
    CAMERA_INDEX = 0  # 摄像头索引
    DEBUG_MODE = False  # 是否开启调试模式
    CALIBRATE = False  # 是否重新标定
    USE_PNP = False  # 是否使用PnP高精度追踪
    SENSITIVITY = 1.0  # 灵敏度系数
    MODEL_PATH = None  # face_landmarker.task 模型路径

    calibration_data = np.load(r'E:\python\PythonProject1\mediapipe_usage\camera_calibration.npz')
    camera_matrix = calibration_data['camera_matrix']
    focal_length_xy = (camera_matrix[0, 0], camera_matrix[1, 1])

    screen_config = ScreenConfig(
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT
    )

    # 初始化追踪器（Tasks API）
    tracker = EyeTracker(
        camera_index=CAMERA_INDEX,
        screen_width_m=screen_config.width,
        screen_height_m=screen_config.height,
        model_path=MODEL_PATH,
        use_pnp=USE_PNP,
        smoothing=True,
        focal_length_xy=focal_length_xy
    )

    # 标定
    config = Calibrator.load_config()
    if CALIBRATE or config is None:
        config = Calibrator.interactive_calibration(tracker)

    projection = OffAxisProjection(screen_config)

    if FULLSCREEN:
        renderer = Renderer(fullscreen=True)
    else:
        renderer = Renderer(window_width=1280, window_height=720)

    print("  移动头部 → 改变3D视角")
    print("G → 切换网格  |  ESC → 退出")

    clock = pygame.time.Clock()
    start_time = time.time()
    sensitivity = SENSITIVITY
    default_eye = (0.0, 0.0, 0.5)

    frame_count = 0
    fps_time = time.time()
    running = True

    while running:
        running = renderer.handle_events()

        eye_pos = tracker.get_eye_position()

        if eye_pos is not None:
            eye_x = (eye_pos.x - config.get('camera_offset_x', 0)) * sensitivity
            eye_y = (eye_pos.y - config.get('camera_offset_y', 0)) * sensitivity
            eye_z = eye_pos.z * config.get('depth_scale', 1.0)
            eye_z = max(0.15, min(eye_z, 2.0))
        else:
            eye_x, eye_y, eye_z = default_eye

        proj_matrix = projection.compute_projection_matrix(eye_x, eye_y, eye_z)
        view_matrix = projection.compute_view_matrix(eye_x, eye_y, eye_z)

        current_time = time.time() - start_time
        renderer.render_frame(
            proj_matrix, view_matrix, current_time,
            screen_config.half_width, screen_config.half_height
        )

        if DEBUG_MODE:
            debug_frame = tracker.get_debug_frame()
            if debug_frame is not None:
                cv2.imshow("Eye Tracking Debug", debug_frame)
                cv2.waitKey(1)

        frame_count += 1
        if time.time() - fps_time > 2.0:
            fps = frame_count / (time.time() - fps_time)
            pygame.display.set_caption(
                f"Bare-Eye 3D | FPS: {fps:.1f} | "
                f"Eye: ({eye_x:.3f}, {eye_y:.3f}, {eye_z:.3f})"
            )
            frame_count = 0
            fps_time = time.time()

        clock.tick(60)

    tracker.release()
    if DEBUG_MODE:
        cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()