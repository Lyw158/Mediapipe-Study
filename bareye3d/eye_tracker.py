# eye_tracker.py
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import urllib.request


@dataclass
class EyePosition:
    """眼睛在物理空间中的位置（单位：米，相对于屏幕中心）"""
    x: float   # 水平方向，右为正
    y: float   # 垂直方向，上为正
    z: float   # 深度方向，屏幕外为正
    left_eye: Tuple[float, float, float] = (0, 0, 0)
    right_eye: Tuple[float, float, float] = (0, 0, 0)
    confidence: float = 0.0


class ModelDownloader:
    """自动下载 MediaPipe 模型"""

    MODELS = {
        "face_landmarker": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            "filename": "face_landmarker.task"
        }
    }

    @staticmethod
    def ensure_model(model_name: str = "face_landmarker") -> str:
        info = ModelDownloader.MODELS[model_name]
        path = info["filename"]

        if not os.path.exists(path):
            print(f"下载 {model_name} 模型...")
            urllib.request.urlretrieve(info["url"], path)
            print(f"已保存到 {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

        return path


class EyeTracker:
    """
    基于 MediaPipe Tasks FaceLandmarker 的人眼3D位置追踪器

    使用 Tasks API (非旧版 Solutions)
    检测 478 个面部关键点（含虹膜），估算眼睛相对屏幕的3D位置
    """

    # FaceLandmarker 关键点索引（与旧版一致）
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # PnP 用的关键点
    FACE_2D_INDICES = [1, 152, 33, 263, 61, 291, 10]

    # 人脸平均尺寸参考（米）
    AVG_INTER_PUPIL_DIST = 0.063

    # 标准面部3D模型点（PnP用，单位：米）
    FACE_3D_MODEL = np.array([
        [0.0, 0.0, 0.0],            # 鼻尖 (1)
        [0.0, -0.0636, -0.0125],     # 下巴 (152)
        [-0.0435, 0.0327, -0.026],   # 左眼左角 (33)
        [0.0435, 0.0327, -0.026],    # 右眼右角 (263)
        [-0.0289, -0.0289, -0.0241], # 嘴左角 (61)
        [0.0289, -0.0289, -0.0241],  # 嘴右角 (291)
        [0.0, 0.0672, -0.0127],      # 额头 (10)
    ], dtype=np.float64)

    def __init__(
        self,
        camera_index: int = 0,
        camera_resolution: Tuple[int, int] = (1280, 720),
        camera_fov_h: float = 60.0,
        screen_width_m: float = 0.344,
        screen_height_m: float = 0.194,
        camera_offset: Tuple[float, float] = (0.0, 0.01),
        smoothing: bool = True,
        model_path: str = None,
        use_pnp: bool = False,
        focal_length_xy: Optional[Tuple[float, float]] = None
    ):
        self.screen_width_m = screen_width_m
        self.screen_height_m = screen_height_m
        self.camera_offset = camera_offset
        self.smoothing = smoothing
        self.use_pnp = use_pnp

        # ========== 初始化摄像头 ==========
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 估算摄像头内参
        if focal_length_xy:
            self.focal_length_px = focal_length_xy[0]
            self.focal_length_py = focal_length_xy[1]
        else:
            self.camera_fov_h = np.radians(camera_fov_h)
            self.focal_length_px = self.frame_width / (2 * np.tan(self.camera_fov_h / 2))
            self.focal_length_py = self.frame_width / (2 * np.tan(self.camera_fov_h / 2))

        cx, cy = self.frame_width / 2, self.frame_height / 2
        self.camera_matrix = np.array([
            [self.focal_length_px, 0, cx],
            [0, self.focal_length_py, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(4)

        # ========== 初始化 MediaPipe Tasks FaceLandmarker ==========
        if model_path is None:
            model_path = ModelDownloader.ensure_model("face_landmarker")

        base_options = mp_tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=self.use_pnp  # PnP模式可获取变换矩阵
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

        # ========== Kalman 滤波器 ==========
        if smoothing:
            self.kf = self._create_kalman_filter()
            self.kf_initialized = False

        self.last_position: Optional[EyePosition] = None
        self._last_frame: Optional[np.ndarray] = None

    def _create_kalman_filter(self) -> KalmanFilter:
        """3D位置追踪的卡尔曼滤波器"""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = 1.0 / 30

        kf.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        kf.Q = np.eye(6) * 0.005
        kf.Q[3:, 3:] *= 2
        kf.R = np.diag([0.001, 0.001, 0.005])
        kf.P *= 0.1

        return kf

    def _detect_landmarks(self, frame: np.ndarray):
        """
        使用 Tasks API 检测面部关键点

        关键区别：
        - Tasks API 使用 mp.Image 而非直接传 numpy array
        - VIDEO 模式需要递增的 timestamp_ms
        - 返回结构是 FaceLandmarkerResult 而非 SolutionOutputs
        """
        # 转为 RGB（MediaPipe Tasks 需要 RGB）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建 MediaPipe Image 对象
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # 递增时间戳（VIDEO 模式要求严格递增）
        self._timestamp_ms += 33  # ~30fps

        # 检测
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        return result

    def _estimate_depth(self, landmarks: list) -> float:
        """通过瞳距估算深度"""
        left_iris = landmarks[self.LEFT_IRIS_CENTER]
        right_iris = landmarks[self.RIGHT_IRIS_CENTER]

        left_px = np.array([
            left_iris.x * self.frame_width,
            left_iris.y * self.frame_height
        ])
        right_px = np.array([
            right_iris.x * self.frame_width,
            right_iris.y * self.frame_height
        ])

        diff = left_px - right_px
        dx_norm = diff[0] / self.focal_length_px
        dy_norm = diff[1] / self.focal_length_py
        angular_dist = np.sqrt(dx_norm ** 2 + dy_norm ** 2)

        if angular_dist < 1e-2:
            return 0.6

        depth = self.AVG_INTER_PUPIL_DIST / angular_dist
        return float(np.clip(depth, 0.15, 2.0))

    def _estimate_position_basic(self, landmarks: list) -> EyePosition:
        """基础位置估算（虹膜中心 + 深度反投影）"""
        left_iris = landmarks[self.LEFT_IRIS_CENTER]
        right_iris = landmarks[self.RIGHT_IRIS_CENTER]

        left_px = np.array([
            left_iris.x * self.frame_width,
            left_iris.y * self.frame_height
        ])
        right_px = np.array([
            right_iris.x * self.frame_width,
            right_iris.y * self.frame_height
        ])

        center_px = (left_px + right_px) / 2
        z = self._estimate_depth(landmarks)

        cx = self.frame_width / 2
        cy = self.frame_height / 2

        x_cam = (center_px[0] - cx) * z / self.focal_length_px
        y_cam = (center_px[1] - cy) * z / self.focal_length_py

        x_screen = -x_cam - self.camera_offset[0]
        y_screen = -y_cam + self.camera_offset[1]

        # 左右眼独立
        left_x = -(left_px[0] - cx) * z / self.focal_length_px - self.camera_offset[0]
        left_y = -(left_px[1] - cy) * z / self.focal_length_py + self.camera_offset[1]
        right_x = -(right_px[0] - cx) * z / self.focal_length_px - self.camera_offset[0]
        right_y = -(right_px[1] - cy) * z / self.focal_length_py + self.camera_offset[1]

        return EyePosition(
            x=x_screen, y=y_screen, z=z,
            left_eye=(left_x, left_y, z),
            right_eye=(right_x, right_y, z),
            confidence=1.0
        )

    def _estimate_position_pnp(self, landmarks: list) -> EyePosition:
        """PnP 高精度位置估算"""
        points_2d = []
        for idx in self.FACE_2D_INDICES:
            lm = landmarks[idx]
            points_2d.append([
                lm.x * self.frame_width,
                lm.y * self.frame_height
            ])
        points_2d = np.array(points_2d, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.FACE_3D_MODEL,
            points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return self._estimate_position_basic(landmarks)

        rmat, _ = cv2.Rodrigues(rvec)

        left_eye_local = np.array([-0.032, 0.032, 0.0])
        right_eye_local = np.array([0.032, 0.032, 0.0])

        left_eye_cam = rmat @ left_eye_local + tvec.flatten()
        right_eye_cam = rmat @ right_eye_local + tvec.flatten()
        center_cam = (left_eye_cam + right_eye_cam) / 2

        x_screen = -center_cam[0] - self.camera_offset[0]
        y_screen = -center_cam[1] + self.camera_offset[1]
        z_screen = float(np.clip(center_cam[2], 0.15, 2.0))

        return EyePosition(
            x=float(x_screen),
            y=float(y_screen),
            z=z_screen,
            left_eye=(
                float(-left_eye_cam[0]),
                float(-left_eye_cam[1]),
                float(left_eye_cam[2])
            ),
            right_eye=(
                float(-right_eye_cam[0]),
                float(-right_eye_cam[1]),
                float(right_eye_cam[2])
            ),
            confidence=1.0
        )

    def get_eye_position(self) -> Optional[EyePosition]:
        """获取当前眼睛位置（主接口）"""
        ret, frame = self.cap.read()
        if not ret:
            return self.last_position

        self._last_frame = frame.copy()

        # ===== Tasks API 检测 =====
        result = self._detect_landmarks(frame)

        # Tasks API 返回结构：result.face_landmarks 是 list[list[NormalizedLandmark]]
        if not result.face_landmarks:
            return self.last_position

        # 获取第一张脸的关键点列表
        landmarks = result.face_landmarks[0]

        # 检查是否有虹膜点（需要478个点）
        if len(landmarks) < 478:
            print(f"警告：仅检测到 {len(landmarks)} 个关键点，需要478个（含虹膜）")
            return self.last_position

        # 估算位置
        if self.use_pnp:
            raw_position = self._estimate_position_pnp(landmarks)
        else:
            raw_position = self._estimate_position_basic(landmarks)

        # Kalman 滤波平滑
        if self.smoothing:
            measurement = np.array([
                raw_position.x, raw_position.y, raw_position.z
            ])

            if not self.kf_initialized:
                self.kf.x[:3] = measurement.reshape(3, 1)
                self.kf_initialized = True

            self.kf.predict()
            self.kf.update(measurement)

            smoothed = self.kf.x[:3].flatten()
            raw_position.x = float(smoothed[0])
            raw_position.y = float(smoothed[1])
            raw_position.z = float(smoothed[2])

        self.last_position = raw_position
        return raw_position

    def get_debug_frame(self) -> Optional[np.ndarray]:
        """获取带标注的调试画面"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        result = self._detect_landmarks(frame)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # 绘制虹膜
            if len(landmarks) >= 478:
                for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                    lm = landmarks[idx]
                    px = int(lm.x * self.frame_width)
                    py = int(lm.y * self.frame_height)
                    cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

                # 绘制PnP参考点
                for idx in self.FACE_2D_INDICES:
                    lm = landmarks[idx]
                    px = int(lm.x * self.frame_width)
                    py = int(lm.y * self.frame_height)
                    cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

            if self.last_position:
                pos = self.last_position
                cv2.putText(
                    frame,
                    f"X:{pos.x:+.3f}m Y:{pos.y:+.3f}m Z:{pos.z:.3f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Landmarks: {len(landmarks)} | Conf: {pos.confidence:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )

        return frame

    def release(self):
        """释放资源"""
        self.cap.release()
        self.landmarker.close()