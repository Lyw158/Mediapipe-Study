# perspective.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ScreenConfig:
    """物理屏幕配置"""
    width: float = 0.344  # 屏幕宽度（米）
    height: float = 0.194  # 屏幕高度（米）

    @property
    def half_width(self) -> float:
        return self.width / 2

    @property
    def half_height(self) -> float:
        return self.height / 2


class OffAxisProjection:
    """
    离轴透视投影（Off-Axis Perspective Projection）

    关键原理：
    - 标准透视投影假设观察者在屏幕正前方中心
    - 离轴投影根据观察者的实际位置调整视锥体
    - 使得屏幕表现得像一扇真实的窗户

    参考论文：Robert Kooima - "Generalized Perspective Projection"
    """

    def __init__(self, screen: ScreenConfig = None):
        self.screen = screen or ScreenConfig()
        self.near = 0.01  # 近裁面（米）
        self.far = 100.0  # 远裁面（米）

    def compute_projection_matrix(
            self,
            eye_x: float,
            eye_y: float,
            eye_z: float
    ) -> np.ndarray:
        """
        计算离轴透视投影矩阵

        原理：
        屏幕四角在眼睛坐标系中的位置决定了视锥体的形状。
        当眼睛不在屏幕正前方中心时，视锥体是不对称的（off-axis）。

        Args:
            eye_x: 眼睛相对屏幕中心的水平位置（米）
            eye_y: 眼睛相对屏幕中心的垂直位置（米）
            eye_z: 眼睛到屏幕的距离（米）

        Returns:
            4x4 投影矩阵（列主序，OpenGL格式）
        """
        if eye_z < 0.01:
            eye_z = 0.01  # 防止除零

        hw = self.screen.half_width
        hh = self.screen.half_height

        # 屏幕边缘相对于眼睛的位置
        # 屏幕在 z=0 平面，眼睛在 z=eye_z
        left = (-hw - eye_x)  # 屏幕左边缘
        right = (hw - eye_x)  # 屏幕右边缘
        bottom = (-hh - eye_y)  # 屏幕下边缘
        top = (hh - eye_y)  # 屏幕上边缘

        # 将视锥体参数缩放到近裁面
        near_over_dist = self.near / eye_z

        l = left * near_over_dist
        r = right * near_over_dist
        b = bottom * near_over_dist
        t = top * near_over_dist

        # 构建 OpenGL glFrustum 矩阵
        # 这是标准的不对称视锥体投影矩阵
        projection = np.zeros((4, 4), dtype=np.float32)

        projection[0, 0] = 2 * self.near / (r - l)
        projection[0, 2] = (r + l) / (r - l)
        projection[1, 1] = 2 * self.near / (t - b)
        projection[1, 2] = (t + b) / (t - b)
        projection[2, 2] = -(self.far + self.near) / (self.far - self.near)
        projection[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        projection[3, 2] = -1.0

        return projection

    def compute_view_matrix(
            self,
            eye_x: float,
            eye_y: float,
            eye_z: float
    ) -> np.ndarray:
        """
        计算视图矩阵

        将世界坐标转换到以眼睛为原点的坐标系
        屏幕位于 z=0 平面
        """
        view = np.eye(4, dtype=np.float32)
        view[0, 3] = -eye_x  # 平移使眼睛在原点
        view[1, 3] = -eye_y
        view[2, 3] = -eye_z
        return view

    def compute_mvp(
            self,
            eye_x: float,
            eye_y: float,
            eye_z: float,
            model: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算完整的 Model-View-Projection 矩阵组

        Returns:
            (projection, view, model) 矩阵元组
        """
        projection = self.compute_projection_matrix(eye_x, eye_y, eye_z)
        view = self.compute_view_matrix(eye_x, eye_y, eye_z)

        if model is None:
            model = np.eye(4, dtype=np.float32)

        return projection, view, model


class GeneralizedProjection(OffAxisProjection):
    """
    广义透视投影（支持屏幕任意朝向）

    基于 Kooima 的方法，使用屏幕三个角点定义屏幕平面
    适用于屏幕倾斜的情况
    """

    def __init__(self, screen: ScreenConfig = None, tilt_angle: float = 0.0):
        """
        Args:
            tilt_angle: 屏幕倾斜角度（度），笔记本屏幕通常 90-135 度
        """
        super().__init__(screen)
        self.tilt_angle = np.radians(tilt_angle)

    def compute_projection_and_view(
            self,
            eye_x: float,
            eye_y: float,
            eye_z: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用屏幕角点方法计算投影和视图矩阵

        屏幕角点（屏幕坐标系）：
        pa = 左下角, pb = 右下角, pc = 左上角
        """
        hw = self.screen.half_width
        hh = self.screen.half_height

        # 考虑屏幕倾斜
        cos_t = np.cos(self.tilt_angle)
        sin_t = np.sin(self.tilt_angle)

        # 屏幕角点（世界坐标）
        pa = np.array([-hw, -hh * cos_t, hh * sin_t])  # 左下
        pb = np.array([hw, -hh * cos_t, hh * sin_t])  # 右下
        pc = np.array([-hw, hh * cos_t, -hh * sin_t])  # 左上

        pe = np.array([eye_x, eye_y, eye_z])  # 眼睛位置

        # 屏幕坐标轴
        vr = pb - pa  # 屏幕右方向
        vu = pc - pa  # 屏幕上方向
        vn = np.cross(vr, vu)  # 屏幕法线

        vr = vr / np.linalg.norm(vr)
        vu = vu / np.linalg.norm(vu)
        vn = vn / np.linalg.norm(vn)

        # 眼睛相对于屏幕角点的向量
        va = pa - pe
        vb = pb - pe
        vc = pc - pe

        # 眼睛到屏幕平面的距离
        d = -np.dot(vn, va)

        if abs(d) < 0.001:
            d = 0.001

        # 视锥体参数
        near_over_d = self.near / d
        l = np.dot(vr, va) * near_over_d
        r = np.dot(vr, vb) * near_over_d
        b = np.dot(vu, va) * near_over_d
        t = np.dot(vu, vc) * near_over_d

        # 投影矩阵
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2 * self.near / (r - l)
        P[0, 2] = (r + l) / (r - l)
        P[1, 1] = 2 * self.near / (t - b)
        P[1, 2] = (t + b) / (t - b)
        P[2, 2] = -(self.far + self.near) / (self.far - self.near)
        P[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        P[3, 2] = -1.0

        # 旋转矩阵（世界坐标 → 屏幕坐标）
        M = np.eye(4, dtype=np.float32)
        M[0, :3] = vr
        M[1, :3] = vu
        M[2, :3] = vn

        # 平移（移动到眼睛位置）
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = -pe[0]
        T[1, 3] = -pe[1]
        T[2, 3] = -pe[2]

        view = M @ T

        return P, view