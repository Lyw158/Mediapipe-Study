# renderer.py
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import List, Tuple


class Scene:
    """3D 演示场景"""

    def __init__(self):
        self.rotation_angle = 0
        self.show_grid = False

    def draw_grid(self, size: float = 0.5, divisions: int = 20):
        """绘制地面网格（在屏幕后方）"""
        glColor4f(1, 1, 1, 1)
        glLineWidth(1.0)

        step = size * 2 / divisions
        glBegin(GL_LINES)
        for i in range(divisions + 1):
            x = -size + i * step
            # 网格在 z = -0.3 平面
            glVertex3f(x, -0.1, -0.05)
            glVertex3f(x, -0.1, -size)

            z = -0.05 - i * step / 2
            glVertex3f(-size, -0.1, z)
            glVertex3f(size, -0.1, z)
        glEnd()

    def draw_walls(self, half_w: float, half_h: float, depth: float = 0.5, divisions: int = 7):
        """绘制两侧墙壁"""
        glColor4f(1, 1, 1, 0.7)
        glLineWidth(1.0)

        step_x = half_w * 2 / divisions
        step_y = half_h * 2 / divisions
        step_z = depth / divisions
        glBegin(GL_LINES)

        # 左右墙壁（x方向）
        for i in range(divisions + 1):
            y = -half_h + i * step_y
            x = -half_w + i * step_x
            glVertex3f(-half_w, y, 0)
            glVertex3f(-half_w, y, -depth)

            glVertex3f(half_w, y, 0)
            glVertex3f(half_w, y, -depth)

            glVertex3f(x, -half_h, 0)
            glVertex3f(x, -half_h, -depth)

            glVertex3f(x, half_h, 0)
            glVertex3f(x, half_h, -depth)

            glVertex3f(x, half_h, -depth)
            glVertex3f(x, -half_h, -depth)

            glVertex3f(half_w, y, -depth)
            glVertex3f(-half_w, y, -depth)

        glEnd()

        for i in range(divisions + 1):
            z = -i * step_z
            glBegin(GL_LINE_LOOP)
            glVertex3f(half_w, half_h, z)
            glVertex3f(half_w, -half_h, z)
            glVertex3f(-half_w, -half_h, z)
            glVertex3f(-half_w, half_h, z)
            glEnd()

    def draw_cube(self, x: float, y: float, z: float,
                  size: float, color: Tuple[float, float, float]):
        """绘制彩色立方体"""
        s = size / 2

        vertices = [
            (x - s, y - s, z - s), (x + s, y - s, z - s),
            (x + s, y + s, z - s), (x - s, y + s, z - s),
            (x - s, y - s, z + s), (x + s, y - s, z + s),
            (x + s, y + s, z + s), (x - s, y + s, z + s),
        ]

        faces = [
            (0, 1, 2, 3), (4, 5, 6, 7),  # front, back
            (0, 1, 5, 4), (2, 3, 7, 6),  # bottom, top
            (0, 3, 7, 4), (1, 2, 6, 5),  # left, right
        ]

        face_colors = [
            (color[0], color[1] * 0.8, color[2] * 0.6),
            (color[0] * 0.8, color[1], color[2] * 0.6),
            (color[0] * 0.6, color[1] * 0.6, color[2]),
            (color[0], color[1] * 0.6, color[2] * 0.8),
            (color[0] * 0.7, color[1], color[2] * 0.7),
            (color[0] * 0.9, color[1] * 0.9, color[2]),
        ]

        glBegin(GL_QUADS)
        for i, face in enumerate(faces):
            glColor3f(*face_colors[i])
            for vertex_idx in face:
                glVertex3f(*vertices[vertex_idx])
        glEnd()

        # 边框
        glColor3f(0, 0, 0)
        glLineWidth(1.5)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        glBegin(GL_LINES)
        for e in edges:
            glVertex3f(*vertices[e[0]])
            glVertex3f(*vertices[e[1]])
        glEnd()

    def draw_sphere(self, x: float, y: float, z: float,
                    radius: float, color: Tuple[float, float, float]):
        """绘制球体"""
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)

        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, radius, 32, 32)
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def draw_floating_text_card(self, x, y, z, width, height, color):
        """绘制浮动卡片（模拟UI元素）"""
        hw, hh = width / 2, height / 2
        glColor4f(*color, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(x - hw, y - hh, z)
        glVertex3f(x + hw, y - hh, z)
        glVertex3f(x + hw, y + hh, z)
        glVertex3f(x - hw, y + hh, z)
        glEnd()

        glColor3f(1, 1, 1)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex3f(x - hw, y - hh, z)
        glVertex3f(x + hw, y - hh, z)
        glVertex3f(x + hw, y + hh, z)
        glVertex3f(x - hw, y + hh, z)
        glEnd()

    def draw_demo_scene(self, time: float, screen_half_w: float, screen_half_h: float):
        """
        绘制完整的演示场景

        场景布局：
        - 屏幕平面(z=0)作为"窗户"
        - 物体分布在屏幕前方(z>0)和后方(z<0)
        - 屏幕前方的物体会产生"弹出"效果
        - 屏幕后方的物体会产生"深度"效果
        """

        # 地面网格（屏幕后方）
        if self.show_grid:
            pass
            # self.draw_grid()

        # === 屏幕后方的物体（深度效果）===

        # 旋转的立方体（屏幕后方）
        """angle = time * 30
        glPushMatrix()
        glTranslatef(0, 0, -0.2)
        glRotatef(angle, 0.5, 1, 0.3)
        self.draw_cube(0, 0, 0, 0.06, (0.2, 0.6, 1.0))
        glPopMatrix()"""

        # 后方的小立方体组
        """for i in range(5):
            a = time * 20 + i * 72
            r = 0.12
            cx = r * math.cos(math.radians(a))
            cy = 0.03 * math.sin(time * 2 + i)
            cz = -0.15 - r * abs(math.sin(math.radians(a)))
            self.draw_cube(cx, cy, cz, 0.025,
                           (0.9, 0.3 + i * 0.1, 0.2))"""

        # 远处的球体
        # self.draw_sphere(0.1, 0.05, -0.35, 0.03, (0.2, 0.8, 0.3))
        # self.draw_sphere(-0.08, -0.02, -0.25, 0.02, (0.9, 0.9, 0.1))

        # === 屏幕前方的物体（弹出效果）===

        # 浮动球体（在屏幕前方弹出）
        # bob_y = 0.02 * math.sin(time * 1.5)
        # self.draw_sphere(0, bob_y, 0.08, 0.025, (1.0, 0.3, 0.3))

        # 轨道运动的小球
        """orbit_r = 0.07
        orbit_x = orbit_r * math.cos(time * 1.2)
        orbit_z = 0.05 + orbit_r * math.sin(time * 1.2) * 0.5
        self.draw_sphere(orbit_x, 0.04, orbit_z, 0.015, (0.3, 0.8, 1.0))"""

        # 浮动卡片（UI元素效果）
        """self.draw_floating_text_card(-0.1, 0.05, 0.03, 0.06, 0.04,
                                     (0.2, 0.2, 0.8))
        self.draw_floating_text_card(0.1, -0.03, 0.05, 0.05, 0.03,
                                     (0.8, 0.2, 0.2))"""

        self.draw_cube(0, 0, 0, 0.04, (1, 0, 0))
        self.draw_walls(screen_half_w, screen_half_h)

class Renderer:
    """OpenGL渲染器"""

    def __init__(
            self,
            window_width: int = 1920,
            window_height: int = 1080,
            fullscreen: bool = False
    ):
        self.window_width = window_width
        self.window_height = window_height
        self.fullscreen = fullscreen
        self.scene = Scene()

        self._init_pygame()
        self._init_opengl()

    def _init_pygame(self):
        """初始化 Pygame 窗口"""
        pygame.init()

        flags = DOUBLEBUF | OPENGL
        if self.fullscreen:
            flags |= FULLSCREEN
            info = pygame.display.Info()
            self.window_width = info.current_w
            self.window_height = info.current_h

        pygame.display.set_mode(
            (self.window_width, self.window_height), flags
        )
        pygame.display.set_caption("Bare-Eye 3D - Head Tracking")

    def _init_opengl(self):
        """初始化 OpenGL 状态"""
        glClearColor(0.05, 0.05, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # 简单光照
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

    def set_projection(self, projection_matrix: np.ndarray):
        """设置投影矩阵"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # OpenGL 使用列主序
        glMultMatrixf(projection_matrix.T.flatten())

    def set_view(self, view_matrix: np.ndarray):
        """设置视图矩阵"""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(view_matrix.T.flatten())

    def render_frame(
            self,
            projection: np.ndarray,
            view: np.ndarray,
            time: float,
            screen_half_w: float,
            screen_half_h: float
    ):
        """渲染一帧"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.set_projection(projection)
        self.set_view(view)

        # 绘制场景
        self.scene.draw_demo_scene(time, screen_half_w, screen_half_h)

        pygame.display.flip()

    def handle_events(self) -> bool:
        """处理事件，返回是否继续运行"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                if event.key == K_g:
                    self.scene.show_grid = not self.scene.show_grid
        return True