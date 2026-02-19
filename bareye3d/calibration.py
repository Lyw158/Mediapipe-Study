# calibration.py
import cv2
import numpy as np
import json
from pathlib import Path


class Calibrator:
    """
    摄像头-屏幕标定工具

    标定内容：
    1. 摄像头相对于屏幕中心的位置偏移
    2. 摄像头焦距（可选，使用棋盘格）
    3. 用户瞳距（提高深度估算精度）
    """

    CONFIG_FILE = "calibration.json"

    @staticmethod
    def interactive_calibration(tracker) -> dict:
        """
        交互式标定流程

        步骤：
        1. 用户将脸置于屏幕正前方已知距离处
        2. 系统记录此时的测量值并计算修正参数
        """
        print("\n=== 裸眼3D 标定 ===")
        print("请坐在屏幕正前方，距离约 50cm")
        print("按 'c' 开始标定，按 'q' 跳过")

        config = {
            "camera_offset_x": 0.0,
            "camera_offset_y": 0.01,  # 默认摄像头在屏幕上方约1cm
            "depth_scale": 1.0,
            "user_ipd": 0.063,
        }

        import time
        samples = []
        collecting = False

        while True:
            frame = tracker.get_debug_frame()
            if frame is not None:
                pos = tracker.last_position

                status = "Collecting..." if collecting else "Press 'c' to calibrate"
                cv2.putText(frame, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if collecting and pos:
                    samples.append((pos.x, pos.y, pos.z))
                    cv2.putText(frame, f"Samples: {len(samples)}/30",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)

                    if len(samples) >= 30:
                        # 计算平均值
                        avg = np.mean(samples, axis=0)
                        # 正前方50cm处，x和y应该接近0
                        config["camera_offset_x"] = float(avg[0])
                        config["camera_offset_y"] = float(avg[1])
                        config["depth_scale"] = 0.5 / float(avg[2])

                        print(f"\n标定完成:")
                        print(f"  X偏移: {config['camera_offset_x']:.4f}m")
                        print(f"  Y偏移: {config['camera_offset_y']:.4f}m")
                        print(f"  深度缩放: {config['depth_scale']:.4f}")

                        Calibrator.save_config(config)
                        break

                cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                collecting = True
                samples = []
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        return config

    @staticmethod
    def save_config(config: dict):
        with open(Calibrator.CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"配置已保存到 {Calibrator.CONFIG_FILE}")

    @staticmethod
    def load_config() -> dict:
        path = Path(Calibrator.CONFIG_FILE)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None