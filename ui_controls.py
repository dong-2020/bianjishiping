from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QSlider, 
                           QLabel, QGroupBox, QPushButton, QComboBox)
from PyQt6.QtCore import Qt

class UIControls:
    @staticmethod
    def create_slider(name, minimum, maximum, default, parent=None):
        """创建滑块控件"""
        layout = QVBoxLayout()
        label = QLabel(name)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(default)
        layout.addWidget(label)
        layout.addWidget(slider)
        if parent:
            parent.addLayout(layout)
        return slider
    
    @staticmethod
    def create_main_video_controls():
        """创建主视频控制组"""
        group = QGroupBox("主视频控制")
        layout = QVBoxLayout()
        
        controls = {}
        
        # 主视频缩放控制
        controls['main_scale'] = UIControls.create_slider(
            "主视频缩放 (60-90%)", 60, 90, 80, layout
        )
        
        # 圆角控制
        controls['corner'] = UIControls.create_slider(
            "圆角半径 (0-50)", 0, 50, 0, layout
        )
        
        # 边框宽度
        controls['border'] = UIControls.create_slider(
            "边框宽度 (0-20)", 0, 20, 0, layout
        )
        
        # 边框颜色选择
        controls['border_color_button'] = QPushButton("选择边框颜色")
        layout.addWidget(controls['border_color_button'])
        
        # 透明度控制
        controls['opacity'] = UIControls.create_slider(
            "透明度 (0-50%)", 0, 50, 0, layout
        )
        
        group.setLayout(layout)
        return group, controls
    
    @staticmethod
    def create_background_controls():
        """创建背景控制组"""
        group = QGroupBox("背景控制")
        layout = QVBoxLayout()
        
        controls = {}
        
        # 背景模糊控制
        controls['blur'] = UIControls.create_slider(
            "模糊程度", 1, 99, 45, layout
        )
        
        # 背景放大控制
        controls['bg_scale'] = UIControls.create_slider(
            "背景放大 (110-150%)", 110, 150, 120, layout
        )
        
        # 背景亮度控制
        controls['brightness'] = UIControls.create_slider(
            "背景亮度 (-50 to 50)", -50, 50, 0, layout
        )
        
        # 背景饱和度控制
        controls['saturation'] = UIControls.create_slider(
            "背景饱和度 (0-200%)", 0, 200, 100, layout
        )
        
        # 背景色调控制
        controls['hue'] = UIControls.create_slider(
            "背景色调 (-180 to 180)", -180, 180, 0, layout
        )
        
        # 背景效果选择
        effect_label = QLabel("背景特效")
        controls['effect_combo'] = QComboBox()
        controls['effect_combo'].addItems(["无", "镜像", "左右渐变", "上下渐变"])
        layout.addWidget(effect_label)
        layout.addWidget(controls['effect_combo'])
        
        group.setLayout(layout)
        return group, controls
