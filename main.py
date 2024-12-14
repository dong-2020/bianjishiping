import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtWidgets import QPushButton, QFileDialog, QLabel, QColorDialog, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# 修改导入路径
from src.video_processor import VideoProcessor
from src.ui_controls import VideoControls

class VideoBlurApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频模糊背景效果增强版 2.0")
        self.setGeometry(100, 100, 1280, 900)
        
        # 初始化组件
        self.video_processor = VideoProcessor()
        self.controls = VideoControls()
        self.video_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI"""
        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label)
        
        # 创建控制面板
        self.layout.addLayout(self.controls.create_control_panel())
        
        # 创建按钮
        button_layout = QHBoxLayout()
        
        self.select_button = QPushButton("选择视频")
        self.select_button.clicked.connect(self.select_video)
        button_layout.addWidget(self.select_button)
        
        self.process_button = QPushButton("开始处理")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.process_button)
        
        self.save_button = QPushButton("保存视频")
        self.save_button.clicked.connect(self.save_video)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.layout.addLayout(button_layout)
        
        # 连接边框颜色按钮
        self.controls.border_color_button.clicked.connect(self.choose_border_color)
        
    def display_frame(self, frame):
        """显示视频帧"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def select_video(self):
        """选择视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        if file_name:
            self.video_path = file_name
            self.process_button.setEnabled(True)
            self.save_button.setEnabled(True)
            
            # 打开视频并更新边框宽度范围
            if self.video_processor.open_video(self.video_path):
                # 获取并设置最大边框宽度
                max_border = self.video_processor.get_max_border_width()
                self.controls.update_border_slider(max_border)
                
                # 显示第一帧
                ret, frame = self.video_processor.read_frame()
                if ret:
                    self.display_frame(frame)
                self.video_processor.close_video()

    def save_video(self):
        """保存处理后的视频"""
        if not self.video_path:
            return
            
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存视频",
            "",
            "MP4 Files (*.mp4);;AVI Files (*.avi)"
        )
        
        if output_path:
            try:
                params = self.controls.get_parameters()
                self.video_processor.save_video(self.video_path, output_path, params)
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))
    
    def process_video(self):
        """处理视频"""
        if not self.timer.isActive():
            try:
                if not self.video_processor.open_video(self.video_path):
                    raise Exception("无法打开视频文件")
                self.timer.start(33)  # 约30fps
                self.process_button.setText("停止")
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))
        else:
            self.timer.stop()
            self.video_processor.close_video()
            self.process_button.setText("开始处理")
            
    def update_frame(self):
        """更新视频帧"""
        ret, frame = self.video_processor.read_frame()
        if ret:
            params = self.controls.get_parameters()
            processed = self.video_processor.process_frame(frame, params)
            self.display_frame(processed)
        else:
            self.timer.stop()
            self.video_processor.close_video()
            self.process_button.setText("开始处理")
            
    def choose_border_color(self):
        """选择边框颜色"""
        color = QColorDialog.getColor(self.controls.border_color, self)
        if color.isValid():
            self.controls.border_color = color
            
    def open_video(self):
        """打开视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.video_path = file_path
            # 更新边框宽度的最大值
            if self.video_processor.open_video(self.video_path):
                max_border = self.video_processor.get_max_border_width()
                self.controls.update_border_slider(max_border)
                self.video_processor.close_video()
            self.process_button.setEnabled(True)

    def closeEvent(self, event):
        """关闭事件处理"""
        self.video_processor.close_video()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoBlurApp()
    window.show()
    sys.exit(app.exec())
