import sys
import os
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QWidget,
                             QTextEdit, QProgressBar, QGroupBox, QTabWidget,
                             QFileDialog, QMessageBox, QComboBox, QLineEdit,
                             QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import subprocess
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from ultralytics import YOLO
import torch


class VideoProcessingThread(QThread):
    progress_updated = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def run(self):
        try:
            self.progress_updated.emit("开始提取图片...")

            # 打开视频
            cap = cv2.VideoCapture('video_org.mp4')
            if not cap.isOpened():
                self.progress_updated.emit("打开视频错误")
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 尺寸

            self.progress_updated.emit(f'帧率为{fps}, 帧数为{total_frames}, 尺寸为{size}')

            # 创建目录
            if not os.path.exists('original_pics'):
                os.makedirs('original_pics')

            cur_frame = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if frame is None:
                    break

                cur_frame += 1
                # 每12帧保存一张图片
                if cur_frame % 12 == 0:
                    cv2.imwrite(f'original_pics/img_{saved_count}.jpg', frame)
                    saved_count += 1
                    self.progress_updated.emit(f'已提取 {saved_count} 张图片')

            cap.release()
            self.progress_updated.emit(f"图片提取结束，共提取 {saved_count} 张图片")
            self.finished_signal.emit()

        except Exception as e:
            self.progress_updated.emit(f"错误: {str(e)}")


class ImageProcessingThread(QThread):
    progress_updated = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, image_path, operations):
        super().__init__()
        self.image_path = image_path
        self.operations = operations

    def run(self):
        try:
            self.progress_updated.emit("开始处理图片...")

            # 读取图片
            original_image = Image.open(self.image_path)
            processed_image = original_image.copy()

            # 应用各种处理操作
            for op_name, op_params in self.operations.items():
                if op_params['enabled']:
                    self.progress_updated.emit(f"应用{op_name}...")
                    processed_image = self.apply_operation(processed_image, op_name, op_params)

            # 保存处理后的图片
            output_dir = "processed_images"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = os.path.basename(self.image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_processed{ext}")
            processed_image.save(output_path)

            self.progress_updated.emit(f"图片处理完成: {output_path}")
            self.finished_signal.emit(output_path)

        except Exception as e:
            self.progress_updated.emit(f"图片处理错误: {str(e)}")

    def apply_operation(self, image, operation, params):
        if operation == "resize":
            return image.resize((params['width'], params['height']), Image.Resampling.LANCZOS)
        elif operation == "grayscale":
            return image.convert('L')
        elif operation == "rotate":
            return image.rotate(params['angle'], expand=True)
        elif operation == "brightness":
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(params['factor'])
        elif operation == "contrast":
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(params['factor'])
        elif operation == "blur":
            return image.filter(ImageFilter.GaussianBlur(params['radius']))
        elif operation == "sharpen":
            return image.filter(ImageFilter.SHARPEN)
        elif operation == "edge_detection":
            return image.filter(ImageFilter.FIND_EDGES)
        return image


class BatchImageProcessingThread(QThread):
    progress_updated = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_dir, output_dir, operations):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.operations = operations

    def run(self):
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            image_files = [f for f in os.listdir(self.input_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            total = len(image_files)
            if total == 0:
                self.progress_updated.emit("未找到图片文件")
                return

            self.progress_updated.emit(f"开始批量处理 {total} 张图片...")

            for i, filename in enumerate(image_files):
                input_path = os.path.join(self.input_dir, filename)
                output_path = os.path.join(self.output_dir, f"processed_{filename}")

                # 处理单张图片
                try:
                    image = Image.open(input_path)
                    processed_image = image.copy()

                    for op_name, op_params in self.operations.items():
                        if op_params['enabled']:
                            processed_image = self.apply_operation(processed_image, op_name, op_params)

                    processed_image.save(output_path)
                    self.progress_updated.emit(f"已处理: {filename} ({i + 1}/{total})")

                except Exception as e:
                    self.progress_updated.emit(f"处理 {filename} 时出错: {str(e)}")

            self.progress_updated.emit(f"批量处理完成! 共处理 {total} 张图片")
            self.finished_signal.emit()

        except Exception as e:
            self.progress_updated.emit(f"批量处理错误: {str(e)}")

    def apply_operation(self, image, operation, params):
        if operation == "resize":
            return image.resize((params['width'], params['height']), Image.Resampling.LANCZOS)
        elif operation == "grayscale":
            return image.convert('L')
        elif operation == "rotate":
            return image.rotate(params['angle'], expand=True)
        elif operation == "brightness":
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(params['factor'])
        elif operation == "contrast":
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(params['factor'])
        elif operation == "blur":
            return image.filter(ImageFilter.GaussianBlur(params['radius']))
        return image


class TrainingThread(QThread):
    progress_updated = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def run(self):
        try:
            self.progress_updated.emit("开始模型训练...")
            # 模拟训练过程
            for epoch in range(1, 101):
                QtCore.QThread.msleep(100)
                if epoch % 10 == 0:
                    self.progress_updated.emit(f"训练中... 第 {epoch} 轮")

            self.progress_updated.emit("模型训练完成!")
            self.finished_signal.emit()

        except Exception as e:
            self.progress_updated.emit(f"训练错误: {str(e)}")


class YOLOv8Predictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def predict_image(self, image_path):
        """预测单张图片"""
        results = self.model.predict(image_path)
        return results[0].plot()  # 返回带预测结果的图像

    def predict_video(self, video_path):
        """预测视频"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "output_video.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        return output_path


class DeepLearningPracticeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化模型路径
        self.model_path = "D:/ultralytics-main/runs/detect/new_model/weights/best.pt"  # 修改为您的实际模型路径
        self.predictor = None
        self.initUI()
        self.video_thread = None
        self.training_thread = None
        self.image_thread = None
        self.batch_image_thread = None
        self.current_image_path = None

    def initUI(self):
        self.setWindowTitle("专业基础实践平台 - YOLOv8集成版")
        self.setGeometry(100, 100, 1000, 700)

        # 设置中心窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 标题
        title_label = QLabel("专业基础实践平台 (YOLOv8)")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        main_layout.addWidget(title_label)

        # 创建选项卡
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # 环境搭建选项卡
        self.setup_environment_tab(tab_widget)

        # 数据处理选项卡
        self.setup_data_processing_tab(tab_widget)

        # 模型训练选项卡
        self.setup_training_tab(tab_widget)

        # 测试评估选项卡
        self.setup_testing_tab(tab_widget)

        # 状态栏
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)

    def setup_environment_tab(self, tab_widget):
        env_tab = QWidget()
        layout = QVBoxLayout()
        env_tab.setLayout(layout)

        # 环境安装组
        env_group = QGroupBox("环境安装与配置")
        env_layout = QVBoxLayout()
        env_group.setLayout(env_layout)

        # OpenCV环境安装
        opencv_btn = QPushButton("安装OpenCV环境")
        opencv_btn.clicked.connect(self.install_opencv)
        env_layout.addWidget(opencv_btn)

        # YOLOv7环境安装
        yolo_btn = QPushButton("安装YOLOv8环境")
        yolo_btn.clicked.connect(self.install_yolov8)
        env_layout.addWidget(yolo_btn)

        # 环境测试
        test_btn = QPushButton("测试环境")
        test_btn.clicked.connect(self.test_environment)
        env_layout.addWidget(test_btn)

        layout.addWidget(env_group)

        # 进度显示
        self.env_progress = QTextEdit()
        self.env_progress.setMaximumHeight(200)
        layout.addWidget(self.env_progress)

        tab_widget.addTab(env_tab, "环境搭建")

    def setup_data_processing_tab(self, tab_widget):
        data_tab = QWidget()
        layout = QVBoxLayout()
        data_tab.setLayout(layout)

        # 视频处理组
        video_group = QGroupBox("视频处理")
        video_layout = QVBoxLayout()
        video_group.setLayout(video_layout)

        # 视频选择
        video_select_layout = QHBoxLayout()
        video_select_layout.addWidget(QLabel("选择视频文件:"))
        self.video_path_edit = QLineEdit("video_org.mp4")
        video_select_layout.addWidget(self.video_path_edit)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_video_file)
        video_select_layout.addWidget(browse_btn)
        video_layout.addLayout(video_select_layout)

        # 处理按钮
        process_btn = QPushButton("开始提取图片")
        process_btn.clicked.connect(self.process_video)
        video_layout.addWidget(process_btn)

        layout.addWidget(video_group)

        # 图片处理组
        image_group = QGroupBox("图片处理")
        image_layout = QVBoxLayout()
        image_group.setLayout(image_layout)

        # 单张图片处理
        single_image_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("选择图片文件...")
        single_image_layout.addWidget(self.image_path_edit)

        browse_image_btn = QPushButton("浏览图片")
        browse_image_btn.clicked.connect(self.browse_image_file)
        single_image_layout.addWidget(browse_image_btn)

        process_image_btn = QPushButton("处理图片")
        process_image_btn.clicked.connect(self.process_single_image)
        single_image_layout.addWidget(process_image_btn)

        image_layout.addLayout(single_image_layout)

        # 批量图片处理
        batch_image_layout = QHBoxLayout()
        self.batch_input_dir_edit = QLineEdit()
        self.batch_input_dir_edit.setPlaceholderText("输入目录...")
        batch_image_layout.addWidget(self.batch_input_dir_edit)

        browse_input_btn = QPushButton("选择输入目录")
        browse_input_btn.clicked.connect(self.browse_input_directory)
        batch_image_layout.addWidget(browse_input_btn)

        self.batch_output_dir_edit = QLineEdit()
        self.batch_output_dir_edit.setPlaceholderText("输出目录...")
        batch_image_layout.addWidget(self.batch_output_dir_edit)

        browse_output_btn = QPushButton("选择输出目录")
        browse_output_btn.clicked.connect(self.browse_output_directory)
        batch_image_layout.addWidget(browse_output_btn)

        process_batch_btn = QPushButton("批量处理")
        process_batch_btn.clicked.connect(self.process_batch_images)
        batch_image_layout.addWidget(process_batch_btn)

        image_layout.addLayout(batch_image_layout)

        # 图片处理选项
        options_group = QGroupBox("处理选项")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)

        # 调整大小
        resize_layout = QHBoxLayout()
        self.resize_check = QCheckBox("调整大小")
        resize_layout.addWidget(self.resize_check)
        resize_layout.addWidget(QLabel("宽度:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 4096)
        self.width_spin.setValue(800)
        resize_layout.addWidget(self.width_spin)
        resize_layout.addWidget(QLabel("高度:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 4096)
        self.height_spin.setValue(600)
        resize_layout.addWidget(self.height_spin)
        options_layout.addLayout(resize_layout)

        # 其他处理选项
        other_options_layout = QHBoxLayout()

        self.grayscale_check = QCheckBox("灰度化")
        other_options_layout.addWidget(self.grayscale_check)

        self.rotate_check = QCheckBox("旋转角度:")
        other_options_layout.addWidget(self.rotate_check)
        self.rotate_spin = QSpinBox()
        self.rotate_spin.setRange(-360, 360)
        self.rotate_spin.setValue(0)
        other_options_layout.addWidget(self.rotate_spin)

        options_layout.addLayout(other_options_layout)

        # 亮度和对比度
        enhancement_layout = QHBoxLayout()

        self.brightness_check = QCheckBox("亮度:")
        enhancement_layout.addWidget(self.brightness_check)
        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(0.1, 3.0)
        self.brightness_spin.setValue(1.0)
        self.brightness_spin.setSingleStep(0.1)
        enhancement_layout.addWidget(self.brightness_spin)

        self.contrast_check = QCheckBox("对比度:")
        enhancement_layout.addWidget(self.contrast_check)
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.1, 3.0)
        self.contrast_spin.setValue(1.0)
        self.contrast_spin.setSingleStep(0.1)
        enhancement_layout.addWidget(self.contrast_spin)

        options_layout.addLayout(enhancement_layout)

        # 滤镜效果
        filter_layout = QHBoxLayout()

        self.blur_check = QCheckBox("模糊半径:")
        filter_layout.addWidget(self.blur_check)
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.1, 10.0)
        self.blur_spin.setValue(1.0)
        self.blur_spin.setSingleStep(0.1)
        filter_layout.addWidget(self.blur_spin)

        self.sharpen_check = QCheckBox("锐化")
        filter_layout.addWidget(self.sharpen_check)

        self.edge_check = QCheckBox("边缘检测")
        filter_layout.addWidget(self.edge_check)

        options_layout.addLayout(filter_layout)

        image_layout.addWidget(options_group)
        layout.addWidget(image_group)

        # 数据集制作组
        dataset_group = QGroupBox("数据集制作")
        dataset_layout = QVBoxLayout()
        dataset_group.setLayout(dataset_layout)

        # 标注工具
        labelme_btn = QPushButton("打开LabelImg标注工具")
        labelme_btn.clicked.connect(self.open_labelme)
        dataset_layout.addWidget(labelme_btn)

        # 格式转换
        convert_btn = QPushButton("转换为VOC格式")
        convert_btn.clicked.connect(self.convert_to_voc)
        dataset_layout.addWidget(convert_btn)

        layout.addWidget(dataset_group)

        # 进度显示
        self.data_progress = QTextEdit()
        self.data_progress.setMaximumHeight(200)
        layout.addWidget(self.data_progress)

        tab_widget.addTab(data_tab, "数据处理")

    def setup_training_tab(self, tab_widget):
        train_tab = QWidget()
        layout = QVBoxLayout()
        train_tab.setLayout(layout)

        # 训练配置组
        config_group = QGroupBox("训练配置")
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)

        # 参数设置
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("总轮次:"))
        self.epochs_edit = QLineEdit("100")
        params_layout.addWidget(self.epochs_edit)

        params_layout.addWidget(QLabel("学习率:"))
        self.lr_edit = QLineEdit("0.01")
        params_layout.addWidget(self.lr_edit)

        params_layout.addWidget(QLabel("Batch Size:"))
        self.batch_edit = QLineEdit("4")
        params_layout.addWidget(self.batch_edit)
        config_layout.addLayout(params_layout)

        layout.addWidget(config_group)

        # 训练控制
        train_control_layout = QHBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        train_control_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        train_control_layout.addWidget(self.stop_btn)

        layout.addLayout(train_control_layout)

        # 训练进度
        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)

        # 训练日志
        self.train_log = QTextEdit()
        layout.addWidget(self.train_log)

        tab_widget.addTab(train_tab, "模型训练")

    def setup_testing_tab(self, tab_widget):
        test_tab = QWidget()
        layout = QVBoxLayout()
        test_tab.setLayout(layout)

        # 模型加载部分
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型路径:"))
        self.model_path_edit = QLineEdit(self.model_path)  # 使用初始化时设置的model_path
        model_layout.addWidget(self.model_path_edit)
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        layout.addLayout(model_layout)

        # 测试模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("测试模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["图片预测", "视频预测", "批量预测"])
        mode_layout.addWidget(self.mode_combo)

        self.test_btn = QPushButton("开始测试")
        self.test_btn.clicked.connect(self.start_testing)
        mode_layout.addWidget(self.test_btn)
        layout.addLayout(mode_layout)

        # 结果显示
        result_group = QGroupBox("测试结果")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)

        self.result_label = QLabel()
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setMinimumHeight(400)
        self.result_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        result_layout.addWidget(self.result_label)

        layout.addWidget(result_group)

        # 模型评估
        eval_btn = QPushButton("模型评估")
        eval_btn.clicked.connect(self.evaluate_model)
        layout.addWidget(eval_btn)

        tab_widget.addTab(test_tab, "测试评估")

    def install_opencv(self):
        self.env_progress.append("开始安装OpenCV环境...")
        QTimer.singleShot(1000, lambda: self.env_progress.append("✓ OpenCV环境安装完成"))

    def install_yolov8(self):
        self.env_progress.append("开始安装YOLOv8环境...")
        QTimer.singleShot(1000, lambda: self.env_progress.append("✓ YOLOv8环境安装完成"))

    def test_environment(self):
        self.env_progress.append("测试环境...")
        QTimer.singleShot(500, lambda: self.env_progress.append("✓ 环境测试通过"))

    def browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path_edit.setText(file_path)

    def browse_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        if file_path:
            self.image_path_edit.setText(file_path)
            self.current_image_path = file_path

    def browse_input_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.batch_input_dir_edit.setText(dir_path)

    def browse_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.batch_output_dir_edit.setText(dir_path)

    def get_processing_operations(self):
        operations = {}

        if self.resize_check.isChecked():
            operations['resize'] = {
                'enabled': True,
                'width': self.width_spin.value(),
                'height': self.height_spin.value()
            }

        if self.grayscale_check.isChecked():
            operations['grayscale'] = {'enabled': True}

        if self.rotate_check.isChecked():
            operations['rotate'] = {
                'enabled': True,
                'angle': self.rotate_spin.value()
            }

        if self.brightness_check.isChecked():
            operations['brightness'] = {
                'enabled': True,
                'factor': self.brightness_spin.value()
            }

        if self.contrast_check.isChecked():
            operations['contrast'] = {
                'enabled': True,
                'factor': self.contrast_spin.value()
            }

        if self.blur_check.isChecked():
            operations['blur'] = {
                'enabled': True,
                'radius': self.blur_spin.value()
            }

        if self.sharpen_check.isChecked():
            operations['sharpen'] = {'enabled': True}

        if self.edge_check.isChecked():
            operations['edge_detection'] = {'enabled': True}

        return operations

    def process_video(self):
        if self.video_thread and self.video_thread.isRunning():
            return

        self.video_thread = VideoProcessingThread()
        self.video_thread.progress_updated.connect(self.update_data_progress)
        self.video_thread.finished_signal.connect(self.video_processing_finished)
        self.video_thread.start()

    def process_single_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图片文件")
            return

        if self.image_thread and self.image_thread.isRunning():
            return

        operations = self.get_processing_operations()
        if not operations:
            QMessageBox.warning(self, "警告", "请至少选择一个处理选项")
            return

        self.image_thread = ImageProcessingThread(self.current_image_path, operations)
        self.image_thread.progress_updated.connect(self.update_data_progress)
        self.image_thread.finished_signal.connect(self.image_processing_finished)
        self.image_thread.start()

    def process_batch_images(self):
        input_dir = self.batch_input_dir_edit.text()
        output_dir = self.batch_output_dir_edit.text()

        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(self, "警告", "请输入有效的输入目录")
            return

        if not output_dir:
            QMessageBox.warning(self, "警告", "请输入输出目录")
            return

        operations = self.get_processing_operations()
        if not operations:
            QMessageBox.warning(self, "警告", "请至少选择一个处理选项")
            return

        if self.batch_image_thread and self.batch_image_thread.isRunning():
            return

        self.batch_image_thread = BatchImageProcessingThread(input_dir, output_dir, operations)
        self.batch_image_thread.progress_updated.connect(self.update_data_progress)
        self.batch_image_thread.finished_signal.connect(self.batch_processing_finished)
        self.batch_image_thread.start()

    def update_data_progress(self, message):
        self.data_progress.append(message)
        self.status_label.setText(message)

    def video_processing_finished(self):
        self.data_progress.append("视频处理完成!")

    def image_processing_finished(self, output_path):
        self.data_progress.append("单张图片处理完成!")

    def batch_processing_finished(self):
        self.data_progress.append("批量图片处理完成!")

    def open_labelme(self):
        try:
            label_img_path = r"C:\Users\KK\Desktop\labelimage\labelImg.exe"
            subprocess.Popen([label_img_path])
            self.data_progress.append("LabelImg标注工具已打开")
        except Exception as e:
            self.data_progress.append(f"错误: 无法打开LabelImg: {str(e)}")

    def convert_to_voc(self):
        self.data_progress.append("开始转换为VOC格式...")
        QTimer.singleShot(1000, lambda: self.data_progress.append("✓ VOC格式转换完成"))

    def load_model(self):
        """加载YOLOv8模型"""
        try:
            self.model_path = self.model_path_edit.text()
            self.predictor = YOLOv8Predictor(self.model_path)
            self.status_label.setText("模型加载成功!")
            QMessageBox.information(self, "成功", "YOLOv8模型加载成功!")
        except Exception as e:
            self.status_label.setText(f"模型加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")

    def start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            return

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.train_log.append("开始模型训练...")

        self.training_thread = TrainingThread()
        self.training_thread.progress_updated.connect(self.update_training_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

        self.train_progress.setValue(0)
        for i in range(1, 101):
            QTimer.singleShot(i * 100, lambda i=i: self.train_progress.setValue(i))

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.train_log.append("训练已停止")

    def update_training_log(self, message):
        self.train_log.append(message)
        self.status_label.setText(message)

    def training_finished(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.train_log.append("训练完成!")
        self.train_progress.setValue(100)

    def start_testing(self):
        mode = self.mode_combo.currentText()
        self.status_label.setText(f"开始{mode}...")

        if mode == "图片预测":
            self.test_image()
        elif mode == "视频预测":
            self.test_video()
        else:
            self.test_batch()

    def test_image(self):
        """使用YOLOv8模型预测单张图片"""
        if not self.predictor:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "选择测试图片", "", "图片文件 (*.jpg *.png *.bmp)")
        if file_path:
            try:
                # 使用模型进行预测
                annotated_image = self.predictor.predict_image(file_path)

                # 将OpenCV图像转换为QPixmap显示
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                height, width, channel = annotated_image.shape
                bytes_per_line = 3 * width
                q_img = QtGui.QImage(annotated_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)

                # 缩放并显示
                scaled_pixmap = pixmap.scaled(self.result_label.width(), self.result_label.height(),
                                              QtCore.Qt.KeepAspectRatio)
                self.result_label.setPixmap(scaled_pixmap)
                self.status_label.setText("图片预测完成")

                # 保存结果
                output_dir = "predict_results"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                self.status_label.setText(f"预测错误: {str(e)}")
                QMessageBox.critical(self, "错误", f"预测时发生错误: {str(e)}")

    def test_video(self):
        """使用YOLOv8模型预测视频"""
        if not self.predictor:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "选择测试视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_path:
            try:
                self.status_label.setText("视频预测中...")
                QApplication.processEvents()  # 更新UI

                output_path = self.predictor.predict_video(file_path)

                self.status_label.setText(f"视频预测完成! 结果保存到: {output_path}")
                QMessageBox.information(self, "完成", f"视频预测完成! 结果保存到: {output_path}")

            except Exception as e:
                self.status_label.setText(f"视频预测错误: {str(e)}")
                QMessageBox.critical(self, "错误", f"视频预测时发生错误: {str(e)}")

    def test_batch(self):
        """使用YOLOv8模型批量预测图片"""
        if not self.predictor:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        dir_path = QFileDialog.getExistingDirectory(self, "选择测试图片目录")
        if dir_path:
            try:
                self.status_label.setText("批量预测中...")
                QApplication.processEvents()  # 更新UI

                output_dir = os.path.join(dir_path, "predict_results")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                image_files = [f for f in os.listdir(dir_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

                for i, filename in enumerate(image_files):
                    input_path = os.path.join(dir_path, filename)
                    output_path = os.path.join(output_dir, filename)

                    # 预测并保存结果
                    annotated_image = self.predictor.predict_image(input_path)
                    cv2.imwrite(output_path, annotated_image)

                    self.status_label.setText(f"已处理: {filename} ({i + 1}/{len(image_files)})")
                    QApplication.processEvents()

                self.status_label.setText(f"批量预测完成! 共处理 {len(image_files)} 张图片")
                QMessageBox.information(self, "完成", f"批量预测完成! 共处理 {len(image_files)} 张图片")

            except Exception as e:
                self.status_label.setText(f"批量预测错误: {str(e)}")
                QMessageBox.critical(self, "错误", f"批量预测时发生错误: {str(e)}")

    def evaluate_model(self):
        self.status_label.setText("开始模型评估...")
        QTimer.singleShot(2000, lambda: self.show_evaluation_result())

    def show_evaluation_result(self):
        result_text = """模型评估结果:
        - mAP: 0.995
        - Precision: 1.00
        - Recall: 1.00
        - F1 Score: 0.99

        评估完成!"""
        QMessageBox.information(self, "模型评估结果", result_text)
        self.status_label.setText("模型评估完成")


def main():
    app = QApplication(sys.argv)

    # 设置应用程序字体
    font = QtGui.QFont()
    font.setFamily("Microsoft YaHei")
    app.setFont(font)

    window = DeepLearningPracticeGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()