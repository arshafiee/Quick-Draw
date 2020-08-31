import sys
import typing
import os
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cairocffi as cairo
import math

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QBasicTimer, QDate, QMimeData, QSize, QPoint, QTimer, QTime
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QMessageBox, QDesktopWidget, QMainWindow
from PyQt5.QtWidgets import qApp, QAction, QMenu, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QLineEdit, QTextEdit
from PyQt5.QtWidgets import QLCDNumber, QSlider, QInputDialog, QFrame, QColorDialog, QSizePolicy, QFontDialog
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QProgressBar, QCalendarWidget, QSplitter
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QDrag, QPainter, QImage, QPen
from PyQt5.QtCore import QObject, pyqtSignal, QBuffer, QIODevice

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


import random


def vector_to_raster(vector_image, side=28, line_diameter=16, padding=16, bg_color=(0, 0, 0), fg_color=(1, 1, 1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """

    original_side = 256.

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    bbox = np.hstack(vector_image).max(axis=1)
    offset = ((original_side, original_side) - bbox) / 2.
    offset = offset.reshape(-1, 1)
    centered = [stroke + offset for stroke in vector_image]

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)
    for xv, yv in centered:
        ctx.move_to(xv[0], yv[0])
        for x, y in zip(xv, yv):
            ctx.line_to(x, y)
        ctx.stroke()

    data = surface.get_data()
    raster_image = np.copy(np.asarray(data)[::4])
    raster_images.append(raster_image)

    return raster_images


class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super(SimpleCNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         ).cuda(device)

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # ).cuda(device)
        #
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # ).cuda(device)

        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
#         out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class QuickDraw(QMainWindow):

    def __init__(self, classes, model):

        super().__init__()

        self.num_tests = 3
        self.items = random.sample(classes, self.num_tests)
        self.turn = 0

        self.correct_pred = 0
        self.pred_name = ''

        col = QColor(200, 200, 50)
        self.setStyleSheet("QWidget { background-color: %s }" % col.name())

        self.back_color = QColor(50, 200, 200)

        self.start_button = QPushButton('Start', self)
        self.start_button.setGeometry(875, 1120, 100, 100)

        self.start_button.clicked.connect(self.doAction)

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setGeometry(875, 1250, 100, 100)

        self.quit_button.clicked.connect(self.quit_pressed)

        self.color_button = QPushButton('Color', self)
        self.color_button.setGeometry(900, 70, 100, 100)

        self.color_button.clicked.connect(self.showDialog)

        self.restart_button = QPushButton('Restart', self)
        self.restart_button.setGeometry(750, 1120, 100, 100)

        self.restart_button.clicked.connect(self.restart)

        self.next_button = QPushButton('Next', self)
        self.next_button.setGeometry(750, 1250, 100, 100)

        self.next_button.clicked.connect(self.next)

        self.setGeometry(800, 200, 1024, 1400)
        self.setWindowTitle('Quick Draw')
        self.setWindowIcon(QIcon('Icon/paint.png'))

        self.Image = QImage(1024, 1024, QImage.Format_RGB32)
        self.Image.fill(self.back_color)

        self.start_flag = False
        self.drawing = False
        self.brush_size = 20
        self.brush_color = Qt.black
        self.moving = False
        self.stroke = [[],[]]
        self.strokes = []
        self.xs = []
        self.ys = []

        self.last_point = QPoint()

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("File")

        save_action = QAction(QIcon('Icon/save.png'), 'Save', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)
        save_action.triggered.connect(self.save)

        clear_action = QAction(QIcon('Icon/clear.png'), 'Clear', self)
        clear_action.setShortcut('Ctrl+C')
        file_menu.addAction(clear_action)
        clear_action.triggered.connect(self.clear)

        self.lcd = QLCDNumber(self)
        self.lcd.setGeometry(50, 1270, 300, 80)

        font = QFont()
        font.setBold(True)
        font.setPointSize(8)

        self.label = QLabel('Remaining Time:', self)
        self.label.setGeometry(50, 1200, 300, 40)
        # self.label.setAlignment(Qt.AlignCenter)

        self.label2 = QLabel('Draw : {0} '.format(self.items[self.turn]), self)
        self.label2.setGeometry(50, 1130, 350, 40)
        # self.label2.setAlignment(Qt.AlignLeft)

        self.label3 = QLabel('Prediction : None', self)
        self.label3.setGeometry(50, 1060, 400, 40)
        # self.label3.setAlignment(Qt.AlignLeft)
        self.label3.setFont(font)

        self.label4 = QLabel('Correct Predictions\n{0}/{1}'.format(self.correct_pred, self.num_tests), self)
        self.label4.setGeometry(350, 1115, 350, 100)
        self.label4.setAlignment(Qt.AlignCenter)
        self.label4.setFont(font)

        self.seconds = 30
        self.milliseconds = 0

        self.timer = QTimer(self)
        self.timer.setInterval(10)
        self.timer.start()
        self.timer.timeout.connect(self.showTime)

        self.timer2 = QTimer(self)
        self.timer2.setInterval(10)
        self.timer2.start()
        # self.timer2.timeout.connect(self.predict)
        self.pred_sec = 0

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def quit_pressed(self):

        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            qApp.quit()

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):

        if (event.buttons() & Qt.LeftButton) & self.drawing & self.start_flag:
            painter = QPainter(self.Image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.xs.append(event.x())
            self.ys.append(event.y())
            self.stroke[0].append(event.x())
            self.stroke[1].append(event.y())
            self.moving = True
            self.update()

    def mouseReleaseEvent(self, event):

        if (event.button() == Qt.LeftButton) & self.moving:
            self.drawing = False
            self.moving = False
            self.strokes.append(self.stroke)
            self.stroke = [[], []]
            strokes = self.cropsize(self.xs, self.ys, self.strokes)
            raster = vector_to_raster(strokes)
            image = raster[0].reshape(28, 28)
            tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((56, 56)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,))
            ])
            t_image = tfms(image)
            tens_image = torch.empty((1, 1, 56, 56))
            tens_image[0] = t_image
            # plt.imshow(tens_image[0,0])
            # plt.show()
            output = model(tens_image)
            _, pred = torch.max(output.data, 1)
            print(output.data)
            print(classes[pred])
            self.label3.setText('Predict : {0}'.format(classes[pred]))
            if classes[pred] == self.items[self.turn]:
                if self.pred_name != classes[pred]:
                    self.pred_name = classes[pred]
                    self.correct_pred = self.correct_pred + 1
                    self.label4.setText('Correct Predictions\n{0}/{1}'.format(self.correct_pred, self.num_tests))
                self.doAction()
                reply = QMessageBox.question(self, 'Message', 'Oh, I know! That\'s {0}. '.format(classes[pred]), QMessageBox.Ok,
                                             QMessageBox.Ok)
                self.doAction()

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.Image.rect(), self.Image, self.Image.rect())

    def doAction(self):

        if self.start_flag:
            self.start_flag = False
            self.start_button.setText('Start')
        else:
            self.start_flag = True
            self.start_button.setText('Pause')

    def save(self):

        file_path,_=QFileDialog.getSaveFileName(self, 'Save Image', '', 'PNG(*.png);;JPG(*.jpg *.jpeg);;All Files(*.*)')
        if file_path == "":
            return
        self.Image.save(file_path)

    def clear(self):

        self.Image.fill(self.back_color)
        self.strokes = []
        self.xs = []
        self.ys = []
        self.update()

    def showTime(self):

        if not self.start_flag:
            return

        if self.milliseconds < 10:
            time = '{0}:0{1}'.format(self.seconds, self.milliseconds)
        else:
            time = '{0}:{1}'.format(self.seconds, self.milliseconds)
        self.lcd.display(time)

        if (self.seconds == 0) & (self.milliseconds == 0):
            self.seconds = 30
            self.doAction()
            self.turn = self.turn + 1
            if self.turn == self.num_tests:
                reply = QMessageBox.question(self, 'The End', 'Do You Want to Play a new Game?', QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.restart()
                if reply == QMessageBox.No:
                    qApp.quit()

            self.label2.setText('Draw : {0} '.format(self.items[self.turn]))
            self.label3.setText('Predict : None')
            self.Image.fill(self.back_color)
            self.strokes = []
            self.xs = []
            self.ys = []
            self.update()
            return

        if self.milliseconds == 0:
            self.milliseconds = 99
            self.seconds = self.seconds - 1
            return

        self.milliseconds = self.milliseconds - 1

    def showDialog(self):

        self.back_color = QColorDialog.getColor()
        self.strokes = []
        self.xs = []
        self.ys = []

        if self.back_color.isValid():
            self.Image.fill(self.back_color)

    def restart(self):

        self.start_flag = True
        self.doAction()
        self.drawing = False
        self.brush_size = 20
        self.brush_color = Qt.black
        self.correct_pred = 0
        self.pred_name = ''
        self.moving = False
        self.stroke = [[], []]
        self.strokes = []
        self.xs = []
        self.ys = []
        self.last_point = QPoint()
        self.seconds = 30
        self.milliseconds = 0
        self.clear()
        self.items = random.sample(classes, self.num_tests)
        self.turn = 0
        self.label4.setText('Correct Predictions\n{0}/{1}'.format(self.correct_pred, self.num_tests))
        self.label2.setText('Draw : {0} '.format(self.items[self.turn]))
        self.label3.setText('Predict : None')
        time = '{0}:0{1}'.format(self.seconds, self.milliseconds)
        self.lcd.display(time)

    def next(self):

        self.start_flag = True
        self.doAction()
        self.drawing = False
        self.brush_size = 20
        self.brush_color = Qt.black
        self.moving = False
        self.stroke = [[], []]
        self.strokes = []
        self.xs = []
        self.ys = []
        self.last_point = QPoint()
        self.seconds = 30
        self.milliseconds = 0
        self.clear()
        self.turn = self.turn + 1
        if self.turn == self.num_tests:
            reply = QMessageBox.question(self, 'The End', 'Do You Want to Play a new Game?',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.restart()
            if reply == QMessageBox.No:
                qApp.quit()

        self.label2.setText('Draw : {0} '.format(self.items[self.turn]))
        self.label3.setText('Predict : None')
        time = '{0}:0{1}'.format(self.seconds, self.milliseconds)
        self.lcd.display(time)

    def cropsize(self, x, y, strokes):

        margin = 10

        x_min = min(x)
        if x_min >= margin:
            x_min = x_min - margin
        else:
            x_min = 0

        x_max = max(x)
        if x_max + margin <= 1024:
            x_max = x_max + margin
        else:
            x_max = 1024

        y_min = min(y)
        if y_min >= margin:
            y_min = y_min - margin
        else:
            y_min = 0

        y_max = max(y)
        if y_max + margin <= 1024:
            y_max = y_max + margin
        else:
            y_max = 1024

        height = y_max - y_min
        width = x_max - x_min

        if height > width:
            crop_size = height
        else:
            crop_size = width

        # print(crop_size)
        crop_factor = crop_size / 256.
        # print(crop_factor)
        out_strokes = []
        for stroke in strokes:
            print(stroke)
            print(x_min)
            print(y_min)
            np_stroke = (np.array(stroke) - [[x_min], [y_min]]) // crop_factor
            list_stroke = np_stroke.tolist()
            out_strokes.append(list_stroke)

        return out_strokes


if __name__ == '__main__':

    DATA_DIR = "E:/internship/QuickDraw/numpy_bitmap"
    class_names = os.listdir(DATA_DIR)
    num_classes = 10
    classes = [""]
    for j in range(num_classes - 1):
        classes.append("")
    for i in range(num_classes):
        classes[i] = class_names[i].split('.')[0]

    model = SimpleCNN(num_classes)
    model_save_name = 'classifier-10.pt'
    path = f"E:/internship/QuickDraw/trained models/{model_save_name}"
    model.load_state_dict(torch.load(path, map_location='cpu'))

    app = QApplication(sys.argv)
    ex = QuickDraw(classes, model)
    ex.show()
    sys.exit(app.exec_())

