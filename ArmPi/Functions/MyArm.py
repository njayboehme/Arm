#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')
import logging
import cv2
import time
import Camera
import threading
from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *
from ColorTracking import getAreaMaxContour

logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO,datefmt="%H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)# This is the file I will do all my code in

class My_Arm:
    def __init__(self):
        self.target_color = 'red'
        self.start_pickup = False
        self.size = (640, 480)
        self.h = 0
        self.w = 0
        self.is_running = False
        self.servo_1 = 500
        self.range_rgb = {
                        'red': (0, 0, 255),
                        'blue': (255, 0, 0),
                        'green': (0, 255, 0),
                        'black': (0, 0, 0),
                        'white': (255, 255, 255),
                        }
        self.AK = ArmIK()
        
    def init(self):
        Board.setBusServoPulse(1, self.servo_1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

    def reset(self):
        self.start_pickup = False

    def start(self):
        self.reset()
        self.is_running = True

    def set_target_color(self, color):
        if color == 'r':
            self.target_color = 'red'
        elif color == 'g':
            self.target_color = 'green'
        else:
            self.target_color = 'blue'

    def get_image(self, cam):
        img = cam.frame
        self.h = img.shape[0]
        self.w = img.shape[1]
        return img

    def resize_and_smooth(self, img):
        img_resize = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        img_gb = cv2.GaussianBlur(img_resize, (11, 11), 11)
        return cv2.cvtColor(img_gb, cv2.COLOR_BGR2LAB)

    def detect_target_color(self, img_lab):
        # color_range is a dict in LABConfig.py
        for i in color_range:
            if i == self.target_color:
                frame_mask = self.threshold(img_lab)
                closed_img = self.remove_noise(frame_mask)
                return self.detect_outline(closed_img)
        return 0, 0


    def threshold(self, img_lab):
        return cv2.inRange(img_lab, color_range[self.target_color][0], color_range[self.target_color][1])

    def remove_noise(self, frame_mask):
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
                

    def detect_outline(self, closed_img):
        contours = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        return getAreaMaxContour(contours)

    def detect_box(self, area_max_cont):
        rect = cv2.minAreaRect(area_max_cont)
        box = np.int0(cv2.boxPoints(rect))
        self.box = box
        roi = getROI(box)
        img_cent_x, img_cent_y = getCenter(rect, roi, self.size, square_length)
        return convertCoordinate(img_cent_x, img_cent_y, self.size)

    def pick_up(self):
        pass

    def draw(self, img, world_x, world_y):
        cv2.drawContours(img, [self.box], -1, self.range_rgb[self.target_color], 2)
        cv2.putText(img, '(' + str(world_x) + ',' + str(world_y) + ')', (min(self.box[0, 0], self.box[2, 0]), self.box[2, 1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[self.target_color], 1)

    def run(self):
        # Need to run init and start
        logging.debug("Init")
        self.init()
        logging.debug("Start")
        self.start()
        logging.debug("Opening camera")
        cam = Camera.Camera()
        cam.camera_open()
        if not self.is_running:
            logging.error("Not Running!!")
            return None
        while input("Press q to quit ") != 'q':
            inp = input("What color to detect (r, g, b)? ")
            logging.debug(f'Setting target value to {inp}')
            self.set_target_color(inp)
            logging.debug("Getting img")
            img = self.get_image(cam)
            logging.debug("Resizing and smoothing")
            lab_img = self.resize_and_smooth(img)
            logging.debug("detecting target color")
            area_max_cont, area_max = self.detect_target_color(lab_img)
            if area_max > 2500:
                logging.debug("detecting box")
                world_x, world_y = self.detect_box(area_max_cont)
                logging.debug("Drawing")
                self.draw(img, world_x, world_y)
            cv2.imshow("img", img)
            cv2.waitKey(1)
        cam.camera_close()
        cv2.destroyAllWindows()






if __name__ == '__main__':
    arm = My_Arm()
    arm.run()