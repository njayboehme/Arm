#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')
import logging
import traceback
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
from ColorTracking import setTargetColor

logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO,datefmt="%H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)# This is the file I will do all my code in

class My_Arm:
    def __init__(self):
        self.target_color = ()
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

        # For moving arm
        self.count = 0
        self.track = False
        self._stop = False
        self.get_roi = False
        self.center_list = []
        self.first_move = True
        self.detect_color = 'None'
        self.action_finish = True
        self.start_pick_up = False
        self.start_count_t1 = True
        
        self.t1 = 0
        self.roi = ()
        self.rect = None
        self.rotation_angle = 0
        self.unreachable = False

        self.world_X = 0
        self.world_Y = 0

        self.world_x = 0
        self.world_y = 0

        self.last_x = 0
        self.last_y = 0
        self.coordinate = {
                            'red':   (-15 + 0.5, 12 - 0.5, 1.5),
                            'green': (-15 + 0.5, 6 - 0.5,  1.5),
                            'blue':  (-15 + 0.5, 0 - 0.5,  1.5),
                          }

    # General Functions
    def init(self):
        Board.setBusServoPulse(1, self.servo_1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

    def reset(self):
        self.start_pick_up = False
        self.count = 0
        self._stop = False
        self.track = False
        self.get_roi = False
        self.center_list = []
        self.first_move = True
        self.target_color = ()
        self.detect_color = 'None'
        self.action_finish = True
        self.start_count_t1 = True

    def start(self):
        self.reset()
        self.is_running = True

    def setBuzzer(self, timer):
        Board.setBuzzer(0)
        Board.setBuzzer(1)
        time.sleep(timer)
        Board.setBuzzer(0)


    # Perception Code
    def get_image(self, img):
        copy_img = img.copy()
        self.h = img.shape[0]
        self.w = img.shape[1]
        return copy_img

    def resize_and_smooth(self, img):
        img_resize = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        img_gb = cv2.GaussianBlur(img_resize, (11, 11), 11)
        if self.get_roi and self.start_pick_up:
            self.get_roi = False
            img_gb = getMaskROI(img_gb, self.roi, self.size)
        return cv2.cvtColor(img_gb, cv2.COLOR_BGR2LAB)

    def detect_target_color(self, img_lab):
        # color_range is a dict in LABConfig.py
        for i in color_range:
            if i in self.target_color:
                self.detect_color = i
                frame_mask = self.threshold(img_lab)
                closed_img = self.remove_noise(frame_mask)
                return self.detect_outline(closed_img)
        return 0, 0


    def threshold(self, img_lab):
        return cv2.inRange(img_lab, color_range[self.detect_color][0], color_range[self.detect_color][1])

    def remove_noise(self, frame_mask):
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
                

    def detect_outline(self, closed_img):
        contours = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        return getAreaMaxContour(contours)

    def detect_box(self, area_max_cont):
        self.rect = cv2.minAreaRect(area_max_cont)
        box = np.int0(cv2.boxPoints(self.rect))
        self.box = box
        self.roi = getROI(box)
        self.get_roi = True
        img_cent_x, img_cent_y = getCenter(self.rect, self.roi, self.size, square_length)
        return convertCoordinate(img_cent_x, img_cent_y, self.size)

    def draw(self, img, world_x, world_y):
        cv2.drawContours(img, [self.box], -1, self.range_rgb[self.detect_color], 2)
        cv2.putText(img, '(' + str(world_x) + ',' + str(world_y) + ')', (min(self.box[0, 0], self.box[2, 0]), self.box[2, 1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[self.detect_color], 1)
        

    def do_perception(self, img):
        logging.debug(f"input image is type {type(img)}")
        img_copy = self.get_image(img)
        logging.debug(f"copied image is type {type(img_copy)}")
        cv2.line(img, (0, int(self.h / 2)), (self.w, int(self.h / 2)), (0, 0, 200), 1)
        cv2.line(img, (int(self.w / 2), 0), (int(self.w / 2), self.h), (0, 0, 200), 1)
    
        if not self.is_running:
            return img
        logging.debug("Entering resize_and_smooth")
        lab_img = self.resize_and_smooth(img_copy)
        logging.debug("Leaving resize_and_smooth")
        area_max = 0
        area_max_cont = 0
        logging.debug(f"start_pick_up is {self.start_pick_up}")
        if not self.start_pick_up:
            logging.debug('Entering detect_target_color')
            area_max_cont, area_max = self.detect_target_color(lab_img)
            if area_max > 2500:
                logging.debug('Entering detect_box')
                self.world_x, self.world_y = self.detect_box(area_max_cont)
                logging.debug('About to draw')
                self.draw(img, self.world_x, self.world_y)

                distance = math.sqrt(pow(self.world_x - self.last_x, 2) + pow(self.world_y - self.last_y, 2))
                self.last_x, self.last_y = self.world_x, self.world_y
                self.track = True
                logging.debug('Entering finish_move')
                self.finish_move(distance)
        return img

        

    # All the functions below here are used for arm movement (except run, which is for everything)
        
    def start_move_thread(self):
        th = threading.Thread(target=self.move)
        th.setDaemon(True)
        th.start
    
    
    def stop(self):
        self._stop = True
        self.is_running = True

    
    def exit(self):
        self._stop = True
        self.is_running = False


    def set_color(self):
        if self.detect_color == "red":
            Board.RGB.setPixelColor(0, Board.PixelColor(255, 0, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(255, 0, 0))
            Board.RGB.show()
        elif self.detect_color == "green":
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 255, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 255, 0))
            Board.RGB.show()
        elif self.detect_color == "blue":
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 255))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 255))
            Board.RGB.show()
        else:
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 0))
            Board.RGB.show()

    def reset_arm_position(self):
        Board.setBusServoPulse(1, self.servo_1 - 70, 300)
        time.sleep(0.5)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)
        time.sleep(1.5)

    def move_arm(self):

        if not self.is_running: # 停止以及退出标志位检测
            return True
        Board.setBusServoPulse(1, self.servo_1 - 280, 500)  # 爪子张开
        # 计算夹持器需要旋转的角度
        servo2_angle = getAngle(self.world_X, self.world_Y, self.rotation_angle)
        Board.setBusServoPulse(2, servo2_angle, 500)
        time.sleep(0.8)
        
        if not self.is_running:
            return True
        self.AK.setPitchRangeMoving((self.world_X, self.world_Y, 2), -90, -90, 0, 1000)  # 降低高度
        time.sleep(2)
        
        if not self.is_running:
            return True
        Board.setBusServoPulse(1, self.servo_1, 500)  # 夹持器闭合
        time.sleep(1)
        
        if not self.is_running:
            return True
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((self.world_X, self.world_Y, 12), -90, -90, 0, 1000)  # 机械臂抬起
        time.sleep(1)
        
        if not self.is_running:
            return True
        # 对不同颜色方块进行分类放置
        result = self.AK.setPitchRangeMoving((self.coordinate[self.detect_color][0], self.coordinate[self.detect_color][1], 12), -90, -90, 0)   
        time.sleep(result[2]/1000)
        
        if not self.is_running:
            return True
        servo2_angle = getAngle(self.coordinate[self.detect_color][0], self.coordinate[self.detect_color][1], -90)
        Board.setBusServoPulse(2, servo2_angle, 500)
        time.sleep(0.5)

        if not self.is_running:
            return True
        self.AK.setPitchRangeMoving((self.coordinate[self.detect_color][0], self.coordinate[self.detect_color][1], self.coordinate[self.detect_color][2] + 3), -90, -90, 0, 500)
        time.sleep(0.5)
        
        if not self.is_running:
            return True
        self.AK.setPitchRangeMoving((self.coordinate[self.detect_color]), -90, -90, 0, 1000)
        time.sleep(0.8)
        
        if not self.is_running:
            return True
        Board.setBusServoPulse(1, self.servo_1 - 200, 500)
        time.sleep(0.8)
        
        if not self.is_running:
            return True                    
        self.AK.setPitchRangeMoving((self.coordinate[self.detect_color][0], self.coordinate[self.detect_color][1], 12), -90, -90, 0, 800)
        time.sleep(0.8)
        return False

    def check_unreachable(self, result):
        if result == False:
            self.unreachable = True
        else:
            self.unreachable = False

    def finish_move(self, distance):
        if self.action_finish:
            if distance < 0.3:
                self.center_list.extend((self.world_x, self.world_y))
                self.count += 1

                if self.start_count_t1:
                    self.start_count_t1 = False
                    self.t1 = time.time()
                if time.time() - self.t1 > 1.5:
                    self.rotation_angle = self.rect[2]
                    self.start_count_t1 = True
                    self.world_X, self.world_Y = np.mean(np.array(self.center_list).reshape(self.count, 2), axis=0)
                    self.count = 0
                    self.center_list = []
                    self.start_pick_up = True
            else:
                self.t1 = time.time()
                self.start_count_t1 = True
                self.count = 0
                self.center_list = []

    # This runs all the code for the movement of the arm
    def move(self):
        while True:
            if self.is_running:
                if self.first_move and self.start_pick_up:
                    self.action_finish = False
                    self.set_color()
                    self.setBuzzer(0.1)
                    result = self.AK.setPitchRangeMoving((self.world_X, self.world_Y - 2, 5), -90, -90, 0)
                    
                    self.check_unreachable(result)
                    time.sleep(result[2] / 1000)
                    self.start_pick_up = False
                    self.first_move = False
                    self.action_finish = True
                elif not self.first_move and not self.unreachable:
                    self.set_color()
                    if self.track:
                        if not self.is_running:
                            continue
                        self.AK.setPitchRangeMoving((self.world_x, self.world_y - 2, 5), -90, -90, 0, 20)
                        time.sleep(0.02)
                        self.track = False
                    if self.start_pick_up:
                        self.action_finish = False
                        if self.move_arm():
                            continue

                        self.init()
                        time.sleep(1.5)

                        self.detect_color = 'None'
                        self.first_move = True

                        self.get_roi = False 
                        self.action_finish = True
                        self.start_pick_up = False
                        self.set_color()
                    else:
                        time.sleep(0.01)
            else:
                if self._stop:
                    self._stop = False
                    self.reset_arm_position()
                time.sleep(0.01)


    def run(self):
        self.start_move_thread()
        self.init()
        self.start()
        self.target_color = ('red', )
        cam = Camera.Camera()
        cam.camera_open()
        while True:
            img = cam.frame
            if img is not None:
                frame = img.copy()
                logging.debug("Entering do_perception")
                Frame = self.do_perception(frame)
                logging.debug("Leaving do_perception")
            
                cv2.imshow("img", Frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
        cam.camera_close()
        cv2.destroyAllWindows()






if __name__ == '__main__':
    arm = My_Arm()
    arm.run()