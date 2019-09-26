#!/usr/bin/env python
import rospy
import numpy as np
import sys
# OpenCV
import cv2
from std_msgs.msg import String
# Ros Messages
from sensor_msgs.msg import CompressedImage
try:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
except:
    print 'You must install the vision_msgs library!!!!!!!!!!!!!!!!!!!! '
    print 'sudo apt-get install ros-DISTRO-vision-msgs'

    print 'sudo apt-get install ros-kinetic-vision-msgs'
    exit(0)
# sys.path.insert(1, 'darknet/')
sys.path.insert(1, 'darknet/python')
import rospkg
rospack = rospkg.RosPack()
WORKING_PATH = rospack.get_path('ros_yolo')
from my_darknet_handler import *
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError



class ROSHandler:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # subscribed Topic
        self.latest_finish_time = rospy.Time.now()
        self.net = load_net(WORKING_PATH+'/src/darknet/'+"cfg/yolov3-tiny.cfg", WORKING_PATH+'/src/darknet/'+"data/yolov3-tiny.weights", 0)
        self.meta = load_meta(WORKING_PATH+'/src/darknet/'+"cfg/coco.data")
        self.res_pub = rospy.Publisher('/yolo/results', Detection2DArray, queue_size=1)
        rospy.Subscriber("/image_raw/compressed",
                         CompressedImage, self.callback, queue_size=2)
        self.database = []
        with open(WORKING_PATH+'/src/darknet/'+"data/coco.names", 'r')  as f:
            for line in f:
                self.database.append(line[:-1])
        print self.database

    def callback(self, ros_data):
        '''Callback function of subscribed topic. '''
        if ros_data.header.stamp < self.latest_finish_time:
            return
        #### direct conversion to CV2 ####
        now = rospy.Time.now()
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # dim = (int(len(image_np / 20)), int(len(image_np[0]) / 20))
        # cv2.resize(image_np, dim)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        results = detect(self.net, self.meta, image_np)
        detections = Detection2DArray()
        for result in results:
            detection = Detection2D()
            detection.header = ros_data.header
            res = ObjectHypothesisWithPose()
            res.id = self.get_id(result[0])
            res.score = result[1]
            detection.results.append(res)
            detection.bbox.size_x = result[2][2]
            detection.bbox.size_y = result[2][3]
            detection.bbox.center.x = result[2][0]
            detection.bbox.center.y = result[2][1]
            detections.detections.append(detection)
        self.res_pub.publish(detections)
        rospy.loginfo_throttle(1, 'Took yolo %s to process image' % ((rospy.Time.now() - now).to_sec()))
        self.latest_finish_time =rospy.Time.now()
        #### Feature detectors using CV2 ####


    def get_id(self, string):
        return self.database.index(string)


def main():
    '''Initializes and cleanup ros node'''
    rospy.init_node('ros_yolo', anonymous=True)
    rh = ROSHandler()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print
        "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()