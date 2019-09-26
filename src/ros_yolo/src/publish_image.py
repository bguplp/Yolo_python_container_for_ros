#!/usr/bin/env python
import sys, time

# numpy and scipy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage

# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE = False


class ImagePublisher:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
	self.img = cv2.imread('src/darknet/data/dog.jpg', cv2.IMREAD_COLOR)	
        self.image_pub = rospy.Publisher("/image_raw/compressed",
                                         CompressedImage)
        # self.bridge = CvBridge()

    def publish_image(self):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''

        #### direct conversion to CV2 ####
        image_np = self.img 
        # np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

        #### Feature detectors using CV2 ####
        # "","Grid","Pyramid" +
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

        # self.subscriber.unregister()


def main(args):
    '''Initializes and cleanup ros node'''
    ic = ImagePublisher()
    rospy.init_node('simple_image_publisher', anonymous=True)
    try:
        while not rospy.is_shutdown():
            ic.publish_image()
            rospy.sleep(0.01)
    except KeyboardInterrupt:
        print
        "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
