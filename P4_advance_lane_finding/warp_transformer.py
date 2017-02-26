import cv2
import numpy as np

# for perspective transforms
class WarpTransformer():
    def __init__(self, src, dst):
        self.birdeye_matrix = cv2.getPerspectiveTransform(src, dst)
        self.normal_matrix = cv2.getPerspectiveTransform(dst, src)

    # tranform to birdeye view
    def to_birdview(self, img):
        return cv2.warpPerspective(img, self.birdeye_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    # tranform to normal view
    def to_normal(self, img):
        return cv2.warpPerspective(img, self.normal_matrix, (img.shape[1], img.shape[0]))
