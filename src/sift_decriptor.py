import cv2
import numpy as np
import matplotlib.pyplot as plt


def sift_descriptor(image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return des

def downsample(image,length = 50):
    height, width = image.shape
    if height < width:
        ratio = length/height
        new_height = length
        new_width = int(width*ratio)
    else:
        ratio = length/width
        new_width = length
        new_height = int(height*ratio)
    return cv2.resize(image, (new_width, new_height))

