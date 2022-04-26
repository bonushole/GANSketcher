from PIL import Image, ImageFilter
import PIL.ImageOps
import cv2
import numpy as np

MY_FILTER = ImageFilter.Kernel((5, 5),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,), 1, 0)

def extract_edges(image):
    image = image.convert('L')
    filter_stack = [ImageFilter.SHARPEN, ImageFilter.SMOOTH, MY_FILTER]
    for filter in filter_stack:
        image = image.filter(filter)
    image = PIL.ImageOps.invert(image)
    return image

def new_extract(cv_img):
    blur = cv2.GaussianBlur(cv_img, (3, 3), 0)
    sigma = np.std(blur)
    mean = np.mean(blur)
    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))

    edge = cv2.Canny(blur, lower, upper)
    edge = cv2.bitwise_not(edge)
    return edge
