from PIL import Image, ImageFilter
import PIL.ImageOps

MY_FILTER = ImageFilter.Kernel((5, 5),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,), 1, 0)

def extract_edges(image):
    image = image.convert('L')
    filter_stack = [ImageFilter.SHARPEN, ImageFilter.SMOOTH, MY_FILTER]
    for filter in filter_stack:
        image = image.filter(filter)
    image = PIL.ImageOps.invert(image)
    return image

