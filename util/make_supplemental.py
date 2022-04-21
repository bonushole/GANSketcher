import os
from PIL import Image
from extract_edges import extract_edges

def resize_image(img):
    target_ratio = 1.0
    image_ratio = img.width/img.height

    offsetx = img.height * (image_ratio - target_ratio) / 2 if image_ratio > target_ratio else 0
    offsety = img.width * (image_ratio - target_ratio) / 2 if image_ratio < target_ratio else 0
    img = img.crop((offsetx, offsety, img.width - offsetx, img.height - offsety))
    img = img.resize((512, 512))
    return img
    
def get_combined(filename):
    resized_image = resize_image(Image.open(filename))
    edges_image = extract_edges(resized_image)
    combined = Image.new('RGB', (512*2, 512))
    combined.paste(resized_image)
    combined.paste(edges_image, (512, 0))
    return combined
    

ARTISTS = ['Michelangelo', 'Mikhail_Vrubel', 'Frida_Kahlo', 'Titian', 'Pablo_Picasso', 'Salvador_Dali', 'Diego_Rivera', 'Vincent_van_Gogh']

SOURCE_DIR = '../images/raw/images/'
SKIP_DIR = '../images/generated/skip'
TEST_DIR = '../images/generated/test'
TRAIN_DIR = '../images/generated/train'
SUPPLEMENTAL_DIR = '../images/generated/supplemental'


files = [os.path.join(a,f) for a in ARTISTS for f in os.listdir(os.path.join(SOURCE_DIR, a))]
already_used = set([f for d in [SKIP_DIR, TEST_DIR, TRAIN_DIR] for f in os.listdir(d)])
unused = [f for f in files if f.split('/')[1] not in already_used]
for unused_image in unused:
    comb = get_combined(os.path.join(SOURCE_DIR, unused_image))
    comb.save(os.path.join(SUPPLEMENTAL_DIR, unused_image.split('/')[1]))
    print('.', end='')


