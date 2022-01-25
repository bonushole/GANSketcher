#!/usr/bin/env python3

import base64
import os
import json
import sys
import random
from write_log import write_log

random.seed()#a=4324325)

sys.stderr = sys.stdout

ARTISTS = ['Michelangelo', 'Mikhail_Vrubel', 'Frida_Kahlo', 'Titian']

SOURCE_DIR = '../images/raw/images/'
SKIP_DIR = '../images/generated/skip'
TEST_DIR = '../images/generated/test'
TRAIN_DIR = '../images/generated/train'

print("Content-Type: text/html;charset=utf-8")
print()

files = [os.path.join(a,f) for a in ARTISTS for f in os.listdir(os.path.join(SOURCE_DIR, a))]
already_used = set([f for d in [SKIP_DIR, TEST_DIR, TRAIN_DIR] for f in os.listdir(d)])
unused = [f for f in files if f not in already_used]
#write_log(str(unused))

file_name = random.choice(unused)
image_file = open(os.path.join(SOURCE_DIR, file_name), 'rb')
image_base64 = base64.b64encode(image_file.read()).decode('ascii')
print(json.dumps({'img': image_base64, 'name': file_name}))
