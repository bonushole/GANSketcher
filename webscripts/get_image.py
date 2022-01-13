#!/usr/bin/env python3

import base64
import os
import json
import sys
from write_log import write_log

sys.stderr = sys.stdout

SOURCE_DIR = '../images/raw/images/Michelangelo'
SKIP_DIR = '../images/generated/skip'
TEST_DIR = '../images/generated/test'
TRAIN_DIR = '../images/generated/train'

print("Content-Type: text/html;charset=utf-8")
print()

files = os.listdir(SOURCE_DIR)
already_used = set([f for d in [SKIP_DIR, TEST_DIR, TRAIN_DIR] for f in os.listdir(d)])
unused = [f for f in files if f not in already_used]
write_log(str(unused))

file_name = unused[0]
image_file = open(os.path.join(SOURCE_DIR, file_name), 'rb')
image_base64 = base64.b64encode(image_file.read()).decode('ascii')
print(json.dumps({'img': image_base64, 'name': file_name}))
