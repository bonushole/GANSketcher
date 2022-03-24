#!/usr/bin/env python3

import base64
import os
import json
import sys
from write_log import write_log

sys.stderr = open('./log', 'a')

write_log('save_image')

GENERATED_FOLDER = '../images/generated/train'

print("Content-Type: text/html;charset=utf-8")
print()

post_args = json.loads(sys.stdin.read())

img_bytes = base64.b64decode(post_args['img'].encode('utf-16'))
######

# This is where the generator should be run.
# 'result_bytes' should be the generator result converted to bytes.

result_bytes = img_bytes
######

image_base64 = base64.b64encode(result_bytes).decode('ascii')
print(json.dumps({'img': image_base64}))

        

