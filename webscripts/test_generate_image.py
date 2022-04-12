#!/usr/bin/env python3
import base64
import os
import json
import sys
import random
from write_log import write_log
from subprocess import Popen, PIPE, STDOUT
from PIL import Image
import io

image_file = open('./test.jpg', 'rb')
image = Image.open(io.BytesIO(image_file.read()))
image = image.crop((image.width//2, 0, image.width, image.height))
img_bytes_io = io.BytesIO()
image.save(img_bytes_io, format='JPEG')


image_base64 = base64.b64encode(img_bytes_io.getvalue()).decode('ascii')
img_str = json.dumps({'img': image_base64})


p = Popen(['./generate_image.py'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)    
stdout = p.communicate(img_str.encode('ascii'))[0].decode()
print(stdout)
cleaned = stdout.split('\n')[2]

post_args = json.loads(cleaned)

with open('./test_output.jpg', 'wb') as f:
    img_bytes = base64.b64decode(post_args['img'].encode('utf-16'))
    f.write(img_bytes)
