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

#post_args = {arg.split('=')[0]: arg.split('=')[1]
#    for arg in sys.stdin.read().split('&')}

post_args = json.loads(sys.stdin.read())

save_name = post_args['name']
if save_name in os.listdir(GENERATED_FOLDER):
    name, extension = post_args['name'].split('.')
    i = 1
    while True:
        save_name = '{}({}).{}'.format(name, i, extension)
        if new_name not in os.listdir(GENERATED_FOLDER):
            break
        i += 1
#write_log(post_args['img'])
write_log('\n')
write_log(len(post_args['img']))
write_log('\n')

#write_log(img_bytes)


img_bytes = base64.b64decode(post_args['img'].encode('utf-16'))
image = Image.open(io.BytesIO(img_bytes))
image = image.crop((image.width//2 + 1, 0, image.width, image.height))
write_log(f'number of colors: {len(image.getcolors())}\n')
if len(image.getcolors()) != 1:
    with open(os.path.join(GENERATED_FOLDER, save_name), 'wb') as f:
        f.write(img_bytes)

print('nothing else to return')
        

