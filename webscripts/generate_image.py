#!/usr/bin/env python3

import os
import base64
import os
import json
#from gan.models import Generator
from write_log import write_log
import sys
import io
import socket


sys.stderr = open('./log', 'a')
write_log('save_image')
GENERATED_FOLDER = '../images/generated/train'

print("Content-Type: text/html;charset=utf-8")
print()

post_args = json.loads(sys.stdin.read())
img_bytes = base64.b64decode(post_args['img'].encode('utf-16'))


# opens and saves file locally bacasue to convert object to tensor flow as below it need to have a file name
with open('./temp_file.jpg', 'wb') as f:
    f.write(img_bytes)
    
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 1234

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(os.path.abspath('./temp_file.jpg'.encode('ascii')))
    all_data = b''
    while True:
        data = s.recv(1024)
        if data:
            all_data += data
        if len(data) < 1024:
            break
write_log(f'received {all_data}')

result_bytes = all_data

# encodes and prints bacl to standard out
image_base64 = base64.b64encode(result_bytes).decode('ascii')
print(json.dumps({'img': image_base64}))


