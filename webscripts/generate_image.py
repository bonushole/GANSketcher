#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import base64
from email import generator
import os
import json
from gan.models import Generator
from write_log import write_log
import sys
from PIL import Image
import io
import numpy as np


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

# takes in saved file and converts to tensor obj
img_bytes = tf.io.read_file('temp_file.jpg')
tensor_flow_obj = tf.image.decode_jpeg(img_bytes)

# resises image and sets peper dimensions to mathc the expected [none, 256, 256, 3]
tensor_flow_obj = tf.image.resize(tensor_flow_obj, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
tensor_flow_obj = tf.expand_dims(tensor_flow_obj, 0)

# creates generator and loads last checkpoint
generator = Generator()
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint('../training/training_checkpoints/paintings')).expect_partial()

# gets image prediction from generator
prediction = generator(tensor_flow_obj, training=False)
result_bytes_tensor_obj = prediction[0]

# converts reults from tensorflow object back to jpeg
result_bytes = tf.image.convert_image_dtype(result_bytes_tensor_obj, dtype=tf.uint8)
img_bytes_io = io.BytesIO()
Image.fromarray(np.array(result_bytes)).save(img_bytes_io, format='JPEG')
result_bytes = img_bytes_io.getvalue()

# encodes and prints bacl to standard out
image_base64 = base64.b64encode(result_bytes).decode('ascii')
print(json.dumps({'img': image_base64}))


