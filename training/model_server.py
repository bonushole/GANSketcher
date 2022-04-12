import selectors
import socket
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from PIL import Image
import numpy as np
import io

sel = selectors.DefaultSelector()
generator = tf.keras.models.load_model('saved_model', compile=False)

def model_predict(filename):
    # takes in saved file and converts to tensor obj
    img_bytes = tf.io.read_file(filename)
    tensor_flow_obj = tf.image.decode_jpeg(img_bytes)
    tensor_flow_obj = tf.cast(tensor_flow_obj, tf.float32)
    tensor_flow_obj = (tensor_flow_obj / 127.5) - 1


    # resises image and sets peper dimensions to mathc the expected [none, 256, 256, 3]
    tensor_flow_obj = tf.image.resize(tensor_flow_obj, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tensor_flow_obj = tf.expand_dims(tensor_flow_obj, 0)


    #test_dataset = tf.data.Dataset.list_files('./test_file.jpg')
    #test_dataset = test_dataset.map(load_image_test)

    # creates generator and loads last checkpoint
    #generator = Generator()
    #checkpoint = tf.train.Checkpoint(generator=generator)
    #write_log(tf.train.latest_checkpoint('../training/training_checkpoints/edges2handbags_transfer_paintings/'))
    #checkpoint.restore(tf.train.latest_checkpoint('../training/training_checkpoints/edges2handbags_transfer_paintings/'))

    # gets image prediction from generator
    #prediction = generator(tensor_flow_obj, training=True)

    prediction = generator(tensor_flow_obj, training=True)
    prediction = (prediction*.5 + .5)
    #write_log(np.array(prediction))
    result_bytes_tensor_obj = prediction[0]

    # converts reults from tensorflow object back to jpeg
    result_bytes = tf.image.convert_image_dtype(result_bytes_tensor_obj, dtype=tf.uint8)
    img_bytes_io = io.BytesIO()
    Image.fromarray(np.array(result_bytes)).save(img_bytes_io, format='JPEG')
    result_bytes = img_bytes_io.getvalue()
    return result_bytes

def accept(sock, mask):
    conn, addr = sock.accept()  # Should be ready
    print('accepted', conn, 'from', addr)
    conn.setblocking(False)
    sel.register(conn, selectors.EVENT_READ, read)

def read(conn, mask):
    data = conn.recv(1000)  # Should be ready
    if data:
        print('echoing', repr(data), 'to', conn)
        conn.send(model_predict(data))  # Hope it won't block
    else:
        print('closing', conn)
        sel.unregister(conn)
        conn.close()

sock = socket.socket()
sock.bind(('localhost', 1234))
sock.listen(100)
sock.setblocking(False)
sel.register(sock, selectors.EVENT_READ, accept)

while True:
    events = sel.select()
    for key, mask in events:
        callback = key.data
        callback(key.fileobj, mask)
