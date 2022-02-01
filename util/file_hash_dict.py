import sys
import hashlib
import os
import json

# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
BASE_PATH = '../images/generated/'

def get_file_hash(filename):
    md5 = hashlib.md5()
    sha1 = hashlib.sha1()

    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
            sha1.update(data)

    return md5.hexdigest()

file_dict = {}
for d in [x for x in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, x))]:
    file_dict[d] = {}
    dir_path = os.path.join(BASE_PATH, d)
    for f in os.listdir(dir_path):
        file_dict[d][f] = get_file_hash(os.path.join(dir_path, f))

with open(os.path.join(BASE_PATH, 'hash_record.json'), 'w') as f:
    f.write(json.dumps(file_dict))

