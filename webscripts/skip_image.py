#!/usr/bin/env python3

import base64
import os
import json
import sys
from write_log import write_log

sys.stderr = open('./log', 'a')

write_log('\n\n')
write_log('skip_image')
write_log('\n\n')

GENERATED_FOLDER = '../images/generated/train'

print("Content-Type: text/html;charset=utf-8")
print()

get_args = {arg.split('=')[0]: arg.split('=')[1]
    for arg in os.getenv('QUERY_STRING').split('&')}

write_log(get_args)

print('nothing else to return')
        

