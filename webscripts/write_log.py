
def write_log(s):
    with open('./log', 'a') as f:
        f.write(str(s))

def redirect_stderr():
    sys.stderr = open('./log', 'a')
