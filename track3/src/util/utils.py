def str2bool(string):
    if string == "True":
        return True
    else:
        return False
    
def generate_timestamp():
    import time
    timestamp = time.strftime("%m%d%H%M")
    return timestamp

