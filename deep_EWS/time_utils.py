import time 

def timing_decorator(func):
    def wrapper(*args):
        start=time.time()
        r=func(*args)
        print('Total time of execution=%f seconds' % (time.time() - start))
        return r
    return wrapper
