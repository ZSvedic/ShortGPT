####################################################################################################

import time

class Timer:
    ''' Usage example: 
        with Timer('Running my_function...'):
            my_function(*args, **kwargs) '''
    
    def __init__(self, msg: str):
        print(msg)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        Timer.last_elapsed_time = time.time() - self.start_time
        print(f"...total time: {Timer.last_elapsed_time:.2f} sec")

    # Static variable, safe to use since Python is single-threaded:  
    last_elapsed_time = 0

####################################################################################################