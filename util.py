import os 

class Logger:
    def __init__(self, logger):
        self.logger = logger 
        cur_dir = ''
        
        for d in logger.split(os.sep)[:-1]:
            directory = os.sep.join(cur_dir, d)
            create_dir(directory)

        with open(logger, 'w+') as f:
            pass 

    def write(self, msg):
        with open(self.logger, 'a') as f:
            print(str(msg), file = f)
    
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)