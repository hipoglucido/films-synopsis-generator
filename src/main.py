import os
import data
import settings
import train

import numpy as np

def test_generator():
    c = data.Generator()

    from time import sleep 
    t = 5000
    for i in range(1):
        a,b = c.generate().__next__()
        t-=1
        for i in range(a[0].shape[0]):
            print(c.to_genre(a[0][i]))
            print(c.to_synopsis(a[1][i]))
            print(c.to_synopsis(np.nonzero(b[i])[0]))
            print('_______________________________________')
        
        #sleep(0.1)
if __name__ == '__main__':
    network = train.Network()
    network.load_generator()
    network.build()
    network.load_weights()
    network.compile()
    network.train()
