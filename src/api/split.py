import os
import random

import os

def split(datadir='src/data/open/'):
    set_size = len(os.listdir(datadir))

    test_ratio = int(set_size * 0.25)

    print(set_size)
    print(test_ratio)

    unique_sequence = random.sample(list(range(set_size)), len(list(range(set_size))))

    test_set = unique_sequence[:test_ratio]

    count = 0
    for i in test_set:
        os.rename(datadir+'open.'+str(i)+".jpg", 'src/data/test/open/open' + '.' + str(count) + ".jpg")
        count += 1
        
split()