import numpy as np
import copy as cp

class Mark:
    def __init__(self, movie, value, timestamp):
        self.movie = movie
        self.value = value
        self.timestamp = timestamp


users = {}
line_list = []
with open('u.data') as file:
    for line in file:
        line_list = line.split('/t')
        #line_mark = Mark(int(line[1]), int([line[2]]), int(line[3]))
        if(line[0] not in users):
            #users[int(line[0])] = [].append(line_mark)
            break

dict = {}

m = Mark(1, 2, 3)
print(m)
dict[20] = [].append(cp.deepcopy(m))
print(dict)
