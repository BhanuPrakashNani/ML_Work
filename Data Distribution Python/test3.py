import numpy as np
import matplotlib.pyplot as plb
import random
import csv

from test import arr
from test2 import arr1

a1 = []
a2 = []

t = []

for i in range(1000):
    a3 = random.choice(arr)
    a1.append(a3)
    
for i in range(1000):
    a4 = random.choice(arr1)
    a2.append(a4)
    
t = a1 + a2
t = np.array([a1,a2])
t = np.average(t,axis=0)
bins = plb.hist(t,14,normed=True)
plb.grid(axis='x',alpha=1)
plb.grid(axis='y',alpha=1)
plb.show()

with open('points.csv', 'w') as csvfile:
    fieldnames = ['points']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in t:
        writer.writerow({'points':i})

