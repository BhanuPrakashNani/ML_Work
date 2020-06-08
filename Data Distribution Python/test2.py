import numpy as npy
import csv
import matplotlib.pyplot as plt
arr1 = npy.random.normal(10, 20, 300)
bins = plt.hist(arr1, 14, normed=True)
plt.show()

with open('points2.csv', 'w') as csvfile:
    fieldnames = ['points']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in arr1:
        writer.writerow({'points':i})
