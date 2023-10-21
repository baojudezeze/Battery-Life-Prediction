import csv

import numpy as np

a = np.arange(0, 470, 10)
b = np.random.normal(-1, 4, (47,))
c = a - b

with open('./111.csv', "w", encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(list(c))

print(1)
