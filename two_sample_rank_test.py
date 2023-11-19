import numpy as np
import seaborn as sbn
from matplotlib import  pyplot as plt
n=5
m=7

def get_rank_dist():
    rank_vec=np.random.permutation(n+m)
    rank_sum_y=0
    for i in range(m):
        rank_sum_y=rank_sum_y+rank_vec[n+i]
    return rank_sum_y

statistic_vector=np.zeros(1000)
for i in range(1000):
    statistic_vector[i]=get_rank_dist()

K=np.quantile(statistic_vector,0.95)
print("95th quantile :",K)

sbn.kdeplot(statistic_vector)
plt.show()