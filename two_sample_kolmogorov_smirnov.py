import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt
n=8
m=11


def get_two_sample_tks():
    def ecdf_x(t):
        count = 0
        for i in range(n):
            if (rank_x[i] <= t):
                count = count + 1
        return count / n

    def ecdf_y(t):
        count = 0
        for i in range(m):
            if (rank_y[i] <= t):
                count = count + 1
        return count / m

    pooled_ranks = np.random.permutation(m + n)
    rank_x = pooled_ranks[0:n]
    rank_y = pooled_ranks[n:n + m]
    rank_x = np.sort(rank_x)
    rank_y = np.sort(rank_y)
    j=0
    arr=np.zeros(2*(m+n))
    for i in range(m+n):
        arr[j]=abs(ecdf_x(i)-ecdf_y(i))
        j=j+1
        arr[j]=abs(ecdf_x(i-0.1)-ecdf_y(i-0.1))
        j=j+1
    return np.max(arr)

statistic_vector=np.zeros(100)
for i in range(100):
    statistic_vector[i]=get_two_sample_tks()

sbn.kdeplot(statistic_vector)
plt.show()