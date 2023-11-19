import numpy as np
import scipy.stats as sps
from matplotlib import pyplot as plt

#nominal size fixed at 5 percent

p=np.zeros(180)
k=0

for i in range(100,1000,5):
    p[k]=i/1000
    k=k+1

p0=0.9
N=50
n=1000

def actual_size_est(p):
    t1 = np.zeros(n)
    t2 = np.zeros(n)
    X = np.random.binomial(N, p, n)
    for i in range(n):
        t1[i] = ((X[i] / N) - p) / np.sqrt((X[i] / N) * (1 - X[i] / N) / N)
        t2[i] = ((X[i] / N) - p) / np.sqrt(p * (1 - p) / N)
    k = sps.norm.ppf(0.975, 0, 1)

    count1=0
    count2=0
    for i in range(n):
        if abs(t1[i]>k):
            count1+=1
        if abs(t2[i]>k):
            count2+=1

    return np.array([count1/n,count2/n])

def power_est(p0,p):
    t1=np.zeros(n)
    t2=np.zeros(n)
    X=np.random.binomial(N,p,n)
    for i in range(n):
        t1[i]=((X[i]/N)-p0)/np.sqrt((X[i]/N)*(1-X[i]/N)/N)
        t2[i]=((X[i]/N)-p0)/np.sqrt(p0*(1-p0)/N)
    k=sps.norm.ppf(0.975,0,1)

    count1=0
    count2=0
    for i in range(n):
        if abs(t1[i])<=k:
            count1+=1
        if abs(t2[i])<=k:
            count2+=1

    return np.array([(n-count1)/n,(n-count2)/n])

print("Size of [X/N-p0]/sqrt[X/N(1-X/N)*1/N], Size of [X/N-p0]/sqrt[p0(1-p0)/N]:",actual_size_est(p0))
print("Power of [X/N-p0]/sqrt[X/N(1-X/N)*1/N], Power of [X/N-p0]/sqrt[p0(1-p0)/N]:")
for i in range(np.size(p)):
    print("p :",p[i],power_est(p0,p[i]))

arr=np.zeros(np.size(p))
for i in range(np.size(p)):
    arr[i]=power_est(p0,p[i])[0]-power_est(p0,p[i])[1]
plt.plot(p,arr)
plt.show()