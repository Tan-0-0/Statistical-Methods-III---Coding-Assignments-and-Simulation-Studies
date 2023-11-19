import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt
n=100
def getT_KS():
    u=np.random.random(n)
    u=np.sort(u)
    arr=np.zeros(2*n)
    arr[0]=u[0]
    arr[2*n-1]=1-u[n-1]
    k=1
    for i in range(n):
        arr[k]=abs(u[i]-(i+1)/n)
        if (i!=n-1):
            arr[k+1]=abs(u[i+1]-(i+1)/n)
        k=k+2
    return (np.max(arr))

def getT_KS_sample(u):
    u=np.sort(u)
    n=np.size(u)
    arr=np.zeros(2*n)
    arr[0]=u[0]
    arr[2*n-1]=1-u[n-1]
    k=1
    for i in range(n):
        arr[k]=abs(u[i]-(i+1)/n)
        if (i!=n-1):
            arr[k+1]=abs(u[i+1]-(i+1)/n)
        k=k+2
    return (np.max(arr))

vec=np.zeros(100)

for i in range(100):
    vec[i]=getT_KS()

K=np.quantile(vec,0.95)

print("95th percentile: ",K)

#now estimating actual size

sample_vec=np.zeros([100,100])
T_ks=np.zeros(100)
count=0
for i in range(100):
    sample_vec[i]=np.random.uniform(0,1,100)
    T_ks[i]=getT_KS_sample(sample_vec[i])
    if T_ks[i]>K:
        count=count+1

print("Approximate Actual Size for nominal size 5%: ",count/100)

#now estimating power

#a. beta(2,3)
count=0
for i in range(100):
    sample_vec[i]=np.random.beta(2,3,100)
    T_ks[i]=getT_KS_sample(sample_vec[i])
    if T_ks[i]<=K:
        count=count+1
print("Power for beta(2,3) is :",(100-count)/100)

#b. u(0,0.9)
#a. beta(2,3)
count=0
for i in range(100):
    sample_vec[i]=np.random.uniform(0,0.9,100)
    T_ks[i]=getT_KS_sample(sample_vec[i])
    if T_ks[i]<=K:
        count=count+1
print("Power for uniform(0,0.9) is :",(100-count)/100)