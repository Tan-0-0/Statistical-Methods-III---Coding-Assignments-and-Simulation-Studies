import numpy as np

siz=15
x=np.random.standard_cauchy(siz)

def der_log_like(theta):
    s=0
    for i in range(siz):
        s=s+((x[i]-theta)/(1+(x[i]-theta)**2))
    return s

n_iterations=2
theta=np.median(x)
print("Initial value :",np.median(x))

for i in range(n_iterations):
    theta=theta+4*der_log_like(theta)*1/siz
    print("After ", i+1 , "iterations: estimate of location is : ", theta)


