import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t
sample=np.random.multivariate_normal(mean=[0,0],cov=[[1,0.5],[0.5,1]],size=15)
x=np.zeros(15)
y=np.zeros(15)
for i in range(15):
    x[i]=sample[i][0]
    y[i]=sample[i][1]
model=LinearRegression()
model.fit(x.reshape(-1,1),y.reshape(-1,1))
beta_0_est=model.intercept_[0]
beta_1_est=model.coef_[0]
sse=0
for i in range(15):
    sse=sse+(y[i]-beta_0_est-beta_1_est*x[i])**2
l=np.sqrt(sse/(13*np.sqrt(np.var(x))))
tval=t.ppf(1-0.05,13)
interval_vec=np.array([beta_1_est-tval*l,beta_1_est+tval*l])
print("Actual value of beta_1:",0.5)
print("95% Confidence Interval:",interval_vec)
