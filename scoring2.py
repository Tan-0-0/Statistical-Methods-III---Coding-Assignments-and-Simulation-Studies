import numpy as np
import math
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

X=np.array([-1,-0.5,0,0.5,1])
def gen_ber(X,n):
    X=logistic_cdf(0.5+0.5*X)
    r_vec=np.empty(n)
    for i in range(n):
        s=np.random.rand()
        if (s<=X):
            r_vec[i]=1
        else:
            r_vec[i]=0
    return r_vec

def logistic_cdf(x):
    return np.exp(x)/(1+np.exp(x))

def logistic_pdf(x):
    return np.exp(x)/((1+np.exp(x))**2)

x_vec=np.repeat(-1,10)
y_vec=gen_ber(-1,10)

for i in range(4):
    x=-0.5+0.5*i
    x_vec = np.append(x_vec, np.repeat(x, 10))
    y_vec=np.append(y_vec,gen_ber(x,10))


para_vec=np.zeros(2)
p0=0
p1=0

prob_vec=np.zeros(5)
for i in range(5):
    for j in range(10):
        prob_vec[i]=prob_vec[i]+y_vec[10*i+j]
prob_vec=prob_vec/10

for i in range(5):
    prob_vec[i]=np.log(prob_vec[i]/(1-prob_vec[i]))

model=LinearRegression()
model.fit(X.reshape(-1,1),prob_vec)
prob_pred=np.zeros(5)
for i in range(5):
    prob_pred[i]=LinearRegression.predict(model,X[i].reshape(-1,1))[0]

para_vec[0]=model.intercept_
para_vec[1]=model.coef_[0]

plt.scatter(X,prob_vec)
plt.plot(X,prob_pred)

print("Initial estimate: alpha=",para_vec[0]," beta= ",para_vec[1])
a11=0
a12=0
a22=0

for i in range(50):
    a11=a11-logistic_pdf(para_vec[0]+para_vec[1]*x_vec[i])
    a12=a12-logistic_pdf(para_vec[0]+para_vec[1]*x_vec[i])*x_vec[i]
    a22=a22-logistic_pdf(para_vec[0]+para_vec[1]*x_vec[i])*x_vec[i]*x_vec[i]
det=a11*a22-a12*a12
T_inverse=np.array([a22/det, -a12/det,-a12/det, a11/det]).reshape(2,2)

def grad_vec(a,b):
    grad=np.zeros(2)
    for i in range(50):
        grad=grad+(y_vec[i]-logistic_cdf(a+b*x_vec[i]))/(logistic_cdf(a+b*x_vec[i])*(1-logistic_cdf(a+b*x_vec[i])))*logistic_pdf(a+b*x_vec[i])*np.array([1,x_vec[i]])
    return grad

N=1
for i in range(N):
    para_vec=para_vec-np.transpose(np.matmul(T_inverse,np.transpose(grad_vec(para_vec[0],para_vec[1]))))
    for i in range(50):
        a11 = a11 - logistic_pdf(para_vec[0] + para_vec[1] * x_vec[i])
        a12 = a12 - logistic_pdf(para_vec[0] + para_vec[1] * x_vec[i]) * x_vec[i]
        a22 = a22 - logistic_pdf(para_vec[0] + para_vec[1] * x_vec[i]) * x_vec[i] * x_vec[i]
    det = a11 * a22 - a12 * a12
    T_inverse = np.array([a22 / det, -a12 / det, -a12 / det, a11 / det]).reshape(2, 2)
print("Estimate after ",N," iteration(s): alpha =",para_vec[0]," beta =",para_vec[1])
plt.show()