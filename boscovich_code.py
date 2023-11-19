import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

class line:
    def __init__(self,slope,intercept):
        self.slope=slope
        self.intercept=intercept

class Geodesic:
    sine = np.array([0, 3014, 4648, 5762, 8386]).reshape(-1, 1)
    length = np.array([56751, 57037, 56979, 57074, 57422]).reshape(-1, 1)

    def abs_dev(self,l):
        len_pred=np.repeat(l.intercept,len(Geodesic.sine)).reshape(-1,1)+l.slope*Geodesic.sine
        error=abs(len_pred-Geodesic.length)
        return sum(error)

    def coeff_perf(self,l):
        tsd=sum(abs(Geodesic.length-np.repeat(np.mean(Geodesic.length),len(Geodesic.length)).reshape(-1,1)))
        return (tsd-Geodesic.abs_dev(self,l))/tsd

    def boscovich_length_ols(self):
        model=LinearRegression()
        model.fit(Geodesic.sine,Geodesic.length)
        print("R squared:",model.score(Geodesic.sine,Geodesic.length))
        print("Slope:",model.coef_)
        print("Intercept:",model.intercept_)
        length_predicted=LinearRegression.predict(model,Geodesic.sine)
        plt.scatter(Geodesic.sine,Geodesic.length,color='r')
        plt.plot(Geodesic.sine,length_predicted,label="OLS")
        c=np.array(range(90)).reshape(-1,1)
        print("Meridian quadrant:",sum(np.repeat(model.intercept_,90).reshape(-1,1)+model.coef_*c))

    def boscovich_length_lad(self):
        lad_best=line(0,0)
        ts=int(len(Geodesic.sine)*(len(Geodesic.sine)-1)/2)
        k=0
        l=line(0,0)
        for i in range(len(Geodesic.sine)):
            for j in range(i+1,len(Geodesic.sine)):
                l.slope=(Geodesic.length[j]-Geodesic.length[i])/(Geodesic.sine[j]-Geodesic.sine[i])
                l.intercept=(Geodesic.length[j]-l.slope*Geodesic.sine[j])
                k+=1
                if (i==1 and j==2):
                    lad_best=l
                elif (Geodesic.abs_dev(self,l)<Geodesic.abs_dev(self,lad_best)):
                    lad_best=l
        l_pred=np.repeat(lad_best.intercept,len(Geodesic.sine)).reshape(-1,1)+lad_best.slope*Geodesic.sine
        print("Coefficient of performance:",Geodesic.coeff_perf(self,lad_best))
        print("Slope:",lad_best.slope)
        print("Intercept:",lad_best.intercept)
        plt.scatter(Geodesic.sine,Geodesic.length,color='r')
        plt.plot(Geodesic.sine,l_pred,label="LAD")
        c = np.array(range(90)).reshape(-1,1)
        print("Meridian quadrant:", sum(np.repeat(lad_best.intercept, 90).reshape(-1, 1) + lad_best.slope * c))


gd=Geodesic()
gd.boscovich_length_lad()
gd.boscovich_length_ols()
plt.legend()
plt.xlabel("(sine of latitude)^2 * 10^4")
plt.ylabel("arc length (toise)")
plt.show()