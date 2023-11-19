import numpy as np

complete_data=np.array([(78,128),(89,118),(93,134),(76,117),(85,130),(84,122),(86,131),(73,137),(97,119),(88,123),(79,135)])
missing_f=np.array([127,129,125,121])
missing_pp=np.array([91,77,95,92,81])

def exp_y_given_x(meanx,meany,varx,vary,cor,x):
    return (meany+cor*vary**0.5/varx**0.5 * (x-meanx))

class em_algorithm:

    def __init__(self,mean,cor,var,data):
        self.mean_vec=mean
        self.cor=cor
        self.var_vec=var
        self.data=data

    def getMLE(self):
        self.mean_vec=sum(self.data)/np.size(self.data,0)
        self.cor=np.corrcoef(self.data[:,0],self.data[:,1])
        self.var_vec=np.array(self.data,axis=0)

    def e_m_steps(self):
        f_complete = np.zeros((len(missing_f), 2))
        for i in range(len(missing_f)):
            f_complete[i, 1] = missing_f[i]
            f_complete[i, 0] = exp_y_given_x(meanx=self.mean_vec[1], meany=self.mean_vec[0], varx=self.var_vec[1],
                                             vary=self.var_vec[0], cor=self.cor, x=missing_f[i])
        pp_complete = np.zeros((len(missing_pp), 2))
        for i in range(len(missing_pp)):
            pp_complete[i, 0] = missing_pp[i]
            pp_complete[i, 1] = exp_y_given_x(self.mean_vec[0], self.mean_vec[1], self.var_vec[0], self.var_vec[1], self.cor,
                                              missing_pp[i])
        self.data = np.append(complete_data, f_complete)
        self.data = np.append(self.data, pp_complete).reshape(-1, 2)
        self.mean_vec=sum(self.data)/np.size(self.data,0)
        self.var_vec=np.array([np.var(self.data[:,0]),np.var(self.data[:,1])])
        self.cor=np.corrcoef(self.data[:,0],self.data[:,1])[0,1]
e_obj=em_algorithm(mean=sum(complete_data)/np.size(complete_data,0),var=np.array([np.var(complete_data[:,0]),np.var(complete_data[:,1])]),cor=np.corrcoef(complete_data[:,0],complete_data[:,1])[0,1],data=complete_data)
for i in range(1000):
    e_obj.e_m_steps()
print("Mean of fasting:",e_obj.mean_vec[0])
print("Mean of pp:",e_obj.mean_vec[1])
print("Correlation:",e_obj.cor)
print("Variance of fasting:",e_obj.var_vec[0])
print("Variance of pp:",e_obj.var_vec[1])

