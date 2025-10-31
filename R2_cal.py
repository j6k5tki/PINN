import numpy as np
import matplotlib.pyplot as plt

Label = [12507,116019,106491]
DATE = [202503082212,202503082109,202503092219]
NUM = ['JET','W7-X','KSTAR']

def RMSE(A,B):
    return np.sqrt(np.mean((A - B) ** 2))

def Rel_err(A,B):
    return np.mean(np.abs((A - B) / B))

def R_2(A,B):
    ss_res = np.sum((A - B) ** 2)
    ss_tot = np.sum((B - np.mean(B)) ** 2)
    return 1 - (ss_res / ss_tot)

for label,num,date in zip(Label,NUM,DATE):

    a = np.loadtxt('./%s/DV_data_%d.txt'%(num,label))
    X = np.load('./%s/inverse/inverse_X_%d.npy'%(num,date))
    Y = np.load('./%s/inverse/inverse_Y_%d.npy'%(num,date))
    
    R = X[:,0].reshape(50,50)[0,:]/0.43
    d = Y[:,1].reshape(50,50)[0,:]
    v = Y[:,2].reshape(50,50)[0,:]
    
    D = np.interp(R,a[0],a[1])
    V = np.interp(R,a[0],a[2])
    
    print('%s R^2'%num)
    print('- D(R2): %.2f'%(R_2(D,d)))
    print('- V(R2): %.2f'%(R_2(V,v)))
    print('- V/D(R2): %.2f\n'%(R_2(V/D,v/d)))
    
    ind = np.argmin(abs(R-0.95))
    
    print('%s R^2(0.00-0.95)'%num)
    print('- D(R2): %.2f'%(R_2(D[:ind],d[:ind])))
    print('- V(R2): %.2f'%(R_2(V[:ind],v[:ind])))
    print('- V/D(R2): %.2f\n'%(R_2(V[:ind]/D[:ind],v[:ind]/d[:ind])))