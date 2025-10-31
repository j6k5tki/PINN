# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:27:39 2025

@author: j6k5tki
"""
import numpy as np

import matplotlib.pyplot as plt

X = np.load('inverse_X_202503202235.npy')
Y = np.load('inverse_Y_202503202235.npy')

R =X[:,0].reshape(50,50)[0]/0.43
n = Y[:,0].reshape(50,50)
D = Y[:,1].reshape(50,50)[0]
V = Y[:,2].reshape(50,50)[0]

plt.pcolormesh(n)
plt.show()

plt.plot(R,D)
plt.plot(R,V)
plt.show()

plt.plot(R,V/D,'k')
plt.grid()
plt.show()

fig,axes=plt.subplots(1,1,figsize=(3,3),dpi=300)
plt.plot(R,(0.43*R+1.84)*V/D,'k')
plt.xlabel(r'r/a',fontweight='bold',fontsize=15)
plt.ylim([-150,100])

plt.tick_params(axis='x',labelsize=15)
plt.tick_params(axis='y',labelcolor='k',labelsize=15)

plt.title('Krypton RV/D',fontsize=15, fontweight='bold')
plt.xlim([0.0,0.8])
plt.ylim([-50,30])
plt.xticks([0,0.4,0.8])

# Remove the top spines
axes.spines['top'].set_visible(False)
plt.show()

