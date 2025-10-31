# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:58:34 2025

@author: j6k5tki
"""
import numpy as np
import matplotlib.pyplot as plt

R = np.load('R_bol.npy')
T = np.load('T_bol.npy')
Z = np.load('IMA2D_bol.npy')

print(np.shape(Z))

fig=plt.figure(dpi=300)
plt.pcolormesh(R,T,Z,cmap='jet',vmin=0,vmax=4e17)
plt.ylim([3,6])
plt.colorbar()
plt.show()