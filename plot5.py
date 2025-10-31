import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig,axes = plt.subplots(3, 1, figsize=(4,9), dpi=300)
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 15})

##################################################
# origin plot (3.0 - 6.0 s)

r0 = np.load('./KSTAR_exp/26861/radius.npy')
t0 = np.load('./KSTAR_exp/26861/time.npy')
Z = np.load('./KSTAR_exp/26861/imp1d.npy')/1e17 # (50,70)

R0, T0 = np.meshgrid(r0, t0)            # (50,70) 형태의 좌표

pc1 = axes[0].pcolormesh(R0,T0,Z.reshape(71,50),cmap='jet',vmax=4)
axes[0].set_xlim([0,0.8])
axes[0].set_ylim([3.2,6])

##################################################
# new plot (2.5 - 6.5 s)

#R0 = np.load('./KSTAR_exp_new/R_bol.npy')
#T0 = np.load('./KSTAR_exp_new/T_bol.npy')
#Z = np.load('./KSTAR_exp_new/IMA2D_bol.npy')/1e17

#c1 = axes[0].pcolormesh(R0,T0,Z,cmap='jet',vmax=4)
#axes[0].set_ylim([2.5,6.5])

cax0 = inset_axes(axes[0], width="3%", height="100%",  # 가로 3 %, 세로 100 %
                  loc='lower left',
                  bbox_to_anchor=(1.02, 0., 1, 1),      # 오른쪽 바깥 2 % 지점
                  bbox_transform=axes[0].transAxes,
                  borderpad=0)
fig.colorbar(pc1, cax=cax0)
cax0.tick_params(labelsize=15)

axes[0].set_xticks([0, 0.4, 0.8])
axes[0].set_title('Experimental krypton density', fontsize=15, fontweight='bold')

axes[0].set_xlabel(r'$\mathbf{r/a}$', fontsize=15, fontweight='bold')
axes[0].set_ylabel(r'$\mathbf{t [s]}$', fontsize=15, fontweight='bold')
axes[0].tick_params(labelsize=15)

##################################################
#axes[0].axhline(y=3,color='w',linewidth=1,linestyle='dashed')
#axes[0].axhline(y=6,color='w',linewidth=1,linestyle='dashed')
##################################################
X = np.load('./KSTAR_exp/inverse_X_202503202235.npy')
Y = np.load('./KSTAR_exp/inverse_Y_202503202235.npy')

R = X[:,0].reshape(50,50) / 0.43
T = X[:,1].reshape(50,50) + 3
N = Y[:,0].reshape(50,50)
D = Y[:,1].reshape(50,50)
V = Y[:,2].reshape(50,50)

pc2 = axes[1].pcolormesh(R,T,N,cmap='jet',vmax=4)

cax1 = inset_axes(axes[1], width="3%", height="100%",
                  loc='lower left',
                  bbox_to_anchor=(1.02, 0., 1, 1),
                  bbox_transform=axes[1].transAxes,
                  borderpad=0)
fig.colorbar(pc2, cax=cax1)
cax1.tick_params(labelsize=15)

axes[1].set_xlim([0,0.8])
axes[1].set_ylim([3.2,6])
axes[1].set_xticks([0,0.4,0.8])
axes[1].set_title('Reconstructed krypton density', fontsize=15, fontweight='bold')

axes[1].set_xlabel(r'$\mathbf{r/a}$', fontsize=15, fontweight='bold')
axes[1].set_ylabel(r'$\mathbf{t [s]}$', fontsize=15, fontweight='bold')

axes[1].tick_params(axis='x',labelsize=15)
axes[1].tick_params(axis='y',labelsize=15)

#################################################
r = R[0,:]
d = D[0,:]
v = V[0,:]

axes[2].plot(r,d,'r')
ax2 = axes[2].twinx()
ax2.plot(r,v,'b')
axes[2].set_xlabel(r'r/a',fontweight='bold',fontsize=15)
axes[2].set_ylim([0,0.07])

axes[2].tick_params(axis='x',labelsize=15)
axes[2].tick_params(axis='y',labelcolor='r',labelsize=15)
ax2.tick_params(axis='y',labelcolor='b',labelsize=15)

axes[2].set_title('Krypton D and V',fontsize=15, fontweight='bold')
axes[2].set_xticks([0,0.4,0.8])
axes[2].set_xlim([0.0,0.8])

# Remove the top spines
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

##################################################

plt.tight_layout()
plt.show()
plt.savefig('Fig5.png', dpi=300)