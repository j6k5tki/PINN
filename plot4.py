import numpy as np
import matplotlib.pyplot as plt

Label = [12507,116019,106491]
DATE = [202503082212,202503082109,202503092219]
NUM = ['JET','W7-X','KSTAR']

fig,axes = plt.subplots(3, 3, figsize=(9,9), dpi=300)
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 15})

i = 0
for label,num,date in zip(Label,NUM,DATE):

    a = np.loadtxt('./%s/DV_data_%d.txt'%(num,label))
    X = np.load('./%s/inverse/inverse_X_%d.npy'%(num,date))
    Y = np.load('./%s/inverse/inverse_Y_%d.npy'%(num,date))
    
    R = X[:,0].reshape(50,50)[0,:]
    D = Y[:,1].reshape(50,50)[0,:]
    V = Y[:,2].reshape(50,50)[0,:]

    axes[0][i].plot(a[0],a[1],'k')
    axes[0][i].plot(R/0.43, D, 'r')
    axes[1][i].plot(a[0],a[2],'k')
    axes[1][i].plot(R/0.43, V, 'r')
    axes[2][i].plot(a[0],a[2]/a[1],'k')
    axes[2][i].plot(R/0.43, V/D, 'r')

    # Set xlabel
    axes[2][i].set_xlabel(r'r/a',fontweight='bold',fontsize=15)

    # Shade the region r/a=0.95~1.00
    for ax in [axes[0][i], axes[1][i], axes[2][i]]:
        ax.axvspan(0.95, 1.00, color='gray', alpha=0.2)  # add shaded region

    axes[0][i].tick_params(axis='x',labelsize=15)
    axes[0][i].tick_params(axis='y',labelsize=15)
    axes[1][i].tick_params(axis='x',labelsize=15)
    axes[1][i].tick_params(axis='y',labelsize=15)
    axes[2][i].tick_params(axis='x',labelsize=15)
    axes[2][i].tick_params(axis='y',labelsize=15)

    # Remove the top spines
    axes[0][i].spines['top'].set_visible(False)
    axes[0][i].spines['right'].set_visible(False)
    axes[1][i].spines['top'].set_visible(False)
    axes[1][i].spines['right'].set_visible(False)
    axes[2][i].spines['top'].set_visible(False)
    axes[2][i].spines['right'].set_visible(False)

    # Add the panel labels (e.g., (a), (b), ...) at the top of each subplot
    #axes[0][i].text(0.4, 1.2, num, transform=axes[0][i].transAxes, 
    #            fontsize=15, fontweight='bold', va='top', ha='left')
    
    axes[0][1].set_ylim([0.0,0.4])

    i += 1

axes[0][0].set_ylabel(r'$\mathbf{D(m^2/s)}$', fontsize=15, fontweight='bold')
axes[1][0].set_ylabel(r'$\mathbf{V(m/s)}$', fontsize=15, fontweight='bold')
axes[2][0].set_ylabel(r'$\mathbf{V/D(m^{-1})}$', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.show()
#plt.savefig('Fig4.png', dpi=300)