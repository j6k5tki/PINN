import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DEVICE = ['JET','W7-X','KSTAR']
DATE = [202503081701,202503082006,202503092203]
NUM = [12507,116019,106491]

fig,axes = plt.subplots(2, 3, figsize=(9,6), dpi=300)
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 13})

# Set common limits for D and V across all plots
min_y_D = float('inf')
max_y_D = float('-inf')
min_y_V = float('inf')
max_y_V = float('-inf')

# First loop to find global y-limits
for device,num in zip(DEVICE,NUM):
    a = np.loadtxt('./%s/DV_data_%d.txt'%(device,num))
    min_y_D = min(min_y_D, np.min(a[1]))  # for D values
    max_y_D = max(max_y_D, np.max(a[1]))
    min_y_V = min(min_y_V, np.min(a[2]))  # for V values
    max_y_V = max(max_y_V, np.max(a[2]))
##########################################################################
i = 0
for device,num in zip(DEVICE,NUM):

    a = np.loadtxt('./%s/DV_data_%d.txt'%(device,num))
    #grad_D = np.gradient(a[1],a[0])
    #grad_V = np.gradient(a[2],a[0])

    axes[0][i].plot(a[0],a[1],'r')
    ax2 = axes[0][i].twinx()
    ax2.plot(a[0],a[2],'b')
    axes[0][i].set_xlabel(r'r/a',fontweight='bold',fontsize=15)

    axes[0][i].tick_params(axis='x',labelsize=15)
    axes[0][i].tick_params(axis='y',labelcolor='r',labelsize=15)
    ax2.tick_params(axis='y',labelcolor='b',labelsize=15)
    
    axes[0][i].set_xlim([0,1])

    # Set the same y-limits for all subplots
    #axes[i].set_ylim([min_y_D, max_y_D])
    #ax2.set_ylim([min_y_V, max_y_V])

    # Remove the top spines
    axes[0][i].spines['top'].set_visible(False)
    axes[0][i].spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Ensure equal sizes for all subplots
    axes[0][i].set_box_aspect(1)  # 1:1 aspect ratio

    # Add the panel labels (e.g., (a), (b), ...) at the top-left corner of each subplot
    #axes[0][i].text(0.4, 1.2, DEVICE[i], transform=axes[0][i].transAxes, 
    #            fontsize=15, fontweight='bold', va='top', ha='left')
    
    i += 1
#########################################################################
i = 0
for device,date in zip(DEVICE,DATE):

    grid = np.load('./%s/forward/data_1_%d.npy'%(device,date))
    data = np.load('./%s/forward/Q_1_%d.npy'%(device,date))

    R = grid[:,0].reshape(50,50)
    T = grid[:,1].reshape(50,50)
    Q = data.reshape(50,50)

    c = axes[1][i].pcolormesh(R / 0.43, T, Q, cmap='jet')
    axes[1][i].set_xlabel(r'$\mathbf{r/a}$',fontsize=13)
    axes[1][i].set_ylabel(r'$\mathbf{t \ (s)}$',fontsize=13)

    # Explicitly set x-ticks to include 1.0
    axes[1][i].set_xticks([0, 0.5, 1.00])
    axes[1][i].set_yticks([0, 0.5, 1.00])

    axes[1][i].tick_params(axis='x',labelsize=13)
    axes[1][i].tick_params(axis='y',labelsize=13)
    
    cax1 = inset_axes(axes[1][i], width="5%", height="100%",
                  loc='lower left',
                  bbox_to_anchor=(1.02, 0., 1, 1),
                  bbox_transform=axes[1][i].transAxes,
                  borderpad=0)
    fig.colorbar(c, cax=cax1)
    cax1.tick_params(axis='both',which='major',labelsize=13)
    
    # Add the panel labels (e.g., (a), (b), ...) at the top-left corner of each subplot
    #axes[1][i].text(0.4, 1.2, device, transform=axes[1][i].transAxes, 
    #            fontsize=15, fontweight='bold', va='top', ha='left')
    
    i += 1

plt.tight_layout()
plt.savefig('Fig2.png', dpi=300)