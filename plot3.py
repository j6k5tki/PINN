import numpy as np
import matplotlib.pyplot as plt

DEVICE = ['   No ECH','600kW ECH','800kW ECH']
DATE = [202503071453,202503071644,202503071702]
DATE_NEW = [202504031344,202504031409,202504031423]
NUM = [106477,106491,106401]

fig,axes = plt.subplots(2, 3, figsize=(15,6), dpi=300)
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 13})

# Set common limits for D and V across all plots
min_y_D = float('inf')
max_y_D = float('-inf')
min_y_V = float('inf')
max_y_V = float('-inf')

# First loop to find global y-limits
for num in NUM:
    a = np.loadtxt('./KSTAR_forward_val/%d/DV_data_%d.txt'%(num,num))
    min_y_D = min(min_y_D, np.min(a[1]))  # for D values
    max_y_D = max(max_y_D, np.max(a[1]))
    min_y_V = min(min_y_V, np.min(a[2]))  # for V values
    max_y_V = max(max_y_V, np.max(a[2]))
##########################################################################
i = 0
for device,num in zip(DEVICE,NUM):

    a = np.loadtxt('./KSTAR_forward_val/%d/DV_data_%d.txt'%(num,num))

    axes[0][i].plot(a[0],a[1],'r')
    ax2 = axes[0][i].twinx()
    ax2.plot(a[0],a[2],'b')
    axes[0][i].set_xlabel(r'r/a',fontweight='bold',fontsize=15)

    axes[0][i].tick_params(axis='x',labelsize=15)
    axes[0][i].tick_params(axis='y',labelcolor='r',labelsize=15)
    ax2.tick_params(axis='y',labelcolor='b',labelsize=15)

    # Remove the top spines
    axes[0][i].spines['top'].set_visible(False)
    axes[0][i].spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    

    # Ensure equal sizes for all subplots
    axes[0][i].set_box_aspect(1)  # 1:1 aspect ratio

    # Add the panel labels (e.g., (a), (b), ...) at the top-left corner of each subplot
    #axes[0][i].text(0.1, 1.2, device, transform=axes[0][i].transAxes, 
    #            fontsize=15, fontweight='bold', va='top', ha='left')
    
    i += 1
#########################################################################
i = 0
for device,num,date in zip(DEVICE,NUM,DATE_NEW):

    grid = np.load('./KSTAR_forward_val/%d_new/data_1_%d.npy'%(num,date))
    data = np.load('./KSTAR_forward_val/%d_new/Q_1_%d.npy'%(num,date))

    R = grid[:,0].reshape(50,50)
    T = grid[:,1].reshape(50,50) + 4.55
    Q = data.reshape(50,50) * 1.3*1e16

    c = axes[1][i].pcolormesh(R / 0.43, T, Q, cmap='jet',vmax=1.9*1e16)
    axes[1][i].set_xlabel(r'$\mathbf{r/a}$',fontsize=13)
    axes[1][i].set_ylabel(r'$\mathbf{t \ (s)}$',fontsize=13)

    # Explicitly set x-ticks to include 1.0
    axes[1][i].set_xticks([0, 0.4, 0.8])
    #axes[1][i].set_yticks([0, 0.5, 1.00])
    axes[1][i].set_xlim([0.00,0.90])
    axes[1][i].set_ylim([4.55,4.90])

    # Adjust tick font size
    axes[1][i].tick_params(axis='both', which='major', labelsize=13)

    # Add a colorbar for each subplot
    fig.colorbar(c, ax=axes[1][i])
    
    # Add the panel labels (e.g., (a), (b), ...) at the top-left corner of each subplot
    #axes[1][i].text(0.1, 1.2, device, transform=axes[1][i].transAxes, 
    #            fontsize=15, fontweight='bold', va='top', ha='left')
    
    i += 1

plt.tight_layout()
plt.savefig('Fig3.png', dpi=300)
