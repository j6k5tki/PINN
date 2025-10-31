import deepxde as dde
import numpy as np
from math import gamma
import matplotlib.pyplot as plt
from math import sqrt
from scipy.interpolate import griddata
import scipy as sp
from deepxde.backend import tf
import csv
import sys
import os
from datetime import datetime

dde.config.set_default_dtype("float64")
tf.keras.backend.set_floatx("float64")

tf.config.optimizer.set_jit(True)

np.random.seed(42)
tf.compat.v1.set_random_seed(42)

now    = datetime.now()
nowstr = now.strftime('%Y%m%d%H%M')
cmap   = plt.get_cmap('jet')
###############################################################################
# input parameters and files
time_dep      = False # time-dependent/time-independent D,V
a             = 0.43  # [m]

tmin          = 0.00  # [s]
tmax          = 3.00  # [s]
t_num         = 50

xmin          = 0.00*a # [m]
xmax          = 0.99*a # [m]
x_num         = 50

alpha         = 1.5#3.0
beta          = 1.8#5.0
decay_len     = 50.0
tau           = 1e-3 # parallel loss time of impurities [s]
lamb          = 5e-2 # decay length of density near SOL [m]

num_domain    = int(float(sys.argv[4])) #2000
num_boundary  = int(float(sys.argv[5])) #50
num_initial   = 1000
num_test      = 1000
layer_size    = [2] + [64] * 5 + [1] # (r,t) -> (n) + (D,V)
activation    = "tanh"
initializer   = "Glorot uniform"
optimizer1    = "adam"
optimizer2    = "L-BFGS"
learning_rate = 0.0001
iterations    = 30000
threshold_err = 1e-2
loss_weights = [int(float(sys.argv[6])),1,1,int(float(sys.argv[7]))] # [5,1,1,200], [1,1,1,100]

RAR_num      = 5
refine_num    = 5000

shot_num        = int(float(sys.argv[1]))
Source_file     = 'Gamma'
mag_source      = int(float(sys.argv[3]))

###################################################################################

#18_smooth_0530_1e4_0~1.0_0~2.0_50, discrete(2024)
training_input  = './results/%d/outputs/data_1_%d.npy'%(shot_num,int(float(sys.argv[2])))
training_output = './results/%d/outputs/Q_1_%d.npy'%(shot_num,int(float(sys.argv[2])))

###############################################################################
# governing equation (n(x,t))
# dn/dt = (1/r)*d(r*D*dn/dr-r*V*n)/dr - n/tau + Q
# -> r*dn/dt = d(r*(D*dn/dr - r*V))/dr - r*n/tau + r*Q
# -> r*dn/dt = (D*dn/dr - V*n) + r*(dD/dr*dn/dr + D*d2n/dr2 - dV/dr*n - V*dn/dr) - r*n/tau + r*Q

# space(m) x time(s) domain
# [0,a*0.99] x [0,1]

# tau = 1e-3                       at open surface
# tau = infinite(100 in this code) at closed surface

# initial condition
# n(x,0) = 0

# boundary conditions
# dn(0,t)/dx = 0
# dn(a*1.1,t)/dx  = -n/lambda
###############################################################################

# gamma distribution function
def gammaftn(x,k,th):
    return 1/(gamma(k)*th**k)*x**(k-1)*np.exp(-x/th)

def tanh(x,a):
    return (np.exp(a*x)-np.exp(-a*x))/(np.exp(a*x)+np.exp(-a*x))

# Create folders
def CreateFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

# load imp. Density
def gen_traindata():
    ob_x = np.load(training_input) # 10000 x 2 (r,t)
    ob_n = np.load(training_output) # 10000 x 1 (n)
    return ob_x, ob_n

def Source_gen(XX,TT,alpha,beta):
    # Gamma distribution function
    S = mag_source*(10*gammaftn(11*TT,alpha,beta) + 0.5*tanh(TT,10))
    Source = S*np.exp(decay_len*(XX-xmax))
    return Source

###############################################################################
CreateFolder('./results/%d_inverse'%shot_num)
CreateFolder('./results/%d_inverse/inputs'%shot_num)
CreateFolder('./results/%d_inverse/outputs'%shot_num)

space  = np.linspace(xmin, xmax, x_num)
time   = np.linspace(tmin, tmax, t_num)

if 'Gamma' in Source_file:
    XX,TT  = np.meshgrid(space,time)
    Source = Source_gen(XX,TT,alpha,beta)
###############################################################################
fig = plt.figure(figsize=(10,5))

ax0 = fig.add_subplot(1,2,1)
ax0.plot(time,np.sum(Source,axis=1),'k')
ax0.set_xlabel('time [s]')
ax0.set_ylabel('Q')
ax0.set_title('Source rate')
ax0.grid()

ax1 = fig.add_subplot(122,projection="3d")
p = ax1.plot_surface(XX,TT,Source,cmap=cmap)
ax1.contour(XX,TT,Source,levels=20, colors='k')
ax1.set_xlabel('r [m]')
ax1.set_ylabel('t [s]')
ax1.set_title('Source rate')
ax1.grid()
fig.colorbar(p,shrink=0.5)
plt.tight_layout()
plt.savefig('./results/%d_inverse/inputs/Source rate_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
##############################################################################
def pde(x, y): # x:(r,t), y:(n,D,V)

    r = x[:,0:1] # space
    t = x[:,1:2] # time

    n = y[:,0:1] # impurity dens.
    D = y[:,1:2] # diffusion coeff.
    V = y[:,2:3] # convection vel.

    k  = alpha
    th = beta
    gammaft1 = 10/(gamma(k)*th**k)*(11*t)**(k-1)*tf.exp(-(11*t)/th)
    gammaft2 = 0.5*(tf.exp(10*t)-tf.exp(-10*t))/(tf.exp(10*t)+tf.exp(-10*t))
    gammaft = mag_source*(gammaft1 + gammaft2)
    Q       = gammaft * tf.exp(decay_len*(r-xmax))

    dn_t  = dde.grad.jacobian(y, x, i=0, j=1)
    dn_x  = dde.grad.jacobian(y, x, i=0, j=0)

    #res1 = r * dn_t
    #res2 = -r * Q
    #res3 = - dde.grad.jacobian(r*(D*dn_x - V*y), x, j=0)

    #def _rms(z):
    #    return tf.sqrt(tf.reduce_mean(tf.square(z)) + 1e-12)

    #n1 = tf.stop_gradient(_rms(res1))
    #n2 = tf.stop_gradient(_rms(res2))
    #n3 = tf.stop_gradient(_rms(res3))

    #residual = res1/n1 + res2/n2 + res3/n3

    residual = r*dn_t - r*Q - dde.grad.jacobian(r*(D*dn_x - V*y), x, j=0)

    return residual

def pde_res(x, y): # x:(x,t), y:(n,D,V)

    # x
    r    = x[:,0:1]
    t    = x[:,1:2]

    # y
    n = y[:,0:1]
    D = y[:,1:2]
    V = y[:,2:3]

    k  = alpha
    th = beta
    gammaft1 = 10/(gamma(k)*th**k)*(11*t)**(k-1)*tf.exp(-(11*t)/th)
    gammaft2 = 0.5*(tf.exp(10*t)-tf.exp(-10*t))/(tf.exp(10*t)+tf.exp(-10*t))
    gammaft  = mag_source*(gammaft1 + gammaft2)
    Q  = gammaft * tf.exp(decay_len*(r-xmax))

    dn_t  = dde.grad.jacobian(y, x, i=0, j=1)
    dn_x  = dde.grad.jacobian(y, x, i=0, j=0)

    residual1 = r*dn_t
    residual2 = -r*Q
    residual3 = -dde.grad.jacobian(r*(D*dn_x - V*y), x, j=0)
    residual4 = residual1 + residual2 + residual3

    return [residual1, residual2, residual3, residual4]
###############################################################################

def boundary_x_l(x, on_boundary): return on_boundary and np.isclose(x[0], xmin)

def boundary_x_r(x, on_boundary): return on_boundary and np.isclose(x[0], xmax)

def func(x): return 0

def func_x_r(x, y, _): return (dde.grad.jacobian(y,x,i=0,j=0) + y[:,0:1]/lamb)


# geometry of training points
geom       = dde.geometry.Interval(xmin, xmax)
timedomain = dde.geometry.TimeDomain(tmin, tmax)
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

# boundary condition at r/a = xmin
bc_x_l = dde.NeumannBC(geomtime, func, boundary_x_l, component = 0)

# boundary condition at r/a = xmax
bc_x_r = dde.OperatorBC(geomtime, func_x_r, boundary_x_r)

# training data in the domain
ob_x, n = gen_traindata()
ob_n   = dde.PointSetBC(ob_x, n, component = 0)

data = dde.data.TimePDE(geomtime,
                        pde,
                        [bc_x_l,bc_x_r, ob_n],
                        num_domain   = num_domain,
                        num_boundary = num_boundary,
                        anchors      = ob_x,
                        num_test     = num_test,
                        train_distribution = 'uniform',
                        )
###############################################################################
Imp_dens = griddata(ob_x,n,(XX,TT),method='nearest')

fig = plt.figure()
plt.pcolormesh(XX,TT,Imp_dens[:,:,0],cmap=cmap)
plt.savefig('./results/%d_inverse/inputs/Impurity density_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()

train_datasets = data.train_points()

fig = plt.figure()
plt.scatter(train_datasets[:len(ob_x),0],train_datasets[:len(ob_x),1], s=5, c='k', label='training') # domain training data
plt.scatter(train_datasets[len(ob_x):len(ob_x)+num_boundary,0],train_datasets[len(ob_x):len(ob_x)+num_boundary,1], s=5, c='r', label='boundary') # boundary training data
plt.scatter(train_datasets[len(ob_x)+num_boundary:,0],train_datasets[len(ob_x)+num_boundary:,1], s=5, c='b', label='domain') # domain training data
plt.xlabel('r [m]')
plt.ylabel('t [s]')
plt.title('training data points')
plt.legend()
plt.grid()
plt.savefig('./results/%d_inverse/inputs/training data points_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
###############################################################################
net = dde.nn.FNN(layer_size, activation, initializer)

def output_transform(x,y):
    
    n = y[:,0:1]
    r = x[:,0:1]
    t = x[:,1:2]

    layer_output_size = 64
    layers_activation = tf.nn.tanh

    DV1 = tf.layers.dense(r,layer_output_size,layers_activation)
    DV2 = tf.layers.dense(DV1,layer_output_size,layers_activation)
    DV3 = tf.layers.dense(DV2,layer_output_size,layers_activation)
    DV4 = tf.layers.dense(DV3,layer_output_size,layers_activation)
    DV = tf.layers.dense(DV4,2,None)

    D = DV[:,0:1]
    V = DV[:,1:2]

    return tf.concat([t*tf.nn.softplus(n),tf.nn.softplus(D),r*V], axis = 1)

net.apply_output_transform(output_transform)

model = dde.Model(data, net)

# before training

model.compile(optimizer1, lr=learning_rate, loss_weights = loss_weights)

model.train(iterations=iterations)

model.compile(optimizer2, loss_weights = loss_weights)

losshistory,train_state = model.train()

################################################################################
# residual-based adaptive refinement (RAR) method

for i in range(RAR_num):

    X_refine = geomtime.random_points(refine_num)

    err = abs(model.predict(X_refine,operator=pde)).ravel()
    err_threshold = np.percentile(err,80) # select top 20% residuals dynamically
    idx_top = np.where(err > err_threshold)[0]

    X_new   = X_refine[idx_top]

    data.add_anchors(X_new)

    #early_stopping = dde.callbacks.EarlyStopping(min_delta=threshold_err, patience=10000)

    print('#----------------------------------------#')
    model.compile(optimizer1,lr=learning_rate, loss_weights = loss_weights)

    #model.train(iterations=iterations, disregard_previous_best=True, callbacks=[early_stopping])

    model.train(iterations=iterations)

    model.compile(optimizer2, loss_weights = loss_weights)

    losshistory, train_state = model.train()
    print('#----------------------------------------#')

################################################################################
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir='./results/%d_inverse/outputs'%shot_num)
###############################################################################
# after training
xx, tt = np.meshgrid(np.linspace(xmin,xmax,x_num), np.linspace(tmin,tmax,t_num)) # 100 x 100, 100 x 100
X = np.vstack((np.ravel(xx), np.ravel(tt))).T # 10000 x 2

y_pred = model.predict(X) # 10000 x 3

# save grid and (density,D,V) info

np.save('./results/%d_inverse/outputs/inverse_X_%s.npy'%(shot_num,nowstr),X)
np.save('./results/%d_inverse/outputs/inverse_Y_%s.npy'%(shot_num,nowstr),y_pred)

res = model.predict(X, operator=pde)     # 10000 x 1
res2= model.predict(X, operator=pde_res) # 10000 x 4
print("#-----------------------------------#")
print('res shape : {}'.format(np.shape(res)))
#print("# Mean residual: {}".format(np.mean(np.absolute(res))))
print("#-----------------------------------#")
##################################################################################
fig,axes = plt.subplots(1,2)

ax1 = axes[0].pcolormesh(XX,TT,Source,cmap=cmap)
axes[0].set_xlabel('r/a')
axes[0].set_ylabel('t [s]')
axes[0].set_title('Source term')
fig.colorbar(ax1, ax = axes[0])
plt.tight_layout()
fig.savefig('./results/%d_inverse/inputs/Source2D_%s.png'%(shot_num,nowstr))

fig,axes = plt.subplots(2,2)
# imp. density
ax1 = axes[0][0].pcolormesh(xx,tt,y_pred[:,0].reshape(x_num,t_num),cmap=cmap) # 100 x 100
axes[0][0].set_xlabel('r/a')
axes[0][0].set_ylabel('t [s]')
axes[0][0].set_title('impurity density')
fig.colorbar(ax1,ax = axes[0][0])
# residual of PDE
ax2 = axes[0][1].pcolormesh(xx,tt,res.reshape(x_num,t_num),cmap='seismic') # 100 x 100
axes[0][1].set_xlabel('r/a')
axes[0][1].set_ylabel('t [s]')
axes[0][1].set_title('residual of PDE')
fig.colorbar(ax2,ax = axes[0][1])
# Diff. coeff.
ax3 = axes[1][0].plot(np.linspace(xmin,xmax,x_num),(y_pred[:,1].reshape(x_num,t_num))[0,:],'k') # 100
axes[1][0].set_xlabel('r/a')
axes[1][0].set_ylabel('D [m^2/s]')
axes[1][0].set_title('Diffusion coefficient')
# Conv. vel.
ax4 = axes[1][1].plot(np.linspace(xmin,xmax,x_num),(y_pred[:,2].reshape(x_num,t_num))[0,:],'k') # 100 x 100
axes[1][1].set_xlabel('r/a')
axes[1][1].set_ylabel('V [m/s]')
axes[1][1].set_title('Convection velocity')

plt.tight_layout()

plt.savefig('./results/%d_inverse/outputs/impurity density_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
#################################################################################################
fig,axes = plt.subplots(2,2)

res2 = np.asarray(res2)

ax1 = axes[0][0].pcolormesh(xx,tt,res2[0,:,0].reshape(x_num,t_num),cmap='seismic') # 100 x 100
axes[0][0].axvline(x=a,c='k')
axes[0][0].set_xlabel('r [m]')
axes[0][0].set_ylabel('t [s]')
axes[0][0].set_title('r*dn/dt')
fig.colorbar(ax1, ax = axes[0][0])

ax2 = axes[0][1].pcolormesh(XX,TT,res2[1,:,0].reshape(x_num,t_num),cmap='seismic') # 100 x 100
axes[0][1].axvline(x=a,c='k')
axes[0][1].set_xlabel('r [m]')
axes[0][1].set_ylabel('t [s]')
axes[0][1].set_title('-r*Q')
fig.colorbar(ax2, ax = axes[0][1])

ax3 = axes[1][0].pcolormesh(XX,TT,res2[2,:,0].reshape(x_num,t_num),cmap='seismic') # 100 x 100
axes[1][0].axvline(x=a,c='k')
axes[1][0].set_xlabel('r [m]')
axes[1][0].set_ylabel('t [s]')
axes[1][0].set_title('transport term')
fig.colorbar(ax3, ax = axes[1][0])

ax4 = axes[1][1].pcolormesh(XX,TT,res2[3,:,0].reshape(x_num,t_num),cmap='seismic') # 100 x 100
axes[1][1].axvline(x=a,c='k')
axes[1][1].set_xlabel('r [m]')
axes[1][1].set_ylabel('t [s]')
axes[1][1].set_title('PDE loss')
fig.colorbar(ax4, ax = axes[1][1])

plt.tight_layout()

plt.savefig('./results/%d_inverse/outputs/residual_of_PDE_terms_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
#################################################################################################
print("################################")
print("shot#: %d"%shot_num)
print("num_domain: %d"%num_domain)
print("num_boundary: %d"%num_boundary)
print("num_initial: %d"%num_initial)
print("mag_source: %d"%mag_source)
print("learning_rate: {}".format(learning_rate))
print("loss_weights: {}".format(loss_weights))
print("layer size: {}".format(layer_size))
print("Epoch: %d"%iterations)
print("activation: %s"%activation)
print("%s PINN training is over!"%nowstr)
print("################################")
