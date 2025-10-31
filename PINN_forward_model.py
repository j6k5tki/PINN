import deepxde as dde
import numpy as np
from math import gamma
import matplotlib.pyplot as plt
from math import sqrt
import scipy as sp
from deepxde.backend import tf
import sys
import os
from datetime import datetime

np.random.seed(1142)
tf.compat.v1.set_random_seed(1142)

now    = datetime.now()
nowstr = now.strftime('%Y%m%d%H%M')
cmap   = plt.get_cmap('jet')
###############################################################################
# input parameters and files
a             = 0.43   # [m]

tmin          = 0.0    # [s]
tmax          = 3.0    # [s]
t_num         = 50

xmin          = 0.00*a  # [m]
xmax          = 0.99*a  # [m]
x_num         = 50

alpha         = 3.0 #5.0
beta          = 5.0 #12.0
decay_len     = 50.0
tau           = 1e-3 # parallel loss time of impurities [s]
lamb          = 5e-2 # decay length of density near SOL [m] # 5e-2

num_domain    = 4000
num_boundary  = 100
num_initial   = 1000
num_test      = 1000
layer_size    = [2] + [64] * 5 + [1] # (r,t) -> n
activation    = "tanh" # "tanh", "silu", "sin", "relu"
initializer   = "Glorot uniform"
optimizer1    = "adam"
optimizer2    = "L-BFGS"
learning_rate = 1e-4
iterations    = 30000
threshold_err = 1.0e-3
loss_weights  = [10,1,1]

RAR_num       = 3
refine_num    = 5000

shot_num      = int(sys.argv[1])
#DV_file       = "./DV_data/DV_data_%d_inv2.txt"%shot_num
DV_file       = "./DV_data/DV_data_%d.txt"%shot_num
Source_file   = "Gamma"
mag_source    = int(sys.argv[2])
###############################################################################
# governing equation (n(x,t))
# dn/dt = (1/r)*d(r*D*dn/dr-r*V*n)/dr - n/tau + Q
# -> r*dn/dt = d(r*(D*dn/dr - r*V))/dr - r*n/tau + r*Q
# -> r*dn/dt = (D*dn/dr - V*n) + r*(dD/dr*dn/dr + D*d2n/dr2 - dV/dr*n - V*dn/dr) - r*n/tau + r*Q

# space(m) x time(s) domain
# [0,a*1.1] x [0,1]

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

def CreateFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

def Source_gen(XX,TT,alpha,beta):
    # Gamma distribution function
    S = mag_source*gammaftn(50*TT,alpha,beta) # mag_source
    XX = np.clip(XX, 1e-6, None)
    Source = S/XX*np.exp(decay_len*(XX-xmax))
    return Source
###############################################################################
CreateFolder('./results/%d'%shot_num)
CreateFolder('./results/%d/inputs'%shot_num)
CreateFolder('./results/%d/outputs'%shot_num)

space  = np.linspace(xmin, xmax, x_num)
time   = np.linspace(tmin, tmax, t_num)

DV_data = np.loadtxt(DV_file)

r_data = DV_data[0]*a # r/a -> r
d_data = DV_data[1]
v_data = DV_data[2]
grad_v = np.gradient(v_data,r_data)
grad_d = np.gradient(d_data,r_data)

R_data,T_data = np.meshgrid(r_data,time)
D_data = np.zeros_like(R_data)
grad_D = np.zeros_like(R_data)
V_data = np.zeros_like(R_data)
grad_V = np.zeros_like(R_data)

for i in range(len(time)):
    D_data[i,:] = d_data
    V_data[i,:] = v_data
    grad_D[i,:] = grad_d
    grad_V[i,:] = grad_v

if 'Gamma' in Source_file:
    XX,TT  = np.meshgrid(space,time)
    Source = Source_gen(XX,TT,alpha,beta)
###############################################################################
fig,axes = plt.subplots(2,2,sharex=True)

axes[0][0].plot(r_data,d_data,'k')
axes[0][0].set_title('D[m^2/s]')
axes[0][0].grid()
axes[0][1].plot(r_data,v_data,'k')
axes[0][1].set_title('V[m/s]')
axes[0][1].grid()
axes[1][0].plot(r_data,grad_d,'k')
axes[1][0].set_title('grad_D[m/s]')
axes[1][0].grid()
axes[1][1].plot(r_data,grad_v,'k')
axes[1][1].set_title('grad_V[/s]')
axes[1][1].grid()
plt.tight_layout()
plt.savefig('./results/%d/inputs/transport coefficient profiles_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()

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
plt.savefig('./results/%d/inputs/Source rate_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
###############################################################################
_y1 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),D_data.ravel()) # D
_y2 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),grad_D.ravel()) # dD/dr
_y3 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),V_data.ravel()) # V
_y4 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),grad_V.ravel()) # dV/dr

def Coeff(x):

    r = x[:,0:1]
    t = x[:,1:2]

    #y1 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),D_data.ravel()) # D
    #y2 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),grad_D.ravel()) # dD/dr
    #y3 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),V_data.ravel()) # V
    #y4 = sp.interpolate.Rbf(R_data.ravel(),T_data.ravel(),grad_V.ravel()) # dV/dr

    #return np.hstack([y1(r,t), y2(r,t), y3(r,t), y4(r,t)])
    return np.hstack([_y1(r, t), _y2(r, t), _y3(r, t), _y4(r, t)])

def pde(x, y, Coeff): # x:(x,t), y:n

    r    = x[:,0:1]
    t    = x[:,1:2]

    # coefficients of PDE
    D    = Coeff[:,0:1]
    dD_x = Coeff[:,1:2]
    V    = Coeff[:,2:3]
    dV_x = Coeff[:,3:4]

    k  = alpha
    th = beta
    gammaft = mag_source/(gamma(k)*th**k)*(50*t)**(k-1)*tf.exp(-(50*t)/th)
    Q    = gammaft*tf.exp(decay_len*(r-xmax))

    dy_t  = dde.grad.jacobian(y, x, j=1)
    dy_x  = dde.grad.jacobian(y, x, j=0)
    dy_xx = dde.grad.hessian(y, x, j=0)

    res1 = r*dy_t
    res2 = -r*Q
    res3 = - ((D*dy_x - V*y) + r*(dD_x*dy_x +D*dy_xx - dV_x*y - V*dy_x))

    def _rms(z):
        return tf.sqrt(tf.reduce_mean(tf.square(z)) + 1e-12)

    n1 = tf.stop_gradient(_rms(res1))
    n2 = tf.stop_gradient(_rms(res2))
    n3 = tf.stop_gradient(_rms(res3))

    #residual = res1/n1 + res2/n2 + res3/n3

    residual = res1 + res2 + res3

    #residual = r*dy_t - r*Q - ((D*dy_x - V*y) + r*(dD_x*dy_x +D*dy_xx - dV_x*y - V*dy_x))

    return residual

def pde_res(x, y, Coeff): # x:(x,t), y:n

    r    = x[:,0:1]
    t    = x[:,1:2]

    # coefficients of PDE
    D    = Coeff[:,0:1]
    dD_x = Coeff[:,1:2]
    V    = Coeff[:,2:3]
    dV_x = Coeff[:,3:4]

    k  = alpha
    th = beta
    gammaft = mag_source/(gamma(k)*th**k)*(50*t)**(k-1)*tf.exp(-(50*t)/th)
    Q  = gammaft * tf.exp(decay_len*(r-xmax))

    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_x = dde.grad.jacobian(y, x, j=0)
    dy_xx= dde.grad.hessian(y, x, j=0)

    residual = r*dy_t - r*Q - ((D*dy_x - V*y) + r*(dD_x*dy_x +D*dy_xx - dV_x*y - V*dy_x))

    residual1 = r*dy_t
    residual2 = -r*Q
    residual3 = -((D*dy_x - V*y) + r*(dD_x*dy_x +D*dy_xx - dV_x*y - V*dy_x))
    residual4 = residual

    return [residual1, residual2, residual3, residual4]
###############################################################################

def boundary_x_l(x, on_boundary): return on_boundary and np.isclose(x[0], xmin)

def boundary_x_r(x, on_boundary): return on_boundary and np.isclose(x[0], xmax)

def func(x): return 0

def func_x_r(x, y, _): return (dde.grad.jacobian(y,x,j=0) + y/lamb)


# geometry of training points
geom       = dde.geometry.Interval(xmin, xmax)
timedomain = dde.geometry.TimeDomain(tmin, tmax)
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

# boundary condition at r/a = xmin
bc_x_l = dde.NeumannBC(geomtime, func, boundary_x_l)

# --- helper: call RBF in TF graph
#def _tf_rbf(r, t, rbf):
#    # r, t: (N,1) tensors
#    r_flat = tf.reshape(r, (-1,))
#    t_flat = tf.reshape(t, (-1,))
#    def _call(rr, tt):
#        return rbf(rr, tt).astype(np.float32)
#    out = tf.numpy_function(_call, [r_flat, t_flat], tf.float32)
#    return tf.reshape(out, (-1, 1))  # (N,1)

#def flux_bc_left(x, y, _):
#    r = x[:, 0:1]
#    t = x[:, 1:2]
#    D = _tf_rbf(r, t, _y1)  # D(r,t)
#    V = _tf_rbf(r, t, _y3)  # V(r,t)
#    dy_x = dde.grad.jacobian(y, x, j=0)
#    # J = -D*dn/dr + V*n = 0  ¡æ-D*dy_x + V*y
#    return -D * dy_x + V * y

#bc_x_l = dde.OperatorBC(geomtime, flux_bc_left, boundary_x_l)

# boundary condition at r/a = xmax
bc_x_r = dde.OperatorBC(geomtime,func_x_r, boundary_x_r)

# initial condition (t = tmin)
# ic     = dde.IC(geomtime, func, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime,
                        pde,
                        [bc_x_l,bc_x_r],
                        num_domain   = num_domain,
                        num_boundary = num_boundary,
                        num_test     = num_test,
                        auxiliary_var_function = Coeff,
                        train_distribution = 'uniform'
                        )
###############################################################################
train_datasets = data.train_points()

fig = plt.figure()
plt.scatter(train_datasets[num_boundary:num_boundary+num_domain,0],train_datasets[num_boundary:num_boundary+num_domain,1],s=10, c='r', label='domain') # domain training data
plt.scatter(train_datasets[:num_boundary,0],train_datasets[:num_boundary,1], s=10, c='k', label='boundary') # boundary training data
plt.xlabel('r [m]')
plt.ylabel('t [s]')
plt.title('training data points')
plt.legend()
plt.grid()
plt.savefig('./results/%d/inputs/training data points_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
###############################################################################
net = dde.nn.FNN(layer_size, activation, initializer)

def output_transform(x,y):
    t = x[:,1:2]
    return t * tf.nn.softplus(y)

net.apply_output_transform(output_transform)

model = dde.Model(data, net)

# before training

def lr_scheduler(epoch, lr):
    return lr * 0.95 if epoch % 10000 == 0 else lr

cb_resample = dde.callbacks.PDEPointResampler(period=1000)

model.compile(optimizer1, lr=learning_rate, loss_weights = loss_weights)

model.train(iterations=iterations,callbacks=[cb_resample])

model.compile(optimizer2, loss_weights = loss_weights)

losshistory,train_state = model.train()
################################################################################
# residual-based adaptive refinement (RAR) method

def log_loss_scaling(loss):
    return np.log(1+loss)

def adjust_weights(losshistory):
    pde_loss  = log_loss_scaling(losshistory.loss_train[-1][0])  # Last PDE loss
    bc_l_loss = log_loss_scaling(losshistory.loss_train[-1][1])  # Last BC Left loss
    bc_r_loss = log_loss_scaling(losshistory.loss_train[-1][2])  # Last BC Right loss

    total_loss = pde_loss + bc_l_loss + bc_r_loss

    w_pde = max(pde_loss / total_loss, 0.9)
    w_bcl = (bc_l_loss / total_loss) * (1 - w_pde)
    w_bcr = (bc_r_loss / total_loss) * (1 - w_pde)

    return [w_pde,w_bcl,w_bcr]

for i in range(RAR_num):

    X_refine = geomtime.random_points(refine_num)

    err = abs(model.predict(X_refine,operator=pde)).ravel()
    err_threshold = np.percentile(err,80) # select top 20% residuals dynamically
    idx_top = np.where(err > err_threshold)[0]

    X_new   = X_refine[idx_top]

    data.add_anchors(X_new)

    #early_stopping = dde.callbacks.EarlyStopping(min_delta=threshold_err, patience=10000)

    print('#----------------------------------------#')
    print(f"\nRAR iteration {i+1}: Added {len(idx_top)} points.")

    #loss_weights = adjust_weights(losshistory)

    model.compile(optimizer1,lr=learning_rate, loss_weights = loss_weights)

    model.train(iterations=iterations,callbacks=[cb_resample])

    #loss_weights = adjust_weights(losshistory)

    model.compile(optimizer2, loss_weights = loss_weights)

    losshistory, train_state = model.train()
    print('#----------------------------------------#')
################################################################################
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir='./results/%d/outputs'%shot_num)
#################
loss_train = np.sum(losshistory.loss_train, axis=1)
loss_test = np.sum(losshistory.loss_test, axis=1)

loss_train_PDE = [i[0] for i in losshistory.loss_train]
loss_train_BCL = [i[1] for i in losshistory.loss_train]
loss_train_BCR = [i[2] for i in losshistory.loss_train]

loss_test_PDE  = [i[0] for i in losshistory.loss_test]
loss_test_BCL  = [i[1] for i in losshistory.loss_test]
loss_test_BCR  = [i[2] for i in losshistory.loss_test]

with open('./results/%d/outputs/Loss_history_%s.dat'%(shot_num,nowstr),'w') as f:
    f.write('PDE(train)\tBCL(train)\tBCR(train)\tPDE(test)\tBCL(test)\tBCR(test)\t{}\n'.format(loss_weights))
    for pde_1,bcl_1,bcr_1,pde_2,bcl_2,bcr_2 in zip(loss_train_PDE,loss_train_BCL,loss_train_BCR,loss_test_PDE,loss_test_BCL,loss_test_BCR):
        f.write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n'%(pde_1,bcl_1,bcr_1,pde_2,bcl_2,bcr_2))

from mpl_toolkits.mplot3d import Axes3D
#################
def _pack_data(train_state):
    def merge_values(values):
        if values is None:
            return None
        return np.hstack(values) if isinstance(values, (list, tuple)) else values

    y_train = merge_values(train_state.y_train)
    y_test = merge_values(train_state.y_test)
    best_y = merge_values(train_state.best_y)
    best_ystd = merge_values(train_state.best_ystd)
    return y_train, y_test, best_y, best_ystd
################
plt.figure()
plt.semilogy(losshistory.steps, loss_train, label="Train loss")
plt.semilogy(losshistory.steps, loss_test, label="Test loss")
for i in range(len(losshistory.metrics_test[0])):
    plt.semilogy(losshistory.metrics_test[:,i],label="Test metric")
plt.xlabel("# Steps")
plt.legend()
plt.savefig("./results/%d/outputs/Loss_history_%s.png"%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()

plt.figure()
plt.semilogy(losshistory.steps, loss_train_PDE,'k--',label="Train loss(PDE)")
plt.semilogy(losshistory.steps, loss_train_BCL,'r--',label="Train loss(B.C. left)")
plt.semilogy(losshistory.steps, loss_train_BCR,'b--',label="Train loss(B.C. right)")
plt.semilogy(losshistory.steps, loss_test_PDE,'k',label="Test loss(PDE)")
plt.semilogy(losshistory.steps, loss_test_BCL,'r',label="Test loss(B.C. left)")
plt.semilogy(losshistory.steps, loss_test_BCR,'b',label="Test loss(B.C. right)")
plt.xlabel("# Steps")
plt.legend()
plt.savefig("./results/%d/outputs/Loss_history(each term)_%s.png"%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
################
if isinstance(train_state.X_train, (list, tuple)):
    print("Error!!!!!: The network has multiple inputs, and plotting such result hasn't been implemented.")
y_train, y_test, best_y, best_ystd = _pack_data(train_state)
y_dim = best_y.shape[1]

# Regression plot
# 1D
if train_state.X_test.shape[1] == 1:
    idx = np.argsort(train_state.X_test[:, 0])
    X = train_state.X_test[idx, 0]
    plt.figure()
    for i in range(y_dim):
        if y_train is not None:
            plt.plot(train_state.X_train[:, 0], y_train[:, i], "ok", label="Train")
        if y_test is not None:
            plt.plot(X, y_test[idx, i], "-k", label="True")
        plt.plot(X, best_y[idx, i], "--r", label="Prediction")
        if best_ystd is not None:
            plt.plot(
                X, best_y[idx, i] + 2 * best_ystd[idx, i], "-b", label="95% CI"
            )
            plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], "-b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
# 2D
elif train_state.X_test.shape[1] == 2:
    for i in range(y_dim):
        plt.figure()
        ax = plt.axes(projection=Axes3D.name)
        ax.plot3D(
            train_state.X_test[:, 0],
            train_state.X_test[:, 1],
            best_y[:, i],
            ".",
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$y_{}$".format(i + 1))
################
#plt.show()
plt.cla()
plt.clf()
plt.close()

# model.restore("model/model.ckpt-" + str(train_state.best_step), verbose = 1)
###############################################################################
# after training
xx, tt = np.meshgrid(np.linspace(xmin,xmax,x_num), np.linspace(tmin,tmax,t_num)) # 100 x 100, 100 x 100
X = np.vstack((np.ravel(xx), np.ravel(tt))).T # 10000 x 2

y_pred = model.predict(X) # 10000 x 1

# save grid and density info

np.save('./results/%d/outputs/data_1_%s.npy'%(shot_num,nowstr),X)
np.save('./results/%d/outputs/Q_1_%s.npy'%(shot_num,nowstr),y_pred)

res = model.predict(X, operator=pde) # 10000 x 1
res2= model.predict(X, operator=pde_res)

print("#-----------------------------------#")
print("# Mean residual:{}".format(np.mean(np.absolute(res))))
print("#-----------------------------------#")
##############################################################################################
fig,axes = plt.subplots(1,2)

ax1 = axes[0].pcolormesh(XX,TT,Source,cmap=cmap)
axes[0].set_xlabel('r [m]')
axes[0].set_ylabel('t [s]')
axes[0].set_title('Source term')
fig.colorbar(ax1, ax = axes[0])
plt.tight_layout()
fig.savefig('./results/%d/inputs/Source2D_%s.png'%(shot_num,nowstr))

fig,axes = plt.subplots(1,2)

ax1 = axes[0].pcolormesh(xx,tt,y_pred.reshape(x_num,t_num),cmap=cmap) # 100 x 100
axes[0].axvline(x=a,c='k')
axes[0].set_xlabel('r [m]')
axes[0].set_ylabel('t [s]')
axes[0].set_title('impurity density')
fig.colorbar(ax1,ax = axes[0])

ax2 = axes[1].pcolormesh(xx,tt,(res.reshape(x_num,t_num)),cmap='seismic', vmin=-np.max(abs(res)), vmax=np.max(abs(res))) # 100 x 100
axes[1].axvline(x=a,c='k')
axes[1].set_xlabel('r [m]')
axes[1].set_ylabel('t [s]')
axes[1].set_title('residual of PDE')
fig.colorbar(ax2,ax = axes[1])

plt.tight_layout()

plt.savefig('./results/%d/outputs/impurity density_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()

np.save('./results/%d/outputs/residual_of_PDE_%s.npy'%(shot_num,nowstr),res)
###############################################################################################
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

plt.savefig('./results/%d/outputs/residual_of_PDE_terms_%s.png'%(shot_num,nowstr))
plt.cla()
plt.clf()
plt.close()
##############################################################################################
# 2D grid (r,t)
rr = np.linspace(xmin, xmax, x_num)
tt = np.linspace(tmin, tmax, t_num)
xx, ttg = np.meshgrid(rr, tt)
Xg = np.vstack([xx.ravel(), ttg.ravel()]).T.astype(np.float32)

# 네트워크 출력 n에 대한 도함수 (PDE에서 쓰는 방식 그대로)
dn_dt_flat = model.predict(Xg, operator=lambda x,y: dde.grad.jacobian(y, x, j=1)).ravel()
dn_dr_flat = model.predict(Xg, operator=lambda x,y: dde.grad.jacobian(y, x, j=0)).ravel()

# 계수에서 가져온 dD/dr, dV/dr (forward는 Coeff(X)로 이미 계산해 사용)
# Coeff(x) -> [D, dD/dr, V, dV/dr] (물리 단위 기준인 경우가 많음)
Cv = Coeff(Xg)
dD_dr_flat = Cv[:, 1]
dV_dr_flat = Cv[:, 3]

dn_dt = dn_dt_flat.reshape(t_num, x_num)
dn_dr = dn_dr_flat.reshape(t_num, x_num)
dD_dr = dD_dr_flat.reshape(t_num, x_num)
dV_dr = dV_dr_flat.reshape(t_num, x_num)

# ---- (옵션) 무차원 스케일로, PDE에서 실제 곱하는 형태 확인 ----
# 스크립트에 R0,T0가 정의되어 있으면 Star-스케일 파생량도 함께 저장
have_scales = ('R0' in globals()) and ('T0' in globals())
if have_scales:
    dn_dts = T0 * dn_dt      # ∂t* n*
    dn_drs = R0 * dn_dr      # ∂r* n*
    dDs_drs = T0/R0 * dD_dr  # ∂r* D*
    dVs_drs = T0     * dV_dr # ∂r* V*

# ---- 그림 저장 (물리 단위) ----
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
im = axes[0,0].pcolormesh(xx, ttg, dn_dt, shading='auto'); plt.colorbar(im, ax=axes[0,0]); axes[0,0].set_title('∂n/∂t (forward)')
im = axes[0,1].pcolormesh(xx, ttg, dn_dr, shading='auto'); plt.colorbar(im, ax=axes[0,1]); axes[0,1].set_title('∂n/∂r (forward)')
im = axes[1,0].pcolormesh(xx, ttg, dD_dr, shading='auto'); plt.colorbar(im, ax=axes[1,0]); axes[1,0].set_title('∂D/∂r (forward)')
im = axes[1,1].pcolormesh(xx, ttg, dV_dr, shading='auto'); plt.colorbar(im, ax=axes[1,1]); axes[1,1].set_title('∂V/∂r (forward)')
for ax in axes.ravel():
    ax.set_xlabel('r'); ax.set_ylabel('t')
plt.tight_layout()
plt.savefig(f'./results/{shot_num}/outputs/derivatives_forward_phys.png', dpi=200)
plt.close()
##############################################################################################
print("################################")
print("shot#: %d"%shot_num)
print("num_domain: %d"%num_domain)
print("num_boundary: %d"%num_boundary)
print("num_initial: %d"%num_initial)
print("loss_weights: {}".format(loss_weights))
print("layer size: {}".format(layer_size))
print("Epoch: %d"%iterations)
print("activation: %s"%activation)
print("%s PINN training is over!"%nowstr)
print("################################")
