import numpy as np
import matplotlib.pyplot as plt
import ot

import matplotlib
matplotlib.rc('xtick',labelsize = 20)
matplotlib.rc('ytick',labelsize = 20)

def ProximalPointOT(mu, nv, C, beta = 2, maxiter = 1000, inner_maxiter = 1, use_path = True, return_map = True, return_loss = True):
    n = len(mu)
    a = np.ones([n,])
    b = a
    T = np.ones((n,n))/n**2
    G = np.exp(-(C/beta))
    loss = []
    for outer_iter in range(maxiter):
        Q = G * T
        for inner_iter in range(inner_maxiter):
            a = mu/(np.matmul(Q,b) + 1e-3)
            b = nv/(np.matmul(np.transpose(Q),a) + 1e-3)
        T = np.expand_dims(a,axis = 1)*Q*np.expand_dims(b,axis = 0)
        WD = np.sum(T*C)
        loss.append(WD)
    return T,loss

np.random.seed(123456)

## numerical expertiments
colors=['k','b','g','r']

n = 100
x = np.arange(n,dtype = np.float64)

p1 = ot.datasets.get_1D_gauss(n,20,8)#0.55 * ot.datasets.get_1D_gauss(n,20,8) + 0.45 * ot.datasets.get_1D_gauss(n,70,9)
p2 = 0.55 * ot.datasets.get_1D_gauss(n,35,9) + 0.45 * ot.datasets.get_1D_gauss(n,55,5)

plt.plot(x,p1,'o-',color='blue')
plt.plot(x,p2,'o-',color='red')
plt.tight_layout()
plt.title('Two given marginals', fontsize=16)
plt.show()

# ## c = -xy
c = -x*np.expand_dims(x,axis=1)
c =  c - c.min() + 10
c/=c.max()


# print(c)
plt.imshow(c)
plt.colorbar()
# plt.clim(0, 1)
plt.title('Cost function', fontsize=16)
plt.show()

T_emd = ot.emd(p1,p2,c)
ground_truth = np.sum(T_emd*c)

maxiter = 2000
beta_list = [0.001,0.01,0.1,1]
inner_maxiter = 1
use_path = True

### proximal point OT
T_pp_list  = []
loss_pp_list = []

for beta in beta_list:
    # T, loss = fpot.fpot_wd(p1, p2, C, beta=beta, maxiter=maxiter, inner_maxiter=inner_maxiter, use_path=use_path)
    T,loss = ProximalPointOT(p1,p2,c,beta = beta, maxiter = maxiter, inner_maxiter = inner_maxiter, use_path = use_path)
    loss_pp_list.append(loss)
    T_pp_list.append(T)
### colormap
import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('Reds')
new_cmap = truncate_colormap(cmap, 0., 0.8)


f,axarr = plt.subplots(1,len(beta_list),figsize = (9,3))
for i, beta in enumerate(beta_list):
    axarr[i].imshow(T_pp_list[i],cmap = new_cmap)
    axarr[i].imshow(T_emd,cmap = plt.get_cmap('binary'),alpha = 0.7)
    axarr[i].xaxis.set_ticks([])
    axarr[i].yaxis.set_ticks([])
    axarr[i].set_title(r'$\beta$ = ' + str(beta), fontsize = 20)
plt.show()

print('done')