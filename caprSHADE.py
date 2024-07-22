import numpy as np
import cst
import operators as op
import astliNN_vae as av
import astliNN_mlp as am
import readerGraphics as rg
import matplotlib.pyplot as plt
import matplotlib as mpl
from cv2 import split
import time

#%% Import ASTLI-NN
tStart = time.time()

vae = av.astliNN_VAE()
vae.load_weights('path_input_7')

mlp = am.astliNN_MLP()
mlp.load_weights('path_mlp_7')

_, scaler = op.norm_inputs()

#%% Optimization conditions

Omega = np.array([[0.3, 1.0],
                  [0.2, 1.0],
                  [0.2, 1.0],
                  [0.2, 1.0],
                  [0.2, 1.0],
                  [0.2, 1.0],
                  [0.2, 1.0],
                  [0.0, 0.7],
                  [0.0, 0.7],
                  [0.0, 0.7],
                  [0.0, 0.7],
                  [0.0, 0.7],
                  [0.0, 0.7],
                  [0.0, 0.7]])
cld = 0.59
ytmax_min = 0.11
alpha_max = 4
d = Omega[:,0].size
NP = 20*d
H = 6
p = 0.11
NPmin = 4
G = 1000
Uest = 0.02
nameTest = 'Test3_5'

#%% Initial population
NPG = np.array([])
Lavg = np.array([])
Lopt = np.array([])
E = np.array([])

Px = op.initial_Population(NP)
Pxcst = scaler.inverse_transform(Px)

zPred = mlp.predict(Px)
imgPred = vae.decoder.predict(zPred)

mx_alpha = np.array([])
mx_cm = np.array([])
mx_cd = np.array([])
mx_yt = np.array([])
fx = np.array([])
for i in range(NP):
    Dg, Mg, Yg = split(imgPred[i])
    alpha = rg.searchAlphawithCl(cld, Yg)
    cm = rg.searchCmwithAlpha(alpha, Mg)
    D = rg.searchDwithAlpha(alpha, Dg)
    mx_alpha = np.append(mx_alpha, alpha)
    mx_cm = np.append(mx_cm, cm)
    mx_cd = np.append(mx_cd, cld**1.5/D)
    fx = np.append(fx, 1/D)
    yt, _ = cst.cst_ST_SC(Pxcst[i])
    mx_yt = np.append(mx_yt, max(yt))
    
#%% Define Fitness
psix = op.Psi(mx_yt, ytmax_min, mx_alpha, alpha_max, NP)
Lx = op.penalty_func(psix, fx, Uest, NP)
Uest = op.updateUest(psix, fx, Uest, NP)

NPG = np.append(NPG, NP)
Lavg = np.append(Lavg, np.mean(Lx))
Lopt = np.append(Lopt, min(Lx))
E = np.append(E, max(Lx)-min(Lx))

print('Generation 0 \n')
print('Lx = ' + str(Lopt[-1])+'\n')
print('cy^1.5/cx = ' + str(1/fx[np.argmin(Lx)])+'\n')
print('alpha = ' + str(mx_alpha[np.argmin(Lx)])+'\n')
print('cd = ' + str(mx_cd[np.argmin(Lx)])+'\n')
print('cm = ' + str(mx_cm[np.argmin(Lx)])+'\n')
print('ytmax = ' + str(mx_yt[np.argmin(Lx)])+'\n')

#%% Next generations
MF = np.ones(H)*0.5
A = np.empty([0,d])
NA = round(2.6*NP)
k = 0
for g in range(1, G):
    Deltaf = np.array([])
    SF = np.array([])
    Pv = np.empty([0, d])
    fv = np.array([])
    mv_alpha = np.array([])
    mv_cm = np.array([])
    mv_cd = np.array([])
    mv_yt = np.array([])
    Fi = np.array([])
    
    for i in range(NP):
        Fi = np.append(Fi, op.operF(H, MF))  
        xBest = op.pBest(p, NP, Px, Lx)
        vi = op.mutation(Omega, xBest, Fi[-1], NP, Px, A, i)
        Pv = np.row_stack((Pv, vi))

    Pvcst = scaler.inverse_transform(Pv)
    zPred = mlp.predict(Pv)
    imgPred = vae.decoder.predict(zPred)    
    for i in range(NP):
        Dg, Mg, Yg = split(imgPred[i])
        alpha = rg.searchAlphawithCl(cld, Yg)
        cm = rg.searchCmwithAlpha(alpha, Mg)
        D = rg.searchDwithAlpha(alpha, Dg)
        mv_alpha = np.append(mv_alpha, alpha)
        mv_cm = np.append(mv_cm, cm)
        mv_cd = np.append(mv_cd, cld**1.5/D)
        fv = np.append(fv, 1/D)
        yt, _ = cst.cst_ST_SC(Pvcst[i])
        mv_yt = np.append(mv_yt, max(yt))
        
    psiv = op.Psi(mv_yt, ytmax_min, mv_alpha, alpha_max, NP)
    Lv = op.penalty_func(psiv, fv, Uest, NP)
    Uest = op.updateUest(psiv, fv, Uest, NP)

    Pxn = np.empty([0, d])
    Lxn = np.array([])
    fxn = np.array([])
    mxn_alpha = np.array([])
    mxn_cm = np.array([])
    mxn_cd = np.array([])
    mxn_yt = np.array([])
    for i in range(NP):
        if Lv[i] < Lx[i]:
            Pxn = np.row_stack((Pxn, Pv[i]))
            Lxn = np.append(Lxn, Lv[i])
            fxn = np.append(fxn, fv[i])
            mxn_alpha = np.append(mxn_alpha, mv_alpha[i])
            mxn_cm = np.append(mxn_cm, mv_cm[i])
            mxn_cd = np.append(mxn_cd, mv_cd[i])
            mxn_yt = np.append(mxn_yt, mv_yt[i])
            A = np.row_stack((A, Px[i]))
            SF = np.append(SF, Fi[i])
            Deltaf = np.append(Deltaf, abs(Lv[i] - Lx[i]))
        else:
            Pxn = np.row_stack((Pxn, Px[i]))
            Lxn = np.append(Lxn, Lx[i])
            fxn = np.append(fxn, fx[i])
            mxn_alpha = np.append(mxn_alpha, mx_alpha[i])
            mxn_cm = np.append(mxn_cm, mx_cm[i])
            mxn_cd = np.append(mxn_cd, mx_cd[i])
            mxn_yt = np.append(mxn_yt, mx_yt[i])
    if SF.size!=0:
        mSF = op.meanWS(SF, Deltaf)
        MF[k] = mSF
        if k>=H-1:
            k = 0
        else:
            k += 1
    A = op.popA(NA, A)
    Lavg = np.append(Lavg, np.mean(Lxn))
    NP = op.newNP_CAPR(Lavg, NP, NPmin)
    Px, Lx, fx, mx_alpha, mx_cm, mx_cd, mx_yt = op.newGen(NP, Pxn, Lxn, fxn, 
                                                    mxn_alpha, mxn_cm, 
                                                    mxn_cd, mxn_yt)
    NPG = np.append(NPG, NP)
    Lopt = np.append(Lopt, min(Lx))
    E = np.append(E, max(Lx)-min(Lx))

    print('Generation '+str(g)+' \n')
    print('Lx = ' + str(Lopt[-1])+'\n')
    print('cy^1.5/cx = ' + str(1/fx[np.argmin(Lx)])+'\n')
    print('alpha = ' + str(mx_alpha[np.argmin(Lx)])+'\n')
    print('cd = ' + str(mx_cd[np.argmin(Lx)])+'\n')
    print('cm = ' + str(mx_cm[np.argmin(Lx)])+'\n')
    print('ytmax = ' + str(mx_yt[np.argmin(Lx)])+'\n')
    
    # if g in [99, 249, 499]:
    #     np.save(nameTest+'/D'+str(g)+'.npy', 1/fx[np.argmin(Lx)])
    #     np.save(nameTest+'/alpha'+str(g)+'.npy', mx_alpha[np.argmin(Lx)])
    #     np.save(nameTest+'/cd'+str(g)+'.npy', mx_cd[np.argmin(Lx)])
    #     np.save(nameTest+'/cm'+str(g)+'.npy', mx_cm[np.argmin(Lx)])
    #     np.save(nameTest+'/yt'+str(g)+'.npy', mx_yt[np.argmin(Lx)])
    #     np.save(nameTest+'/A'+str(g)+'.npy', Px[np.argmin(Lx)])
    
#%%
import pandas as pd

Pxcst = scaler.inverse_transform(Px)
Gx = np.column_stack((Lx, 1/fx))
Gx = np.column_stack((Gx, mx_alpha))
Gx = np.column_stack((Gx, mx_cd))
Gx = np.column_stack((Gx, mx_cm))
Gx = np.column_stack((Gx, mx_yt))
Gx = np.column_stack((Gx, Pxcst))
Columns = ['Lx', 'D', 'alpha', 'cx', 'mz', 'yt', 'A0', 'A1', 'A2', 'A3', 'A4',
           'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13']
GX = pd.DataFrame(Gx, columns=Columns)
GX = GX.sort_values(by=['Lx'], ascending=True)

font = {'family' : 'Liberation Serif',
        'weight' : 'normal',
        'size'   : 10}
cm=1/2.54
mpl.rc('font', **font)
mpl.rc('axes', linewidth=1)
mpl.rc('lines', lw=1)  

fig = plt.figure(66, figsize=(16*cm, 8*cm))
ax = fig.add_subplot(121)
ax.plot(np.arange(g+1), E, '-k')
ax.set_xlabel('Поколения')
ax.set_ylabel('Ошибка')
ax.grid()
ax = fig.add_subplot(122)
ax.plot(np.arange(g+1), Lopt, '-b')
ax.set_xlabel('Поколения')
ax.set_ylabel('$Lx$')
ax.grid()
plt.tight_layout()
fig.savefig('convergence.svg', format='svg')
plt.show()

# np.save(nameTest+'/E.npy', E)
# np.save(nameTest+'/L.npy', Lopt)

# Results = GX[:5].to_numpy()
# for i in range(5):
#     X, YU, YL = cst.cstN6(Results[i,6:])
#     fig = plt.figure(i+1, figsize=(10*cm, 4*cm))
#     ax = fig.add_subplot(111)
#     ax.plot(X, YU, '-k')
#     ax.plot(X, YL, '-k')
#     ax.set_xlabel('x/b')
#     ax.set_ylabel('y/b')
#     ax.set_title('Профиль '+str(i))
#     ax.grid()
#     ax.set_aspect('equal')
#     plt.tight_layout()
#     fig.savefig('airfoil'+str(i)+'.svg')
#     plt.show()

# B5x = op.best5Polars(Lx, Px, d)
# zPred = mlp.predict(B5x)
# imgPred = vae.decoder.predict(zPred)
# for i in range(5):
#     Dg, Mg, Yg = split(imgPred[i])
#     fig = plt.figure(i+10, figsize=(21*cm, 7*cm))
#     ax = fig.add_subplot(131)
#     ax.imshow(Dg, cmap='bone')
#     ax = fig.add_subplot(132)
#     ax.imshow(Mg, cmap='bone')
#     ax = fig.add_subplot(133)
#     ax.imshow(Yg, cmap='bone')
#     plt.tight_layout()
#     fig.savefig('airfoil_graphs'+str(i)+'.svg')
#     plt.show()
    
tEnd = time.time()
DeltaT = tEnd - tStart