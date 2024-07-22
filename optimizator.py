import numpy as np
import operators as op
import astliNN_vae as av
import astliNN_mlp as am
import readerGraphics as rg
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#%% Import ASTLI-NN
vae = av.astliNN_VAE()
vae.load_weights('path_input_7')

mlp = am.astliNN_MLP()
mlp.load_weights('path_mlp_7')

#%% Optimization conditions


d = np.array([[0.3, 1.0],
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
v = d[:,0].size
G = 250
pc = 0.95
N = 100
nameTest = 'Test5'

_, scaler = op.norm_inputs()
dScaler = scaler.inverse_transform(d.T)

#%% Initial population
Pt = op.initial_Population(2*N)

zPred = mlp.predict(Pt)
imgPred = vae.decoder.predict(zPred)

f_cm = np.array([])
f_D = np.array([])
for i in range(2*N):
    Dg, Mg, Yg = cv2.split(imgPred[i])
    alpha = rg.searchAlphawithCl(cld, Yg)
    cm = rg.searchCmwithAlpha(alpha, Mg)
    D = rg.searchDwithAlpha(alpha, Dg)
    f_cm = np.append(f_cm, abs(cm))
    f_D = np.append(f_D, 1/D)

f_cm, f_D = op.suprEqualities(f_cm, f_D)
#%% Fitness initial population

Pt_sort, f_cm_sort, f_D_sort, Fronts = op.nonDominatedSort2(Pt, f_cm, f_D, v)
Pt_sortI, f_cm_sortI, f_D_sortI, Idt = op.crowDist(Pt_sort, f_cm_sort, f_D_sort, Fronts, v)

fig = plt.figure(1)
ax = fig.add_subplot(111)
# ax.scatter(f_cm, f_D, marker='x')
colors = ['black', 'grey', 'lightgrey', 'darkred', 'red', 'salmon',
          'chocolate', 'sandybrown', 'tan', 'moccasin',
          'goldenrod', 'gold', 'yellow', 'darkgreen', 'yellowgreen',
          'lawngreen', 'lightseagreen', 'turquoise', 'cyan', 'darkblue',
          'blue', 'skyblue', 'darkviolet', 'magenta', 'violet']
k = 0
for i in range(int(max(Fronts))+1):
    ax.scatter(f_cm_sortI[np.where(Fronts==i)[0]], 
               f_D_sortI[np.where(Fronts==i)[0]], c=colors[k], label='F'+str(i))
    k += 1
    if k>24:
        k = 0
ax.set_xlabel('$|m_z|$')
ax.set_ylabel('$c_x/c_y^1.5$')
ax.grid()
# ax.legend(loc='best')
plt.tight_layout()
fig.savefig(nameTest+'/generation0.png')
plt.show()

Pt1 = Pt_sortI[:N]
f_cm1 = f_cm_sortI[:N]
f_D1 = f_D_sortI[:N]
#%%
xG = [49, 99, 149, 199, 249, 299, 349, 399, 449, 499]
for g in range(1,G):
    Ct1 = op.SBX(Pt1, N, v)
    Qt1 = op.mutation(Ct1, N, d, v)
    
    zQt1 = mlp.predict(Qt1)
    imgQt1 = vae.decoder.predict(zQt1)
    
    f_cmQ = np.array([])
    f_DQ = np.array([])
    for i in range(N):
        Dg, Mg, Yg = cv2.split(imgQt1[i])
        alpha = rg.searchAlphawithCl(cld, Yg)
        cm = rg.searchCmwithAlpha(alpha, Mg)
        D = rg.searchDwithAlpha(alpha, Dg)
        f_cmQ = np.append(f_cmQ, abs(cm))
        f_DQ = np.append(f_DQ, 1/D)
        
    Rt1 = np.row_stack((Pt1, Qt1))
    f_cmR = np.append(f_cm1, f_cmQ)
    f_DR = np.append(f_D1, f_DQ)
    
    f_cmR, f_DR = op.suprEqualities(f_cmR, f_DR)
    
    Rt1_sort, f_cmR_sort, f_DR_sort, FrontsR = op.nonDominatedSort2(Rt1, f_cmR, f_DR, v)
    Rt1_sortI, f_cmR_sortI, f_DR_sortI, IdtR = op.crowDist(Rt1_sort, f_cmR_sort, f_DR_sort, FrontsR, v)
    
    #%%
    if g in xG:
        fig = plt.figure(g+1)
        ax1 = fig.add_subplot(111)
        colors = ['black', 'grey', 'lightgrey', 'darkred', 'red', 'salmon',
                  'chocolate', 'sandybrown', 'tan', 'moccasin',
                  'goldenrod', 'gold', 'yellow', 'darkgreen', 'yellowgreen',
                  'lawngreen', 'lightseagreen', 'turquoise', 'cyan', 'darkblue',
                  'blue', 'skyblue', 'darkviolet', 'magenta', 'violet']
        k = 0
        for i in range(int(max(FrontsR))+1):
            ax1.scatter(f_cmR_sortI[np.where(FrontsR==i)[0]], 
                        f_DR_sortI[np.where(FrontsR==i)[0]], c=colors[k], label='F'+str(i))
            k += 1
            if k>24:
                k = 0
        ax1.set_xlabel('$|m_z|$')
        ax1.set_ylabel('$c_x/c_y^1.5$')
        ax1.grid()
        plt.tight_layout()
        # ax1.legend(loc='best')
        fig.savefig(nameTest+'/generation'+str(g)+'.png')
        plt.show()
        
    #%%
    Pt1 = Rt1_sortI[:N]
    f_cm1 = f_cmR_sortI[:N]
    f_D1 = f_DR_sortI[:N]
    Frontst1 = FrontsR[:N]
    
    if g in xG:
        np.save(nameTest+'/Pt_pareto'+str(g)+'.npy', Pt1[np.where(Frontst1==0)[0]])
        np.save(nameTest+'/fcm_pareto'+str(g)+'.npy', f_cm1[np.where(Frontst1==0)[0]])
        np.save(nameTest+'/fD_pareto'+str(g)+'.npy', f_D1[np.where(Frontst1==0)[0]])
    

# #%%
