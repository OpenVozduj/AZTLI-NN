import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import cst

def norm_inputs():
    XT = np.load('airfoilsCSTplus.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normXT = scaler.fit_transform(XT)
    return normXT, scaler 

def initial_Population(NP):
    Real = np.load('airfoilsCST_sub.npy')
    Fake1 = np.load('airfoilsCSTgan_sub.npy')
    Fake2 = np.load('airfoilsCSTgan_sub2.npy')    
    airfoils = np.concatenate((Real, Fake1, Fake2), axis=0)
    xsel = np.random.choice(len(airfoils), NP, replace=False)
    P = airfoils[xsel]
    _, scaler = norm_inputs()
    Pn = scaler.fit_transform(P)
    return Pn

def suprEqualities(f1, f2):
    fs = np.column_stack((f1, f2))
    for p in range(f1.size):
        for q in range(f1.size):
            if p != q:
                if np.array_equal(fs[p], fs[q]) == True:
                    fs[q] = fs[q] + np.random.rand()/100
    return fs[:,0], fs[:,1]

def nonDominatedSort2(Rt, f1, f2, D):
    Rtc = Rt.copy()
    f1c = f1.copy()
    f2c = f2.copy()
    ff = np.array([])
    Rt_sort = np.empty([0, D])
    f1_sort = np.array([])
    f2_sort = np.array([])
    F = 0
    while f1c.size > 0:
        if f1c.size > 1:
            d = []
            for p in range(f1c.size):
                Np = 0
                for q in range(f1c.size):
                    if p != q:
                        sp = 0
                        if f1c[p] >= f1c[q]:
                            sp += 1
                        if f2c[p] >= f2c[q]:
                            sp += 1
                        if sp == 2:
                            Np += 1
                if Np == 0:
                    Rt_sort = np.row_stack((Rt_sort, Rtc[p]))
                    f1_sort = np.append(f1_sort, f1c[p])
                    f2_sort = np.append(f2_sort, f2c[p])
                    ff = np.append(ff, F)
                    d.append(p)
            if len(d) == 0:
                Rt_sort = np.row_stack((Rt_sort, Rtc))
                f1_sort = np.append(f1_sort, f1c)
                f2_sort = np.append(f2_sort, f2c)
                ff = np.append(ff, F)
                print('F'+str(F)+' '+str(f1c.size)+' elements')
                Rtc = np.empty([0, D])
                f1c = np.array([])
                f2c = np.array([])
            else:
                Rtc = np.delete(Rtc, d, axis=0)
                f1c = np.delete(f1c, d)
                f2c = np.delete(f2c, d)
                print('F'+str(F)+' '+str(len(d))+' elements')
                F += 1
        else:
            Rt_sort = np.row_stack((Rt_sort, Rtc))
            f1_sort = np.append(f1_sort, f1c)
            f2_sort = np.append(f2_sort, f2c)
            ff = np.append(ff, F)
            print('F'+str(F)+' '+str(f1c.size)+' elements')
            Rtc = np.empty([0, D])
            f1c = np.array([])
            f2c = np.array([])
    return Rt_sort, f1_sort, f2_sort, ff

def nonDominatedSort(Rt, f1, f2, D):
    Rtc = Rt.copy()
    f1c = f1.copy()
    f2c = f2.copy()
    ff = np.array([])
    Rt_sort = np.empty([0, D])
    f1_sort = np.array([])
    f2_sort = np.array([])
    F = 0
    while f1c.size > 0:
        d = []
        for p in range(f1c.size):
            Np = 0
            for q in range(f1c.size):                
                if p != q:
                    sp = 0
                    sq = 0
                    if f1c[p] >= f1c[q]:
                        sp += 1
                    else:
                        sq += 1
                    if f2c[p] >= f2c[q]:
                        sp += 1
                    else:
                        sq += 1
                    if sp > sq:
                        Np += 1
            if Np == 0:
                Rt_sort = np.row_stack((Rt_sort, Rtc[p]))
                f1_sort = np.append(f1_sort, f1c[p])
                f2_sort = np.append(f2_sort, f2c[p])
                ff = np.append(ff, F)
                d.append(p)
        if len(d) == 0:
            for p in range(f1c.size):
                Rt_sort = np.row_stack((Rt_sort, Rtc[p]))
                f1_sort = np.append(f1_sort, f1c[p])
                f2_sort = np.append(f2_sort, f2c[p])
                ff = np.append(ff, F)
            Rtc = np.empty([0, D])
            f1c = np.array([])
            f2c = np.array([])
        else:
            Rtc = np.delete(Rtc, d, axis=0)
            f1c = np.delete(f1c, d)
            f2c = np.delete(f2c, d)
        F += 1
    return Rt_sort, f1_sort, f2_sort, ff

def crowDist(Rt_sort, f1_sort, f2_sort, ff, D):
    Rt_sortI = np.empty([0, D])
    f1_sortI = np.array([])
    f2_sortI = np.array([])
    Idt = np.array([])
    for i in range(int(max(ff))+1):
        Fi = np.where(ff==i)[0]
        Rt_Fi = Rt_sort[Fi]
        f1_Fi = f1_sort[Fi]
        f2_Fi = f2_sort[Fi]
        FiGroup = np.column_stack((f1_Fi, f2_Fi))
        FiGroup = np.column_stack((FiGroup, Rt_Fi))
        Fidf = pd.DataFrame(FiGroup)
        Fidf = Fidf.sort_values(by=0, ascending=True)
        FiGroup_sort = Fidf.to_numpy()
        Id = np.zeros_like(f1_Fi)
        if Fi.size>2:
            Id[0] = 100
            Id[Fi.size-1] = 100
            for l in range(1, Fi.size-1):
                sm = 0
                for m in range(2):
                    if max(FiGroup_sort[:,m])-min(FiGroup_sort[:,m]) !=0:
                        sm += abs(FiGroup_sort[l+1,m]-FiGroup_sort[l-1,m])/(max(FiGroup_sort[:,m])-min(FiGroup_sort[:,m]))
                Id[l] = sm
        FiGroup_sort = np.column_stack((Id, FiGroup_sort))
        Fidf2 = pd.DataFrame(FiGroup_sort)
        Fidf2 = Fidf2.sort_values(by=0, ascending=False)
        FiGroup_sortI = Fidf2.to_numpy()
        Idt = np.append(Idt, FiGroup_sortI[:,0])
        f1_sortI = np.append(f1_sortI, FiGroup_sortI[:,1])
        f2_sortI = np.append(f2_sortI, FiGroup_sortI[:,2])
        Rt_sortI = np.row_stack((Rt_sortI, FiGroup_sortI[:,3:]))
        
    return Rt_sortI, f1_sortI, f2_sortI, Idt

def newPt(Rt_sort, f1_sort, f2_sort, ff, N, D):
    Pt1 = np.empty([0, D])
    f1_1 = np.array([])
    f2_1 = np.array([])
    Rtc = Rt_sort.copy()
    f1c = f1_sort.copy()
    f2c = f2_sort.copy()
    F = 0
    Np = 0
    while f1_1.size<N:
        nF = np.where(ff==F)[0].size
        Np += nF
        if Np <= N:
            Pt1 = np.row_stack((Pt1, Rtc[np.where(ff==F)[0]]))
            f1_1 = np.append(f1_1, f1c[np.where(ff==F)[0]])
            f2_1 = np.append(f2_1, f2c[np.where(ff==F)[0]])
            Rtc = np.delete(Rtc, np.where(ff==F)[0], axis=0)
            f1c = np.delete(f1c, np.where(ff==F)[0])
            f2c = np.delete(f2c, np.where(ff==F)[0])
            F += 1
        else:
            Np -= nF
            break
    
    return Pt1, f1_1, f2_1

def tournament(N, pt=0.8):
    xr = np.random.choice(np.arange(N), 2, replace=False)
    r = np.random.rand()
    if r < pt:
        if xr[0] < xr[1]:
            p = xr[0]
        else:
            p = xr[1]
    else:
        if xr[0] < xr[1]:
            p = xr[1]
        else:
            p = xr[0]
    return p

def SBX(Pt1, N, D, eta_c=20):
    C = np.empty([0, D])
    for i in range(int(N/2)):
        while 2==2:
            while 1==1:
                p1 = tournament(N)
                p2 = tournament(N)
                if p1!=p2:
                    break
            P1 = Pt1[p1]
            P2 = Pt1[p2]
            C1 = np.zeros_like(P1)
            C2 = np.zeros_like(P2)
            for j in range(D):
                u = np.random.uniform()
                if u < 0.5:
                    beta = (2*u)**(1/(eta_c+1))
                else:
                    beta = 1/((2*(1-u))**(1/(eta_c+1)))
                C1[j] = 0.5*((1-beta)*P1[j] + (1+beta)*P2[j])
                C2[j] = 0.5*((1+beta)*P1[j] + (1-beta)*P2[j])
            _, scaler = norm_inputs()
            A1 = scaler.inverse_transform(C1.reshape(1, -1))
            A2 = scaler.inverse_transform(C2.reshape(1, -1))
            yt1, _ = cst.cst_ST_SC(A1[0])
            yt2, _ = cst.cst_ST_SC(A2[0])
            if np.where(yt1<0)[0].size == 0 and np.where(yt2<0)[0].size == 0:
                break
        C = np.row_stack((C, C1, C2))
    return C

def mutation(C, N, Omega, D, eta_m=20):
    Q = np.empty([0, D])
    for i in range(N):
        p = C[i]
        while 1==1:
            m = np.zeros_like(p)
            for j in range(D):
                r = np.random.uniform()
                if r < 0.5:
                    delta = (2*r)**(1/(eta_m+1))-1
                else:
                    delta = 1 - (2*(1-r))**(1/(eta_m+1))
                m[j] = p[j] + delta*(Omega[j,1] - Omega[j,0])
            m = np.clip(m, Omega[:,0], Omega[:,1])
            _, scaler = norm_inputs()
            Am = scaler.inverse_transform(m.reshape(1, -1))
            yt, _ = cst.cst_ST_SC(Am[0])
            if np.where(yt<0)[0].size == 0:
                break
        Q = np.row_stack((Q, m))
    return Q