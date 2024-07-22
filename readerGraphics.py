import numpy as np

def searchClwithAlpha(alpha, graph):    
    alpha1 = -4
    pxa1 = 0
    alpha2 = 20
    pxa2 = 255 
    px_alpha = (alpha-alpha1)*(pxa2-pxa1)/(alpha2-alpha1)+pxa1
    px_alpha = round(px_alpha)
    
    pxc = np.argmax(graph[:,px_alpha])
    cl1 = 2
    pxc1 = 0
    cl2 = 0
    pxc2 = 255
    cl = (pxc-pxc1)*(cl2-cl1)/(pxc2-pxc1)+cl1
    return cl

def searchAlphawithCl(cl, graph):
    cl1 = 2
    pxc1 = 0
    cl2 = 0
    pxc2 = 255
    px_cl = (cl-cl1)*(pxc2-pxc1)/(cl2-cl1)+pxc1
    px_cl = round(px_cl)
    
    px_alpha = np.argmax(graph[px_cl])
    alpha1 = -4
    pxa1 = 0
    alpha2 = 20
    pxa2 = 255 
    alpha = (px_alpha-pxa1)*(alpha2-alpha1)/(pxa2-pxa1)+alpha1
    return alpha

def searchCmwithAlpha(alpha, graph):    
    alpha1 = -4
    pxa1 = 0
    alpha2 = 20
    pxa2 = 255 
    px_alpha = (alpha-alpha1)*(pxa2-pxa1)/(alpha2-alpha1)+pxa1
    px_alpha = round(px_alpha)
    
    pxc = np.argmax(graph[:,px_alpha])
    cm1 = 0.10
    pxc1 = 0
    cm2 = -0.35
    pxc2 = 255
    cm = (pxc-pxc1)*(cm2-cm1)/(pxc2-pxc1)+cm1
    return cm

def searchDwithAlpha(alpha, graph):    
    alpha1 = -4
    pxa1 = 0
    alpha2 = 20
    pxa2 = 255 
    px_alpha = (alpha-alpha1)*(pxa2-pxa1)/(alpha2-alpha1)+pxa1
    px_alpha = round(px_alpha)
    
    pxc = np.argmax(graph[:,px_alpha])
    D1 = 110
    pxc1 = 0
    D2 = 0
    pxc2 = 255
    D = (pxc-pxc1)*(D2-D1)/(pxc2-pxc1)+D1
    return D
