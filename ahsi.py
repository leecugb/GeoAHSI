import json
import pandas as pd
import numpy as np
from spectral import open_image
from osgeo import gdal

with open('./expert_rules.json') as f:
    rf=json.load(f)
    
with open('./colors_rules.json') as f:
    colors_dic=json.load(f)
    
splib=pd.read_json('./splib.json')
splib['wave'] = splib['wave'].apply(lambda x:np.array(x))
splib['rel'] = splib['rel'].apply(lambda x:np.array(x))

class InvalidRangeError(BaseException):
    def __init__(self):
        pass
    def __str__(self):
        return "the spectrum range of this feature contains no channels"

class InvalidLeftEndPointError(BaseException):
    def __init__(self):
        pass
    def __str__(self):
        return "this left end point range covers no channels"

class InvalidRightEndPointError(BaseException):
    def __init__(self):
        pass
    def __str__(self):
        return "this right end point range covers no channels"

def get_quadratic_center(spectrum, wl, CONTINUUM_ENDPTS, mask_const):
  
    return center
    
def diagnostic_feature(spectrum, reference, wl, CONTINUUM_ENDPTS, FEATURE_WEIGHT,
            CONTINUUM_CONSTRAINTS = [None]*8, FIT_CONSTRAINTS = None,
            DEPTH_CONSTRAINTS = [None,None]):
    # wl is a 1-d array containing p wavelengths
    # spectrum is an array of nxp, n is the number of pixels
    # reference is a 1-d array,containing p elements
    # CONTINUUM_ENDPTS is a list containing four elements, first two elements define the boundary of the left continuum point range,
    # and the last two define the right continuum point range
    # FEATURE_WEIGHT is a scalar,
    # CONTINUUM_CONSTRAINTS is 8-elements list
    # FIT_CONSTRAINTS is a scalar
    # DEPTH_CONSTRAINTS is a 2-elements list, defining the range of absorption depth



   
    return r2_*FEATURE_WEIGHT,depth_*FEATURE_WEIGHT,r2_*FEATURE_WEIGHT*depth_

def get_fit(spectrum, reference, wl, CONTINUUM_ENDPTS):
    
   
    return r2, depth

def not_absolute_feature(spectrum, reference, wl, CONTINUUM_ENDPTS, NOT_FEATURE_FIT_CONSTRAINTS,
            NOT_FEATURE_ABSOLUTE_DEPTH_CONSTRAINTS):

    return mask_c

def not_relative_feature(spectrum, reference, wl, CONTINUUM_ENDPTS, NOT_FEATURE_FIT_CONSTRAINTS,
            RELATIVE_FEATURE_DEPTH, NOT_FEATURE_RELATIVE_DEPTH_CONSTRAINTS):

    return mask_c

def judge_reference_entry(spectrum, wl, key, resampled1, chanels):

 
   
    return fit, fd

def geo_map(pathname,b_n):
    
  

def rgb2hsi(rgb):
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    num = 0.5*((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b)*(g - b))
    theta = np.arccos(num/(den + np.finfo(float).eps))
    H = theta
    H[b > g] = 2*np.pi - H[b > g]
    H = H/(2*np.pi)
    num=rgb.min(1)
    den = r + g + b
    den[den == 0] = np.finfo(float).eps
    S = 1 - 3* num/den
    H[S == 0] = 0
    I = (r + g + b)/3
    return np.vstack([H, S, I]).T

def hsi2rgb(hsi):
    H = hsi[:, 0] * 2 * np.pi
    S = hsi[:, 1]
    I = hsi[:, 2]
    R = np.zeros(len(hsi))
    G = np.zeros(len(hsi))
    B = np.zeros(len(hsi))
    mask = (0 <= H) & (H < 2*np.pi/3)
    B[mask] = I[mask] * (1 - S[mask])
    R[mask] = I[mask] * (1 + S[mask] * np.cos(H[mask])/np.cos(np.pi/3 - H[mask]))
    G[mask] = 3*I[mask] - (R[mask] + B[mask])
    mask = (2*np.pi/3 <= H) & (H < 4*np.pi/3)
    R[mask] = I[mask] * (1 - S[mask])
    G[mask] = I[mask] * (1 + S[mask] * np.cos(H[mask] - 2*np.pi/3) /np.cos(np.pi - H[mask]))
    B[mask] = 3*I[mask] - (R[mask] + G[mask])
    mask = (4*np.pi/3 <= H) & (H <= 2*np.pi)
    G[mask] = I[mask] * (1 - S[mask])
    B[mask] = I[mask] * (1 + S[mask] * np.cos(H[mask] - 4*np.pi/3) /np.cos(5*np.pi/3 - H[mask]))
    R[mask] = 3*I[mask] - (G[mask] + B[mask])
    rgb=np.vstack([R,G,B]).T
    rgb[rgb>1]=1
    rgb[rgb<0]=0
    return rgb
