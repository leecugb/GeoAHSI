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
    mask = (wl <= CONTINUUM_ENDPTS[3]) & (wl >= CONTINUUM_ENDPTS[0])
    mask_left_end = (wl <= CONTINUUM_ENDPTS[1]) & (wl >= CONTINUUM_ENDPTS[0])
    mask_right_end = (wl <= CONTINUUM_ENDPTS[3]) & (wl >= CONTINUUM_ENDPTS[2])
    x_av = np.array([wl[mask_left_end].mean(), wl[mask_right_end].mean()], dtype='float64')
    y_av = np.array([spectrum[mask_const][:, mask_left_end].mean(1), spectrum[mask_const][:, mask_right_end].mean(1)], dtype='float64').T
    con = y_av[:, [0]]+(y_av[:, [1]]-y_av[:, [0]])/(x_av[1]-x_av[0])*(wl[mask].reshape([1,-1])-x_av[0])
    con = spectrum[mask_const][:, mask]/con
    index = con.argmin(1)
    mas=((index-1)>=0)&(index+1<mask.sum())
    con=con[mas]
    x2=wl[mask][index[mas]]
    x1=wl[mask][index[mas]-1]
    x3=wl[mask][index[mas]+1]
    r,c=con.shape
    y2=con.ravel()[index[mas]+c*np.arange(r)]
    y1=con.ravel()[index[mas]-1+c*np.arange(r)]
    y3=con.ravel()[index[mas]+1+c*np.arange(r)]
    temp=((y2-y1)*(x3**2-x2**2)-(y3-y2)*(x2**2-x1**2))/((y3-y2)*(x2-x1)-(y2-y1)*(x3-x2))/2
    cen=np.zeros(len(index))*np.nan
    cen[mas]=-1*temp
    center= np.zeros(len(spectrum))*np.nan
    center[mask_const] = cen
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
    mask = (wl <= CONTINUUM_ENDPTS[3]) & (wl >= CONTINUUM_ENDPTS[0])
    if not mask.any():
        raise InvalidRangeError()
    mask_left_end = (wl <= CONTINUUM_ENDPTS[1]) & (wl >= CONTINUUM_ENDPTS[0])
    if not mask_left_end.any():
        raise InvalidLeftEndPointError()
    mask_right_end = (wl <= CONTINUUM_ENDPTS[3]) & (wl >= CONTINUUM_ENDPTS[2])
    if not mask_right_end.any():
        raise InvalidRightEndPointError()
    x_av = np.array([wl[mask_left_end].mean(), wl[mask_right_end].mean()], dtype='float64')
    y_av = np.array([spectrum[:, mask_left_end].mean(1), spectrum[:, mask_right_end].mean(1)], dtype='float64').T
    r_l=reference[mask_left_end]
    r_r=reference[mask_right_end]
    y_r_av = np.array([r_l[~np.isnan(r_l)].mean(), r_r[~np.isnan(r_r)].mean()], dtype='float64')
    mask_const = np.ones(len(spectrum), dtype='bool')
    if CONTINUUM_CONSTRAINTS[0] is not None:
        mask_const = mask_const & (y_av[:, 0] >= CONTINUUM_CONSTRAINTS[0])
    if CONTINUUM_CONSTRAINTS[1] is not None:
        mask_const = mask_const & (y_av[:, 0] <= CONTINUUM_CONSTRAINTS[1])
    if CONTINUUM_CONSTRAINTS[2] is not None:
        mask_const = mask_const & ((y_av.mean(1)) >= CONTINUUM_CONSTRAINTS[2])
    if CONTINUUM_CONSTRAINTS[3] is not None:
        mask_const = mask_const & ((y_av.mean(1)) <= CONTINUUM_CONSTRAINTS[3])
    if CONTINUUM_CONSTRAINTS[4] is not None:
        mask_const = mask_const & (y_av[:,1] >= CONTINUUM_CONSTRAINTS[4])
    if CONTINUUM_CONSTRAINTS[5] is not None:
        mask_const = mask_const & (y_av[:,1] <= CONTINUUM_CONSTRAINTS[5])
    if CONTINUUM_CONSTRAINTS[6] is not None:
        mask_const = mask_const & ((y_av[:, 1]/y_av[:, 0]) >= CONTINUUM_CONSTRAINTS[6])
    if CONTINUUM_CONSTRAINTS[7] is not None:
        mask_const = mask_const & ((y_av[:, 1]/y_av[:, 0]) <= CONTINUUM_CONSTRAINTS[7])
    con = y_av[mask_const][:, [0]]+(y_av[mask_const][:, [1]]-y_av[mask_const][:, [0]])/(x_av[1]-x_av[0])*(wl[mask].reshape([1,-1])-x_av[0])
    con = spectrum[mask_const][:, mask]/con
    con_ = y_r_av[0]+(y_r_av[1]-y_r_av[0])/(x_av[1]-x_av[0])*(wl[mask]-x_av[0])
    con_ = reference[mask]/con_
    A = np.array([con_, np.ones(len(con_))],dtype='float64').T
    k = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(con.T)
    mask_fit = k[0] > 0   # check signs of depths
    y_fit = k[0][mask_fit].reshape([-1, 1])*(con_.reshape([1,-1]))+k[1][mask_fit].reshape([-1, 1])
    ss_reg = ((y_fit-con[mask_fit].mean(1).reshape([-1, 1]))**2).sum(1)
    ss_tot = ((con[mask_fit]-con[mask_fit].mean(1).reshape([-1, 1]))**2).sum(1)
    r2 = np.zeros(len(con))*np.nan
    r2[mask_fit] = ss_reg/ss_tot
    depth = np.zeros(len(con))*np.nan
    if con_.mean()<1:
        depth[mask_fit] = (1-con_.min())*k[0][mask_fit]
    else:
        depth[mask_fit] = (con_.max()-1)*k[0][mask_fit]
    mask_c = np.ones(len(con),dtype='bool')
    if FIT_CONSTRAINTS is not None:
        mask_c=mask_c &(r2 >= FIT_CONSTRAINTS)
    if DEPTH_CONSTRAINTS[0] is not None:
        mask_c=mask_c & (depth >= DEPTH_CONSTRAINTS[0])
    if DEPTH_CONSTRAINTS[1] is not None:
        mask_c=mask_c&(depth <= DEPTH_CONSTRAINTS[1])   
    r2[~mask_c] = np.nan
    depth[~mask_c] = np.nan
    r2_ = np.zeros(len(spectrum))*np.nan
    r2_[mask_const] = r2
    depth_ = np.zeros(len(spectrum))*np.nan
    depth_[mask_const] = depth
    return r2_*FEATURE_WEIGHT,depth_*FEATURE_WEIGHT,r2_*FEATURE_WEIGHT*depth_

def get_fit(spectrum, reference, wl, CONTINUUM_ENDPTS):
    mask = (wl <= CONTINUUM_ENDPTS[3]) & (wl >= CONTINUUM_ENDPTS[0])
    if not mask.any():
        raise InvalidRangeError()
    mask_left_end = (wl <= CONTINUUM_ENDPTS[1]) & (wl >= CONTINUUM_ENDPTS[0])
    if not mask_left_end.any():
        raise InvalidLeftEndPointError()
    mask_right_end = (wl <= CONTINUUM_ENDPTS[3]) & (wl >= CONTINUUM_ENDPTS[2])
    if not mask_right_end.any():
        raise InvalidRightEndPointError()
    x_av = np.array([wl[mask_left_end].mean(),wl[mask_right_end].mean()],dtype='float64')
    y_av = np.array([spectrum[:,mask_left_end].mean(1),spectrum[:,mask_right_end].mean(1)],dtype='float64').T
    r_l=reference[mask_left_end]
    r_r=reference[mask_right_end]
    y_r_av = np.array([r_l[~np.isnan(r_l)].mean(), r_r[~np.isnan(r_r)].mean()], dtype='float64')
    con = y_av[:,[0]]+(y_av[:,[1]]-y_av[:,[0]])/(x_av[1]-x_av[0])*(wl[mask].reshape([1,-1])-x_av[0])
    con = spectrum[:,mask]/con
    con_ = y_r_av[0]+(y_r_av[1]-y_r_av[0])/(x_av[1]-x_av[0])*(wl[mask]-x_av[0])
    con_ = reference[mask]/con_
    A = np.array([con_,np.ones(len(con_))],dtype='float64').T
    k = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(con.T)
    mask_fit = k[0]>0 # check signs of depths
    y_fit = k[0][mask_fit].reshape([-1,1])*(con_.reshape([1,-1]))+k[1][mask_fit].reshape([-1,1])
    ss_reg = ((y_fit-con[mask_fit].mean(1).reshape([-1,1]))**2).sum(1)
    ss_tot=((con[mask_fit]-con[mask_fit].mean(1).reshape([-1,1]))**2).sum(1)
    r2=np.zeros(len(con))*np.nan
    r2[mask_fit]=ss_reg/ss_tot
    depth=np.zeros(len(con))*np.nan
    if con_.mean()<1:
        #depth[mask_fit]=(1-con_.min())*k[0][mask_fit]
        depth[mask_fit]=1-(con_.min()*k[0]+k[1])[mask_fit]
    else:
        #depth[mask_fit]=(con_.max()-1)*k[0][mask_fit]
        depth[mask_fit]=(con_.max()*k[0]+k[1])[mask_fit]-1
    return r2, depth

def not_absolute_feature(spectrum, reference, wl, CONTINUUM_ENDPTS, NOT_FEATURE_FIT_CONSTRAINTS,
            NOT_FEATURE_ABSOLUTE_DEPTH_CONSTRAINTS):
    r2, depth = get_fit(spectrum, reference, wl, CONTINUUM_ENDPTS)
    mask_c= (r2>=NOT_FEATURE_FIT_CONSTRAINTS)
    mask_c=mask_c & (depth>=NOT_FEATURE_ABSOLUTE_DEPTH_CONSTRAINTS)
    return mask_c

def not_relative_feature(spectrum, reference, wl, CONTINUUM_ENDPTS, NOT_FEATURE_FIT_CONSTRAINTS,
            RELATIVE_FEATURE_DEPTH, NOT_FEATURE_RELATIVE_DEPTH_CONSTRAINTS):
    r2, depth = get_fit(spectrum, reference, wl, CONTINUUM_ENDPTS)
    mask_c= (r2>=NOT_FEATURE_FIT_CONSTRAINTS)
    mask_c=mask_c & (depth>=(RELATIVE_FEATURE_DEPTH*NOT_FEATURE_RELATIVE_DEPTH_CONSTRAINTS))
    return mask_c

def judge_reference_entry(spectrum, wl, key, resampled1, chanels):
    reference = resampled1[key][chanels]
    fit = np.zeros(len(spectrum))
    depth = np.zeros(len(spectrum))
    fd = np.zeros(len(spectrum))
    i=rf[key]['diagnostic'][0]
    FEATURE_WEIGHT=i[0]
    CONTINUUM_ENDPTS=i[1]
    CONTINUUM_CONSTRAINTS=[j if j!=-99.99 else None for j in i[2]]
    FIT_CONSTRAINTS=(lambda x:x if x!=-99.99 else None)(i[3])
    DEPTH_CONSTRAINTS=[j if j!=-99.99 else None for j in i[4]]
    r1,d1,fd1=diagnostic_feature(spectrum,reference,wl,CONTINUUM_ENDPTS,FEATURE_WEIGHT,
        CONTINUUM_CONSTRAINTS,FIT_CONSTRAINTS,
        DEPTH_CONSTRAINTS)
    fit=fit+r1
    depth=depth+d1
    fd=fd+fd1
    for i in rf[key]['diagnostic'][1:]:
        FEATURE_WEIGHT=i[0]
        CONTINUUM_ENDPTS=i[1]
        CONTINUUM_CONSTRAINTS=[j if j!=-99.99 else None for j in i[2]]
        FIT_CONSTRAINTS=(lambda x:x if x!=-99.99 else None)(i[3])
        DEPTH_CONSTRAINTS=[j if j!=-99.99 else None for j in i[4]]
        r,d,fdd=diagnostic_feature(spectrum,reference,wl,CONTINUUM_ENDPTS,FEATURE_WEIGHT,
            CONTINUUM_CONSTRAINTS,FIT_CONSTRAINTS,
            DEPTH_CONSTRAINTS)
        fit=fit+r
        depth=depth+d
        fd=fd+fdd
    for i in rf[key]['not_abs']:
        reference = resampled1[i[0]][chanels]
        CONTINUUM_ENDPTS = i[1]
        NOT_FEATURE_FIT_CONSTRAINTS=i[2]
        NOT_FEATURE_ABSOLUTE_DEPTH_CONSTRAINTS=i[3]
        mask_c=not_absolute_feature(spectrum, reference, wl, CONTINUUM_ENDPTS, NOT_FEATURE_FIT_CONSTRAINTS,
            NOT_FEATURE_ABSOLUTE_DEPTH_CONSTRAINTS)
        fit[mask_c]=np.nan
        depth[mask_c]=np.nan
        fd[mask_c]=np.nan
    for i in rf[key]['not_rel']:
        reference = resampled1[i[0]][chanels]
        CONTINUUM_ENDPTS=i[1]
        NOT_FEATURE_FIT_CONSTRAINTS=i[2]
        RELATIVE_FEATURE_DEPTH=d1
        NOT_FEATURE_RELATIVE_DEPTH_CONSTRAINTS=i[4]
        mask_c=not_relative_feature(spectrum, reference, wl, CONTINUUM_ENDPTS, NOT_FEATURE_FIT_CONSTRAINTS,
                         RELATIVE_FEATURE_DEPTH, NOT_FEATURE_RELATIVE_DEPTH_CONSTRAINTS)
        fit[mask_c]=np.nan
        depth[mask_c]=np.nan
        fd[mask_c]=np.nan
    minimum_weighted_fit,minimum_weighted_depth,maximum_weighted_depth,minimum_fit_depth=[i if i !=-99.99 else None for i in rf[key]['WEIGHTED_FIT_DEPTH_CONSTRAINTS']]
    mask_c = np.ones(len(spectrum), dtype='bool')
    if minimum_weighted_fit is not None:
        mask_c = fit >= minimum_weighted_fit
    if minimum_weighted_depth is not None:
        mask_c= mask_c&(depth>=minimum_weighted_depth)
    if maximum_weighted_depth is not None:
        mask_c=mask_c&(depth<=maximum_weighted_depth)
    if minimum_fit_depth is not None:
        mask_c=mask_c&(fd>minimum_fit_depth)
    fit[~mask_c]=np.nan
    depth[~mask_c]=np.nan
    fd[~mask_c]=np.nan
    return fit, fd

def geo_map(pathname,b_n):
    img = open_image(pathname)
    spectrum=img.load()
    w=np.array([float(i) for i in img.metadata['wavelength']])/1000
    bp=np.array([float(i) for i in img.metadata['fwhm']])/1000
    resampled1={}
    for i in splib.index:
        res1=[]
        for x,y in zip(w,bp):
            mask=(splib.loc[i,'wave']<x+y/2)&(splib.loc[i,'wave']>x-y/2)
            if not mask.any():
                res1.append(np.nan)
            else:
                lamda=y/2
                sigma=(lamda**2/np.log(2))**0.5
                xx=splib.loc[i,'wave'][mask]-x
                wt=np.exp(-1*xx**2/sigma**2)
                temp=splib.loc[i,'rel'].astype('double')[mask]
                wt=wt[temp>0]
                temp=temp[temp>0]
                if len(temp)==0:
                    res1.append(np.nan)
                else:
                    res1.append((temp*wt).sum()/wt.sum())
        resampled1[i]=np.array(res1)
    t=spectrum.reshape([-1,b_n]).mean(0)
    mask=t<0.02
    chanels=np.arange(b_n)[~mask]
    wl=w[chanels]
    r,c,s=spectrum.shape
    mus_center=np.zeros(r*c,dtype='float')
    image=np.zeros([r*c,3],dtype='uint8')
    FIT_=np.zeros(r*c,dtype='float')
    DEPTH_=np.zeros(r*c,dtype='float')
    NUM_=np.zeros(r*c,dtype='uint8')
    for a,b in [(0,int(r/3)),(int(r/3),int(r*2/3)),(int(r*2/3),r)]:
        spectrum1=spectrum[a:b]
        spectrum1=spectrum1.reshape([-1,s])
        spectrum1=spectrum1[:,chanels]
        index=[]
        FIT=np.zeros([len(spectrum1),len(rf)])
        DEPTH=np.zeros([len(spectrum1),len(rf)])      
        k=0
        for key in rf.keys():
            index.append(key)
            try:
                fit,depth=judge_reference_entry(spectrum1, wl, key, resampled1, chanels)
            except (InvalidRangeError, InvalidLeftEndPointError, InvalidRightEndPointError):
                print(key)
                k = k+1
                continue
            FIT[:,k]=fit
            DEPTH[:,k]=depth
            k=k+1
        FIT[np.isnan(FIT)]=0
        DEPTH[np.isnan(DEPTH)]=0
        im=FIT.argmax(1)
        FIT_[a*c:b*c]   =FIT.flatten()[im+np.arange(len(im))*len(rf)]
        DEPTH_[a*c:b*c] =DEPTH.flatten()[im+np.arange(len(im))*len(rf)]
        NUM_[a*c:b*c]   =im 
        images=np.zeros([len(spectrum1),3],dtype='uint8')
        k=0
        for i in im:
            images[k]=colors_dic[index[i]]
            k=k+1
        images[FIT.max(1)==0]=[0,0,0]
        image[a*c:b*c,:3]=images
        index_d=dict(zip(index,np.arange(index.__len__())))
        muscovite=[
             'calcite.7+muscovite.3',
             'chlorite+muscovite',
             'muscovite_lowAl',
             'muscovite_medAl',
             'muscovite_medhighAl',
             'muscovite_Fe-rich',
             'illite',
             'illite_gds4',
             'kaolin.5+muscovite_medAl',
             'kaolin+muscovite_mix_intimate',
             'kaolin.5+muscovite_medhighAl'
        ]
        for item in muscovite:
            t_mask=(im==index_d[item])&(FIT.max(1)>0)
            center=get_quadratic_center(spectrum1, wl, rf[item]['diagnostic'][0][1], t_mask)
            center[np.isnan(center)]=0
            mus_center[a*c:b*c]=mus_center[a*c:b*c]+center
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(os.path.split(pathname)[0]+'/mineral_map.tiff', c,r,3 ,gdal.GDT_Byte)
    raster.SetMetadataItem('AREA_OR_POINT', 'Point')
    for i in range(3):
        raster.GetRasterBand(i+1).WriteArray(image[:,i].reshape([r,c]))
        raster.FlushCache()
    raster=None
    table=[        
        [255,  30,   0],
        [255, 115,   0],
        [255, 191,   0], 
        [251, 255,  25],
        [207, 255, 110],
        [143, 255, 186],
        [ 15, 251, 255],
        [ 59, 167, 255],
        [ 54,  94, 255],
        [  8,   8, 255]
          ]
    image_=np.zeros([r*c,3],dtype='uint8')
    image_[mus_center<2.199]=table[0]
    image_[(mus_center>=2.199)&(mus_center<2.2)]=table[1]
    image_[(mus_center>=2.2)&(mus_center<2.201)]=table[2]
    image_[(mus_center>=2.201)&(mus_center<2.202)]=table[3]
    image_[(mus_center>=2.202)&(mus_center<2.203)]=table[4]
    image_[(mus_center>=2.203)&(mus_center<2.204)]=table[5]
    image_[(mus_center>=2.204)&(mus_center<2.205)]=table[6]
    image_[(mus_center>=2.205)&(mus_center<2.206)]=table[7]
    image_[(mus_center>=2.206)&(mus_center<2.207)]=table[8]
    image_[mus_center>=2.207]=table[9]
    image_[mus_center<2.195]=[0,0,0]
    image_[mus_center>=2.226]=[0,0,0]
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(os.path.split(pathname)[0]+'/muscovite_wv.tiff', c,r,3 ,gdal.GDT_Byte)
    raster.SetMetadataItem('AREA_OR_POINT', 'Point')
    for i in range(3):
        raster.GetRasterBand(i+1).WriteArray(image_[:,i].reshape([r,c]))
        raster.FlushCache()
    raster=None
    mask=FIT_>0
    images=np.zeros([r*c,3],dtype='uint8')
    for i in index:
        mask1=mask & (NUM_==index_d[i])
        if not mask1.any():
            continue
        dep=DEPTH_[mask1]
        dep=np.log(dep)
        dep=(dep-dep.min())/(dep.max()-dep.min())
        hsi=rgb2hsi((np.array(colors_dic[i])/255).reshape([-1,3]))[0]
        dep=dep*hsi[-1]/3+hsi[-1]*2/3
        #dep=dep*0.5/2+0.5/2
        im=np.zeros([mask1.sum(), 3])
        im[:,0]=hsi[0]
        im[:,1]=hsi[1]
        im[:,2]=dep
        images[mask1]=hsi2rgb(im)*255
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(os.path.split(pathname)[0]+'/mineral_color_enhanced.tiff', c,r,3 ,gdal.GDT_Byte)
    raster.SetMetadataItem('AREA_OR_POINT', 'Point')
    for i in range(3):
        raster.GetRasterBand(i+1).WriteArray(images[:,i].reshape([r,c]))
        raster.FlushCache()
    raster=None

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
