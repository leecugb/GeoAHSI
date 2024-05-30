from spectral import *
from ahsi_algorithm import *
from osgeo import gdal
import os,re,shutil
from xml.dom.minidom import parse
import xml.dom.minidom
def batch_process(pathname,b_n):
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
    raster = driver.Create(os.path.split(pathname)[0]+'/mica.tiff', c,r,3 ,gdal.GDT_Byte)
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
    raster = driver.Create(os.path.split(pathname)[0]+'/mus_wv.tiff', c,r,3 ,gdal.GDT_Byte)
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
    raster = driver.Create(os.path.split(pathname)[0]+'/mica_color_enhanced.tiff', c,r,3 ,gdal.GDT_Byte)
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
