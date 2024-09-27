"""
Lambda1 detector and block subspace filter, for radio interference detection and removal in single-look-complex SAR images
Author: Huizhang Yang
Email: hzyang@njust.edu.cn
Date: 2024.9.27
Reference list:
[1] Huizhang Yang et al., "Lambda-1 Detector for Interference Detection in Synthetic Aperture Radar Images," in peer review.
[2] Huizhang Yang et al., "Robust Block Subspace Filtering for Efficient Removal of Radio Interference in Synthetic Aperture Radar Images," IEEE TGRS, 2024.
[3] Huizhang Yang et al., "BSF: Block Subspace Filter for Removing Narrowband and Wideband Radio Interference Artifacts in Single-Look Complex SAR Images," IEEE TGRS, 2024.
"""
import numpy as np
from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from shutil import copy
import os
import getopt
import sys

def plot_img(data,ref):
    # plot SAR intensity img
    img = np.power(np.abs(data),2)
    ref_img = np.power(np.abs(ref),2)
    std = np.std(ref_img)
    mean = np.mean(ref_img)
    img = np.clip(img, 0, mean + 1.0 * std)
    skip = np.fix(np.max(np.array(data.shape))/2000)
    np.clip(skip,0,float('inf'))
    plt.imshow(img[::int(skip), ::int(skip)], cmap='gray')
    
    # adjust tick labels
    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.set_yticks(ax.get_yticks())
    ylabel = np.array(ax.get_yticks()) * skip
    ax.set_yticklabels(ylabel.astype('int'))
    ax.set_ylim(ylim[0],ylim[1])
    
    xlim = ax.get_xlim()
    xtick = ax.get_xticks()
    xlabel = np.array(ax.get_xticks()) * skip
    ax.set_xticks(xtick[[1,-2]])
    ax.set_xticklabels(xlabel[[1,-2]].astype('int'))
    ax.set_xlim(xlim[0],xlim[1])
    
def get_eig_th(var,n,p,gamma=2,ups1=1.5,ups2=1.5,bdr1=0.7,bdr2=0.7):
    # get lambda1 threshold
    # reference:
    # Huizhang Yang et al., "Lambda-1 Detector for Interference Detection in Synthetic Aperture Radar Images," in peer review.
    ups1 = ups1/bdr1
    ups2 = ups2/bdr2
    n = np.fix(n/ups1)
    p = np.fix(p/ups2)
    var = var*ups1*ups2*gamma
    TW = 2.6672 # pfa = 1e-5
    u = np.square(np.sqrt(n)+np.sqrt(p));
    sigma = np.sqrt(u) * np.power(1/np.sqrt(n)+1/np.sqrt(p),1/3)
    eig_th = (TW*sigma + u)*var
    return eig_th

def lamdba1_bsf(data_complex,lblk=[500,500],lpd=[50,50],
                            gamma=2,ns_max=20,ns_mul=2,
                            ups1=1.5,ups2=1.5,bdr1=0.7,bdr2=0.7):
    # lambda1 detector and block subspace filter   
    # data_complex: 2-D numpy array
    # reference list:
    # [1] Huizhang Yang et al., "Lambda-1 Detector for Interference Detection in Synthetic Aperture Radar Images," in peer review.
    # [2] Huizhang Yang et al., "Robust Block Subspace Filtering for Efficient Removal of Radio Interference in Synthetic Aperture Radar Images," IEEE TGRS, 2024.
    # [3] Huizhang Yang et al., "BSF: Block Subspace Filter for Removing Narrowband and Wideband Radio Interference Artifacts in Single-Look Complex SAR Images," IEEE TGRS, 2024.
    
    lblk = np.array(lblk)
    lpd = np.array(lpd)
    ldata = np.array(data_complex.shape) 
    nblk = np.fix((ldata-2*lpd)/lblk)
    nblk = np.fix(ldata/lblk).astype(int) 

    data_bsf = data_complex.copy() 
    rfi_mask = np.zeros(ldata,dtype=int)
    for kk in range(0,nblk[0]):
        if kk == 0:             
            id0   = (0, lblk[0]+lpd[0])
            id0_i = (0, lblk[0])
            id0_o = (0, lblk[0])
        elif kk == nblk[0]-1:             
            id0 = (kk*lblk[0]-lpd[0], ldata[0])
            id0_i = (lpd[0], ldata[0]-kk*lblk[0] + lpd[0])
            id0_o = (kk*lblk[0],ldata[0])   
        else:
            id0 = kk*lblk[0] + (-lpd[0], lblk[0]+ lpd[0])
            id0_i = (lpd[0],lblk[0]+lpd[0])
            id0_o = kk*lblk[0] + (0,lblk[0])
        # id0 = np.arange(id0[0],id0[1])
        # id0_i = np.arange(id0_i[0],id0_i[1])
        # id0_o = np.arange(id0_o[0],id0_o[1])
        for nn in range(0,nblk[1]):
            if nn == 0:             
                id1   = (0, lblk[1]+lpd[1])
                id1_i = (0, lblk[1])
                id1_o = (0, lblk[1])
            elif nn == nblk[1]-1:             
                id1 = (nn*lblk[1]-lpd[1], ldata[1])
                id1_i = (lpd[1], ldata[1]-nn*lblk[1] + lpd[1])
                id1_o = (nn*lblk[1],ldata[1])               
            else:
                id1 = nn*lblk[1] + (-lpd[1], lblk[1]+ lpd[1])
                id1_i = (lpd[1],lblk[1]+lpd[1])
                id1_o = nn*lblk[1] + (0,lblk[1])
            # id1 = np.arange(id1[0],id1[1])
            # id1_i = np.arange(id1_i[0],id1_i[1])
            # id1_o = np.arange(id1_o[0],id1_o[1])
            
            data_patch = data_complex[id0[0]:id0[1],id1[0]:id1[1]]
            patch_complex = data_patch.copy()
            patch_amplitude = np.abs(patch_complex)
            std = np.std(patch_amplitude)        
            mn = np.mean(patch_amplitude)
            th = mn + 8*std
            patch_complex[patch_amplitude>th] = 0
            
            # plt.figure(1)
            # plt.imshow(patch_amplitude>th,cmap='binary')
            # plt.show()
            
            var = np.var(patch_complex);
            n, p = patch_complex.shape        
            th = get_eig_th(var,n,p,gamma,ups1,ups2,bdr1,bdr2)
            _,tmp,_ = svds(patch_complex, k = 1, maxiter=20)
            lambda1 = np.square(tmp)
            
            if lambda1>th:
                rfi_mask[id0_o[0]:id0_o[1],id1_o[0]:id1_o[1]] = 1
                u1, s1, vh1 = svds(patch_complex, k = ns_max)
                q = np.sum(np.square(s1)>th)*ns_mul
                data_patch = data_patch - np.matmul(u1[:,-q:] * s1[-q:], vh1[-q:,:])
                data_bsf[id0_o[0]:id0_o[1],id1_o[0]:id1_o[1]] = data_patch[id0_i[0]:id0_i[1],id1_i[0]:id1_i[1]]
    return data_bsf, rfi_mask

def tif_block_lambda1_bsf(src_path,dst_prefix='',plot_flag=1,
                          xoff=0,yoff=0,xsize=float('inf'),ysize=float('inf'),
                          l_tif_blk=[float('inf'),float('inf')],lblk=[500,500],lpd=[50,50],
                          gamma=2,ns_max=20,ns_mul=2):
    # segment a tif file into multiple blocks and apply lambda1_bsf  
    # output two tif files: 1) the filtered image, and 2) the other is the removed rfi (channel1) and detection result (channel2).
    dir_name, _ = os.path.split(src_path)
    file_name, extension = os.path.splitext(os.path.basename(src_path))
    dst_path = dir_name + '/' + dst_prefix + file_name + '_filtered' + extension
    rmv_path = dir_name + '/' + dst_prefix + file_name + '_rfi' + extension
    print('output img path:',dst_path)
    print('output rfi path:',rmv_path)
    copy(src_path, dst_path)
    dataset = gdal.Open(dst_path,gdalconst.GA_Update)
    ny = dataset.RasterYSize
    nx = dataset.RasterXSize
    bands = dataset.RasterCount
    #  
    driver = gdal.GetDriverByName("GTiff")
    rmv_dataset = driver.Create(rmv_path,xsize=nx,ysize=ny,bands=bands,eType=gdal.GDT_Float32)
    rmv_dataset.SetProjection(dataset.GetProjection())
    rmv_dataset.SetGeoTransform(dataset.GetGeoTransform())
    # check tif block size
    xsize = int(min([nx-xoff,xsize]))
    ysize = int(min([ny-yoff,ysize]))
    l_tif_blk = [min([l_tif_blk[0],ysize]), min([l_tif_blk[1],xsize])]
    n_tif_blk = np.fix(np.array([ysize,xsize])/l_tif_blk).astype(int)

    print('num. of tif blocks:',n_tif_blk)
    for k in range(0, int(bands/2)):
        band0 = dataset.GetRasterBand(2*k+1)
        band1 = dataset.GetRasterBand(2*k+2)
        # generate tif block start point and size
        for ky in range(0,n_tif_blk[0]):
            yoffk = ky*l_tif_blk[0] + yoff
            ysizek = l_tif_blk[0]
            if ky==n_tif_blk[0]-1:
                ysizek = ysize - ky*l_tif_blk[0]
            for kx in range(0,n_tif_blk[1]):
                xoffk = kx*l_tif_blk[1] + xoff
                xsizek = l_tif_blk[1]
                if kx == n_tif_blk[1]-1:
                    xsizek = xsize - kx*l_tif_blk[1]
                # read tif block by block
                data = np.zeros((ysizek,xsizek,2))
                print('block id:',[ky,kx],',  offset:',[yoffk,xoffk],',  [height, width]:',[ysizek,xsizek])
                data[:,:,0] = band0.ReadAsArray(xoffk,yoffk,xsizek,ysizek)
                data[:,:,1] = band1.ReadAsArray(xoffk,yoffk,xsizek,ysizek)
                data_complex = data[:,:,0] + 1j*data[:,:,1]
                # lambd1 detector and BSF
                data_bsf, rfi_mask = lamdba1_bsf(data_complex,
                                                 lblk,lpd,
                                                 gamma,ns_max,ns_mul)
                # write to tif
                dataset.GetRasterBand(2*k+1).WriteArray(np.real(data_bsf).astype('float32'),xoffk,yoffk)
                dataset.GetRasterBand(2*k+2).WriteArray(np.imag(data_bsf).astype('float32'),xoffk,yoffk)
                
                rmv_dataset.GetRasterBand(2*k+1).WriteArray(np.square(np.abs(data_complex-data_bsf)).astype('float32'),xoffk,yoffk)
                rmv_dataset.GetRasterBand(2*k+2).WriteArray(rfi_mask.astype('int8'),xoffk,yoffk)
                
                if plot_flag==1:
                   if plot_flag==2:
                     plt.ion()
                   plt.figure(k+1)
                    
                   plt.subplot(221)
                   plot_img(data_complex,data_complex)
                   plt.title('source img')
                    
                   plt.subplot(222)
                   plot_img(data_bsf,data_complex)
                   plt.title('filtered img')
                    
                   plt.subplot(223)
                   plot_img(data_complex-data_bsf,data_complex)
                   plt.title('removed rfi')
                    
                   plt.subplot(224)
                   plt.imshow(rfi_mask)
                   plt.title('binary rfi mask')
                   
                   plt.show()
                   
    rmv_dataset = None
    dataset = None

       
def main(argv):  
    src = ''
    prefix = ''
    plot_flag = 1
    
    gamma = 2.5
    ns_max = 15
    ns_mul = 2
    
    xoff = 0 
    yoff = 0 
    xsize = float('inf')
    ysize = float('inf')
    
    l_tif_blk = [8000, 8000]
    lblk = [500, 500]
    lpd = [50, 50]
    
    try:
       opts, args = getopt.getopt(argv,'s:p:f:g:n:m:x:y:w:h:p:q:k:',
                                                  ['src=','prefix=','plot_flag=',
                                                   'gamma=','ns_max=','ns_mul=',
                                                   'xoff=','yoff=','xsize=','ysize=',
                                                   'lblk=','lpd=','l_tif_blk='])
    except getopt.GetoptError:
       print('yang_filter.py -src -prefix -plot_flat -gamma -ns_max -ns_mul -xoff -yoff -xsize -ysize -l_tif_blk -lblk -lpd')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('yang_filter.py -src -prefix -plot_flag -gamma -ns_max -ns_mul -xoff -yoff -xsize -ysize -l_tif_blk -lblk -lpd')
          sys.exit()
       elif opt in ("-s", "--src"):
          src = arg
       elif opt in ("-p", "--prefix"):
          prefix = arg
       elif opt in ("-f", "--plot_flag"):
          plot_flag = int(arg)
       elif opt in ("-g", "--gamma"):
          gamma = float(arg)
       elif opt in ("-n", "--ns_max"):
          ns_max = int(arg)
       elif opt in ("-m", "--ns_mul"):
             ns_mul = int(arg)
       elif opt in ("-x", "--xoff"):
          xoff = int(arg)
       elif opt in ("-y", "--yoff"):
          yoff = int(arg)
       elif opt in ("-w", "--xsize"):
          xsize = float(arg)
       elif opt in ("-h", "--ysize"):
          ysize = float(arg)
       elif opt in ("-p", "--lblk"):
          lblk = list(int(digit) for digit in arg.split(','))
          print(type(lblk))
       elif opt in ("-q", "--lpd"):
          lpd = list(int(digit) for digit in arg.split(','))
          print(type(lpd))
       elif opt in ("-k", "--l_tif_blk"):
          l_tif_blk = list(int(digit) for digit in arg.split(','))
          print(type(lpd))

    tif_block_lambda1_bsf(src,prefix,plot_flag,
                                         xoff,yoff,xsize,ysize,
                                         l_tif_blk,lblk,lpd,
                                         gamma,ns_max,ns_mul)

if __name__ == "__main__":
   main(sys.argv[1:])
   
   # usage:
   # python yang_filter.py --src=E:/20200425_Guangdong_IW1_VH.tif --prefix=demo_ --gamma=2.5 --plot_flag=1 --ns_max=15 --ns_mul=2 --l_tif_blk=8000,8000 --lblk=500,500 --lpd=50,50  --xoff=0 --yoff=0 --xsize=inf --ysize=inf     
