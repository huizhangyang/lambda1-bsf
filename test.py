import yang_filter as yf

src = 'E:/20200425_IW1_VH.tif'
prefix = 'demo_'
plot_flag = 1
# using the default parameters
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
# apply the filter
yf.tif_block_lambda1_bsf(src,prefix,plot_flag,
                           xoff,yoff,xsize,ysize,
                           l_tif_blk,lblk,lpd,
                           gamma,ns_max,ns_mul)
