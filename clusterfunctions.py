import sys
sys.path.append("/home/kaylank/imports/")
from imports import *
import pickle

def convolve_w_map(input_map,psi,psize=0.25):
    resrad=psize*0.000290888
    fft_map = np.fft.fft2( np.fft.fftshift(input_map)*resrad)
    fft_map*=psi
    filtered = np.real (np.fft.fftshift(np.fft.ifft2(fft_map)/resrad ))

def sigma_faster(fsz,sf,nmat,ny,nx,b):
    s = sf*fsz
    s = np.reshape(s, (ny,nx,1,b))
    full_int=(s*np.linalg.inv(nmat)*np.transpose(s,axes=(0,1,3,2))).sum(axis=(2,3)) #blue box
    sigma=np.real((np.sum(np.sum(full_int))/(ny*nx))**(-.5))
    return sigma

#create the optimal matched filter
#returns psi, sigma
def psi_faster(fsz,sf,nmat,ny,nx,b):
    s=sf*fsz
    s = np.reshape(s, (ny,nx,1,b))
    sigma=sigma_faster(fsz,sf,nmat,ny,nx,b)
    temp=(np.linalg.inv(nmat)*s).sum(axis=(3))
    return (sigma**2*temp), sigma

def calc_fsz(freq):
    T = 2.725 #temp of the CMB
    nu = freq*10**9 #GHz to Hz
    h=6.62607015*10**-34 
    k_b = 1.380649*10**-23
    
    x = (h*nu)/(k_b*T)
    curr_fsz = x*((np.e**x+1.)/(np.e**x-1.))-4 
    return curr_fsz

def pkl(filename,matrix='None'):
    if type(matrix) != str:
        pickle.dump(matrix, open(filename, "wb"))
        return
    elif matrix == 'None':
        data = pickle.load(open(filename, "rb"))
        return data
    
def gridtomap(ellgrid,cls,els):
                    kmap=np.zeros(np.shape(ellgrid))
                    ny,nx=np.shape(ellgrid)
                    for i in range(0,nx):
                        for j in range(0,ny):
                            index=ellgrid[j,i]
                            maxi=len(cls)
                            if index<maxi:
                                kmap[j,i]=cls[ellgrid[j,i]]
                    return np.asarray(kmap)
def create_multi_band_s(sfilts):
    #b=len(sfilts)
    return np.stack(sfilts,axis=2)  

def radec_to_spt_name(ra,dec,source_class='cluster'):
    names = []
    if (type(ra) == list):
        for i in range(len(ra)):
            names.append(spt3g.sources.radec_to_spt_name(ra[i]*core.G3Units.deg, dec[i]*core.G3Units.deg, source_class=source_class))
        return names
    else:
        return spt3g.sources.radec_to_spt_name(ra*core.G3Units.deg, dec*core.G3Units.deg, source_class=source_class)
    
def convolve_w_map(input_map,psi,psize=0.25):
    resrad=psize*0.000290888
    fft_map = np.fft.fft2( np.fft.fftshift(input_map)*resrad)
    fft_map*=psi
    filtered = np.real (np.fft.fftshift(np.fft.ifft2(fft_map)/resrad ))
    return filtered


def multi_band_filter(amaps,psi,psize,b):
    kmaps=np.array([np.fft.fft2(np.fft.fftshift(m),norm='ortho') for m in amaps])
    kmat=np.stack(kmaps,axis=2)
    filtered=np.sum(psi*kmat,axis=2)
    filtered = np.fft.fftshift(np.fft.ifft2(filtered,norm='ortho'))
    filtered = np.real(filtered)
    return filtered

def distance(x1,y1,x2,y2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2  )
    
def fft_grid(reso,nx,ny=None,dim=1):
    if dim == 2 and ny is None:
        ny = nx
    if ny is not None:
        dim = 2
    n = nx
    ind = np.arange(n,dtype=float)
    wh_lo = np.where(ind <= n/2.0)[0]
    n_lo = len(wh_lo)
    if np.mod(n,2) == 0:
        n_right = n_lo -1
    else:
        n_right = n_lo
    temp_ind = -ind[wh_lo[1:n_right]]
    ind = np.concatenate((ind[wh_lo],temp_ind[::-1]))
    grid = ind/reso/n
    if dim == 2:
        n = ny
        ind = np.arange(n,dtype=float)
        wh_lo = np.where(ind <= n/2.0)[0]
        n_lo = len(wh_lo)
        if np.mod(n,2) == 0:
            n_right = n_lo -1
        else:
            n_right = n_lo
        temp_ind = -ind[wh_lo[1:n_right]]
        ind = np.concatenate((ind[wh_lo],temp_ind[::-1]))
        gridy = ind/reso/n
        grid2d = np.zeros([nx,ny])
        for i in range(ny):
            grid2d[:,i] = np.sqrt(np.square(grid) + np.square(gridy[i]))
        # returns oscillations per radian on sky
        return grid2d
    else:
        return grid    

def ell_grid(psize,ny,nx=None):
    if nx is None:
        nx=ny
    ellgrid=fft_grid(np.radians(psize/60),ny,nx)*2*np.pi
    ellgrid[np.where(ellgrid==0)] = 2
    return ellgrid.astype(int)

def make_filter_xfer(lpf, hpf, isohpf, nx, ny, reso_arcmin = .25, 
                     return_sep = False):
    y, x = np.indices((ny, nx))
    dlx = 360 / (reso_arcmin / 60 * nx)
    dly = 360 / (reso_arcmin / 60 * ny)
    lx = (x - nx / 2.) * dlx
    ly = (y - ny / 2.) * dly
    ellgrid = np.sqrt(lx**2 + ly**2)
    if hpf is not None:
        print(hpf)
        _hpf = np.exp(-(hpf / lx)**6)
    else:
        _hpf = 1
    if lpf is not None:
        _lpf = np.exp(-(lx / lpf)**6)
    else:
        _lpf = 1

    # TdH's code does this Gaussian thing, whereas more official code treats it
    # like the hpf, but iso
    # _isohpf = np.ones_like(ellgrid)
    # inds = np.where(ellgrid < isohpf)
    # _isohpf[inds] = np.exp(-(ellgrid[inds] - isohpf)**2 / (2 * 20**2))
    if isohpf is not None:
        _isohpf = np.exp(-(isohpf / ellgrid)**6)
    else:
        _isohpf = 1
    if return_sep:
        return _lpf, _hpf, _isohpf
    return _lpf * _hpf * _isohpf

def gridtomap(ellgrid,cls,els):
    kmap=np.zeros(np.shape(ellgrid))
    ny,nx=np.shape(ellgrid)
    for i in range(0,nx):
        for j in range(0,ny):
            index=ellgrid[j,i]
            maxi=len(cls)
            if index<maxi:
                kmap[j,i]=cls[ellgrid[j,i]]
    return np.asarray(kmap)

def clstokmap(cls,els,psize,ny,nx=None):
    kmap=gridtomap(ell_grid(psize,ny,nx),cls,els)
    return kmap

from spt3g.util.fitting import gaussfit_hist
def make_snmap(fmaps,apod,mask,psize,cutoffs=None):
    fmaps=np.asarray(fmaps)
        
    stripwidth=int(90/psize)
    ny=np.shape(fmaps[0])[0]
    
    nstrips=int(ny/stripwidth)
    snmaps=fmaps.copy()
    
    sigmas = []
    
    for sn in snmaps:
        sn*=apod
        sn*=mask
        
        for st in range(0,nstrips):
            curr_strip = sn[st*stripwidth:(st+1)*stripwidth,:]
            masked_pix=apod[st*stripwidth:(st+1)*stripwidth,:]*mask[st*stripwidth:(st+1)*stripwidth,:]
            tmap = curr_strip[masked_pix != 0]
            if len(tmap) == 0:
                sigmas.append(0)
            else:
                sig = np.std(tmap)
                tmap = tmap[np.abs(tmap)<sig*5]

                atemp = gaussfit_hist(tmap,1000,-8.*sig,8.*sig,do_plot = False)
                signoise = atemp['params'][2]
                sigma = abs(signoise)
                sigmas.append(sigma)
                
        sn*=apod
        sn*=mask
        
    median_sig = np.median(sigmas)
    for sn in snmaps:
        sn*=apod
        sn*=mask
        updated_sigmas = []
        for st in range(0,nstrips):
            curr_strip = sn[st*stripwidth:(st+1)*stripwidth,:]
            curr_sigma = sigmas[st]
            
            if curr_sigma == 0:
                curr_sigma = median_sig
            
            if curr_sigma*1.2 < median_sig:
                curr_sigma = median_sig
                
            print('Dividing by:', curr_sigma)
            curr_strip/=curr_sigma
            updated_sigmas.append(curr_sigma)
            
        if ny > (nstrips)*stripwidth:
            print('last bit')
            curr_strip = sn[(nstrips)*stripwidth::,:]
            curr_strip/=updated_sigmas[-2]
        
    return snmaps
    
def make_astro_covar_matrix(astro_cl_dict):
    nbands = len(astro_cl_dict.keys())
    ell_max = 43_200
    ncov1d = np.zeros([nbands,nbands, int(ell_max)])

    for i,key in enumerate(astro_cl_dict):
        for j,key2 in enumerate(astro_cl_dict):
            if key == key2:
                for comp_key in astro_cl_dict[key]:
                    ncov1d[i,j,:]+=astro_cl_dict[key][comp_key]
            else:
                for comp_key in astro_cl_dict[key]:
                    ncov1d[i,j,:]+=np.sqrt(astro_cl_dict[key][comp_key]*astro_cl_dict[key2][comp_key])
    return ncov1d


def create_N_d(noisemaps):
    b=len(noisemaps)
    ny,nx=np.shape(noisemaps[0])
    stacked=np.stack(noisemaps,axis=2)
    stacked=np.reshape(stacked,(ny,nx,b,1))
    nmat=np.nan_to_num(np.sqrt(stacked*np.transpose(stacked,axes=(0,1,3,2))))
    nmat*=np.identity(b)
    return nmat

def cilc_multi_band_filter(cf):
    fmaps = []
    for i in range(len(cf.skymaps)):
        curr_map = np.fft.fft2(np.fft.fftshift(cf.skymaps[i])*cf.resrad,norm='ortho')
        curr_map*=cf.cilc_psi[:,:,i]
        filtered = np.fft.fftshift(np.fft.ifft2(curr_map,norm='ortho'))
        filtered = np.real(filtered)
        fmaps.append(filtered)
    return fmaps
        
    

def multi_band_filter(amaps,psi,psize,b):
    kmaps=np.array([np.fft.fft2(np.fft.fftshift(m),norm='ortho') for m in amaps])
    kmat=np.stack(kmaps,axis=2)
    filtered=np.sum(psi*kmat,axis=2)
    filtered = np.fft.fftshift(np.fft.ifft2(filtered,norm='ortho'))
    filtered = np.real(filtered)
    return filtered

def create_ncovbd(ncov_1d,beam_tf,psize,nbands,ny,nx,ell_max):
    ell_astro=np.arange(int(ell_max))
    ellgrid = ell_grid(psize,ny,nx)
    
    noisemaps=np.reshape(np.array([[gridtomap(ellgrid,ncov_1d[i][j],ell_astro)*beam_tf[i]*beam_tf[j] 
                         for i in range(0,nbands)] for j in range(0,nbands)]),(nbands**2,ny,nx))
    
    nmat=np.reshape(np.stack(noisemaps,axis=2),(ny,nx,nbands,nbands))
    return nmat

 


    
def mycreate_ncovbd(ncov_1d,beams,psize,ny,nx):
    nbands = len(beams)
    ell_max = 43_200
    
    ellgrid = ell_grid(psize, ny, nx)
    ell_astro=np.arange(int(ell_max))
    max_astro = max(ell_astro)
    
    ellgrid[ellgrid > max_astro] = 0
    noisemaps = np.array([[cl_to_grid(nx,ny,ncov_1d[i][j])*beams[i]*beams[j] 
                         for i in range(nbands)] for j in range(nbands)])
    
    noisemaps=np.reshape(noisemaps,(nbands**2,ny,nx))
    nmat=np.reshape(np.stack(noisemaps,axis=2),(ny,nx,nbands,nbands))
    
    return nmat


#much faster than the clusterfunctions version, but only works with square ell_grids
def cl_to_grid(grid_width, grid_height, cl_list):
    """
    Fill a 2D rectangular Fourier grid with C_l values from a list.

    :param grid_width: Width of the 2D grid.
    :param grid_height: Height of the 2D grid.
    :param cl_list: List of C_l values.
    :return: 2D grid filled with C_l values.
    """
    half_width = grid_width // 2
    half_height = grid_height // 2

    # Create an empty grid
    cl_grid = np.zeros((grid_height, grid_width))

    for i in range(grid_height):
        for j in range(grid_width):
            l = int(np.sqrt((i - half_height)**2 + (j - half_width)**2))
            
            # Fetch the value from cl_list if available, else use 0
            cl_value = cl_list[l] if l < len(cl_list) else 0
            cl_grid[i, j] = cl_value

    return cl_grid

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    lx, ly
    """

    nx, ny, dx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly
def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d

def cl2map(flatskymapparams, cl, el = None):
    if el is None:
        el = np.arange(len(cl))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl2d = cl_to_cl2d(el, cl, flatskymapparams) 

    #pixel area normalisation
    dx_rad = np.radians(dx/60.)
    pix_area_norm = np.sqrt(1./ (dx_rad**2.))
    cl2d_sqrt_normed = np.sqrt(cl2d) * pix_area_norm

    #make a random Gaussian realisation now
    gauss_reals = np.random.randn(nx,ny)
    
    #convolve with the power spectra
    flatskymap = np.fft.ifft2( np.fft.fft2(gauss_reals) * cl2d_sqrt_normed).real
    flatskymap = flatskymap - np.mean(flatskymap)

    return flatskymap

  


def check_units():
    herschel_ell_eff = [700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2350, 2650, 2950, 3300, 3700, 4150, 4650, 5200, 5849, 6599, 7399, 8299, 9299, 10399]
    C_ell_600_600 = [18.9, 12.4, 8.88, 7.44, 7.03, 6.42, 6.12, 5.22, 4.67, 4.57, 4.27, 3.797, 3.717, 3.509, 3.428, 3.267, 3.103, 3.030, 2.963, 2.836, 2.796, 2.690]
    C_ell_PLW = np.multiply(C_ell_600_600,1e3)
    
    C_ell_857_857 = [39.3, 30.4, 22.8, 18.9, 17.45, 15.55, 14.31, 12.72, 11.20, 11.22, 9.91, 9.32, 8.88, 8.72, 8.30, 7.82, 7.460, 7.350, 7.290, 6.914, 6.650, 6.583]
    C_ell_PMW = np.multiply(C_ell_857_857,1e3)
    
    C_ell_1200_1200 = [57.0, 43.3, 31.9, 28.6, 24.1, 23.9, 21.05, 18.91, 17.28, 16.87, 14.99, 14.04, 13.77, 13.31, 12.80, 12.18, 11.68, 11.59, 11.36, 10.91, 10.56, 10.43]
    C_ell_PSW = np.multiply(C_ell_1200_1200,1e3)
    
    
    spt_ell = [2068, 2323, 2630, 2932, 3288, 3690, 4143, 4645, 5198, 5851, 6604, 7406, 8309, 9312, 10416]
    spt_90 = [18.33, 9.14, 4.04, 2.12, 1.36, 0.63, 1.17, 0.26, 0.67, 0.37, 0.10, 1.01, 0, 0, 0]
    spt_150 = [52.08, 23.86, 11.84, 6.19, 3.59, 2.50, 1.88, 1.67, 1.48, 1.40, 1.38, 1.33, 1.09, 1.31, 0]
    spt_220 = [94.55, 51.74, 34.95, 23.61, 23.24, 19.64, 17.49, 17.09, 16.81, 16.14, 14.81, 15.07, 14.29, 14.06, 15.06]


    return herschel_ell_eff, C_ell_PLW,C_ell_PMW,C_ell_PSW,spt_ell, spt_90,spt_150,spt_220

def bin_radially_in_ell(input, reso_arcmin=0.25, delta_ell=None, ellvec=None, mask=None, trans_func_grid=None, tfthresh=0.5, nosquare=False):
    ngridx = np.shape(input)[1]
    ngridy = np.shape(input)[0]

    reso = np.radians(reso_arcmin / 60.)
    # make grid of ell values
    elltemp = ell_grid(reso_arcmin, ngridy,ngridx)
    llp1_grid = elltemp*(elltemp+1.)/2./np.pi
    # make ell bins
    if delta_ell is None:
        delta_ell = np.max([1, np.floor(2. * np.pi / reso / np.max([ngridx,ngridy]))])
    binned_ell = np.floor(elltemp / np.float(delta_ell))
    binned_ell = np.array(binned_ell,dtype=int)
    maxbin = binned_ell.max()
    clsize = np.int(maxbin + 1)
    cl = np.zeros(clsize)
    nsamp = np.zeros(clsize)
    ellvec = np.arange(clsize) * delta_ell + np.round(delta_ell / 2)
    
    # deal with kmask & transfer function
    if mask is  None:   
        mask = np.zeros([ngridy, ngridx]) + 1.
    if trans_func_grid is not None:  
        ngtf = len(trans_func_grid[0,:])
        if is_square:
            tfg2use = np.copy(trans_func_grid)
        else:
            tfg2use = np.zeros([ngridy,ngridx])
            if ngridx > ngridy:
                x1 = np.arange(ngridy)/np.float(ngridy)
                x2 = np.arange(ngrid)/np.float(ngrid)
                for q in np.arange(ngridx):
                    tf1 = trans_func_grid[:,q]
                    tfg2use[:,q] = np.interp(x2,x1,tf1)
            else:
                x1 = np.arange(ngridx)/np.float(ngridx)
                x2 = np.arange(ngrid)/np.float(ngrid)
                for q in np.arange(ngridy):
                    tf1 = trans_func_grid[q,:]
                    tfg2use[q,:] = np.interp(x2,x1,tf1)
      # zero mask elements where transfer function is below some threshold
        whlt50 = np.where(np.ravel(tfg2use < tfthresh))[0]
        nlt50 = len(whlt50)
        if nlt50 > 0:
            mask[whlt50] = 0.
      # calculate inverse transfer function
        itfg2use = tfg2use
        tfthresh01 = np.max(np.concatenate([0.01, tfthresh]))
        whlt01 = np.where(np.ravel(tfg2use < tfthresh01))[0]
        nlt01 = len(whlt01)
        if nlt01 > 0:
            itfg2use[whlt01] = 1.e6
        itfg2use = 1. / itfg2use
    else:   
        itfg2use = np.zeros([ngridy, ngridx]) + 1.
    
    nsamp = np.bincount(binned_ell.ravel(),weights=mask.ravel())
    if nosquare:
        pstemp = input * itfg2use
        ps = pstemp*mask
    else:
        pstemp = (np.abs(input) * itfg2use) ** 2
        ps = pstemp*mask
    
    cl = np.bincount(binned_ell.ravel(),weights=ps.ravel())
    cl /= nsamp
    outdict = {}
    if ellvec is not None:
        outdict['ell'] = ellvec
    outdict['cl'] = cl
    
    return outdict

def Ri(thetae, x):
    A = [0.0, 4.13674e-03, -3.31208e-02, 1.10852e-01, -8.50340e-01, 9.01794e+00,
         -4.66592e+01, 1.29713e+02, -2.09147e+02, 1.96762e+02, -1.00443e+02,
         2.15317e+01, -4.40180e-01, 3.06556e+00, -1.04165e+01, 2.57306e+00,
         8.52640e+01, -3.02747e+02, 5.40230e+02, -5.58051e+02, 3.10522e+02,
         -6.68969e+01, -4.02135e+00, 1.04215e+01, -5.17200e+01, 1.99559e+02,
         -3.79080e+02, 3.46939e+02, 2.24316e+02, -1.36551e+03, 2.34610e+03,
         -1.98895e+03, 8.05039e+02, -1.15856e+02, -1.04701e+02, 2.89511e+02,
         -1.07083e+03, 1.78548e+03, -2.22467e+03, 2.27992e+03, -1.99835e+03,
         5.66340e+02, -1.33271e+02, 1.22955e+02, 1.03703e+02, 5.62156e+02,
         -4.18708e+02, 2.25922e+03, -1.83968e+03, 1.36786e+03, -7.92453e+02,
         1.97510e+03, -6.95032e+02, 2.44220e+03, -1.23225e+03, -1.35584e+03,
         -1.79573e+03, -1.89408e+03, -1.77153e+03, -3.27372e+03, 8.54365e+02,
         -1.25396e+03, -1.51541e+03, -3.26618e+03, -2.63084e+03, 2.45043e+03,
         5.10306e+03, 3.58624e+03, 9.51532e+03, 1.91833e+03, 9.66009e+03,
         6.12196e+03, 1.12396e+03, 3.46686e+03, 4.91340e+03, -2.76135e+02,
         -5.50214e+03, -7.96578e+03, -4.52643e+03, -1.84257e+04, -9.27276e+03,
         -9.39242e+03, -1.34916e+04, -6.12769e+03, 3.49467e+02, 7.13723e+02,
         7.73758e+03, 5.62142e+03, 4.89986e+03, 3.50884e+03, 1.86382e+04,
         1.71457e+04, 1.45701e+03, -1.32694e+03, -5.84720e+03, -6.47538e+03,
         -9.17737e+03, -7.39415e+03, -2.89347e+03, 1.56557e+03, -1.52319e+03,
         -9.69534e+03, -1.26259e+04, 5.42746e+03, 2.19713e+04, 2.26855e+04,
         1.43159e+04, 4.00062e+03, 2.78513e+02, -1.82119e+03, -1.42476e+03,
         2.82814e+02, 2.03915e+03, 3.22794e+03, -3.47781e+03, -1.34560e+04,
         -1.28873e+04, -6.66119e+03, -1.86024e+03, 2.44108e+03, 3.94107e+03,
         -1.63878e+03]

    THE = (thetae - 0.01) * 100.0 / 4.0
    Z = (x - 2.5) / 17.5
    Rr = (A[0] + A[1]*THE + A[2]*THE**2 + A[3]*THE**3 + A[4]*THE**4 +
          A[5]*THE**5 + A[6]*THE**6 + A[7]*THE**7 + A[8]*THE**8 +
          A[9]*THE**9 + A[10]*THE**10) + (A[11] + A[12]*THE + A[13]*THE**2 +
          A[14]*THE**3 + A[15]*THE**4 + A[16]*THE**5 + A[17]*THE**6 +
          A[18]*THE**7 + A[19]*THE**8 + A[20]*THE**9 + A[21]*THE**10) * Z + \
         (A[22] + A[23]*THE + A[24]*THE**2 + A[25]*THE**3 + A[26]*THE**4 +
          A[27]*THE**5 + A[28]*THE**6 + A[29]*THE**7 + A[30]*THE**8 +
          A[31]*THE**9 + A[32]*THE**10) * Z**2 + (A[33] + A[34]*THE +
          A[35]*THE**2 + A[36]*THE**3 + A[37]*THE**4 + A[38]*THE**5 +
          A[39]*THE**6 + A[40]*THE**7 + A[41]*THE**8 + A[42]*THE**9 +
          A[43]*THE**10) * Z**3 + (A[44] + A[45]*THE + A[46]*THE**2 +
          A[47]*THE**3 + A[48]*THE**4 + A[49]*THE**5 + A[50]*THE**6 +
          A[51]*THE**7 + A[52]*THE**8 + A[53]*THE**9 + A[54]*THE**10) * Z**4 + \
         (A[55] + A[56]*THE + A[57]*THE**2 + A[58]*THE**3 + A[59]*THE**4 +
          A[60]*THE**5 + A[61]*THE**6 + A[62]*THE**7 + A[63]*THE**8 +
          A[64]*THE**9 + A[65]*THE**10) * Z**5 + (A[66] + A[67]*THE +
          A[68]*THE**2 + A[69]*THE**3 + A[70]*THE**4 + A[71]*THE**5 +
          A[72]*THE**6 + A[73]*THE**7 + A[74]*THE**8 + A[75]*THE**9 +
          A[76]*THE**10) * Z**6 + (A[77] + A[78]*THE + A[79]*THE**2 +
          A[80]*THE**3 + A[81]*THE**4 + A[82]*THE**5 + A[83]*THE**6 +
          A[84]*THE**7 + A[85]*THE**8 + A[86]*THE**9 + A[87]*THE**10) * Z**7 + \
         (A[88] + A[89]*THE + A[90]*THE**2 + A[91]*THE**3 + A[92]*THE**4 +
          A[93]*THE**5 + A[94]*THE**6 + A[95]*THE**7 + A[96]*THE**8 +
          A[97]*THE**9 + A[98]*THE**10) * Z**8 + (A[99] + A[100]*THE +
          A[101]*THE**2 + A[102]*THE**3 + A[103]*THE**4 + A[104]*THE**5 +
          A[105]*THE**6 + A[106]*THE**7 + A[107]*THE**8 + A[108]*THE**9 +
          A[109]*THE**10) * Z**9 + (A[110] + A[111]*THE + A[112]*THE**2 +
          A[113]*THE**3 + A[114]*THE**4 + A[115]*THE**5 + A[116]*THE**6 +
          A[117]*THE**7 + A[118]*THE**8 + A[119]*THE**9 + A[120]*THE**10) * Z**10
    return Rr
def FINK(thetae, x):
    SH = (math.exp(x / 2) - math.exp(-x / 2)) / 2
    CH = (math.exp(x / 2) + math.exp(-x / 2)) / 2
    CTH = CH / SH
    XT = x * CTH
    ST = x / SH

    Y0 = -4.0 + XT
    Y1 = -10.0 + 47.0 * XT / 2.0 - 42.0 * (XT ** 2.0) / 5.0 + 7.0 * (XT ** 3.0) / 10.0 + (ST ** 2.0) * (-21.0 / 5.0 + 7.0 * XT / 5.0)
    Y2 = -15.0 / 2 + 1023.0 * XT / 8.0 - 868.0 * (XT ** 2.0) / 5.0 + 329.0 * (XT ** 3.0) / 5.0 - 44.0 * (XT ** 4.0) / 5.0 + 11.0 * (XT ** 5.0) / 30.0 + (ST ** 2.0) * (-434.0 / 5.0 + 658.0 * XT / 5.0 - 242.0 * (XT ** 2.0) / 5.0 + 143.0 * (XT ** 3.0) / 30.0) + (ST ** 4.0) * (-44.0 / 5.0 + 187.0 * XT / 60.0)
    Y3 = 15.0 / 2 + 2505.0 * XT / 8.0 - 7098.0 * (XT ** 2.0) / 5.0 + 14253.0 * (XT ** 3.0) / 10.0 - 18594.0 * (XT ** 4.0) / 35.0 + 12059.0 * (XT ** 5.0) / 140.0 - 128.0 * (XT ** 6.0) / 21.0 + 16.0 * (XT ** 7.0) / 105.0 + (ST ** 2.0) * (-7098.0 / 10.0 + 14253.0 * XT / 5.0 - 102267.0 * (XT ** 2.0) / 35.0 + 156767.0 * (XT ** 3) / 140.0 - 1216.0 * (XT ** 4.0) / 7.0 + 64.0 * (XT ** 5.0) / 7.0) + (ST ** 4.0) * (-18594.0 / 35.0 + 205003.0 * XT / 280.0 - 1920.0 * (XT ** 2.0) / 7.0 + 1024.0 * (XT ** 3.0) / 35.0) + (ST ** 6.0) * (-544.0 / 21.0 + 992.0 * XT / 105.0)
    Y4 = -135.0 / 32.0 + 30375.0 * XT / 128.0 - 62391.0 * (XT ** 2.0) / 10.0 + 614727.0 * (XT ** 3.0) / 40.0 - 124389.0 * (XT ** 4.0) / 10.0 + 355703.0 * (XT ** 5.0) / 80.0 - 16568.0 * (XT ** 6.0) / 21.0 + 7516.0 * (XT ** 7.0) / 105.0 - 22.0 * (XT ** 8.0) / 7.0 + 11.0 * (XT ** 9.0) / 210.0 + (ST ** 2.0) * (-62391.0 / 20.0 + 614727.0 * XT / 20.0 - 1368279.0 * (XT ** 2.0) / 20.0 + 4624139.0 * (XT ** 3.0) / 80.0 - 157396.0 * (XT ** 4.0) / 7.0 + 30064.0 * (XT ** 5.0) / 7.0 - 2717.0 * (XT ** 6.0) / 7.0 + 2761.0 * (XT ** 7.0) / 210.0) + (ST ** 4.0) * (-124389.0 / 10.0 + 6046951.0 * XT / 160.0 - 248520.0 * (XT ** 2.0) / 7.0 + 481024.0 * (XT ** 3.0) / 35.0 - 15972.0 * (XT ** 4.0) / 7.0 + 18689.0 * (XT ** 5.0) / 140.0) + (ST ** 6.0) * (-70414.0 / 21.0 + 465992.0 * XT / 105.0 - 11792.0 * (XT ** 2.0) / 7.0 + 19778.0 * (XT ** 3.0) / 105.0) + (ST ** 8.0) * (-682.0 / 7.0 + 7601.0 * XT / 210.0)

    F1 = thetae * x * math.exp(x) / (math.exp(x) - 1)
    FNK = F1 * (Y0 + thetae * Y1 + Y2 * (thetae ** 2) + Y3 * (thetae ** 3) + Y4 * (thetae ** 4))
    return FNK

def fxsz_itoh(nughz, tempkeV, tcmb=None):
    nughz = np.asarray(nughz)
    tempkeV = np.asarray(tempkeV)
    
    h = 6.626076e-34
    k = 1.380658e-23
    kev = 8.617386e-5  # k_boltzmann to convert temp to eV
    m_e = 9.10940e-31
    mc2 = 511.
    if tcmb is None:
        tcmb = 2.725

    c = 2.997925e8
    i_o = 2. * (k * tcmb) ** 3 / h ** 2 / c ** 2

    x_all = h * (nughz * 1e9) / k / tcmb
    thetae_all = tempkeV / mc2
    ntmp_pts = len(thetae_all)
    nx_pts = len(x_all)
    inten = np.zeros((ntmp_pts, nx_pts))
    gx = np.zeros(nx_pts)

    for i in range(ntmp_pts):
        thetae = thetae_all[i]
        for j in range(nx_pts):
            x = x_all[j]

            if thetae < 0.02:
                delI = FINK(thetae, x)
            elif x < 2.5:
                delI = FINK(thetae, x)
            else:
                delI = FINK(thetae, x) + Ri(thetae, x)

            inten[i, j] = delI * (511. / tempkeV) * (np.exp(x) - 1.) / (x * np.exp(x))
            if i == 0:
                gx[j] = i_o * x ** 4 * np.exp(x) / (np.exp(x) - 1.) ** 2 * (x * (np.exp(x) + 1.) / (np.exp(x) - 1.) - 4.)

    return inten

def export_array_as_fits(array,ra_cent,dec_cent,psize,filepath):
    map_obj = maps.FlatSkyMap(array,
                      res=psize*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(ra_cent)*core.G3Units.deg,
                      delta_center=(dec_cent)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)
    spt3g.maps.fitsio.save_skymap_fits(filepath,map_obj)
    return
