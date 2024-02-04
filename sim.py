import numpy as np, sys, os, glob
from scipy.io import readsav
from scipy import interpolate as intrp
import scipy as sc
from scipy import integrate, stats
import ilc
from astropy import units as u

import sys
import importlib
sys.path.append("/home/kaylank/imports/pipelines/")
from class_imports import *

from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import concentration
from colossus.cosmology import cosmology

cosmology.setCosmology('planck15')

def add_beam(wi, final_map):
    final_maps = {}
    for key in final_map:
        final_maps[key] = {}
        
        for field in final_map[key]:
            final_maps[key][field] = clu.convolve_w_map(final_map[key][field], wi.setting_dic[field]['beams'][0])
    return final_maps
            
def convolve_sim_maps_w_beam(wi,sim_maps):
    beam_maps = {}
    for key in sim_maps:
        beam_maps[key] = {}
        for field in sim_maps[key]:
            beam_maps[key][field] = clu.convolve_w_map(sim_maps[key][field][0],wi.setting_dic[field]['beams'][0])
    return beam_maps
    
def add_instrumental_nosie(wi,final_map,noise_path='/sptlocal/user/kaylank/megaclusters/sim_data/kayla_sub9_noise_maps/',
                           apod='/sptlocal/user/kaylank/megaclusters/sim_data/kayla_sub9_noise_maps/jacknife_apod_mask.pkl'):
    apod_mask = clu.pkl(apod)
    final_maps = {}
    noise_PSDs = {}
    
    for key in final_map:
        final_maps[key] = {}
        noise_PSDs[key] = {}
        random_num = np.random.randint(0,24)
        random_num = str(random_num)
        if int(random_num) < 10:
            random_num = '0'+random_num
        
        for field in final_map[key]:
            freq = str(wi.setting_dic[field]['freq'])
            file_name =noise_path+'subfield_noise_map_'+freq+'GHz_0'+random_num+'_sub9.fits'
            file_data = fits.open(file_name)[0].data
            
            noise_PSD_2D = spt3g.mapspectra.map_analysis.calculate_powerspectra(my_misc.make_frame(file_data*apod_mask), return_2d=True,
                                                   apod_mask=apod_mask, flatten=False, real=False)
            
            
            ni = clusterfinder_wrapper.congrid(noise_PSD_2D['TT'], (3360,3360))

            
            fft_map = np.fft.fft2(final_map[key][field])
            fft_map+=4.5e7*(np.sqrt(ni)) #FIGURE OUT NUMBER
            

            final_maps[key][field] = np.real( np.fft.ifft2(fft_map) )
            noise_PSDs[key][field] = ni
         
    return final_maps,noise_PSDs


def make_final_maps(scaled_maps, scaled_cluster_sim, scaled_ps_maps):
    final_maps = {}
    for key in scaled_maps:
        final_maps[key] = {}
        for field in scaled_maps[key]:
            curr_dic = scaled_maps[key][field]
            final_maps[key][field] = curr_dic['CMB']+curr_dic['kSZ']+curr_dic['DG-Po']+curr_dic['DG-Cl']+np.asarray(scaled_cluster_sim[key][field])+scaled_ps_maps[key][field]
    return final_maps

def scale_maps(wi,fg_realizations,cluster_sim,ps_map):
    print('Scaling Foreground Sims')
    scaled_fg_maps = {}
    for key in fg_realizations:
        scaled_fg_maps[key] = {}
        for field in wi.setting_dic:
            scaled_fg_maps[key][field] = {}
            
            for comp in fg_realizations[key]:
                if (wi.setting_dic[field]['freq'] == 150) or comp =='kSZ' or comp == 'CMB':
                    scaled_fg_maps[key][field][comp] = fg_realizations[key][comp]
                else:
                    if comp == 'tSZ':
                        scale = clu.calc_fsz(wi.setting_dic[field]['sz_band_centers'])/clu.calc_fsz(wi.setting_dic['3g150']['sz_band_centers'])
                    else:
                        scale = np.sqrt(wi.astro_cl_dict[field][comp][3000]/wi.astro_cl_dict['3g150'][comp][3000])
                        
                    scaled_fg_maps[key][field][comp] = fg_realizations[key][comp]*scale
                
    print('Scaling Cluster Sims')
    scaled_cluster_sim = {}
    for i in range(len(cluster_sim)):
        scaled_cluster_sim[str(i)] = {}
        
        for field in wi.setting_dic:
        
            if wi.setting_dic[field]['freq'] == 150:
                scaled_cluster_sim[str(i)][field] = cluster_sim[i]
            else:
                scaled_cluster_sim[str(i)][field] = cluster_sim[i]*(clu.calc_fsz(wi.setting_dic[field]['sz_band_centers'])/clu.calc_fsz(wi.setting_dic['3g150']['sz_band_centers']))


    scaled_ps_maps = {}
    for i in range(len(ps_map)):
        scaled_ps_maps[str(i)] = {}
        
        for field in wi.setting_dic:
        
            if wi.setting_dic[field]['freq'] == 150:
                scaled_ps_maps[str(i)][field] = ps_map
                
            elif wi.setting_dic[field]['freq'] == 220:
                
                alpha = -0.9
                random_grid = np.random.normal(loc=alpha, scale=0.3, size=(wi.ny,wi.nx))

                scaled_ps_maps[str(i)][field] = ((220/150)**(alpha)) * np.asarray(ps_map)
                
            elif wi.setting_dic[field]['freq'] == 90:
                alpha = -0.7
                random_grid = np.random.normal(loc=alpha, scale=0.3, size=(wi.ny,wi.nx))
                scaled_ps_maps[str(i)][field] = ((90/150)**(alpha)) * np.asarray(ps_map)

    return scaled_fg_maps,scaled_cluster_sim,scaled_ps_maps


def load_sim_data(sim_path,load_halo_cat=True):
    halo_path = sim_path+'halo_sim/'
    ps_path = sim_path+'point_source_sim/'
    fg_path = sim_path+'fg_sim/'
    
    fg_realizations = clu.pkl(fg_path+'fg_sim_realizations_proj5.pkl')
    cluster_sim = clu.pkl(halo_path+'proj5_sim_realizations.pkl')
    if load_halo_cat:
        halo_cat = clu.pkl(halo_path+'halo_catalog.pkl')
        ra_dec_centers = clu.pkl(halo_path+'ra_dec_centers.pkl')
    else:
        halo_cat = 1
        ra_dec_centers = 1
        
    ps_map = clu.pkl(ps_path+'point_source_proj5_realizations.pkl')
    ps_info = pd.read_csv(ps_path+'ps_info_realization_0.tsv',sep='\t')
    return fg_realizations,cluster_sim,halo_cat,ra_dec_centers,ps_map,ps_info


class package_files():
    def __init__(self,package_path,halo_cat_file,tsz_map, 
                 cutout_ra_deg = 10, cutout_dec_deg=10, cutout_res = 0.25, num_realizations=1,
                 is_fullsky=True,verbose=True,cf=None,sz_band_center=148.85):  
        self.halo_cat_file = halo_cat_file
        self.package_path = package_path
        self.tsz_map = tsz_map
        self.is_fullsky = is_fullsky
        self.cf = cf
        self.cutout_ra_deg = cutout_ra_deg
        self.cutout_dec_deg = cutout_dec_deg
        self.cutout_res = cutout_res
        self.num_realizations = num_realizations
        self.verbose=verbose
        self.sz_band_center = sz_band_center
        return
    
    def package_halo_catalog(self,add_500_info=True):
        self.halo_directory = self.package_path+'halo_sim/'
        if not os.path.exists(self.halo_directory):
            os.makedirs(self.halo_directory)
        if '.sav' in self.halo_cat_file:
            print('IDLsav not implemented')
            
        elif '.fits' in self.halo_cat_file:
            loaded_halo_cat_file = fits.open(self.package_path+self.halo_cat_file)
            try:
                self.halo_cat = loaded_halo_cat_file[0].data
            except:
                self.halo_cat = loaded_halo_cat_file[1].data
            if self.halo_cat is None:
                self.halo_cat = loaded_halo_cat_file[1]
            else:
                self.halo_cat = loaded_halo_cat_file[0]
        self.nested_halo_cat = point.fitstable_to_dict(self.halo_cat)
        if add_500_info:
            self.nested_halo_cat = add_500(self.nested_halo_cat['data'],'M200RED')
        
        clu.pkl(self.halo_directory+'halo_catalog.pkl', self.nested_halo_cat )
        
        return
        
    def get_sim_realizations(self):
        self.halo_directory = self.package_path+'halo_sim/'
        outer_rim_tsz_norm = 0.7209869669630059*1.1
        fsz_scale = clu.calc_fsz(self.sz_band_center)
        
        if not os.path.exists(self.halo_directory):
            os.makedirs(self.halo_directory)
            
        healpix_map = maps.fitsio.load_skymap_fits(self.package_path+self.tsz_map)
        keys = list(healpix_map.keys())
        healpix_map = healpix_map[keys[0]]
        healpix_map*=outer_rim_tsz_norm
        healpix_map*=fsz_scale*-1
        
        if self.is_fullsky is False:
            def find_healpix_bounds_with_data_adjusted(full_sky_map):
                nside = hp.npix2nside(len(full_sky_map))
                pixel_indices_with_data = np.where((full_sky_map != 0) & ~np.isnan(full_sky_map))[0]

                theta, phi = hp.pix2ang(nside, pixel_indices_with_data)
                ra = np.rad2deg(phi)
                dec = 90 - np.rad2deg(theta)

                # Sort RA values
                ra_sorted = np.sort(ra)

                # Calculate differences between successive elements
                ra_diffs = np.diff(ra_sorted)

                # Find indices where the difference is greater than 100
                jump_indices = np.where(abs(ra_diffs) > 0.2)[0]

                ra = ra_sorted[:jump_indices[0]]

                bounds = (min(ra), max(ra), min(dec),max(dec))
                return bounds

            bounds = find_healpix_bounds_with_data_adjusted(np.asarray(healpix_map))
            ra_bounds = (bounds[0],bounds[1])
            dec_bounds = (bounds[2],bounds[3])
        else:
            ra_bounds = (0+self.cutout_ra_deg,360-self.cutout_ra_deg)
            dec_bounds = (-90+self.cutout_dec_deg,90-self.cutout_dec_deg)
            
        
        res = self.cutout_res * core.G3Units.arcmin
        width = self.cutout_ra_deg * core.G3Units.degree
        height = self.cutout_dec_deg * core.G3Units.degree
        x_len = int(width/res)
        y_len = int(height/res)

        self.random_ra = random.sample(range(ra_bounds[0], ra_bounds[1]), self.num_realizations) 
        self.random_dec =  random.sample(range(dec_bounds[0], dec_bounds[1]), self.num_realizations) 
        
        self.final_maps = []
        for i in range(len(self.random_ra)):
            proj5_stub = maps.FlatSkyMap(
                x_len=3360, y_len=3360, res=res,
                proj=maps.MapProjection.Proj5,
                alpha_center=self.random_ra[i]*core.G3Units.deg, 
                delta_center=self.random_dec[i]*core.G3Units.deg)
            
            proj5_map = maps.healpix_to_flatsky(healpix_map, map_stub=proj5_stub)
            self.final_maps.append(proj5_map)
        clu.pkl(self.halo_directory+'proj5_sim_realizations.pkl', self.final_maps)
        clu.pkl(self.halo_directory+'ra_dec_centers.pkl', (self.random_ra, self.random_dec))

        return   
    
    def make_fg_realizations(self):
        self.fg_directory = self.package_path+'fg_sim/'
        if not os.path.exists(self.fg_directory):
            os.makedirs(self.fg_directory)

        realizations = {}
        for i in range(self.num_realizations):
            realizations[str(i)] = {}
            for comp in (self.cf.astro_cl_dict['3g150']):
                curr_map = clu.cl2map((3360,3360,0.25,0.25),self.cf.astro_cl_dict['3g150'][comp],el=self.cf.ell)
                curr_map_proj5 = maps.FlatSkyMap(np.asarray(curr_map,order='C'),
                                  res=0.25*core.G3Units.arcmin,
                                  weighted=False,
                                  alpha_center=(352.5)*core.G3Units.deg,
                                  delta_center=(-55)*core.G3Units.deg,
                                  proj=maps.MapProjection.Proj5)

                realizations[str(i)][comp] = curr_map_proj5
        clu.pkl(self.fg_directory+'fg_sim_realizations_proj5.pkl', realizations)
        return
    
    def make_flux_map(self,min_flux,max_flux):
        import csv
        self.ps_directory = self.package_path+'point_source_sim/'
        if not os.path.exists(self.ps_directory):
            os.makedirs(self.ps_directory)
        self.final_flux_maps = []
        
        self.conversion = spt3g.sources.flux_to_temperature(150*core.G3Units.GHz,value=(1*core.G3Units.Jy/(0.25*core.G3Units.arcmin)**2)) / core.G3Units.uK
        
        s, nsources, dndlns = get_poisson_source_counts(band=150, min_flux_mJy=min_flux,max_flux_mJy=max_flux)
        
        self.area_deg = self.cutout_ra_deg*self.cutout_dec_deg
        self.area_norm = (self.area_deg*u.deg**2).to(u.sr).value
        area_norm = self.area_norm
        
        ds = s[1]-s[0]
        log_s = np.log(s)
        log_ds = log_s[1]-log_s[0]
        
        
        
        npix = (3360*3360)
        
        n_sources_tot = dndlns*log_ds
        self.n_sources_area = n_sources_tot*area_norm
        n_sources_area = self.n_sources_area
        
        n_sources_expected = np.sum(self.n_sources_area)
        n_source_safe = 2*np.sum(self.n_sources_area)
        
        psmap_proj5,combined_data,self.mjy_ps_map = point_ps_on_grid(s,ds,self.conversion,n_sources_area,npix)
        
        headers = ['Flux(Jy)', 'Flux(uK)', 'x', 'y']
        j =0
        with open(self.ps_directory+'ps_info_realization_'+str(j)+'.tsv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(headers)
            writer.writerows(combined_data)
            s
        clu.pkl(self.ps_directory+'point_source_proj5_realizations.pkl',psmap_proj5)
        return
    
    
def add_500(cat,mass_key):
        for key in cat:
            c200 = concentration.concentration(cat[key][mass_key], '200c', cat[key]['REDSHIFT'])
            M500, R500, c500 = mass_defs.changeMassDefinition(cat[key][mass_key], c200, cat[key]['REDSHIFT'], '200m', '500m')
            cat[key]['M500'] = M500
            cat[key]['R500'] = R500
            cat[key]['R500'] = c500
        return cat

 
def point_ps_on_grid(s,ds,conversion,n_sources_area,npix,n_source_safe=1e8):
    ny= int(np.sqrt(npix))
    nx=ny
    nstot = 0
    flux = np.zeros(round(n_source_safe))
    pixno_f = np.zeros(round(n_source_safe))
    n_expecteds = []
    n_actuals = []
    
    for i in range(len(s)):
        n_expected = n_sources_area[i]
        nsource = np.random.poisson(n_expected)
        
        n_expecteds.append(n_expected)
        n_actuals.append(nsource)
        
        lsmin,lsmax = s[i]-ds, s[i]+ds
        
        if nsource > 0:
                lflux = np.random.uniform(lsmin,lsmax,nsource)
                flux[nstot:nstot+nsource] = lflux
                pixno_f[nstot:nstot+nsource] = np.floor(np.random.uniform(0,npix,nsource))
                nstot += nsource
                
    flux = flux[0:nstot]
    pixno_f = pixno_f[0:nstot]
    print('Number expected: ', np.sum(n_expecteds))
    print('Number of source: ', np.sum(n_actuals))
    print(np.sum(n_expecteds)/np.sum(n_actuals))
    
    pixno = pixno_f.astype(int)
    psmap = np.zeros(npix)
    
    [psmtemp,nsperpix] = bin_into_pixels(flux,pixno,return_weighted_map=True)
    psmap[0:len(psmtemp)] = psmtemp
    psmap = psmap.reshape(ny,nx)

    y,x = np.unravel_index(pixno, (ny,nx))
    psmap_uk = psmap*conversion
    
    psmap_proj5 = maps.FlatSkyMap(psmap_uk,
                                  res=0.25*core.G3Units.arcmin,
                                  weighted=False,
                                  alpha_center=(352.5)*core.G3Units.deg,
                                  delta_center=(-55)*core.G3Units.deg,
                                  proj=maps.MapProjection.Proj5)
    
    combined_data = zip(flux, np.multiply(flux, conversion), x, y)

    return psmap_proj5,combined_data, psmap 
\

def interpolate_dn_ds(s, dnds, increasing_spacing_by = 10):
    """
    interpolate dN/ds to increase the resolution.
    This is particularly important for high flux (S150>=2 mJy) sources.

    Parameters
    ----------
    s: array
        flux bins.
    dnds: array
        source count in each flux bin.
    increasing_spacing_by: int
        interploation factor. Default is x10.
    
    Returns
    -------
    s_ip: array
        interpolated flux bins.
    dnds_ip: array
        interpolated source counts in each flux bin.
    """

    lns = np.log(s)
    dlns =  (lns[1]-lns[0])

    #get a new closely separated range using interpolation
    dlns_ip = dlns / increasing_spacing_by
    #lns_ip = np.logspace(min(lns), max(lns), dlns_ip)
    s_ip = np.exp( np.arange(min(lns), max(lns), dlns_ip) )
    dnds_ip = np.interp(s_ip, s, dnds)
        
    return s_ip, dnds_ip


def get_poisson_source_counts(band, min_flux_mJy = -1., max_flux_mJy = 6.4e-3, band0 = 150., which_dnds = 'lagache', spec_index_radio = -0.76, dnds_ip_factor = 10):

    """
    get radio source population dN/ds for the desired band.

    Parameters
    ----------
    band: float
        freqeuncy in GHz.
    min_flux_mJy: float
        min. flux required in mJy.
        default is -1 --> no minimum threshold.
    max_flux_mJy: float
        min. flux required in mJy.
    band0: float
        default band in which the source count file is defined.
        default is 150 GHz. Does not work for others.
    which_dnds: str
        dn/ds model to be used.
        default in lagache.
        options are lagache and dezotti.
    spec_index_radio: float
        radio spectral index to be used to scale from band0 to other bands.
        default is -0.7 (R21 SPT result).
    dnds_ip_factor: int
        interploation factor for dN/ds. Default is x10.
    
    Returns
    -------
    s: array
        flux array.
    nsources: array
        total source in each flux bin.
    dndlns: array
        logrithminc source counts in each flux bin.
    """
    if which_dnds == 'dezotti':
        assert band0 == 150
        de_zotti_number_counts_file = '/home/kaylank/imports/catalogs/pt_src_catalogs/counts_150GHz_de_Zotti_radio.res'
        radio_log_s,radio_x,radio_y,radio_logs25n = np.loadtxt(de_zotti_number_counts_file, skiprows = 12, unpack = True)

        #first get the number of sources in each flux range
        s = 10**radio_log_s

        s150_flux = np.copy(s)
        ###print(s); sys.exit()
        #perform scaling
        s = (band/band0)**spec_index_radio * s
        s25n = 10**radio_logs25n
        dnds = s25n / s**2.5

        #20221101 - perform interpolation
        s_ip, dnds_ip = interpolate_dn_ds(s, dnds, increasing_spacing_by = dnds_ip_factor)
        s, dnds = s_ip, dnds_ip
        s150_flux = np.copy(s)
        s = (band/band0)**spec_index_radio * s

        ###print(s[-50:]); sys.exit()

        #get dn/ds and number of sources
        lns = np.log(s)
        dlns = lns[1] - lns[0]
        dndlns = dnds * s
        nsources = dndlns * dlns  # number of sources obtained

    elif which_dnds == 'lagache':
        assert band0 == 150
        lagache_number_counts_file = '/home/kaylank/imports/catalogs/pt_src_catalogs/lagache_2019_ns150_radio.dat'
        s, dnds = np.loadtxt(lagache_number_counts_file, unpack = True)

        #20221101 - perform interpolation
        s_ip, dnds_ip = interpolate_dn_ds(s, dnds, increasing_spacing_by = dnds_ip_factor)
        s, dnds = s_ip, dnds_ip

        s150_flux = np.copy(s)
        #perform scaling
        s = (band/band0)**spec_index_radio * s
        lns = np.log(s)
        dlns =  (lns[1]-lns[0])
        dndlns = dnds*s
        nsources = dndlns * dlns  # number of sources obtained

    elif which_dnds == 'agora':
        assert band0 == 150
        mdpl2_number_counts_searchstr = 'publish/data/mdpl2_radio/mdpl2_radiocat_len_universemachine_trinity_95_150_220ghz_randflux_datta2018_truncgauss_*'
        ##print(mdpl2_number_counts_searchstr); sys.exit()
        mdpl2_number_counts_flist = sorted( glob.glob( mdpl2_number_counts_searchstr ) )
        band_ind_dic = {90: 3, 95: 3, 150: 6, 220: 9} #total LK + HK sources
        colind = band_ind_dic[band0]
        s_arr, dnds_s2p5_arr = [], []
        for mdpl2_number_counts_fname in mdpl2_number_counts_flist:
            s, dnds_s2p5 = np.loadtxt(mdpl2_number_counts_fname, usecols = [0, colind], unpack = True) #Flux S is in Jy.
            s_arr.append(s)
            dnds_s2p5_arr.append(dnds_s2p5)
        s_arr = np.asarray( s_arr )
        dnds_s2p5_arr = np.asarray( dnds_s2p5_arr )
        s, dnds_s2p5 = np.mean(s_arr, axis = 0), np.mean(dnds_s2p5_arr, axis = 0)
        dnds = dnds_s2p5/s**2.5

        s_ip, dnds_ip = interpolate_dn_ds(s, dnds, increasing_spacing_by = dnds_ip_factor)
        s, dnds = s_ip, dnds_ip
        s150_flux = np.copy(s)
        s = (band/band0)**spec_index_radio * s

        #get dn/ds and number of sources
        lns = np.log(s)
        dlns = lns[1] - lns[0]
        dndlns = dnds * s
        nsources = dndlns * dlns  # number of sources obtained

    #pick the required flux indices
    sinds = np.where((s150_flux>min_flux_mJy) & (s150_flux<=max_flux_mJy))
    s, nsources, dndlns = s[sinds], nsources[sinds], dndlns[sinds]

    return s, nsources, dndlns
h, k, c, temp=6.62607e-34, 1.38065e-23, 2.9979e8, 2.725

################################################################
################################################################
def Jy_Sr_K(band, Jy_K = True, K_Jy = False):
    """
    Conversion from Jy/Sr to Kelvin and vice versa.

    Parameters
    ----------
    band: float
        freqeuncy in GHz.
    Jy_K: bool
        Default is True. Convert from flux to temperature.
    k_Jy: bool
        Default is False. Convert from temperature to flux.

    Returns
    -------
    conv: float
        conversion factor from Jy/Sr or vice versa.
    """
    band *= 1e9

    x=h*band/(k*temp)
    
    dbnu = 2.*k*(band**2/c**2)*(x**2*np.exp(x))/(np.exp(x)-1.)**2.0
    conv=1e26*dbnu #Jy to K

    if Jy_K:
        return 1./conv
    else:
        return conv
def get_poisson_power_with_spectra_index_scatter(band1, band2 = None, min_flux_mJy_band0 = -1., max_flux_mJy_band0 = 6e-3, band0 = 150., which_dnds = 'dezotti', spec_index_radio = -0.76, spec_index_radio_scatter = 0.6, units = 'uk', dnds_ip_factor = 1, max_radio_band = 230.):

    flux_arr_band0, counts_arr_band0, dndlns_arr_band0,_ = get_poisson_source_counts(band0, min_flux_mJy = min_flux_mJy_band0, max_flux_mJy = max_flux_mJy_band0, band0 = band0, which_dnds = which_dnds, spec_index_radio = spec_index_radio, dnds_ip_factor = dnds_ip_factor)
    flux_arr, counts_arr, dndlns_arr = flux_arr_band0, counts_arr_band0, dndlns_arr_band0

    if band2 is None: band2 = band1

    s_arr = flux_arr
    #print(s_arr); sys.exit()
    lns = np.log(s_arr)
    dlns =  (lns[1]-lns[0])

    ##print(dlns, s_arr, min_flux_mJy_band0, max_flux_mJy_band0, s_arr**2. * dndlns_arr, 'hihihi'); sys.exit()
    if spec_index_radio_scatter == 0.:
        min_alpha, max_alpha = spec_index_radio - 5. * 0.6, spec_index_radio + 5. * 0.6
        alpha_arr_for_int = np.arange(min_alpha, max_alpha, 0.01)
    else:
        min_alpha, max_alpha = spec_index_radio - 5. * spec_index_radio_scatter, spec_index_radio + 5. * spec_index_radio_scatter
        alpha_arr_for_int = np.arange(min_alpha, max_alpha, 0.01)

    int_arr = []
    for s, dndlns in zip(s_arr, dndlns_arr):
        def integrand_flux_dalpha(s, dndlns, dlns, alpha, alpha_mean, alpha_sigma):                
            return s**2.*(dndlns) * dlns * (band1 * band2 / band0**2.)**alpha * stats.norm.pdf(alpha, alpha_mean, alpha_sigma)
            
        int_val = integrate.simps( integrand_flux_dalpha(s, dndlns, dlns, alpha_arr_for_int, spec_index_radio, spec_index_radio_scatter), x=alpha_arr_for_int )
        ##int_val = s**2.*(dndlns) * dlns
        int_arr.append( int_val )

    integral = np.sum(int_arr)

    Jy_to_K_conv = Jy_Sr_K(band1) * Jy_Sr_K(band2)
    #print(Jy_Sr_K(band1)*1e6, Jy_Sr_K(band2)*1e6); sys.exit()
    cl_poisson = integral * Jy_to_K_conv

    if units == 'uk': cl_poisson *= 1e12

    if band1 > max_radio_band or band2 > max_radio_band: cl_poisson *= 0.

    return cl_poisson
def bin_into_pixels(data, pointing, dweight=1, 
                    use_histogram=True, return_weighted_map=False, return_weights=True):
    """
    Make simple map, using simplest or next-to-simplest algorithm.
    """

    which_alg = 2
    if use_histogram: which_alg = 1

    indp = pointing.astype(int)
    npixels = np.max(indp) + 1
    map = np.zeros(npixels)
    weights = np.zeros(npixels)

# check if dweight is a vector the same length as data
    if hasattr(dweight,"__len__"):
        if len(dweight) == len(data):
            dw2use = dweight
        else:
            dw2use = np.zeros(len(data)) + dweight[0]
    else:
        dw2use = np.zeros(len(data)) + dweight

    if which_alg == 1:
        sp = np.argsort(indp)
        sindp = indp[sp]
        sdata = data[sp]
        [upix,invind] = np.unique(sindp,return_inverse=True)
        histind = (np.histogram(invind,bins=np.max(invind)+1,range=(0,max(invind)+1)))[0]
        ntot = 0
        for i in np.arange(len(histind)):
            h = histind[i]
            if h > 0:
                theseinds = np.arange(h) + ntot
                thispix = upix[i]
                map[thispix] = np.sum(sdata[theseinds]*dw2use[theseinds])
                weights[thispix] = np.sum(dw2use[theseinds])
                ntot += h

    if which_alg == 2:
        for i in np.arange(len(data)):
            map[indp[i]] += data[i]*dw2use[i]
            weights[indp[i]] += dw2use[i]
    
    if return_weighted_map == False:
        whn0 = (np.where(weights > 0))[0]
        map[whn0] /= weights[whn0]

    if return_weights:
        return [map,weights]
    else:
        return map
    
def make_cumulative_distribution(data,bins):
    bin_counts, bin_edges = np.histogram(data, bins=bins)
    cumulative_counts = np.cumsum(bin_counts)
    cumulative_distribution = cumulative_counts / cumulative_counts[-1]
    return cumulative_distribution