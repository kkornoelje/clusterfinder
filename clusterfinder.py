import sys
import source_subtraction
import pointing
sys.path.append("/home/kaylank/imports/")
from imports import *
import cilc


class cluster_finder():
    def __init__(self,fields,dic,units='uk',verbose=True,psize=0.25):  
        self.beta_profile_file = '/big_scratch/mck74/polmaps/sptpol_100d/'+'profiles_12_beta1_rc025_rc300.sav' #This is where the beta profiles live.
        self.pol_directory='/big_scratch/mck74/polmaps/sptpol_100d/' 
        
        self.tc_arcmin=np.array([ 0.25,  0.5 ,  0.75,  1.  , 1.25,  1.5 ,  1.75,  2.  ,  2.25, 2.5 ,  2.75,  3.  ]) #These are the core sizes for clusters that you iteratively filter over (ie 0.25' - 3')
        
        self.idl_profiles = readsav(self.beta_profile_file,verbose=False)
        self.ell = np.arange(43_200)
        self.idl_ls = np.arange(80001)
        self.ell_max = 43_200
        self.psize = psize
        
        self.units = units
        self.fields = fields
        self.verbose = verbose
        self.nbands = len(self.fields)
        self.initial_field = self.fields[0]
        self.fields = fields

        self.map_setting_dic = dic['map_inputs_dic']
        self.tcs=np.array([tci/self.psize for tci in self.tc_arcmin])
        
        
        #Grab all the values from the google doc and throw them into a dictionary
        for key, value in self.map_setting_dic.items():
            setattr(self, key, value)
            
            
        self.setting_dic = {}
        for key in dic['inputs_dic']:
            if key in self.fields:
                self.setting_dic[key] = copy.deepcopy(dic['inputs_dic'][key])
                
        #Set the global SPT flatskymap (used to convert RA/DEC to x/y and vice versa)
        self.map_obj = maps.FlatSkyMap(np.zeros((self.ny,self.nx)),
                      res=self.psize*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(self.field_ra_cent)*core.G3Units.deg,
                      delta_center=(self.field_dec_cent)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)
                
        if self.verbose:
            print('Running the cluster finder for the fields:', self.fields)
            
        
        if self.units == 'mJy':
            print('Running cluster finder in mJy. Are you using Herschel maps?')
            for key in self.setting_dic:
                from astropy import units as u
                from astropy.cosmology import Planck15

                freq = self.setting_dic[key]['sz_band_centers'] * u.GHz
                equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
                val = (1. * u.uK).to(u.Jy / u.sr, equivalencies=equiv) 
                self.setting_dic[key]['to_mJy'] = val.value
        return
    
    def make_fsz(self,do_rel=False,rel_kev=None):
        #Can change once updated.
        for f in self.setting_dic:
            szband = self.setting_dic[f]['sz_band_centers']
            curr_fsz = clu.calc_fsz(szband)
            
            if self.units == 'mJy':
                if 'W' not in f:
                    curr_fsz*=self.setting_dic[f]['to_mJy']

            self.setting_dic[f]['fsz'] = curr_fsz   
        self.fsz = [self.setting_dic[f]['fsz'] for f in self.fields]
        return
        
    def make_maps(self,what_mask):
        for i,key in enumerate(self.setting_dic):
            print(' d')
            self.ss_map, self.mask, self.mask2 = load_mask(self.setting_dic, key,what_mask,self.ny,self.nx) 
            self.setting_dic[key]['ss_maps'] = self.ss_map
            self.setting_dic[key]['skymaps']+=self.ss_map
            self.setting_dic[key]['skymaps']*=self.mask
            if self.units == 'mJy':
                if 'W' not in key:
                    self.setting_dic[key]['skymaps']*=self.setting_dic[key]['to_mJy']
        return

    def make_beams(self,theory=False):
        for i,f in enumerate(self.setting_dic):
            self.setting_dic[f]['beams'].append(self.setting_dic[f]['beams'][0])
        return  
    
    def make_astro_noise(self,ignore_fgs=[]):
        #Can rewrite when people are updated
        self.ignore_fgs = ignore_fgs
        components = ['tSZ','kSZ','CMB','DG-Po','DG-Cl']
        
        self.astro_cl_dict = {}

        for i, field in enumerate(self.fields):
            self.astro_cl_dict[field] = {}
            band = str(self.setting_dic[field]['freq'])+'GHz'
            
            for index, comp in enumerate(components):
                if comp not in ignore_fgs:
                    curr_fg =foregrounds.get_foreground_power(comp, rescale='True', band1=band, model='reichardt')/core.G3Units.uK**2
                    if self.units == 'mJy':
                        curr_fg*=(self.setting_dic[field]['to_mJy']**2)
                    if 'PLW' in field:
                        curr_fg*=1e-50
                    curr_fg = np.pad(curr_fg, (0,29699), 'constant', constant_values=curr_fg[-1]) 
                    self.astro_cl_dict[field][comp] = curr_fg
        return

    def make_instr_noise(self,scale=1):
        for i,key in enumerate(self.setting_dic):
            curr_noise_instrument = self.setting_dic[key]['NoiseInstrument'][0]
            if self.units == 'mJy':
                curr_noise_instrument*=(self.setting_dic[key]['to_mJy']**2)
                
            if 'W' in key:
                curr_noise_instrument = np.fft.fft2(np.fft.fftshift(self.setting_dic[key]['skymaps'][0])*self.resrad,norm='ortho')
                

            self.setting_dic[key]['NoiseInstrument'][0] = (curr_noise_instrument)
        return
       
    def make_covar(self,my_inver='None',my_ncov1d='None',nmatastro='None',nmatinst='None'):
        self.ncov1d = clu.make_astro_covar_matrix(self.astro_cl_dict)
        self.nmatastro=create_ncovbd(self.ncov1d, [self.setting_dic[f]['beams'][0] for f in self.fields],
                                        self.psize,self.nbands,self.ny,self.nx,self.ell_max)

        self.NoiseInstrument = [self.setting_dic[f]['NoiseInstrument'][0] for f in self.fields]
        self.nmatinst=clu.create_N_d(self.NoiseInstrument)

        self.nmatfull=self.nmatinst + self.nmatastro
        self.inver = np.linalg.pinv(self.nmatfull)
            
        return

    def find_clusters(self,sn_cutoff,tcs=(0,1),bvec='None',point_source_finder=False,add_false_detections=False):
        self.bvec = bvec
        self.skymaps = [self.setting_dic[f]['skymaps'][0] for f in self.fields]
        
        self.psis={};self.fmaps={};self.snmaps={};self.predicteds=np.array([]);self.measureds=np.array([])
        self.maxes=np.array([]);self.xes=np.array([]);self.yes=np.array([]);self.tlocs=np.array([])
        self.results = {}
        
        
        for ti in range(tcs[0],tcs[1]):
            name = self.tc_arcmin[ti]
            self.results[name] = {}
            self.results[name]['psi'] = []
            self.results[name]['predicted_sigma'] = []
            self.results[name]['measured_sigma'] = []
            self.results[name]['snmap'] = []
            self.results[name]['cilc_psi'] = []
            self.results[name]['cilc_fmaps'] = []
            
        print('norm')
        norm_factors =[ 7.66780e-08,  3.06715e-07,  6.90103e-07,  1.22692e-06,  1.91692e-06,  2.76045e-06,  3.75714e-06,  4.90727e-06,  6.21089e-06,
  7.66767e-06,  9.27791e-06,  1.10415e-05]
        for ti in range(tcs[0],tcs[1]):
            name = self.tc_arcmin[ti]
            print(name)
            profile_ft=clu.gridtomap(clu.ell_grid(self.psize,self.ny,self.nx),self.idl_profiles['profs'][ti],self.idl_ls)
            self.profile_ft = profile_ft#*norm_factors[ti]
                    
            if point_source_finder:
                print('Finding Point Sources')
                self.fsz = [1]*self.nbands
                self.ft_signal = np.stack([self.setting_dic[f]['beams'][0] for f in self.fields],axis=2) 
                
            else:     
                print('Finding Clusters')
                ft_signal= clu.create_multi_band_s([profile_ft * self.setting_dic[f]['beams'][0] for f in self.fields])
                self.ft_signal = ft_signal
                
            if type(bvec) is str:
                print('Running Minimum Variance')
                self.psi, self.sigma = clu.psi_faster(self.fsz, self.ft_signal, self.nmatfull, self.ny, self.nx, self.nbands)
                self.fmap = clu.multi_band_filter(self.skymaps, self.psi,self.psize,self.nbands)
                self.snmap = clu.make_snmap([self.fmap],self.apod2,self.mask2,self.psize)[0]
                
            else:
                print('Running cILC')
                self.min_var_weight,self.unit_weights,self.null_weights,_,_ = cilc.get_multipole_weights(self.inver,
                                                                                                         self.fsz,
                                                                                                         np.ones_like(self.ft_signal),
                                                                                                         self.bvec)
                
                self.cilc_fmaps, self.cilc_fmap  = cilc_multi_band_filter(self, self.skymaps, 
                                                                                        self.unit_weights, np.ones_like(self.ft_signal))
                two_d_astro_cl_dict = {}
                
                ellgrid = clu.ell_grid(self.psize,self.ny,self.nx)
                two_d_astro_cl_dict['total'] = {}
                null_key = self.fields[0]

                for comp in self.astro_cl_dict[null_key]:
                    two_d_astro_cl_dict['total'][comp] = np.zeros((self.ny,self.nx))
                    for i,key in enumerate(self.nmatastro):
                        two_d_comp =  gridtomap_optimized(ellgrid, np.sqrt(self.nmatastro[key][comp]))
                        rescaled = two_d_comp*(unit_weights[:,:,i])
                        two_d_astro_cl_dict['total'][comp]+=rescaled

                    two_d_astro_cl_dict['total'][comp] = two_d_astro_cl_dict['total'][comp]**2
                
                self.two_d_astro_cl_dict = two_d_astro_cl_dict
                
                rescaled_ncov1d = np.zeros((3360,3360))

                for comp in two_d_astro_cl_dict['total']:
                    rescaled_ncov1d+=two_d_astro_cl_dict['total'][comp]
                
                self.rescaled_ncov1d = rescaled_ncov1d
                rescaled_nmatastro = rescaled_ncov1d[:,:,np.newaxis,np.newaxis]
                
                rescaled_NoiseInstrument = np.zeros((3360,3360))

                for i in range(len(self.NoiseInstrument)):
                    temp_noise = np.sqrt(self.NoiseInstrument[i])
                    temp_noise*=self.unit_weights[:,:,i]
                    rescaled_NoiseInstrument+=temp_noise

                rescaled_NoiseInstrument = rescaled_NoiseInstrument**2
                self.rescaled_NoiseInstrument = rescaled_NoiseInstrument
                
                rescaled_nmatinst = rescaled_NoiseInstrument[:,:,np.newaxis,np.newaxis]
                rescaled_nmatfull=rescaled_nmatastro + rescaled_nmatinst
                rescaled_inver = np.linalg.pinv(rescaled_nmatfull)
                
                self.psi, unit,null,_,_ = cilc.get_multipole_weights(rescaled_inver, [1], self.ft_signal[:,:,0], deproj=[0])
                self.fmap = clu.multi_band_filter([self.cilc_fmap], self.psi, self.psize,1)
                self.snmap = clu.make_snmap([self.fmap],self.apod2,self.mask2,self.psize)[0]

            groups=sources.source_utils.find_groups(self.snmap,signoise=1,nsigma=sn_cutoff)

            self.maxes=np.append(self.maxes,groups['maxvals'])
            self.xes=np.append(self.xes,groups['xcen'])
            self.yes=np.append(self.yes,groups['ycen'])
            self.tlocs=np.append(self.tlocs,self.tcs[ti]*np.ones(groups['n_detected'])*self.psize)

        if len(self.maxes) == 1:
            print('No Clusters Found')
            return
        else:
            self.cat = make_clu_cat(self.maxes,self.xes,self.yes,self.tlocs,self.map_obj)
            if add_false_detections:
                print('Producing a list of possible false detections')

                self.flags = ['None']*len(self.cat['xvals'])
                ss = source_subtraction.source_subtraction(self,what_field=self.fields[0],
                            sn_threshold = 5,
                            temp_threshold = 7,
                            mel_flux=True,
                            what_pts_file=None,
                            verbose=True)

                ss.make_full_source_catalog()
                ss.make_subfield_catalog(check_mask=False)

                for source in ss.in_subfield_cat:
                    if ss.in_subfield_cat[source]['SN90'] >= 40:
                        source_x = ss.in_subfield_cat[source]['x']
                        source_y = ss.in_subfield_cat[source]['y']

                        for i in range(len(self.cat['xvals'])):
                            clu_x = self.cat['xvals'][i]
                            clu_y = self.cat['yvals'][i]

                            if np.sqrt( (clu_x - source_x)**2 +  (clu_y - source_y)**2) <= (28*0.25):
                                self.flags[i] = 'Near Source'

                for i in range(len(self.cat['xvals'])):
                    if self.cat['sigs'][i] > 60:
                        large_x = self.cat['xvals'][i]
                        large_y = self.cat['yvals'][i]

                        for j in range(len(self.cat['xvals'])):
                            if i!= j:
                                small_x = self.cat['xvals'][j]
                                small_y = self.cat['yvals'][j]

                                if np.sqrt( (clu_x - source_x)**2 +  (clu_y - source_y)**2) <= (28*0.25):
                                    self.flags[j] = 'Near Large Cluster'
                self.cat['comments'] = self.flags
        return

    def plot_results(self):
        plt.rcParams['figure.figsize']=[40,40]
        fig, (ax1) = plt.subplots()
        scale = np.mean(self.snmap[500:1000,500:1000])
        ax1.imshow(self.snmap,vmin=-8,vmax=8,cmap='Greys')
        ax1.scatter(self.cat['xvals'],
                    self.cat['yvals'],
                    facecolors='none',edgecolor='red',
                    s=250)

        ax1.legend(loc=0,fontsize=22)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Detected Clusters Circled on Filtered Map\n'+
                      'Colors Correspond to Core Size\n',
                     fontsize=22)
        return


def cilc_multi_band_filter(cf,skymaps, psi,profile):
    fmaps = []
    
    for i in range(len(skymaps)):
        curr_map = skymaps[i]
        curr_map = np.fft.fft2(np.fft.fftshift(curr_map)*cf.resrad,norm='ortho')
        curr_map*=psi[:,:,i]
        curr_map*=profile[:,:,i]
        
        filtered = np.real(np.fft.fftshift(np.fft.ifft2(curr_map/cf.resrad,norm='ortho')))
        
        fmaps.append(filtered)
    fmap_final = np.zeros_like(skymaps[0])
    for i in range(len(fmaps)):
        fmap_final+=fmaps[i]
    
    return fmaps, fmap_final
def make_clu_cat(maxes,xes,yes,tlocs,map_obj):
    zipped=np.array(sorted(zip(maxes,xes,yes,tlocs),reverse=True))
    cat={}
    cat['sigs']=np.array([])
    cat['xvals']=np.array([])
    cat['yvals']=np.array([])
    cat['core_size']=np.array([])
    
    
    candidates=zipped.copy()
    coords=candidates[:,[1,2]]
    while len(candidates>0):
        cat['sigs']=np.append(cat['sigs'],candidates[0][0])
        cat['xvals']=np.append(cat['xvals'],candidates[0][1])
        cat['yvals']=np.append(cat['yvals'],candidates[0][2])
        cat['core_size']=np.append(cat['core_size'],candidates[0][3])    
        
        coords, candidates=remove_duplicates(0,coords,candidates,16)   
    ra,dec = spt3g.maps.FlatSkyMap.xy_to_angle(map_obj, cat['xvals'],cat['yvals'])
    ra/=spt3g.core.G3Units.deg
    ra = list(ra)
    dec/=spt3g.core.G3Units.deg
    dec = list(dec)
    cat['RA'] = ra
    cat['DEC'] = dec
    
    cat['CAND_NAME']=[]
    for i in range(len(ra)):
    
        cat['CAND_NAME'].append( spt3g.sources.radec_to_spt_name(np.multiply(ra[i],core.G3Units.deg),
                                                           np.multiply(dec[i],core.G3Units.deg),
                                                           source_class='cluster') )
    
    return cat
def remove_duplicates(index,clist,flist,r):
    c1=clist[index]
    mask = np.linalg.norm(clist-c1, axis=1) > r
    return clist[mask],flist[mask]
def export_results(self,name,export_array_as_fits='False'):
    cat_df = pd.DataFrame(self.cat)
    cat_df.to_csv(r'/sptlocal/user/kaylank/megaclusters/temp_results/'+name+'.csv', index=False, header=True)
    if type(export_fits) != False:
        maps.FlatSkyMap(export_array_as_fits,
                      res=self.psize*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(self.field_ra_center)*core.G3Units.deg,
                      delta_center=(self.field_dec_center)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)
        maps.fitsio.save_skymap_fits(r'/home/kaylank/clusterfinding/results/map_'+name+'.csv')
    return
#not done
def plot_inputs(cf,keys):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    axs = [ax1,ax2]
    titles = ['SPTpol 500d 90GHz','SPTpol 500d 150GHz']
    curr_fields = ['pol_5d_90','pol_5d_150']

    fig.suptitle('SPTpol 500d',fontsize=15)

    for i,field in enumerate(curr_fields):
        instrumental_noise = copy.deepcopy(cf.setting_dic[field]['NoiseInstrument'][0])
        instr_ps = spt3g.simulations.quick_flatsky_routines.bin_radially_in_ell(instrumental_noise)
        axs[i].semilogy(instr_ps['ell'], np.sqrt(instr_ps['cl']))
        
        for comp in cf.astro_cl_dict['3g90']:
            axs[i].semilogy(cf.ell, cf.astro_cl_dict[field][comp],label=comp)
            
        axs[i].legend(fontsize=13)
        axs[i].set_xlim(0,10_000)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(r'$\ell$',fontsize=15)
        axs[0].set_ylabel(r'$C\ell (\mu K^2 -rad)$',fontsize=16)
    return
def plot_psi(plot=True):
    ell_min = 0
    ell_max = np.pi / (self.resrad)
    delta_ell = 2 * np.pi / (np.max([self.nx,self.ny]) * self.resrad)
    ell_bins = np.arange(ell_min, ell_max + delta_ell, delta_ell)
    ell_plot=(np.array(ell_bins[1:]) + np.array(ell_bins[:-1]))/2

    pypsi=np.reshape(np.split(self.psi,self.nbands,axis=2),(self.nbands, 3360, 3360)) 
    py = []
    for i in range(len(pypsi)):
        py.append(basicmaputils.av_ps((-1*pypsi[i]),
                                     .25*core.G3Units.arcmin,
                                     ell_bins,s=(self.ny,self.nx),real=False))
    self.py
    if plot:
        x = ell_plot
        plt.semilogx(x,py[0])
        plt.semilogx(x,py[1])
        plt.semilogx(x,py[2])
        plt.semilogx(x, py[0]+py[1]+py[2],color='black')
    return
    
def ilc(cf,do_cilc=False,add_beam=True):
    if do_cilc == False:
        s = np.ones_like(cf.ft_signal) * cf.fsz
        s = np.reshape(s, (cf.ny,cf.nx,1,cf.nbands))
        result_matrix = np.einsum('ijkl,ijkl->ijk', cf.inver, s)
        return result_matrix

    else:
        print('Doing cILC')
        unit_weights,null_weights = cilc.get_multipole_weights(cf, cf.bvec)
        return unit_weights,null_weights
    
def create_ncovbd(ncov_1d,beam_tf,psize,nbands,ny,nx,ell_max):
    ell_astro=np.arange(int(ell_max))
    print([ny,nx,nbands,nbands])
    ellgrid = clu.ell_grid(psize,ny,nx)
    
    noisemaps=np.reshape(np.array([[gridtomap_optimized(ellgrid,ncov_1d[i][j])*beam_tf[i]*beam_tf[j] 
                         for i in range(0,nbands)] for j in range(0,nbands)]),(nbands**2,ny,nx))
    nmat=np.reshape(np.stack(noisemaps,axis=2),(ny,nx,nbands,nbands))
    return nmat

def gridtomap_optimized(ellgrid, cls):
    ny, nx = ellgrid.shape
    kmap = np.zeros((ny, nx))

    # Create a mask where ellgrid values are less than the length of cls
    valid_mask = ellgrid < len(cls)

    # Apply the mask and use advanced indexing for valid elements
    kmap[valid_mask] = np.array(cls)[ellgrid[valid_mask]]

    return kmap        

    












#Not Yet Done




def make_masking_dic(source_list, dictionary):
    masking_dic_1 = {}
    masking_dic_2 = {}

    for source in source_list:
        masking_dic_1[source] = {}
        masking_dic_1[source]['RA'] = self.ss.full_pt_src_cat[source]['RA']*core.G3Units.deg
        masking_dic_1[source]['Dec'] = self.ss.full_pt_src_cat[source]['DEC']*core.G3Units.deg
        masking_dic_1[source]['Radius'] = 4*core.G3Units.arcmin

        masking_dic_2[source] = {} 
        masking_dic_2[source]['RA'] = self.ss.full_pt_src_cat[source]['RA']*core.G3Units.deg
        masking_dic_2[source]['Dec'] = self.ss.full_pt_src_cat[source]['DEC']*core.G3Units.deg
        masking_dic_2[source]['Radius'] = 8*core.G3Units.arcmin

    mask = sources.make_point_source_map_from_dict(self.map_obj,masking_dic_1)
    mask.invert()
    mask2 = sources.make_point_source_map_from_dict(self.map_obj,masking_dic_2)
    mask2.invert()

    return np.asarray(mask), np.asarray(mask2)

def make_mask(flat_sky_map, pts_file=None,ra_list=None,dec_list=None,masking_radius_1 = 4, masking_radius_2 = 8):
    masking_dic_1 = {}
    masking_dic_2 = {}
    
    if pts_file != None:
        pts_list = pd.read_csv(pts_file,comment='#',delim_whitespace=True,names=['Index','RA','DEC','Radius'])
        for i in range(len(pts_list['DEC'])):
            masking_dic_1[str(i)] = {}
            masking_dic_2[str(i)] = {}
            
            masking_dic_1[str(i)]['RA'] = pts_list['RA'][i]*core.G3Units.deg
            masking_dic_2[str(i)]['RA'] = pts_list['RA'][i]*core.G3Units.deg
            
            masking_dic_1[str(i)]['Dec'] = pts_list['DEC'][i]*core.G3Units.deg
            masking_dic_2[str(i)]['Dec'] = pts_list['DEC'][i]*core.G3Units.deg
            
            masking_dic_1[str(i)]['Radius'] = masking_radius_1*core.G3Units.arcmin
            masking_dic_2[str(i)]['Radius'] = masking_radius_2*core.G3Units.arcmin
    else:
        if ra_list != None and dec_list!= None:
            for i in range(len(ra_list)):
                masking_dic_1[str(i)] = {}
                masking_dic_2[str(i)] = {}

                masking_dic_1[str(i)]['RA'] = ra_list[i]*core.G3Units.deg
                masking_dic_2[str(i)]['RA'] = ra_list[i]*core.G3Units.deg

                masking_dic_1[str(i)]['Dec'] = dec_list[i]*core.G3Units.deg
                masking_dic_2[str(i)]['Dec'] = dec_list[i]*core.G3Units.deg

                masking_dic_1[str(i)]['Radius'] = masking_radius_1*core.G3Units.arcmin
                masking_dic_2[str(i)]['Radius'] = masking_radius_2*core.G3Units.arcmin
        else:
            print('Error, unknown format')

    mask = sources.make_point_source_map_from_dict(flat_sky_map,masking_dic_1)
    mask.invert()
    mask2 = sources.make_point_source_map_from_dict(flat_sky_map,masking_dic_2)
    mask2.invert()

    return mask, mask2

def load_mask(setting_dic, key,what_mask,ny,nx):
   
    if what_mask == 'sourcesub':
        try:
            source_sub_file = setting_dic[key]['source_sub_files']
            ss_map = pickle.load(open(source_sub_file, "rb"))
            ss_map = np.asarray(ss_map)
        except:
            ss_map = np.ones((ny,nx))
        
        mask = np.ones((ny,nx))
        mask2 = np.ones((ny,nx))
        
    
    if what_mask == 'no_mask':
        mask = np.ones((ny,nx))
        mask2 = np.ones((ny,nx))
        ss_map = np.zeros((ny,nx))

    if what_mask == 'simple_mask':
        ss_map = np.zeros((ny,nx))
        mask = fits.open(setting_dic[key]['mask1'])[1].data
        mask2 = fits.open(setting_dic[key]['mask2'])[1].data
            
    if what_mask == 'full':
        source_sub_file = setting_dic[key]['source_sub_files']
        if source_sub_file == 'None':
            ss_map = np.zeros((ny,nx))
        else:
            ss_map = pickle.load(open(source_sub_file, "rb"))
            ss_map = np.asarray(ss_map)

        mask = fits.open(setting_dic[key]['mask1'])[1].data
        mask2 = fits.open(setting_dic[key]['mask2'])[1].data

    return ss_map, np.asarray(mask), np.asarray(mask2)


def make_herschel_astro_noise(cf,key):
    herschel_apod = pickle.load(open("/home/kaylank/Temp/clusterfinding/cf_inputs/herschel_apod_mask.bin", "rb"))
    def make_frame(input_map):
        frame= core.G3Frame()
        frame['T'] = maps.FlatSkyMap(np.asarray(input_map,order='C'),
                          res=0.25*core.G3Units.arcmin,
                          weighted=False,
                          alpha_center=(352.5)*core.G3Units.deg,
                          delta_center=(-55)*core.G3Units.deg,
                          proj=maps.MapProjection.Proj0)
        return frame
    del_el = 100
    corrected_map = cf.setting_dic[key]['skymaps'][0]*herschel_apod*(1/np.mean(herschel_apod))
    corrected_map_fft = np.fft.fft2(corrected_map)
    corrected_beam = copy.deepcopy(cf.setting_dic[key]['beams'][0])
    corrected_beam[corrected_beam < 1e-2] = 1e50
    
    corrected_map_fft = (corrected_map_fft/corrected_beam)*np.sqrt(5e-16)
    
    herschel_noise = spt3g.simulations.quick_flatsky_routines.bin_radially_in_ell(corrected_map_fft)

    ps_function = interpolate.interp1d(herschel_noise['ell'], herschel_noise['cl'], bounds_error=False, fill_value="extrapolate")
    astro_cl = ps_function(np.arange(43_200))
#     if 'PSW' in key:
#         astro_cl*=0.5

    return astro_cl



def check_units(self,key,units='mJy'):
        herschel_ell_eff, C_ell_600_600,C_ell_857_857,C_ell_1200_1200,spt_ell, spt_95,spt_150,spt_220 = clu.check_units()
        herschel = [C_ell_1200_1200, C_ell_857_857, C_ell_600_600]
        spt = [spt_95,spt_150,spt_220]

        ps = []
        control_ps = []

        if '90' in key or '95' in key:
            cl = spt_95
            ell = spt_ell
        elif '150' in key:
            cl = spt_150
            ell = spt_ell
        elif '220' in key:
            cl = spt_220
            ell = spt_ell
        elif 'PLW' in key:
            cl = C_ell_600_600
            ell = herschel_ell_eff
        elif 'PMW' in key:
            cl = C_ell_857_857
            ell = herschel_ell_eff
        elif 'PSW' in key:
            cl = C_ell_1200_1200
            ell = herschel_ell_eff

        if 'sri' in key:
            mask_scale_fac = np.sqrt(1/np.mean(self.apod))
        elif 'P' in key:
            herschel_apod = pickle.load(open("/home/kaylank/clusterfinding/cf_inputs/herschel_apod_mask.bin", "rb"))
            joint_apod = self.apod*herschel_apod
            joint_apod = np.asarray(joint_apod)

            k = np.sum(joint_apod == 0)
            mask_scale_fac = np.sqrt(1/np.mean(joint_apod))
        else:
            mask_scale_fac = np.sqrt(1/np.mean(self.apod))

        if units == 'mJy':
            plt.rcParams['figure.figsize'] = (10,5)
            fig, (ax1,ax2) = plt.subplots(1,2)
            ax1.scatter(ell,cl)
            if 'P' in key:
                ni = self.astro_cl_dict[key]['DG-Cl']
                ax1.plot(np.arange(43_200),ni,color='red',label='DG-Cl')
                ax2.plot(np.arange(43_200),ni,color='red',label='DG-Cl')

                ni = np.fft.fft2(np.fft.fftshift(self.setting_dic[key]['skymaps'][0])*self.resrad*mask_scale_fac,norm='ortho')#/self.setting_dic[key]['beams'][0]
                ps = clu.bin_radially_in_ell(ni,delta_ell=500)

                ax1.plot(ps['ell'],ps['cl'],color='green',linestyle='--',label='Map')
                ax2.plot(ps['ell'],ps['cl'],color='green',linestyle='--',label='Map')
                ax2.set_ylim(1e2,1e7)
            else:
                ni = np.fft.fft2(np.fft.fftshift(self.setting_dic[key]['skymaps'][0])*self.resrad*mask_scale_fac,norm='ortho')#/self.setting_dic[key]['beams'][0]
                ps = clu.bin_radially_in_ell(ni,delta_ell=200)

                ax1.plot(ps['ell'],ps['cl'],color='green',linestyle='--')
                ax2.plot(ps['ell'],ps['cl'],color='green',linestyle='--')


            ax1.set_xlim(200,10_000)
            ax1.set_ylim(-10,np.max(cl))

            ax1.set_ylabel(r'$C\ell$ (Jy)$^2$ sr$^{-1}$',fontsize=15)
            ax1.set_xlabel(r'multipole $\ell$',fontsize=15)

            map_key = key
            for comp in self.astro_cl_dict[map_key]:
                plt.semilogy(np.arange(43_200), self.astro_cl_dict[map_key][comp],label=comp)

            ax2.set_xlim(0,4_000)
            ax2.legend()
            fig.suptitle(str(units),fontsize=20)
        else:
            plt.rcParams['figure.figsize'] = (10,10)
            ni = np.fft.fft2(np.fft.fftshift(self.setting_dic[key]['map_objs'][0])*self.resrad*mask_scale_fac*self.apod,norm='ortho')#/self.setting_dic[key]['beams'][0]
            ps = clu.bin_radially_in_ell(ni)

            plt.semilogy(np.arange(43_200),self.astro_cl_dict[key]['CMB'])
            plt.semilogy(ps['ell'],ps['cl'],color='green',linestyle='--')
            plt.xlim(400,2_000)

        return