import sys
import source_subtraction
import pointing

sys.path.append("/home/kaylank/imports/")
from imports import *
import importlib

import clusterfunctions as clu

#========================== CLUSTER FINDER DOCUMENTATION ==========================#

class cluster_finder():
    def __init__(self,fields,dic,units='uk',verbose=True):
        self.beta_profile_file = '/big_scratch/mck74/polmaps/sptpol_100d/'+'profiles_12_beta1_rc025_rc300.sav'
        self.pol_directory='/big_scratch/mck74/polmaps/sptpol_100d/'
        self.tc_arcmin=np.array([ 0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ,  2.25, 2.5 ,  2.75,  3.  ])
        self.idl_profiles = readsav(self.beta_profile_file,verbose=False)
        self.ell = np.arange(43_200)
        
        self.units = units
        self.fields = fields
        self.verbose = verbose
        self.nbands = len(self.fields)
        self.initial_field = self.fields[0]
        self.fields = fields
        
        self.map_setting_dic = dic['map_inputs_dic']
        
        for key, value in self.map_setting_dic.items():
            setattr(self, key, value)
            
            
        self.tcs=np.array([tci/self.psize for tci in self.tc_arcmin])
            
        self.setting_dic = {}
        for key in dic['inputs_dic']:
            if key in self.fields:
                self.setting_dic[key] = copy.deepcopy(dic['inputs_dic'][key])
                
        self.map_obj = maps.FlatSkyMap(np.zeros((self.ny,self.nx)),
                          res=self.psize*core.G3Units.arcmin,
                          weighted=False,
                          alpha_center=(self.field_ra_cent)*core.G3Units.deg,
                          delta_center=(self.field_dec_cent)*core.G3Units.deg,
                          proj=maps.MapProjection.Proj0)
        
        new_cols = ['fsz','ss_maps']
        
        for field in self.setting_dic:
            for col in new_cols:
                self.setting_dic[field][col] = []
                
        if self.verbose:
            print('Running the cluster finder for the fields:', self.fields)
            
        ss = source_subtraction.source_subtraction(self,what_field=self.initial_field,
                        sn_threshold = 5,
                        temp_threshold = 7,
                        mel_flux=False,                
                        what_pts_file='/big_scratch/joshuasobrin/clusters/kayla_files/spt3g_1500d_source_list_ma_19sept23_v4.txt',
                        verbose=True)

        ss.make_full_source_catalog()
        ss.make_subfield_catalog()
        self.ss = ss
            
        return
    
    def make_fsz(self,do_rel=False,rel_kev=None):
        self.do_rel = do_rel
        self.rel_kev = rel_kev
        
        if do_rel:
            if self.verbose:
                print('Running the relativistic cluster finder for temperatures (keV):', rel_kev)
        else:
            if self.verbose:
                print('Running the non-relativistic cluster finder')
        
        for f in self.setting_dic:
            szband = self.setting_dic[f]['sz_band_centers']
            curr_fsz = clu.calc_fsz(szband)
            
            if self.units == 'mJy':
                
                if 'W' not in f:
                    curr_fsz*=self.setting_dic[f]['janksy_conversion']
                    
                else:
                    control_settings = self.setting_dic[self.fields[0]]
                    szband_control = control_settings['sz_band_centers']
                    szband_curr = self.setting_dic[f]['sz_band_centers']

                    scaling_factor = clu.calc_fsz(szband_curr) / clu.calc_fsz(szband_control)
                    curr_fsz = clu.calc_fsz(szband_control) * control_settings['janksy_conversion'] * scaling_factor

            self.setting_dic[f]['fsz'].append(curr_fsz)
            
            if do_rel:   
                self.setting_dic[f]['rel_fsz'] = {}
                if type(rel_kev) == list:
                    for rel in rel_kev:
                        rel_name = 'rel_fsz_'+str(rel)+'kev'
                        curr_rel_fsz = clu.fxsz_itoh([szband],[rel])[0][0]
                        self.setting_dic[f]['rel_fsz'][rel_name] = curr_rel_fsz
                        
                else:
                    rel_name = 'rel_fsz_'+str(rel_kev)+'kev'
                    curr_rel_fsz = clu.fxsz_itoh([szband],[rel_kev])[0][0]
                    self.setting_dic[f]['rel_fsz'][rel_name] = curr_rel_fsz
                    
        return
    
    #NOT CURRENTLY FULLY OPTIMIZED
    def make_mask(self,key,what_mask):
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
            

        if what_mask == 'megaclusters' or what_mask == 'mega_source_sub':
            intepolated_sources =['48','58','61','69','77','79','80','81','85','88','101','103']
            
            high_flux_sources = ['493','447','1071','223','236','249','265','271','308','327','330',
                         '358','377','423','217','248','259','511','405','231','289','118','404','210']
            
            extended_sources = ['510','514','628','978','2708','917','3415']
            
            full_masking_list = intepolated_sources+high_flux_sources+extended_sources
            
            mask, mask2 = make_masking_dic(full_masking_list, self.ss.in_subfield_cat)
            
            if what_mask == 'mega_source_sub':
                ss_map = np.zeros((self.ny,self.nx))
            else:
                source_sub_file = self.setting_dic[key]['source_sub_files']
                ss_map = pickle.load(open(source_sub_file, "rb"))

                if self.units=='mJy':
                    if 'W' not in key:
                        ss_map*=self.setting_dic[key]['janksy_conversion']

            return mask, mask2, ss_map

        if what_mask == 'milo_simple_mask':
            mask = self.map_obj.clone(False) 
            mask = sources.source_utils.make_point_source_map(mask, self.pol_directory+'sources_edited_5arcmin.txt') 
            mask.invert()
            mask = np.asarray(mask)

            mask2 = self.map_obj.clone(False) 
            mask2 = sources.source_utils.make_point_source_map(mask2, self.pol_directory+'sources_edited_8arcmin.txt')
            mask2.invert()
            mask2 = mask2
            mask2 = np.asarray(mask2)

            ss_map = np.zeros((self.ny,self.nx))

            return mask, mask2, ss_map
        
        elif what_mask == 'no_mask':
            mask = np.ones((self.ny,self.nx))
            mask2 = np.ones((self.ny,self.nx))
            ss_map = np.zeros((self.ny,self.nx))

            return mask, mask2, ss_map

        elif ('.txt' in what_mask) or ('.csv' in what_mask):
            read_source_file = pd.read_csv(what_mask,comment='#',names=['RA','DEC','Radius'])
            source_dic = {}
            source_dic_2 = {}

            for i in range(len(read_source_file)):
                source_dic[str(i)] = {}
                source_dic_2[str(i)] = {}

                source_dic[str(i)]['RA'] = read_source_file['RA'][i]*core.G3Units.deg
                source_dic_2[str(i)]['RA'] = read_source_file['RA'][i]*core.G3Units.deg

                source_dic[str(i)]['Dec'] = read_source_file['DEC'][i]*core.G3Units.deg
                source_dic_2[str(i)]['Dec'] = read_source_file['DEC'][i]*core.G3Units.deg

                source_dic[str(i)]['Radius'] = read_source_file['Radius'][i]*core.G3Units.deg
                source_dic[str(i)]['Radius'] = 8*core.G3Units.arcmin


            mask = sources.make_point_source_map_from_dict(self.map_obj,source_dic)
            mask.invert()
            mask2 = sources.make_point_source_map_from_dict(self.map_obj,make_ps_dict(cf,8))
            mask2.invert()

            ss_map = np.zeros((self.ny,self.nx))

            return mask,mask2, ss_map  
        else:
            print('Error, unknown input')
            return
        
    def make_maps(self,what_mask):
        for i,key in enumerate(self.setting_dic):
            self.mask, self.mask2, ss_map = self.make_mask(key,what_mask)
            
            self.setting_dic[key]['skymaps']*=self.mask
            self.setting_dic[key]['skymaps']+=ss_map
            self.setting_dic[key]['ss_maps'].append(ss_map)
            
        return
    
    #=========  BEAM FUNCTIONS (NOT YET OPTIMIZED) =======#
    def make_beams(self,theory=False):
        for i,f in enumerate(self.setting_dic):
            self.setting_dic[f]['beams'].append(self.setting_dic[f]['beams'][0])
        return  

    def make_herschel_astro_noise(self,key):
        herschel_apod = pickle.load(open("/home/kaylank/Temp/clusterfinding/cf_inputs/herschel_apod_mask.bin", "rb"))
        joint_apod = self.apod*herschel_apod
        joint_apod = np.asarray(joint_apod)

        mask_scale_fac = np.sqrt(1/np.mean(joint_apod))
        ni = np.fft.fft2(np.fft.fftshift(self.setting_dic[key]['skymaps'][0])*self.resrad*mask_scale_fac,norm='ortho')/self.setting_dic[key]['beams'][0]

        ps = clu.bin_radially_in_ell(ni,delta_ell=200)
        ps_function = interpolate.interp1d(ps['ell'], ps['cl'], bounds_error=False, fill_value="extrapolate")
        astro_cl = ps_function(self.ell)

        return astro_cl

    def make_astro_noise(self):
        components = ['tSZ','kSZ','CMB','DG-Po','DG-Cl']
        
        self.astro_cl_dict = {}

        for i, field in enumerate(self.fields):
            self.astro_cl_dict[field] = {}
            band = str(self.setting_dic[field]['freq'])+'GHz'
            if 'W' in field:
                band = '220GHz'
            
            for index, comp in enumerate(components):
                curr_fg =foregrounds.get_foreground_power(comp, rescale='True', band1=band, model='reichardt')/core.G3Units.uK**2

                if 'W' in field: 
                    curr_fg/=1e100
                    
                if self.units == 'mJy':
                    curr_fg*=(self.setting_dic[field]['janksy_conversion']**2)
                    
                curr_fg = np.pad(curr_fg, (0,29699), 'constant', constant_values=curr_fg[-1])
                self.astro_cl_dict[field][comp] = curr_fg   
                
            if 'W' in field:
                print('Making Herschel Instrumental Noise')
                if comp == 'DG-Cl':
                    self.astro_cl_dict[field][comp] = self.make_herschel_astro_noise(field)
        return
        
    ##======INSTRUMENTAL NOISE FUNCTIONS =====#

    def make_instr_noise(self):
        for i,key in enumerate(self.setting_dic):
            self.setting_dic[key]['NoiseInstrument'].append(self.setting_dic[key]['NoiseInstrument'])
        return
            
    def make_covar(self,my_inver='None',my_ncov1d='None',nmatastro='None',nmatinst='None'):  
        self.ncov1d = clu.make_astro_covar_matrix(self.astro_cl_dict)
        self.ell_max = 43_200
        self.nmatastro=clu.create_ncovbd(self.ncov1d,
                                        [self.setting_dic[f]['beams'][0] for f in self.fields],
                                        self.psize,self.nbands,self.ny,self.nx,self.ell_max)

        self.NoiseInstrument = [self.setting_dic[f]['NoiseInstrument'][0] for f in self.fields]
        self.nmatinst=clu.create_N_d(self.NoiseInstrument)

        self.nmatfull=self.nmatinst + self.nmatastro
        self.inver = np.linalg.pinv(self.nmatfull)
            
        return
        
    def cilc(self,ft_signal,fsz,do_cilc=False,e_temp_map = False):
        if do_cilc:  
            if self.verbose:
                print('Preforming Constrained ILC')

                if e_temp_map == False:
                    bvec = []
                    control_bvec = mod_bb(self.setting_dic[fields[-1]]['sz_band_center'])

                    for f in fields:
                        if self.units == 'mJy':
                            curr_bvec = ((self.mod_bb(setting_dic[f]['sz_band_center']))/control_bvec)*setting_dic[f]['k_to_j']
                        else:
                            curr_bvec = ((self.mod_bb(setting_dic[f]['sz_band_center']))/control_bvec)
                        bvec.append(curr_bvec)

                    G = np.reshape(np.column_stack((fsz, bvec)),(1,1,self.nbands,2)) 
                    s = G*np.repeat(np.reshape(ft_signal,(self.ny,self.nx,self.nbands,1)),2,axis=3)
                    print(bvec)

                else:
                    if self.verbose:
                        print('Making Electron Temperature Map')
                        fsz = []
                        n_bvec = []
                        pivot_temp = self.rel_kev[0]
                        
                        for key in self.setting_dic:
                            x1 = pivot_temp-0.01
                            x2 = pivot_temp+0.01
                            
                            y1 = clu.fxsz_itoh([self.setting_dic[key]['sz_band_centers']], [x1])[0][0]
                            y2 = clu.fxsz_itoh([self.setting_dic[key]['sz_band_centers']], [x2])[0][0]
                            slope = (y2-y1)/(x2-x1)

                            fsz.append(slope)
                            n_bvec.append(clu.fxsz_itoh([self.setting_dic[key]['sz_band_centers']], [pivot_temp])[0][0])

                        print('pivot temp:', pivot_temp)

                        G = np.reshape(np.column_stack((fsz, n_bvec)),(1,1,self.nbands,2)) 
                        s = G*np.repeat(np.reshape(ft_signal,(self.ny,self.nx,self.nbands,1)),2,axis=3)

        else:
            print('Preforming Minimum Varaince ILC')
            s = ft_signal*fsz
            s = np.reshape(s, (self.ny,self.nx,self.nbands,1))

        inner_product = np.einsum('ijkl,ijkm,ijml->ij', s, self.inver, s)
        sigma = np.real((np.sum(np.sum(inner_product))/(self.ny*self.nx))**(-.5))
        transpose_s = np.reshape(s,(self.ny,self.nx,s.shape[-1],self.nbands))

        if do_cilc:
            nvec = np.reshape(np.asarray([1,0]),(2,1))
            print(nvec.shape)
            transpose_s = np.sum((transpose_s*nvec),axis=2)
            transpose_s = np.reshape(transpose_s,(self.ny,self.nx,1,self.nbands))

        weight = np.sum(np.einsum('ijkl,ijkl->ijkl', transpose_s, self.inver),axis=-1)
        psi = sigma**2 * weight

        return psi, sigma

    def find_clusters(self,sn_cutoff,tcs=1,do_cilc=False,do_rel=False,e_temp_map=False,point_source_finder=False,my_psi='None'):
        self.skymaps = [self.setting_dic[f]['skymaps'][0] for f in self.fields]
        
        self.psis={};self.fmaps={};self.snmaps={};self.predicteds=np.array([]);self.measureds=np.array([])
        self.maxes=np.array([]);self.xes=np.array([]);self.yes=np.array([]);self.tlocs=np.array([])
        
        
        self.results = {}
        for ti in range(0,tcs):
            name = self.tc_arcmin[ti]
            self.results[name] = {}
            self.results[name]['psi'] = []
            self.results[name]['predicted_sigma'] = []
            self.results[name]['measured_sigma'] = []
            self.results[name]['snmap'] = []
            
        fsz = [self.setting_dic[f]['fsz'][0] for f in self.setting_dic]
            
        
        for ti in range(0,tcs):
            name = self.tc_arcmin[ti]
            print(name)
            norm_factors = [0.0371802757428985]
            
            if point_source_finder:
                print('Finding Point Sources')
                fsz = [1]*self.nbands
                self.ft_signal = np.stack([self.setting_dic[f]['beams'][0] for f in self.fields],axis=2) 
                self.psi,self.sigma = self.cilc(self.ft_signal,fsz,do_cilc=False,e_temp_map = False)
            else:     
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
                idl_ls=np.arange(80001)
                
                print('Finding Clusters')
                profile_ft=gridtomap(clu.ell_grid(self.psize,self.ny,self.nx),self.idl_profiles['profs'][ti],idl_ls)
                
                real_profile=np.fft.fftshift(np.fft.ifft2(profile_ft,norm='ortho'))/self.psize
                norm_factor=1/np.max(np.real(real_profile))
                
#                 profile_ft*=norm_factor
                profile_ft/=7.66780e-08
                
                ft_signal=create_multi_band_s([profile_ft * self.setting_dic[f]['beams'][0] for f in self.fields])
                self.ft_signal = ft_signal
                
            self.ft_signal*=norm_factors[ti]
            
            if type(my_psi) != str:
                self.psi = my_psi
                
            else:
                self.psi,self.sigma = self.cilc(self.ft_signal,fsz,do_cilc=do_cilc,e_temp_map = e_temp_map)
                self.psi[0][0]=0.0
                self.results[name]['psi'].append(self.psi)
                self.results[name]['predicted_sigma'].append(self.sigma)

            

            #create filtered maps -- 
            self.fmap=clu.multi_band_filter(self.skymaps,self.psi,self.psize,self.nbands)
            self.results[name]['measured_sigma'].append(np.std(self.fmap))

            #make snmap
            self.snmap = clu.make_snmap([self.fmap],self.apod2,self.mask2,self.psize,cutoffs=None)[0]
            self.results[name]['snmap'].append(self.snmap)
            
            groups=sources.source_utils.find_groups(self.snmap,signoise=1,nsigma=sn_cutoff)

            self.maxes=np.append(self.maxes,groups['maxvals'])
            self.xes=np.append(self.xes,groups['xcen'])
            self.yes=np.append(self.yes,groups['ycen'])
            self.tlocs=np.append(self.tlocs,self.tcs[ti]*np.ones(groups['n_detected'])*self.psize)
        
        self.cat = clu.make_clu_cat(self.maxes,self.xes,self.yes,self.tlocs,self.map_obj)
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
    
    def export_results(self,name):
        cat_df = pd.DataFrame(self.cat)
        cat_df.to_csv(r'/home/kaylank/clusterfinding/results/'+name+'.csv', index=False, header=True)
        return
    
    def make_simulated_maps(self,num_noise_realizations=10,shape=(3360,3360),save_individual_maps=False,add_sources=False):
        self.add_sources = add_sources
        mapparams = [shape[0],shape[1],self.psize,self.psize]
        el = self.ell

        self.simulated_maps = {}

        for i in range(num_noise_realizations):
            if self.verbose:
                print('Preforming Noise Realization: ' + str(i))
                
            self.simulated_maps['rel'+str(i)] = {}
            for key in self.astro_cl_dict:
                if self.verbose:
                    print('Making simulated ' + key + ' maps.')
                          
                self.simulated_maps['rel'+str(i)][key] = {}
                self.simulated_maps['rel'+str(i)][key]['total'] = np.zeros((shape[0],shape[1]))

                for comp in self.astro_cl_dict[key]:
                    if self.verbose:
                        print('Making simulated ' + comp + ' maps.')
                    
                    curr_map = clu.clstokmap(self.astro_cl_dict[key][comp],el,0.25,self.ny,self.nx)
                    if save_individual_maps:
                        self.simulated_maps['rel'+str(i)][key][comp] = curr_map
                        
                    self.simulated_maps['rel'+str(i)][key]['total']+=curr_map
            
                instrumental_noise_copy = copy.deepcopy(self.setting_dic[key]['NoiseInstrument'][0])
                instrumental_noise_copy[:,0:21] = 0
                instrumental_noise_copy[:,(3360-21):3360] = 0
                
                ni = clu.bin_radially_in_ell(instrumental_noise_copy)
                
                
                curr_map = clu.clstokmap(np.sqrt(ni['cl']), ni['ell'],0.25,self.ny,self.nx)
                if save_individual_maps:
                    self.simulated_maps['rel'+str(i)][key]['instrumental_noise'] = curr_map
                self.simulated_maps['rel'+str(i)][key]['total']+=curr_map
                

        if self.add_sources:
            for key in self.astro_cl_dict:
                #simulated_maps[key]['sources'] = np.zeros((self.nx,self.nx))
                source_map = np.zeros((self.nx,self.nx))
                if '90' in key:
                    freq = '90GHz'
                elif '150' in key:
                    freq = '150GHz'
                elif '220' in key:
                    freq = '220GHz'

                fluxes, counts = spt3g.simulations.foregrounds.get_poisson_source_counts('dust',band = freq)
                total_counts = np.sum(counts)
                weights = np.divide(counts,total_counts)

                for i in range(int(total_counts)):
                    pix_x = int(self.nx*np.random.rand()) 
                    pix_y = int(self.nx*np.random.rand()) 

                    source_map[pix_y][pix_x]+=choices(fluxes, weights)

                from scipy import signal

                fft_source_map = np.fft.fft2(np.fft.fftshift(source_map))
                fft_delta_func = np.fft.fft2(np.fft.fftshift(signal.unit_impulse((self.nx, self.nx), 'mid')))
                source_map = np.fft.fftshift(np.real(np.fft.ifft2(fft_source_map*fft_delta_func)))
                self.simulated_maps[key]['sources'] = source_map
                self.simulated_maps[key]['total']+=source_map

        return 
    
    #def preform_pointing_check(self,what_field):
        
    def plot_inputs(self,key):
        plt.rcParams['figure.figsize'] = (10,10)
        total = np.zeros(len(self.ell))
        for comp in self.astro_cl_dict[key]:
            plt.semilogy(self.ell, self.astro_cl_dict[key][comp],label=comp)
            total+=self.astro_cl_dict[key][comp]
            plt.xlabel(r'$\mu K^{2}$')
        plt.semilogy(self.ell, total,label='total')
                
        ni_test = copy.deepcopy(self.setting_dic[key]['NoiseInstrument'][0])
        if '3g' in key:
            print('fixing')
            ni_test[:,0:30] = 0
            ni_test[:,(self.nx-30):self.nx] = 0
        test = clu.bin_radially_in_ell(ni_test,nosquare=True)
        plt.semilogy(test['ell'],test['cl'],label='Instrumental Noise')
        map_corr = 1/np.mean(self.apod[500:2_000,500:2_000])
            
        test2 = spt3g.simulations.quick_flatsky_routines.cl_flatsky(np.asarray(self.setting_dic[key]['map_array'][0][500:2_000,500:2_000]*map_corr))
        plt.semilogy(test2['ell'],test2['cl']['TT'],label='input_map')
            
        plt.legend()
                
        return
    
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