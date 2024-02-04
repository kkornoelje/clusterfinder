import sys
import source_subtraction
import pointing
import clusterfinder

sys.path.append("/home/kaylank/imports/")
from imports import *
import clusterfunctions as clu


class wrap_inputs():
    def __init__(self,SHEET_ID,SHEET_NAME,verbose = True,units='uk',psize = 0.25):
        self.SHEET_ID = SHEET_ID
        self.SHEET_NAME = SHEET_NAME
        self.verbose = verbose
        self.units = units
        self.psize = psize
        
        url = f'https://docs.google.com/spreadsheets/d/{self.SHEET_ID}/gviz/tq?tqx=out:csv&sheet={self.SHEET_NAME}'
        setting_file = pd.read_csv(url)
        self.setting_dic = {}
        new_cols = ['map_array','map_objs','skymaps','beams','NoiseInstrument']
        
        for i in range((setting_file.shape[0])):
            field = setting_file['Field Name'][i]
            self.setting_dic[field] = {}
            for key in setting_file.keys()[1:len(setting_file.keys())]:
                self.setting_dic[field][key] = setting_file[key][i]
            for col in new_cols:
                self.setting_dic[field][col] = [] 
                
        self.control_key = list(self.setting_dic.keys())[0]
        
        return 
    
    def reproj_map(self,key,array,ra_offset,dec_offset,reproj=True):
        if np.sqrt(ra_offset**2 + dec_offset**2) > 15:
            if reproj:
                if self.verbose:
                    print('Reprojecting Map Center for:',key )
                    print('RA offset ("):', ra_offset)
                    print('DEC offset ("):', dec_offset)

                map_obj = maps.FlatSkyMap(np.asarray(array, order='C'),
                                          res=self.psize*core.G3Units.arcmin,
                                          weighted=False,
                                          alpha_center=(self.field_ra_center+(ra_offset/3600))*core.G3Units.deg,
                                          delta_center=(self.field_dec_center+(dec_offset/3600))*core.G3Units.deg,
                                          proj=maps.MapProjection.Proj0)

                map_obj_reproj = maps.FlatSkyMap(np.zeros_like(array),
                                          res=self.psize*core.G3Units.arcmin,
                                          weighted=False,
                                          alpha_center=(self.field_ra_center)*core.G3Units.deg,
                                          delta_center=(self.field_dec_center)*core.G3Units.deg,
                                          proj=maps.MapProjection.Proj0)
                
                spt3g.maps.reproj_map(map_obj,map_obj_reproj,rebin=4)
                return map_obj_reproj
            
        else:
            map_obj = maps.FlatSkyMap(np.asarray(array, order='C'),
                                      res=self.psize*core.G3Units.arcmin,
                                      weighted=False,
                                      alpha_center=(self.field_ra_center)*core.G3Units.deg,
                                      delta_center=(self.field_dec_center)*core.G3Units.deg,
                                      proj=maps.MapProjection.Proj0)
            return map_obj
        
    def get_map_array(self,apod_file = None, apod_pixel_file = None,
                      preform_pointing_check=False,apod_size = 10,radius_arcmin=20,units='uk'):

        if self.verbose:
            print('Initializing map making.')

        for i,key in enumerate(self.setting_dic):
            m = self.setting_dic[key]['map_files']
            self.field_ra_center = self.setting_dic[key]['ra_cent']
            self.field_dec_center = self.setting_dic[key]['dec_cent']
            ra_offset = self.setting_dic[key]['ra_offsets']
            dec_offset = self.setting_dic[key]['dec_offsets']
            
#             if '.g3' in m:
#                 print('Input file type is in .g3. Map format not fully tested and may need debugging.')
#                 g3_file = list(core.G3File(m))[0]
#                 spt3g.maps.remove_weights_t(g3_file['T'], g3_file['Wunpol'],zero_nans=True)
#                 alpha_center = g3_file['T'].alpha_center/core.G3Units.deg
                
#                 if round(alpha_center,2) != self.field_ra_center:
#                     print('Reprojecting to different portion of the map')
                    
#                     self.empty_map = maps.FlatSkyMap(np.zeros((3360,3360)),
#                                           res=0.25*core.G3Units.arcmin,
#                                           weighted=False,
#                                           alpha_center=(self.field_ra_center)*core.G3Units.deg,
#                                           delta_center=(self.field_dec_center)*core.G3Units.deg,
#                                           proj=maps.MapProjection.Proj0)
                    
#                     spt3g.maps.reproj_map(g3_file['T'], self.empty_map,rebin=4)
                    
#                 map_array = np.asarray(g3_file['T'])/core.G3Units.uK
                
            
            if '.fits' in m:
                hdul = fits.open(m)
                map_array = hdul[0].data * 1
                if len(map_array.shape) != 2:
                    map_array = map_array[0]
                    
                    
            elif '.sav' in m:
                map_file = readsav(m)
                keys = list(map_file.keys())
                map_array = map_file[keys[0]]*1
                
            elif '.bin' or '.pkl' in m:
                map_array = pickle.load(open(m, "rb"))*1
                
            elif type(m) != str:
                map_array = m*1
                
            else:
                print('Error: Unknown file type. Please enter a .sav, .fits, ir .g3 file')
            
            map_array*=self.setting_dic[key]['tcal']
            
            if units == 'K':
                map_array*=1e6
            
            if self.units == 'mJy':
                map_array/=self.setting_dic[key]['janksy_conversion']

                
            
            self.ny = map_array.shape[0]
            self.nx = map_array.shape[1]
            
            self.setting_dic[key]['map_array'].append(np.asarray(map_array))
            
            reprojected_map = self.reproj_map(key,map_array,ra_offset,dec_offset)
            
            self.setting_dic[key]['map_objs'].append(reprojected_map)
            self.setting_dic[key]['skymaps'].append(np.asarray(reprojected_map))
            
        if (apod_file == None) or (apod_pixel_file == None):
            print('Generating Apodiation Mask')
            apod_pixel_mask = copy.copy(map_array)
            self.apod_pixel_mask = apod_pixel_mask
            
            apod_pixel_mask[:,0:apod_size] = 0
            apod_pixel_mask[:,(self.nx-apod_size):self.nx] = 0
            apod_pixel_mask[0:apod_size,:] = 0
            apod_pixel_mask[(self.ny-apod_size):self.ny,:] = 0
            
            apod_pixel_mask[apod_pixel_mask != 0] = 1
            
            apod_obj = maps.FlatSkyMap(apod_pixel_mask,
                      res=self.psize*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(self.field_ra_center)*core.G3Units.deg,
                      delta_center=(self.field_dec_center)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)

            apod_mask = spt3g.mapspectra.apodmask.make_border_apodization(apod_obj,radius_arcmin=radius_arcmin)
            
            self.apod = apod_mask
            temp = copy.deepcopy(self.apod)
            temp = np.where(temp != 1, 0, temp)
            self.apod2 = temp
            
        else:
            self.apod_file = apod_file
            self.apod_pixel_file = apod_pixel_file
            
            if '.bin' in self.apod_file or '.pkl' in self.apod_file:
                self.apod = pd.read_pickle(apod_file)
                self.apod2 =pd.read_pickle(apod_pixel_file)
                
            elif '.fits' in self.apod_file:
                self.apod = fits.open(self.apod_file)[0].data
                self.apod2 = fits.open(self.apod_pixel_file)[0].data
            else:
                print('Unknown apodization mask file type. Please input either a .fits or .bin/.pkl file.')
                
            
            if self.apod.shape != map_array.shape:
                diff_ny = int((self.ny - self.apod.shape[0])/2)
                diff_nx = int((self.nx - self.apod.shape[1])/2)
                if self.apod.shape < map_array.shape:
                    print('Apod Shape does not equal map array shape. Padding apod with zeros')
                    
                    self.apod = np.pad(self.apod, ((diff_ny,diff_ny), (diff_nx,diff_nx)))
                    self.apod2 = np.pad(self.apod2, ((diff_ny,diff_ny), (diff_nx,diff_nx)))
                    
                elif self.apod.shape > map_array.shape:
                    print('Apod Shape does not equal map array shape. Cropping apod to map shape')
                    self.apod = self.apod[diff_ny:(self.ny-diff_ny), diff_nx:(self.ny-diff_nx)]
                    self.apod2 = self.apod2[diff_ny:(self.ny-diff_ny), diff_nx:(self.ny-diff_nx)]
                    
        for key in self.setting_dic:
            self.setting_dic[key]['skymaps'][0]*=self.apod
            
        if preform_pointing_check:
            print('Error: Pointing check not yet implemented')
            point = pointing.pointing(self.setting_dic,key)
            point.crossmatch()
            point.plot()  
        return
    
    def extrapolate_beam(self,beam_ell,beam_cl):
        ell = np.arange(43_200)
        beam_function = interpolate.interp1d(beam_ell, beam_cl, bounds_error=False, fill_value="extrapolate")
        interpolated_cl = np.where(beam_function(ell) <= 0, 1e-50, beam_function(ell))
        beam = clu.clstokmap(interpolated_cl, ell,self.psize,ny=self.ny,nx=self.nx)
        beam = np.where(beam<=0,1e-50,beam)
        return beam
    
    def wrap_beams(self,add_transfer = False, theory=False,lpf = 20_000,hpf = 400,isohpf = 500,fwhm=1.2):
        if theory:
            print('Generating theoretical beams')
            xfer=np.fft.fftshift(clu.make_filter_xfer(lpf=lpf,hpf=hpf,isohpf=isohpf,nx=3360,ny=3360,reso_arcmin=0.25))
            beams_ft=beam_mod.Bl_gauss(clu.ell_grid(self.psize,cf.ny,cf.nx),fwhm*core.G3Units.arcmin,1)
            beam=beams_ft*xfer
            return beam

        for i,key in enumerate(self.setting_dic):
            b = str(self.setting_dic[key]['beam_files'])
            if '.fits' in b:
                print('here')
                hdul = fits.open(b)
                beam = hdul[0].data * 1
                if len(beam.shape) >2:
                    print('please split up beam files')
                    beam = beam[0]
                
                
            elif '.sav' in b:
                map_file = readsav(b)
                keys = list(map_file.keys())
                beam = map_file[keys[0]]*1
                
            elif '.tsv' in b or '.txt' in b:
                beams = pd.read_csv(b,delim_whitespace=True,comment='#',names=['ell','cl'])
                beam_ell = beams['ell']
                beam_cl = beams['cl']

                beam = self.extrapolate_beam(beam_ell,beam_cl)
            else:
                print('Error, unknown beam file type')
                
                
            if beam.shape != (self.ny,self.nx):
                print('Beam shape does not equal Megaclusters size. Are you using an old map? Reshaping')
                beam = clusterfinder.clu.resize_input(beam,ell_space=True)

            if '3g' in key:
                print('Adding transfer function')
                xfer=np.fft.fftshift(clu.make_filter_xfer(lpf=lpf,hpf=hpf,isohpf=isohpf,nx=self.nx,ny=self.ny,reso_arcmin=self.psize))
                beam*=xfer
                    
            self.setting_dic[key]['beams'].append(beam)
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
    
    def wrap_instrumental_noise(self,units='uk'):
        for i, key in enumerate(self.setting_dic):
            n = self.setting_dic[key]['instrumental_noise_files']
            
            if '.fits' in n:
                ni = fits.open(n)[0].data
                
            elif '.sav' in n:
                ni_file = readsav(n)
                ni_keys = list(ni_file.keys())
                ni = ni_file[ni_keys[0]]*1
            else:
                if 'W' in key:
                    print('Making Herschel Instrumental Noise')
                    ni = np.random.uniform(-1, 1, size=(self.ny, self.nx))*1e-50
                else:
                    print('Error: unknown file type')
                
                
            if ni.shape != (self.ny,self.nx):
                print('Instrumental noise shape does not equal Megaclusters size. Are you using an old map? Reshaping')
                ni = clusterfinder.clu.congrid(ni,(self.ny,self.nx))
                    
            ni*=(self.setting_dic[key]['tcal'])
            if units=='K':
                print('Instrumental Noise in K')
                ni*=1e6
                
            ni= (ni**2)
            
                    
            if 'W' in key:
                ni = np.random.uniform(1, 2, (3360,3360))*1e-40
            if '3g' in key and 'old' not in key:
                print('scaling')
                scale_factors = [2.092421602791408,2.1530765463680774,2.3736586174413628]
                if '90' in key:
                    ni/=scale_factors[0]
                elif '150' in key:
                    ni/=scale_factors[1]
                else:
                    ni/=scale_factors[2]
                        
            
                        
            

            if self.units == 'mJy':
                ni/=(self.setting_dic[key]['janksy_conversion']**2)

            self.setting_dic[key]['NoiseInstrument'].append(ni)
        return 
    
    def finish_wrapping(self):
        self.final_setting_dic = {}
        self.final_setting_dic['map_inputs_dic']= {}
        self.final_setting_dic['inputs_dic']= {}
        
        self.final_setting_dic['map_inputs_dic']['field_ra_cent'] = self.field_ra_center
        self.final_setting_dic['map_inputs_dic']['field_dec_cent'] = self.field_dec_center
        self.final_setting_dic['map_inputs_dic']['resrad'] = self.psize*0.000290888
        self.final_setting_dic['map_inputs_dic']['psize'] = self.psize
        self.final_setting_dic['map_inputs_dic']['apod'] = self.apod
        self.final_setting_dic['map_inputs_dic']['apod2'] = self.apod2
        self.final_setting_dic['map_inputs_dic']['ny'] = self.ny
        self.final_setting_dic['map_inputs_dic']['nx'] = self.nx
        
        for key in self.setting_dic:
            self.final_setting_dic['inputs_dic'][key] = self.setting_dic[key]
        return
    
    
