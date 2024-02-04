import sys
import source_subtraction
import pointing
import clusterfinder

sys.path.append("/home/kaylank/imports/")
from imports import *
import clusterfunctions as clu

import importlib
importlib.reload(pointing)
import pointing

class wrap_inputs():
    
    '''
    The inputs wrapper for the clusterfinder pipeline
    See: Cluster Finder tutorial for notes on how to use
    '''
    
    #Initialization function creates a nested dictionary with all of the inputs that the cluster finder uses. 
    def __init__(self,SHEET_ID='1st10v2-yRFmoUIurbhtOpm2JyZxobCSApQ6e1DTC7fc',
                 SHEET_NAME='cluster_finder_final',verbose = True,psize = 0.25,units='uk',skip_maps=[]):
        
        
        self.verbose = verbose
        self.psize = psize
        self.ell = np.arange(43_200)
        
        #Loading in the sheet with all of the cluster finder inputs
        if SHEET_ID != None:
            self.SHEET_ID = SHEET_ID
            self.SHEET_NAME = SHEET_NAME
            url = f'https://docs.google.com/spreadsheets/d/{self.SHEET_ID}/gviz/tq?tqx=out:csv&sheet={self.SHEET_NAME}'
            setting_file = pd.read_csv(url)
            self.setting_file = setting_file
            
        else:
            #Improvement
            print('Should implement a non-google doc option here.')
            
        #Creating the settings dictionary
        self.setting_dic = {}
        new_cols = ['map_array','map_objs','skymaps','beams','NoiseInstrument']
        
        #Load in everything from the google doc into a dictionary.
        #self.setting_dic is the hub for basically all your go to information. 
        for i in range((setting_file.shape[0])):
            field = setting_file['Field Name'][i]
            if field not in skip_maps:
                self.setting_dic[field] = {}
                for key in setting_file.keys()[1:len(setting_file.keys())]:
                    self.setting_dic[field][key] = setting_file[key][i]
                for col in new_cols:
                    self.setting_dic[field][col] = [] 
                
        self.control_key = list(self.setting_dic.keys())[0]      
        return 
        
    #Get map array retrieves the input maps and prepares them for clusterfinding.
    #Maps should be in uK. 
    #Will accept files in the formats: .fits, .pkl, .sav and .g3 once someone checks that if statement.
    def get_map_array(self,apod_file =  '/sptlocal/user/kaylank/megaclusters/additional_data_products/milo_apod_mask.bin', apod_pixel_file = '/sptlocal/user/kaylank/megaclusters/additional_data_products/milo_pixel_mask.bin',
                      preform_pointing_check=False,apod_size = 10,radius_arcmin=20,units='uk'):

        if self.verbose:
            print('Initializing map making.')

        for i,key in enumerate(self.setting_dic):
            m = self.setting_dic[key]['map_files']
            
            self.field_ra_center = self.setting_dic[key]['ra_cent']
            self.field_dec_center = self.setting_dic[key]['dec_cent']
            
            if self.verbose:
                print('Starting Map: ', m)

            if '.g3' in m:
                #Improvement: haven't checked to see if this actually works. Someone should check that loading in
                #a .g3 map file will produce an unweighted map in uK. 
                
                coadded_mapframe = list(core.G3File(m))[0]
                if coadded_mapframe.type != core.G3FrameType.Map:
                    sys.exit("Coadded-map file must contain the coadded map in the first frame")
                if coadded_mapframe['T'].weighted == True:
                    maps.RemoveWeights(coadded_mapframe, zero_nans=True)
                coadded_FlatSkyMap = coadded_mapframe.pop('T')
                coadded_FlatSkyMap/=core.G3Units.uK
                map_array = coadded_FlatSkyMap
            
            if '.fits' in m:
                hdul = fits.open(m)
                try:
                    map_array = hdul[0].data * 1
                except:
                    map_array = hdul[1].data * 1
                
            elif '.sav' in m:
                map_file = readsav(m)
                keys = list(map_file.keys())
                map_array = map_file[keys[0]]*1
                
            elif '.bin' or '.pkl' in m:
                map_array = pickle.load(open(m, "rb"))*1
                
            else:
                print('Error: Unknown file type. Please enter a .sav, .fits, or .g3 file')
                
            self.ny = map_array.shape[0]
            self.nx = map_array.shape[1]

            #Multiplying map by the provided tcal factor
            map_array*=self.setting_dic[key]['tcal']

            self.setting_dic[key]['map_array'].append(np.asarray(map_array))

            map_obj = maps.FlatSkyMap(np.asarray(map_array, order='C'),
                                  res=self.psize*core.G3Units.arcmin,
                                  weighted=False,
                                  alpha_center=(self.field_ra_center)*core.G3Units.deg,
                                  delta_center=(self.field_dec_center)*core.G3Units.deg,
                                  proj=maps.MapProjection.Proj0)
            
            self.setting_dic[key]['map_objs'].append(map_obj)
            self.setting_dic[key]['skymaps'].append(np.asarray(map_obj))
            
        #Code that will produce an apodization mask if one is not provided. If your map is shaped weird use caution. 
        if (apod_file == None) or (apod_pixel_file == None):
            self.apod, self.apod2 = generate_apodization_mask(self,map_array)
               
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
                
            #Reshapes the apodization mask if it's not the correct size.
            if self.apod.shape != map_array.shape:
                self.apod,self.apod2 = crop_apod_mask(wi, apod,apod2)
                
        #Add apodization mask to map to avoid FT artifacts.
        for key in self.setting_dic:
            self.setting_dic[key]['skymaps'][0]*=self.apod

        return
    
    def wrap_beams(self,add_transfer = False, theory=False,lpf = 20_000,hpf = 400,isohpf = 500,fwhm=None):
        #If using theory, need to provide a list of fwhm values as:
        #fwhm = [fwhm_map_1, fwhm_map_2,fwhm_map_3...] or it will default to 1.1' beams. 
        if theory:
            print('Generating theoretical beams')
            if fwhm == None:
                print('No Beam FWHM value provided. Defaulting to 1.1 arcmin beams')
                fwhm = [1.1]*len(list(self.setting_dic.keys()))
            
            for i,key in enumerate(self.setting_dic):
                if self.verbose:
                    print('Constructing beam for: ', key)
                    print('FWHM: ', fwhm[i])
                
                beam=beam_mod.Bl_gauss(clu.ell_grid(self.psize,self.ny,self.nx),fwhm[i]*core.G3Units.arcmin,1)
                print('multiplying beam by tcal')
                self.setting_dic[key]['beams'].append(beam*self.setting_dic[key]['tcal'])

        else:
            for i,key in enumerate(self.setting_dic):
                b = str(self.setting_dic[key]['beam_files'])

                if self.verbose:
                    print('Starting Beam: ', b)

                if '.fits' in b:
                    hdul = fits.open(b)
                    try:
                        beam = hdul[1].data * 1
                    except:
                        beam = hdul[0].data * 1

                elif '.sav' in b:
                    map_file = readsav(b)
                    keys = list(map_file.keys())
                    beam = map_file[keys[0]]*1

                elif '.tsv' in b or '.txt' in b:
                    beams = pd.read_csv(b,delim_whitespace=True,comment='#',names=['ell','cl'])
                    beam_ell = beams['ell']
                    beam_cl = beams['cl']
                    beam = extrapolate_beam(beam_ell,beam_cl,self.ny,self.nx,self.psize)

                else:
                    print('Error, unknown beam file type')

                if beam.shape != (self.ny,self.nx):
                    print('Beam shape does not equal map size. Are you using an old map? Reshaping')
                    beam = resize_input(beam,ell_space=True)

                if add_transfer:
                    print('Adding transfer function')
                    xfer=np.fft.fftshift(clu.make_filter_xfer(lpf=lpf,hpf=hpf,isohpf=isohpf,nx=self.nx,ny=self.ny,reso_arcmin=self.psize))
                    beam*=xfer

                print('multiplying beam by tcal')
                self.setting_dic[key]['beams'].append(beam*self.setting_dic[key]['tcal'])

            print(' ')
            return 
        
    def wrap_astro_noise(self):
        print('Constructing Astrophysical Noise spectra')
        components = ['tSZ','kSZ','CMB','DG-Po','DG-Cl']
        
        self.astro_cl_dict = {}
        
        for i, field in enumerate(self.setting_dic):
            self.astro_cl_dict[field] = {}
            band = str(self.setting_dic[field]['freq'])+'GHz'
            
            for index, comp in enumerate(components):
                curr_fg =foregrounds.get_foreground_power(comp, rescale='True', band1=band, model='reichardt')/core.G3Units.uK**2
                if 'PLW' in field or 'PMW' in field or 'PSW' in field:
                    curr_fg*=1e-50
                        
                curr_fg = np.pad(curr_fg, (0,29699), 'constant', constant_values=curr_fg[-1]) 
                self.astro_cl_dict[field][comp] = curr_fg
        return
                    
    def wrap_instrumental_noise(self):
        for i, key in enumerate(self.setting_dic):
            n = self.setting_dic[key]['instrumental_noise_files']
            
            if self.verbose:
                print('Starting Instrumental Noise: ', n)
                
            if n == 'None':
                 self.one_d_noise,ni = make_herschel_instr_noise(self,key)
            elif '.fits' in n:
                try:
                    ni = fits.open(n)[1].data
                except:
                    ni = fits.open(n)[0].data
                
            elif '.sav' in n:
                ni_file = readsav(n)
                ni_keys = list(ni_file.keys())
                ni = ni_file[ni_keys[0]]*1
                
            elif 'pkl' or 'bin' in n:
                ni = clu.pkl(n)*1

            else:
                print('Error: unknown file type')
                
            if ni.shape != (self.ny,self.nx):
                print('Instrumental noise shape does not equal map size. Congridding instrumental noise')
                ni = congrid(ni,(self.ny,self.nx))
                
            ni= (ni**2)
            
            self.setting_dic[key]['NoiseInstrument'].append(ni)
        print(' ')
        return 
    
    def make_fsz(self,do_rel=False,rel_kev=None):
        for f in self.setting_dic:
            szband = self.setting_dic[f]['sz_band_centers']
            curr_fsz = clu.calc_fsz(szband)
            self.setting_dic[f]['fsz'] = curr_fsz   
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
    
    def run_wrapper(self,apod_file='/sptlocal/user/kaylank/megaclusters/additional_data_products/milo_apod_mask.bin',
                    apod_pixel_file='/sptlocal/user/kaylank/megaclusters/additional_data_products/milo_pixel_mask.bin',add_transfer=False, theory=False,lpf = 20_000,hpf = 400,isohpf = 500,fwhm=None):
        self.get_map_array(apod_file = apod_file, apod_pixel_file = apod_pixel_file)
        self.make_fsz()
        self.wrap_beams(add_transfer=add_transfer, theory=theory,lpf=lpf,hpf=hpf,isohpf=500,fwhm=fwhm)
        self.wrap_astro_noise()
        self.wrap_instrumental_noise()
        self.finish_wrapping()
        return
    
def weight_maps(cf,map_list,noise_psds,verbose=True):
    final_map = np.zeros((cf.ny,cf.nx))
    total_noise = 0
    for i in range(len(map_list)):
        std = np.sqrt((noise_psds[i][150,100]))*3437.75
        if verbose:
            print('Noise in map: ' + str(i) + ' is about: '+ str(round(std,2)) +' uK-arcmin')
        curr_weight = 1/(std**2)
        final_map+=(map_list[i]*curr_weight)
        total_noise+=curr_weight
    if verbose:
        print('Final noise level is: ', round(1/np.sqrt(total_noise),2), ' uK-arcmin')
    return final_map/total_noise
    
def export_array_as_fits(array,field_ra_cent,field_dec_cent,file_path,psize=0.25):
    map_obj = maps.FlatSkyMap(array,
                      res=psize*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(field_ra_cent)*core.G3Units.deg,
                      delta_center=(field_dec_cent)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)
    spt3g.maps.fitsio.save_skymap_fits(file_path,map_obj)
    return

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims.astype(int))[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + list(range( ndims - 1 ))
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None 
    
def proj_reproj_map(key,array,field_ra_center, field_dec_center,ra_offset=0,dec_offset=0,psize=0.25):
    ra_offset/=3600
    dec_offset/=3600
    
    if np.sqrt(ra_offset**2 + dec_offset**2) > 15:
        if reproj:
                print('Reprojecting Map Center')
                print('RA offset ("):', ra_offset)
                print('DEC offset ("):', dec_offset)

                map_obj = maps.FlatSkyMap(np.asarray(array, order='C'),
                                          res=psize*core.G3Units.arcmin,
                                          weighted=False,
                                          alpha_center=(field_ra_center+ra_offset)*core.G3Units.deg,
                                          delta_center=(field_dec_center+dec_offset)*core.G3Units.deg,
                                          proj=maps.MapProjection.Proj0)

                map_obj_reproj = maps.FlatSkyMap(np.zeros_like(array),
                                          res=psize*core.G3Units.arcmin,
                                          weighted=False,
                                          alpha_center=(field_ra_center)*core.G3Units.deg,
                                          delta_center=(field_dec_center)*core.G3Units.deg,
                                          proj=maps.MapProjection.Proj0)

                spt3g.maps.reproj_map(map_obj,map_obj_reproj,rebin=4)

                return map_obj_reproj

    else:
        map_obj = maps.FlatSkyMap(np.asarray(array, order='C'),
                                  res=psize*core.G3Units.arcmin,
                                  weighted=False,
                                  alpha_center=(field_ra_center)*core.G3Units.deg,
                                  delta_center=(field_dec_center)*core.G3Units.deg,
                                  proj=maps.MapProjection.Proj0)
        return map_obj
    
def crop_apod_mask(wi, apod,apod2):
    diff_ny = int((wi.ny - wi.apod.shape[0])/2)
    diff_nx = int((wi.nx - wi.apod.shape[1])/2)
    if wi.apod.shape < map_array.shape:
        print('Apod Shape does not equal map array shape. Padding apod with zeros')

        wi.apod = np.pad(wi.apod, ((diff_ny,diff_ny), (diff_nx,diff_nx)))
        wi.apod2 = np.pad(wi.apod2, ((diff_ny,diff_ny), (diff_nx,diff_nx)))

    elif wi.apod.shape > map_array.shape:
        print('Apod Shape does not equal map array shape. Cropping apod to map shape')
        wi.apod = wi.apod[diff_ny:(wi.ny-diff_ny), diff_nx:(wi.ny-diff_nx)]
        wi.apod2 = wi.apod2[diff_ny:(wi.ny-diff_ny), diff_nx:(wi.ny-diff_nx)]
        
    return apod, apod2
    
def generate_apodization_mask(wi,map_array,apod_size=10,radius_arcmin=5):
    print('Generating Apodiation Mask')
    print('Warning: edge cases still need to be explored. May not generate a perfect apodization map')
    apod_pixel_mask = copy.copy(map_array)

    apod_pixel_mask[:,0:apod_size] = 0
    apod_pixel_mask[:,(wi.nx-apod_size):wi.nx] = 0
    apod_pixel_mask[0:apod_size,:] = 0
    apod_pixel_mask[(wi.ny-apod_size):wi.ny,:] = 0

    apod_pixel_mask[apod_pixel_mask != 0] = 1

    apod_obj = maps.FlatSkyMap(apod_pixel_mask,
              res=wi.psize*core.G3Units.arcmin,
              weighted=False,
              alpha_center=(wi.field_ra_center)*core.G3Units.deg,
              delta_center=(wi.field_dec_center)*core.G3Units.deg,
              proj=maps.MapProjection.Proj0)

    apod_mask = spt3g.mapspectra.apodmask.make_border_apodization(apod_obj,radius_arcmin=radius_arcmin)

    apod = apod_mask
    temp = copy.deepcopy(apod)
    temp = np.where(temp != 1, 0, temp)
    apod2 = temp
    
    return apod, apod2

def resize_input(array,ell_space):
    if ell_space:
        array = congrid(array,(3360,3360))
    else:
        y_shape = array.shape[0]; x_shape = array.shape[1]
        y_diff = int((y_shape-3360)/2); x_diff = int((x_shape-3360)/2)
        array = array[y_diff:y_shape-y_diff,x_diff:x_shape-x_diff]
    return array

def make_herschel_instr_noise(wi,key):
    half_1 = wi.setting_dic[key]['map_array'][0][600:2700,600:1650]
    half_2 = wi.setting_dic[key]['map_array'][0][600:2700,1650:2700]
    diff = np.pad((half_1-half_2)/2, ((0,0),(525,525)))
    k = 0
    for i in range(2100):
        for j in range(2100):
            if diff[i,j] == 0:
                k+=1
    instr_noise = spt3g.simulations.quick_flatsky_routines.cl_flatsky(diff/(k/2100**2))
    psd = clu.clstokmap(instr_noise['cl']['TT'], instr_noise['ell'],0.25,wi.ny,wi.nx)
    return instr_noise,psd
    
def extrapolate_beam(beam_ell,beam_cl,ny,nx,psize=0.25):
    ell = np.arange(43_200)
    beam_function = interpolate.interp1d(beam_ell, beam_cl, bounds_error=False, fill_value="extrapolate")
    interpolated_cl = np.where(beam_function(ell) <= 0, 1e-50, beam_function(ell))
    beam = clu.clstokmap(interpolated_cl, ell,psize,ny=ny,nx=nx)
    beam = np.where(beam<=0,1e-50,beam)
    return beam