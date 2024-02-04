import sys
import importlib
sys.path.append("/home/kaylank/imports")
import relativistic_tsz
import sys
import importlib
sys.path.append("/home/kaylank/pipelines/")
from class_imports import *

sys.path.append("/home/kaylank/imports")
from imports import *

SHEET_ID = '1OmF4kCz3ySe76jrlax4b5qFgSQc1PELZcb7IQ_zfi5w'
SHEET_NAME = 'fitting_settings'



    
    
    



def make_frame(input_map,ra_cent=352.5,dec_cent=-55,psize=0.25):
        frame= core.G3Frame()
        frame['T'] = maps.FlatSkyMap(np.asarray(input_map,order='C'),
                          res=psize*core.G3Units.arcmin,
                          weighted=False,
                          alpha_center=(ra_cent)*core.G3Units.deg,
                          delta_center=(dec_cent)*core.G3Units.deg,
                          proj=maps.MapProjection.Proj0)
        return frame
    
def is_close_to_existing_points(x, y, existing_x, existing_y, max_distance_x, max_distance_y):
    for i in range(len(existing_x)):
        distance_x = abs(existing_x[i] - x)
        distance_y = abs(existing_y[i] - y)
        if distance_x < max_distance_x and distance_y < max_distance_y:
            return True
    return False

def fake_cluster_map(cf, num_clusters, core_size = 0.25,psize=0.25,freq_scale=True,temp=0.01,
                     cat=None,beam_convolved=False,verbose=True,x_pixs=None,y_pixs=None,complex_temp_map=False,quadrants=4):
    
    fake_cluster_x = []
    fake_cluster_y = []

    k = 0
    if x_pixs != None and y_pixs != None:
        fake_cluster_x = x_pixs
        fake_cluster_y = y_pixs
    else:
        while k < num_clusters:
            x = random.randint(150, cf.nx-150)
            y = random.randint(150, cf.ny-150)

            if cf.apod2[y][x] != 0:
                if not is_close_to_existing_points(x, y, fake_cluster_x, fake_cluster_y, 300, 200): 
                    fake_cluster_x.append(x)
                    fake_cluster_y.append(y)
                    k += 1
                    if k > 80:
                        print(k)
                    if k % 10 == 0:
                        print(k)
                
    curr_core_size = int(((core_size)/(0.25))-1)
    
    if beam_convolved:
        if verbose:
            print('Convolving cluster profiles with beams')
    
    if complex_temp_map:
        print('Generating complex temp_map')
        
    if freq_scale:
        if verbose:
            print('Scaling maps by temperature: ', temp)
        
        
    fake_cluster_map = {}
    

    for key in cf.setting_dic:
        fake_cluster_map[key] = np.zeros((cf.ny,cf.nx))
        profile_ft=clu.gridtomap(clu.ell_grid(cf.psize,cf.ny,cf.nx),cf.idl_profiles['profs'][curr_core_size],cf.idl_ls)
        if beam_convolved:
            profile_ft*=cf.setting_dic[key]['beams'][0]
            
        real_profile=np.real(np.fft.fftshift(np.fft.ifft2(profile_ft,norm='ortho'))/cf.psize)
        cent_y = int(cf.ny/2)
        cent_x = int(cf.nx/2)
        
        real_profile = real_profile[cent_y - 150: cent_y + 150, cent_x - 150 : cent_x + 150] 
        real_profile/=np.max(real_profile)
        
        
        
        if freq_scale:
            real_profile*=relativistic_tsz.fxsz_itoh(cf.setting_dic[key]['sz_band_centers'],temp)[0][0]

        for i in range(len(fake_cluster_x)):
            curr_x = fake_cluster_x[i]
            curr_y = fake_cluster_y[i]

            fake_cluster_map[key][int(curr_y-(150)):int(curr_y+(150)), int(curr_x-(150)):int(curr_x+(150))] = real_profile
    coords = []
    for i in range(len(fake_cluster_x)):
        coords.append((fake_cluster_x[i], fake_cluster_y[i]))
    
            
    return fake_cluster_map, coords

def export_array_as_fits(array,ra_cent,dec_cent,psize,filepath):
    map_obj = maps.FlatSkyMap(array,
                      res=psize*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(ra_cent)*core.G3Units.deg,
                      delta_center=(dec_cent)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)
    spt3g.maps.fitsio.save_skymap_fits(filepath,map_obj)
    return

def gen_sim_maps(cf,flatskyparams,scaled_copies = True):
    sim_maps = {}
    keys = list(cf.astro_cl_dict.keys())
    null_key = keys[0]
    
    for comp in cf.astro_cl_dict[null_key]:
        sim_maps[comp] = clu.cl2map(flatskyparams, cf.astro_cl_dict[null_key][comp], np.arange(43_200))
        
    final_sim_maps = {}
    for key in cf.astro_cl_dict:
        final_sim_maps[key] = {}
        for comp in sim_maps:
            if comp != 'tSZ' and comp != 'kSZ' and comp != 'CMB':
                print("Sorry, can't get frequency scaling")
            else:
                if comp == 'tSZ':
                    scale = cf.setting_dic[key]['fsz'] /  cf.setting_dic[null_key]['fsz']
                    final_sim_maps[key][comp] == scale*sim_maps[comp]
                else:
                    final_sim_maps[key][comp] == sim_maps[comp]
                    
    return

#===these don't work yet===#
def test_theoretical_spectrum(cf,skymaps,null_map,unit_map,do_cilc=True):
    
    weight = clusterfinder.cilc(cf.fsz,np.ones_like(cf.ft_signal),cf.inver,cf.ny,cf.nx,cf.nbands,do_cilc,projection_scaling=[1,1])
    fmap = clu.multi_band_filter(skymaps, weight,cf.psize,cf.nbands)
    

    del_el = 100
    fmap_map =(-1*fmap*1e4)*cf.apod 
    CMB_full = CMB_map*cf.apod
    tSZ_full = tSZ_map*cf.apod
    
    cross_spectrum_CMB = spt3g.mapspectra.calculate_powerspectra(input1=make_frame(fmap_map),input2=make_frame(CMB_full),delta_l=del_el)['TT']
    cross_spectrum_tSZ = spt3g.mapspectra.calculate_powerspectra(input1=make_frame(fmap_map),input2=make_frame(tSZ_full),delta_l=del_el)['TT']
    
    fmap_ps = spt3g.mapspectra.calculate_powerspectra(input1=make_frame(fmap_map),delta_l=del_el)['TT']
    CMB_ps = spt3g.mapspectra.calculate_powerspectra(input1=make_frame(CMB_full),delta_l=del_el)['TT']
    tSZ_ps = spt3g.mapspectra.calculate_powerspectra(input1=make_frame(tSZ_full),delta_l=del_el)['TT']

    ell = np.arange(len(cross_spectrum_CMB))*del_el
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))
    fig.suptitle('cILC',fontsize=30,y=1.05)
    ax1.semilogy(ell, fmap_ps,color='red',label='Compton-y Signal Map')
    ax1.semilogy(ell, CMB_ps,color='black',linestyle='--',label='CMB')
    ax1.semilogy(ell, tSZ_ps,color='grey',linestyle='--',label='tSZ')
    ax1.set_xlim(1_000,10_000)
    ax1.legend(fontsize=15)
    ax1.set_xlabel(r'$\ell$',fontsize=20,x=1.7)


    ax2.plot(ell,cross_spectrum_CMB/np.sqrt(CMB_ps*fmap_ps),color='slategrey',label=r'$\frac{P_{CMB} x P_{Signal}}{\sqrt{P_{CMB}xP_{Signal}}}$')
    ax2.set_title('CMB',fontsize=14)
    ax2.set_xlim(1_000,10_000)
    ax2.set_ylim(-1,2)
    ax2.hlines(0,0,10_000,color='red',linestyle='--')
    ax2.hlines(1,0,10_000,color='green',linestyle='--')
    ax2.legend(fontsize=20)

    ax3.plot(ell,cross_spectrum_tSZ/np.sqrt(tSZ_ps*fmap_ps),color='slategrey',label=r'$\frac{P_{tSZ} x P_{Signal}}{\sqrt{P_{tSZ}xP_{Signal}}}$')
    ax3.set_title('tSZ',fontsize=14)
    ax3.set_xlim(1_000,10_000)
    ax3.set_ylim(-1,2)
    ax3.hlines(0,0,10_000,color='red',linestyle='--')
    ax3.hlines(1,0,10_000,color='green',linestyle='--')
    ax3.legend(fontsize=20)
    
    return
def test_powerspectra(test_map,control_map,do_plot=True,cross=False):
    

    def cs(n, y):
        return chisquare(n, np.sum(n)/np.sum(y) * y)
    
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
    if cross:
        test_control_cross_spectrum = np.asarray(spt3g.mapspectra.calculate_powerspectra(input1=make_frame(test_map),input2=make_frame(control_map),delta_l=del_el)['TT'])
        
        
    control_auto_spectrum = np.asarray(spt3g.mapspectra.calculate_powerspectra(input1=make_frame(control_map),delta_l=del_el)['TT'])
    test_auto_spectrum = np.asarray(spt3g.mapspectra.calculate_powerspectra(input1=make_frame(test_map),delta_l=del_el)['TT'])
    
    ell = np.arange(len(test_auto_spectrum))*del_el
    
    

    
    if do_plot:
        if cross:
            fig, axs = plt.subplots(nrows=5,ncols=1,gridspec_kw={'height_ratios':[3,1,1,1,1]},figsize=(9,15),sharex=True)
            plt.subplots_adjust(wspace=0, hspace=0)
            full_axs = [0,1,2,3,4]
            for i in full_axs:
                axs[i].set_xlim(0,10_00)
                axs[i].set_ylim(-1,1)

            axs[0].scatter(ell, test_control_cross_spectrum,label='Cross Spectrum',color='slategrey')
            axs[0].scatter(ell, control_auto_spectrum,label='Control Auto Spectrum',color='palevioletred')
            axs[0].scatter(ell, test_auto_spectrum,label='Test Auto Spectrum',facecolor='None',edgecolor='lightseagreen',s=100,linewidth=2)
            axs[0].set_ylim(np.min(control_auto_spectrum[0:12])*1.1, np.max(control_auto_spectrum[0:12])*1.1)
            axs[0].legend()

            cross_control_diff = test_control_cross_spectrum- control_auto_spectrum


            axs[1].plot(ell,cross_control_diff,color='slategrey',label=r'$P^{cross-control} - P^{control-control}$')



            auto_diff = test_auto_spectrum- control_auto_spectrum
            axs[2].plot(ell,auto_diff,color='slategrey',label=r'$P^{test-test} - P^{control-control}$')



            cross_ratio = (test_control_cross_spectrum/control_auto_spectrum) - 1
            axs[3].plot(ell,cross_ratio,color='slategrey',label=r'$P^{cross-control}/P^{control-control} -1$')



            auto_ratio = (test_auto_spectrum/control_auto_spectrum) - 1
            axs[4].plot(ell,auto_ratio,color='slategrey',label=r'$P^{test-test}/P^{control-control} -1$')

            sep_axs = [1,2,3,4]
            for i in sep_axs:
                axs[i].hlines(0,0,10_000,linestyle='--',color='palevioletred')
                axs[i].legend(fontsize=15)
        else:
            fig, axs = plt.subplots(nrows=3,ncols=1,gridspec_kw={'height_ratios':[3,1,1]},figsize=(9,15),sharex=True)
            plt.subplots_adjust(wspace=0, hspace=0)
            full_axs = [0,1,2]
            for i in full_axs:
                axs[i].set_xlim(0,10_000)
                axs[i].set_ylim(-1,1)

            axs[0].semilogy(ell, control_auto_spectrum,label='Control Auto Spectrum',color='palevioletred')
            axs[0].scatter(ell, test_auto_spectrum,label='Test Auto Spectrum',edgecolor='lightseagreen',s=10)
            axs[0].set_ylim(np.min(control_auto_spectrum[0:100])*1.1, np.max(control_auto_spectrum[0:12])*1.1)
            axs[0].legend()

            auto_diff = test_auto_spectrum- control_auto_spectrum
            axs[1].plot(ell,auto_diff,color='slategrey',label=r'$P^{test-test} - P^{control-control}$')
            chi_2 = cs(auto_diff,np.zeros(len(auto_diff)))
            axs[1].plot(np.asarray(ell),np.asarray(auto_diff),color='slategrey',label=r'\Chi^2:'+ str(round(chi_2,2)))
            
            axs[1].plot(ell,auto_diff,color='slategrey',label=r'$P^{test-test} - P^{control-control}$')

            auto_ratio = (test_auto_spectrum/control_auto_spectrum) - 1
            axs[2].plot(ell,auto_ratio,color='slategrey',label=r'$P^{test-test}/P^{control-control} -1$')
            chi_2 = cs(auto_ratio,np.zeros(len(auto_diff)))
            
            axs[2].plot(np.asarray(ell),np.asarray(auto_diff),color='slategrey',label=r'\Chi^2:'+ str(round(chi_2,2)))

            sep_axs = [1,2]
            for i in sep_axs:
                axs[i].hlines(0,0,10_000,linestyle='--',color='palevioletred')
                axs[i].legend(fontsize=15)
            
        
    return

