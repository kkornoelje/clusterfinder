import sys
sys.path.append("/home/kaylank/imports")
from imports import *
import clusterfinder
import clusterfunctions as clu


#Edit: has a little trouble with the map edge cases

class source_subtraction():
    def __init__(self,cf,what_field,sn_threshold,temp_threshold,sn_thresholds=[0],mel_flux=False,what_pts_file=None,verbose=True):
        self.cf = cf
        self.sn_thresholds = sn_thresholds
        
        self.sn_threshold = sn_threshold
        self.temp_threshold = temp_threshold
        self.mel_flux = mel_flux
        self.what_field = what_field
        self.verbose = verbose
        
        
        
        if what_pts_file == None:
            pts_file = '/home/ndhuang/spt_code/spt3g_software/sources/python/main_lists/spt3g_1500d_source_list_ma_19sept23_v4.txt'
        else:
            pts_file=what_pts_file
            
        self.pts_file =pts_file
        self.pts_list = pd.read_csv(pts_file, comment='#', delim_whitespace=True)
        
        if 'W' in self.what_field:
            self.what_flux = 'F220'
            self.what_sn = 'SN220'
        else:
            self.what_flux = 'F'+str(self.cf.setting_dic[self.what_field]['freq'])
            self.what_sn = 'SN'+str(self.cf.setting_dic[self.what_field]['freq'])
            
            
        self.what_pts_file = pts_file
        if self.verbose:
            print("Source Subtracting with Mel Flux?: ", self.mel_flux)
            print('Using Flux: ', self.what_flux)
            print('Using S/N: ', self.what_sn)
        
        return

    def make_full_source_catalog(self):
        self.pts_cat = {}
        old_keys = ['RA(deg)','DEC(deg)','SN90','SN150','SN220','S90raw(mJy)','S150raw(mJy)','S220raw(mJy)']
        new_keys = ['RA','DEC','SN90','SN150','SN220','F90','F150','F220']
        
        for i in range(len(self.pts_list)):
            self.pts_cat[str(i)] = {}
            dec = self.pts_list['DEC(deg)'][i]
            ra = self.pts_list['RA(deg)'][i]
            if ra < 360:
                ra+=360
            
            for j, key in enumerate(old_keys):
                if key == 'RA(deg)':
                    self.pts_cat[str(i)][new_keys[j]] = self.pts_list[key][i]
                else:
                     self.pts_cat[str(i)][new_keys[j]] = self.pts_list[key][i]
                
            x,y = spt3g.maps.FlatSkyMap.angle_to_xy(self.cf.setting_dic[self.what_field]['map_objs'][0],
                                                    ra*core.G3Units.deg,
                                                    dec*core.G3Units.deg)
            self.pts_cat[str(i)]['x'] = x
            self.pts_cat[str(i)]['y'] = y
        
        self.full_pt_src_cat = self.pts_cat
        print('Number of point sources in total catalog: ', len(self.full_pt_src_cat))
        return

    def make_subfield_catalog(self,test_plot=False,check_mask=True):
        self.in_subfield_index = []
        self.not_in_subfield_index = []
        #Check if source in subfield
        for source in self.pts_cat:
            x = round(self.pts_cat[source]['x'])
            y = round(self.pts_cat[source]['y'])
            if (2 <= x < round(self.cf.nx)-2) and (2 <= y < round(self.cf.ny)-2) and (np.sum(np.asarray(self.cf.apod2)[y-5:y+5,x-5:x+5]) != 0):
                if check_mask:
                    if self.cf.mask[y][x] != 0:
                        self.in_subfield_index.append(source)
                    else:
                        self.not_in_subfield_index.append(source)
                else:
                    self.in_subfield_index.append(source)
            else:
                self.not_in_subfield_index.append(source)

        self.in_subfield_cat = copy.deepcopy(self.pts_cat)
                
        for source in self.not_in_subfield_index:
            self.in_subfield_cat.pop(source)
            
        for source in self.in_subfield_cat:
            y = round(self.in_subfield_cat[source]['y'])
            x = round(self.in_subfield_cat[source]['x'])
            
            self.in_subfield_cat[source]['mel_ref_flux'] = self.in_subfield_cat[source][self.what_flux]
            self.in_subfield_cat[source]['mel_ref_sn'] = self.in_subfield_cat[source][self.what_sn]
            
            try:
                self.in_subfield_cat[source]['cf_ref_flux'] = self.cf.fmap[y][x]
                self.in_subfield_cat[source]['cf_ref_sn'] = self.cf.snmap[y][x]
            except:
                continue
            
            
            
        print('Number of point sources in subfield: ', len(self.in_subfield_cat))
        if test_plot:
            plt.rcParams['figure.figsize'] = (40,20)
            plt.imshow(self.cf.apod2*self.cf.mask,vmin=-5,vmax=5,cmap='Greys')
            
            plt.scatter([self.in_subfield_cat[i]['x'] for i in self.in_subfield_cat],
                       [self.in_subfield_cat[i]['y'] for i in self.in_subfield_cat],color='red',s=20,label='Within Subfield Bounds')
            
            plt.scatter([self.full_pt_src_cat[i]['x'] for i in self.full_pt_src_cat],
                       [self.full_pt_src_cat[i]['y'] for i in self.full_pt_src_cat],facecolor='none',edgecolor='blue',s=50,label='All Point Sources')
            
            plt.xlim(self.cf.nx,0)
            plt.ylim(self.cf.ny,0)
            plt.legend()
            plt.show()
                   
        return
    
    def set_sn_threshold(self,test_plot=False):
        if self.verbose:
            print('Subtracting Sources with S/N greater than: ' + str(self.sn_threshold) + ' S/N ')
        self.sources_below_sn_cut = []
        self.sub_sources = copy.deepcopy(self.in_subfield_cat)
        
        for source in self.in_subfield_cat:
            if self.in_subfield_cat[source]['SN90'] < self.sn_threshold:
                if self.in_subfield_cat[source]['SN150'] < self.sn_threshold:
                    if self.in_subfield_cat[source]['SN220'] < self.sn_threshold:
                        self.sources_below_sn_cut.append(source)
                        self.sub_sources.pop(source)

        if test_plot:
            plt.rcParams['figure.figsize'] = (40,20)
            plt.imshow(self.cf.apod2,vmin=-5,vmax=5,cmap='Greys')
            
            plt.scatter([self.in_subfield_cat[i]['x'] for i in self.in_subfield_cat],
                       [self.in_subfield_cat[i]['y'] for i in self.in_subfield_cat],color='red',s=20)
            
            plt.scatter([self.in_subfield_cat[i]['x'] for i in self.sources_below_sn_cut],
                       [self.in_subfield_cat[i]['y'] for i in self.sources_below_sn_cut],color='green',s=20)
            
            plt.scatter([self.full_pt_src_cat[i]['x'] for i in self.full_pt_src_cat],
                       [self.full_pt_src_cat[i]['y'] for i in self.full_pt_src_cat],facecolor='none',edgecolor='blue',s=50)
            
            
            plt.xlim(self.cf.nx,0)
            plt.ylim(self.cf.ny,0)
            plt.show()
            
            
                        
        return
    
    def remove_interpolated_sources(self,test_plot = False,interpol_source_file = None):
        def distance(ra1,dec1,ra2,dec2):
            return np.sqrt( (ra1-ra2)**2 + (dec1-dec2)**2 )
        
        self.interpolated_sources = []
        if interpol_source_file == None:
            self.interpol_source_file = '/home/joshuasobrin/spt3g/spt3g_software/sources/mask_lists/1500d_ptsrc_150GHz_50mJy_2021.txt'
        else:
            self.interpol_source_file = interpol_source_file
            
            
        interpol_source_list = pd.read_csv(self.interpol_source_file,delim_whitespace=True,comment='#')
        for i in range(len(interpol_source_list)):
            interpol_RA = interpol_source_list['RA[deg]'][i]
            interpol_DEC = interpol_source_list['DEC[deg]'][i]
            
            for source in self.sub_sources:
                ra = self.in_subfield_cat[source]['RA']
                dec = self.in_subfield_cat[source]['DEC']
                
                if distance(interpol_RA,interpol_DEC,ra,dec)*60 <= 2:
                    self.interpolated_sources.append(source)

        for key in self.interpolated_sources:
            if key in self.sub_sources:
                self.sub_sources.pop(key)

        if test_plot:
            plt.rcParams['figure.figsize'] = (40,20)
            plt.imshow(self.cf.apod2,vmin=-5,vmax=5,cmap='Greys')
            
            plt.scatter([self.in_subfield_cat[i]['x'] for i in self.in_subfield_cat],
                       [self.in_subfield_cat[i]['y'] for i in self.in_subfield_cat],color='red',s=20)
            
            plt.scatter([self.in_subfield_cat[i]['x'] for i in self.interpolated_sources],
                       [self.in_subfield_cat[i]['y'] for i in self.interpolated_sources],color='green',s=20)
            
            plt.scatter([self.full_pt_src_cat[i]['x'] for i in self.full_pt_src_cat],
                       [self.full_pt_src_cat[i]['y'] for i in self.full_pt_src_cat],facecolor='none',edgecolor='blue',s=50)
            
            
            plt.xlim(self.cf.nx,0)
            plt.ylim(self.cf.ny,0)
            plt.show()
        print('Number of interpolated sources: ', len(self.interpolated_sources))
        return
    
    def remove_double_sources(self):
        print('Depreciated')
        self.double_sources_matches = []
        self.double_source_pop = []
        self.high_flux = []
        self.low_flux = []
        
        def distance(x1,y1,x2,y2):
            return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
        
        for source_1 in self.sub_sources:
            for source_2 in self.sub_sources:
                if source_1 != source_2:
                    
                    if distance(self.in_subfield_cat[source_1]['x'],self.in_subfield_cat[source_1]['y'],
                               self.in_subfield_cat[source_2]['x'],self.in_subfield_cat[source_2]['y']) <= 3:
                        
                        self.double_sources_matches.append((source_1,source_2))
                        
                        if self.in_subfield_cat[source_1][self.what_sn] > self.in_subfield_cat[source_2][self.what_sn]:
                            self.double_source_pop.append(source_2)
                            self.high_flux.append(source_2)
                            self.low_flux.append(source_1)
                            
                        else:
                            self.double_source_pop.append(source_1)
                            self.high_flux.append(source_1)
                            self.low_flux.append(source_2)
                        
                            
        for source in self.double_source_pop:
            if source in self.sub_sources:
                self.sub_sources.pop(source)
        return
            
    def create_cutout_map(self,ra,dec,map_obj,temp_y,temp_x):
        cutout_template = maps.FlatSkyMap((temp_x*4), #nx
                                          (temp_y*4), #ny
                                          (self.cf.psize/4)*core.G3Units.arcmin, #resolution
                                          proj=maps.MapProjection.Proj0,
                                          alpha_center=ra,
                                          delta_center=dec)

        maps.reproj_map(map_obj,cutout_template, rebin=4)
        cutout_template = np.asarray(cutout_template)

        return cutout_template
    
    def make_med_temp(self,temp_dic,mel_flux):
        templates = []

        for i,source in enumerate(temp_dic):
            ra = self.in_subfield_cat[source]['RA']*core.G3Units.deg
            dec = self.in_subfield_cat[source]['DEC']*core.G3Units.deg
            x = round(self.in_subfield_cat[source]['x'])
            y = round(self.in_subfield_cat[source]['y'])

            cutout_template = self.create_cutout_map(ra,dec,self.cf.setting_dic[self.what_field]['map_objs'][0],self.temp_y,self.temp_x)

            if mel_flux:
                flux = self.in_subfield_cat[source]['mel_ref_flux']
            else:
                flux = self.in_subfield_cat[source]['cf_ref_flux']

            cutout_template/=flux 
            templates.append(cutout_template)

        median_template = np.nanmedian(np.dstack(templates), -1)

        return templates,median_template
                        
    def make_ss_template(self,temp_y=25,temp_x=125):
        if self.verbose:
            print('Constructing Template with Sources with S/N greater than: ' + str(self.temp_threshold) + ' S/N')
            
        self.temp_y = temp_y
        self.temp_x = temp_x
        
        self.template_sources = []
                
        for source in self.in_subfield_cat:
            if self.in_subfield_cat[source][self.what_sn] >= self.temp_threshold:
                x, y = round(self.in_subfield_cat[source]['x']), round(self.in_subfield_cat[source]['y'])
                self.template_sources.append(source)

        self.templates, self.median_template = self.make_med_temp(self.template_sources,self.mel_flux)
        
        template_obj = maps.FlatSkyMap(np.ones((self.temp_y*4,self.temp_x*4)),
                                      res=self.cf.psize*core.G3Units.arcmin,
                                      weighted=False,
                                      alpha_center=(self.cf.field_ra_cent)*core.G3Units.deg,
                                      delta_center=(self.cf.field_dec_cent)*core.G3Units.deg,
                                      proj=maps.MapProjection.Proj0)
        
        self.template_apod= spt3g.mapspectra.apodmask.make_border_apodization(template_obj,radius_arcmin=15,apod_threshold=1e-100,zero_border_arcmin=-100)
        self.median_template*=np.asarray(self.template_apod)
        if self.verbose:
            print('Made a point source template with ', len(self.template_sources), ' sources')
        
        return
    
    def run_ss(self,sn_thresholds,insert_temp='None',subtract_list='None'):
        if self.verbose:
            print('Starting Source Subtraction')
            print('Subtracting ', len(self.sub_sources), ' sources')
            
        source_subtract_map = maps.FlatSkyMap(np.zeros((self.cf.ny,self.cf.nx)),
                                       res=self.cf.psize*core.G3Units.arcmin,
                                        weighted=False,
                                        alpha_center = self.cf.field_ra_cent*core.G3Units.deg,
                                        delta_center= self.cf.field_dec_cent*core.G3Units.deg,
                                        proj=maps.MapProjection.Proj0)
        ss = source_subtract_map.copy()

        y_ext = round((self.median_template.shape[0]/4)/2)
        x_ext = round((self.median_template.shape[1]/4)/2)
        
        if self.mel_flux:
            for source in tqdm_notebook(self.sub_sources):
                x = round(self.sub_sources[source]['x'])
                y = round(self.sub_sources[source]['y'])

                flux = self.sub_sources[source]['mel_ref_flux']

                source_array = self.median_template*flux
                source_centered_map = maps.FlatSkyMap(source_array,(self.cf.psize/4)*core.G3Units.arcmin,
                                                                  proj=maps.MapProjection.Proj0,
                                                                  alpha_center=self.sub_sources[source]['RA']*core.G3Units.deg,
                                                                  delta_center=self.sub_sources[source]['DEC']*core.G3Units.deg,
                                                                  weighted=False)

                subfield_centered_map = source_subtract_map.copy()
                subfield_centered_map = subfield_centered_map[y-y_ext:y+y_ext,x-x_ext:x+x_ext]
                maps.reproj_map(source_centered_map,subfield_centered_map,rebin=1,interp=False)

                ss[y-y_ext:y+y_ext,x-x_ext:x+x_ext]-=subfield_centered_map
        else:
            for thresh_index, threshold in enumerate(sn_thresholds):
                if self.verbose:
                    print('Subtracting Sources Greater Than: ', threshold, r'$\sigma$' )
                    
                if thresh_index == 0:
                    self.what_fmap = self.cf.fmap
                else:             
                    self.skymaps = np.asarray([(self.cf.setting_dic[self.what_field]['map_objs'][0]*self.cf.mask  +(ss))*self.cf.apod])
                    self.fmap = clu.multi_band_filter(self.skymaps,self.cf.psi,self.cf.psize,1)
                    self.what_fmap = self.fmap

                for source in tqdm_notebook(self.sub_sources):
                    if abs(self.sub_sources[source][self.what_sn]) > threshold:
                        x = round(self.sub_sources[source]['x'])
                        y = round(self.sub_sources[source]['y'])

                        flux = self.what_fmap[y][x]

                        source_array = self.median_template*flux

                        source_centered_map = maps.FlatSkyMap(source_array,(self.cf.psize/4)*core.G3Units.arcmin,
                                                                  proj=maps.MapProjection.Proj0,
                                                                  alpha_center=self.sub_sources[source]['RA']*core.G3Units.deg,
                                                                  delta_center=self.sub_sources[source]['DEC']*core.G3Units.deg,
                                                                  weighted=False)

                        subfield_centered_map = source_subtract_map.copy()
                        subfield_centered_map = subfield_centered_map[y-y_ext:y+y_ext,x-x_ext:x+x_ext]
                        maps.reproj_map(source_centered_map,subfield_centered_map,rebin=1,interp=False)

                        ss[y-y_ext:y+y_ext,x-x_ext:x+x_ext]-=subfield_centered_map
        self.ss = ss
        return
            
    def preform_source_subtraction(self):
        self.make_full_source_catalog()
        self.make_subfield_catalog()
        self.set_sn_threshold()
        self.remove_interpolated_sources()
        if self.mel_flux == False:
            self.remove_double_sources()
        self.make_ss_template()
        self.run_ss(self.sn_thresholds)
        
        return
    
class test_source_sub():
    def __init__(self,wi,ss,cf,what_map='None'):
        self.ss = ss
        self.wi = wi
        self.cf = cf
        self.what_field = list(wi.setting_dic.keys())[0]
        
        return
        
    def rerun_map(self,matrix,scale=1,sn_cutoff=5):
        cf_new = clusterfinder.cluster_finder([self.what_field], self.wi.final_setting_dic,units='uk')
        cf_new.make_fsz()
        cf_new.make_maps(what_mask='simple_mask')
        cf_new.setting_dic[self.what_field]['skymaps']+=self.ss.ss
        cf_new.make_beams()
        cf_new.make_astro_noise()
        cf_new.make_instr_noise()
        cf_new.make_covar()
        cf_new.find_clusters(sn_cutoff=5,tcs=1,point_source_finder=False)
        self.cf_new = cf_new
        return
    
    def test_subtraction(self,save_fig=False,plot_results=False):
        old_dic = copy.deepcopy(self.ss.in_subfield_cat)
        new_dic = copy.deepcopy(self.ss.in_subfield_cat)
        
        not_in_sub_sources = []
        
        for source in self.ss.in_subfield_cat:
            if source not in self.ss.sub_sources:
                not_in_sub_sources.append(source)

            x = round(self.ss.in_subfield_cat[source]['x'])
            y = round(self.ss.in_subfield_cat[source]['y'])
            
            old_dic[source]['cent_sn'] = self.cf.snmap[y][x]
            new_dic[source]['cent_sn'] = self.cf_new.snmap[y][x]
        
            old_wing = (self.cf.snmap[y][x+10] + self.cf.snmap[y][x-10])/2
            old_dic[source]['wing_sn'] = old_wing
            
            new_wing = (self.cf_new.snmap[y][x+10] + self.cf_new.snmap[y][x-10])/2
            new_dic[source]['wing_sn'] = new_wing
            
        self.new_dic = new_dic
        self.old_dic = old_dic
        return
    
    def make_plot(self,key,title,save_fig,file_name_and_path=''):
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        x_full = [self.old_dic[source][key] for source in self.ss.in_subfield_cat]
        y_full = [self.new_dic[source][key] for source in self.ss.in_subfield_cat]
        
        x = [self.old_dic[source][key] for source in self.ss.sub_sources]
        y = [self.new_dic[source][key] for source in self.ss.sub_sources]
        
        ax.scatter(x_full, y_full,color='red')
        ax.scatter(x,y,color='grey')
        m, b = np.polyfit(x, y, 1)
        
        y_fit = np.multiply(m ,x)
        ax.plot(x, y_fit, color='red', label=f'Best-fit Line: y = {m:.2f}x'+ '+ '+str(round(b,2)),linestyle='--')
        ax.legend()
        
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (round(xymax/binwidth) + 1) * binwidth
        
        ax_histx.hist(x, bins=20,color='slategrey')
        ax_histy.hist(y, bins=20, orientation='horizontal',color='slategrey')
        def rms(y):
            y = np.asarray(y)
            return np.sqrt(np.mean(y**2))

        rms_1 = rms(x)
        rms_2 = rms(y)
        ax_histx.hlines(100, 0+rms_1,0-rms_1,label='RMS: \n'+str(round(rms_1,2)),color='red',linewidth=2)
        ax_histy.vlines(100, 0+rms_2,0-rms_2,label='RMS: \n'+str(round(rms_2,2)),color='red',linewidth=2)
        ax_histx.legend()
        ax_histy.legend(loc='upper center')

        ax.set_xlabel('Uncorrected '+r'$\xi$')
        ax.set_ylabel('Corrected '+r'$\xi$')

        fig.suptitle(title)
        if save_fig:
            fig.savefig(file_name_and_path)
        return
        
    def flux_comparison(self,plot_results=True,bad_flux=None):
        self.recorded_fluxes = []
        self.cluster_finder_fluxes = []
        self.bad_flux_index = []
        self.bad_flux_source = []
        
        for i,source in enumerate(self.ss.sub_sources):
            x = round(self.ss.sub_sources[source]['x'])
            y = round(self.ss.sub_sources[source]['y'])
            
            cf_flux = abs(self.cf.fmap[y][x])/6.3e7
            rec_flux = self.ss.sub_sources[source]['F90']
            
            self.recorded_fluxes.append(rec_flux)
            self.cluster_finder_fluxes.append(cf_flux)
            if bad_flux !=None:
                if abs(mel/mine) > bad_flux or abs(mel/mine) < (1/bad_flux):
                    self.bad_flux_source.append(source)
                    self.bad_flux_index.append(i)
        
            
        self.intercept,self.slope = median_quantile_regression(self.recorded_fluxes,self.cluster_finder_fluxes)
        if plot_results:
            plt.scatter(self.recorded_fluxes,self.cluster_finder_fluxes,color='grey')
            plt.plot(self.recorded_fluxes,np.multiply(self.recorded_fluxes,self.slope)+self.intercept,color='red',
                     linestyle='--',label='y = '+str(round(self.slope,2))+'x +'+str(round(self.intercept,2)))
            
            plt.legend(fontsize=15)
            if bad_flux !=None:
                plt.scatter([self.recorded_fluxes[i] for i in self.bad_flux_index],
                            [self.cluster_finder_fluxes[i] for i in self.bad_flux_index],
                            color='red',label='Flux Ratio Difference >'+str(bad_flux))
                
            plt.legend(fontsize=15)

            plt.xlabel('Recorded Flux',fontsize=15)
            plt.ylabel('Cluster Finder Flux',fontsize=15)
            plt.title('SPT3G 90GHz',fontsize=20)
            
        return