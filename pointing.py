import sys
import source_subtraction
sys.path.append("/home/kaylank/imports/")
from imports import *

testing_path = '/sptlocal/user/kaylank/megaclusters/validation/pointing_validation/temp/'
streetlights = np.load('/home/kaylank/imports/streelights_cmap.npy', allow_pickle=True).item()
cmap = streetlights




def add_tcs(dic):
    cluster_list = pd.read_csv('/home/kaylank/megaclusters/spt_cluster/notebooks/Min_Var_List_9_22.csv')
    
    for i,key in enumerate(dic):
        dic[key]['core_size'] = cluster_list['tcs'][i]
    return dic

def deconvolve_and_filter(cf,raw_maps):
    deconvolved_maps = {}

    for key in raw_maps:
        deconvolved_maps[key] = cib.deconvolve_beam(raw_maps[key],
                                                  cf.setting_dic[key]['beams'][0],
                                                  cf.setting_dic['3g90']['beams'][0])

    filtered_maps = {}
    psi = clu.pkl('/sptlocal/user/kaylank/megaclusters/results/mod_bb_samples/aux_files/filter_inputs/final_psi.pkl')
    new_keys = ['90GHz','150GHz','220GHz','600GHz','857GHz','1200GHz']

    for i,key in enumerate(raw_maps):
        filtered_maps[new_keys[i]] = clu.multi_band_filter([deconvolved_maps[key]], psi*-1, cf.psize,1)
    return deconvolved_maps, filtered_maps

def create_clean_cats(cat1,cat2,match1,match2):
    clean1 = {}
    for key in match1:
        clean1[key] = cat1[key]
    clean2 = {}
    for key in match2:
        clean2[key] = cat2[key]
    return clean1,clean2

def crossmatch_cats(dic1,dic2,threshold=5):
    matches = []
    for key in dic1:
        for key2 in dic2:
            dist = point.calc_distance(dic1[key]['RA'], dic2[key2]['RA'], dic1[key]['DEC'], dic2[key2]['DEC'],units='arcmin')
            if dist < threshold:
                matches.append((key,key2))
    return [matches[i][0] for i in range(len(matches))],[matches[i][1] for i in range(len(matches))]    

def add_optical_pos(cf, mega):
    test = fits.open('/home/kaylank/imports/catalogs/cluster_catalogs/y3_gold_2.2.1_wide_sof-sfd98_run_spt3g_minvar_09_22_redmapper_v0.8.1_zscan_catalog.fit')[1]
    table = point.fitstable_to_dict(test)
    
    for key in table:
        spt_x, spt_y = spt3g.maps.FlatSkyMap.angle_to_xy(cf.map_obj,table[key]['ra']*core.G3Units.deg, table[key]['dec']*core.G3Units.deg )
        table[key]['spt_x'] = round(spt_x)
        table[key]['spt_y'] = round(spt_y)

        opt_x, opt_y = spt3g.maps.FlatSkyMap.angle_to_xy(cf.map_obj,table[key]['ra_opt']*core.G3Units.deg, table[key]['dec_opt']*core.G3Units.deg )
        table[key]['opt_x'] = round(opt_x)
        table[key]['opt_y'] = round(opt_y)
        
    for key2 in mega:
        mega[key2]['opt_ra'] = 10_000
        mega[key2]['opt_dec'] = 10_000
        mega[key2]['opt_x']  =10_000
        mega[key2]['opt_y'] = 10_000
         
        for key in table:
            dist = point.calc_distance(table[key]['ra'], mega[key2]['RA'], table[key]['dec'], mega[key2]['DEC'],units='arcmin')
            if dist < 0.8:
                mega[key2]['opt_ra'] = table[key]['ra_opt']
                mega[key2]['opt_dec'] = table[key]['dec_opt']
                mega[key2]['opt_x']  =table[key]['opt_x']
                mega[key2]['opt_y'] =table[key]['opt_y']
    return mega

def clean_false_detections(dic):
    false_detection = []
    keys = list(dic.keys())
    for key in keys:
        if dic[key]['REDSHIFT'] < 1e-3:
            dic.pop(key)

    return dic

def initialize_plot(cutouts,tag='horizontal',freqs=['90GHz','150GHz','220GHz','600GHz','857GHz','1200GHz'],title='',num_objects='', add_colorbar=False,overplot_guassian=True):
    redshift_bins = list(cutouts.keys())
    print(redshift_bins)
    
    if tag == 'horizontal':
        if 'All' in redshift_bins:
            rows = len(redshift_bins)+1;cols = 6;height = 5;width = 18
        else:
            rows = len(redshift_bins);cols = 6;height = 9;width = 18
        
    elif tag == 'vertical':
        rows = 6;cols = 3;height = 18;width = 9
    else:
        raise ValueError("Tag must be 'horizontal' or 'vertical'")
    
    
    if 'All' in redshift_bins:
        print('all')
        fig, axs = plt.subplots(rows, cols, figsize=(width, height),gridspec_kw={'height_ratios': [3, 1]},sharex=True)
    else:
        fig, axs = plt.subplots(rows, cols, figsize=(width, height))
       
        
        
#     plt.subplots_adjust(wspace=-0.25)
#     plt.subplots_adjust(hspace=0.1)
    
    for r,redshift_key in enumerate(redshift_bins):
        for f,freq in enumerate(freqs):
            int_val = float(freq.split('G')[0])
            if tag == 'horizontal':
                curr_cutout = cutouts[redshift_key][freq]*1e6
        
                im = axs[r,f].imshow(curr_cutout,cmap=cmap)
                
                axs[r,f].set_xticks([])
                axs[r,f].set_yticks([])
                axs[r,f].scatter(20,20,s=50,facecolor='None',edgecolor='white')
                if int_val > 200:
                    max_val = max(max(row) for row in curr_cutout)
                    position = [(ix, iy) for ix, row in enumerate(curr_cutout) for iy, val in enumerate(row) if val == max_val][0]
                else:
                    max_val = min(min(row) for row in curr_cutout)
                    position = [(ix, iy) for ix, row in enumerate(curr_cutout) for iy, val in enumerate(row) if val == max_val][0]

#                 axs[r,f].scatter(position[0],position[1],color='white',s=6)
                
                axs[0,f].set_title(freq)
                axs[r,0].set_ylabel(redshift_bins[r],fontsize=18)
                
                
                if int_val > 300:
                    bbox = axs[r,f].get_position()
                    axs[r,f].set_position([bbox.x0+0.02, bbox.y0, bbox.width, bbox.height])
                if add_colorbar:
                    if r == 0:
                        if f == 1 or f ==4:
                            bbox = axs[r,f].get_position()
                            cbar_ax = fig.add_axes([bbox.x0-0.1, bbox.y0+0.35, 0.3, 0.02]) 

                            c = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                            c.set_label('(arb)',fontsize=10)

            else:
                axs[f,r].imshow(cutouts[redshift_key][freq],cmap=cmap,vmin=vmin,vmax=vmax)
                axs[f,r].set_xticks([])
                axs[f,r].set_yticks([])
                
    beam_size = [1.9,1.2,1.1, 30/60,30/60,30/60]
    if 'All' in redshift_bins:
        for f,freq in enumerate(freqs):
            curr_cutout = cutouts[redshift_key][freq]*1e6
            xslice = curr_cutout[20,0:40]
            axs[1,f].plot(np.arange(40),xslice,color='slategrey')
#             axs[1,f].set_yticks([])
            axs[1,f].vlines(20,np.min(curr_cutout)*1.2,np.max(curr_cutout)*1.2,color='red',linestyle='dotted',alpha=0.2)
            if overplot_guassian:
                guass = gaussian_curve(np.arange(40),20,beam_size[f]/0.25)
                guass/=np.max(guass)
 
                axs[1,f].plot(np.arange(40),guass*xslice[20],color='#618768', linestyle='--' )
            
    fig.suptitle(title+'\n'+  num_objects +' objects',fontsize=17)
    
    plt.show()
    
def get_est_errors(vals,num_samples=1_000):
    k = 0
    curr_meds = []
    while k < num_samples:
        random_choice = np.random.choice(vals,size=int(len(vals)/1.2),replace=True)
        curr_meds.append(np.median(random_choice))
        k+=1
    return np.std(curr_meds)

def gaussian_curve(x, mean=0, fwhm=None):
    """
    This function calculates the Gaussian curve (Normal distribution) values for a given set of x,
    mean (μ), and full width at half maximum (FWHM).
    
    Parameters:
    x (array-like): Input values for which the Gaussian curve is to be calculated.
    mean (float, optional): The mean or expectation of the distribution (and also its median and mode).
                            Defaults to 0.
    fwhm (float, optional): The full width at half maximum, a measure of spread. If provided, it is used
                            to calculate the standard deviation, σ.
    
    Returns:
    array-like: Gaussian function values for each element in x.
    """
    import numpy as np
    # Convert FWHM to standard deviation, σ, if FWHM is provided
    if fwhm is not None:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    else:
        # Default value of sigma if FWHM is not given
        sigma = 1

    # Calculate the Gaussian function
    gaussian = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return gaussian

def create_zbinned_cutouts(curr_dic, final_maps,redshift_edges =[(0,0.5),(0.5,1),(1)],psize=0.25,ext=10,cent_type='tsz'):
    if cent_type != 'tSZ':
        print('using optical centers')
       
    full_cutouts = {}
    med_imgs = {}
    redshift_bins = []
    if type(redshift_edges) == list:
        for i in range(len(redshift_edges)):
            curr_bin = redshift_edges[i]
            try:
                redshift_bins.append(str(curr_bin[0])+'-'+str(curr_bin[1]))  
            except:
                redshift_bins.append(str(curr_bin)+'+')
    else:
        redshift_bins.append('All')
    
    
    for redshift in redshift_bins:
        full_cutouts[redshift] = {}
        med_imgs[redshift] = {}

        for freq in final_maps:
            full_cutouts[redshift][freq] = {}
            med_imgs[redshift][freq] = {}
            
    if 'All' in redshift_bins:
        if cent_type == 'tSZ':
            xcoords = [round(curr_dic[clu]['x']) for clu in curr_dic]
            ycoords = [round(curr_dic[clu]['y']) for clu in curr_dic]
        else:
            xcoords = [round(curr_dic[clu]['opt_x']) for clu in curr_dic if curr_dic[clu]['opt_x'] < 5000]
            ycoords = [round(curr_dic[clu]['opt_y']) for clu in curr_dic if curr_dic[clu]['opt_x'] < 5000]
        
        for freq in final_maps:
            cutouts,med_img = cutout_and_med_stack(list(final_maps.keys()),final_maps,xcoords=xcoords,ycoords=ycoords,ext=ext,psize=psize)
            full_cutouts[redshift_bins[0]][freq] = cutouts[freq]
            med_imgs[redshift_bins[0]][freq] = med_img[freq]
    else:
        for i,redshift in enumerate(redshift_edges):
            xcoords = []
            ycoords = []
            if i == len(redshift_edges)-1:
                for clu in curr_dic:
                    if curr_dic[clu]['REDSHIFT'] > redshift:
                        if cent_type == 'tSZ':
                            xcoords.append(curr_dic[clu]['x'])
                            ycoords.append(curr_dic[clu]['y'])
                        else:
                            xcoords.append(curr_dic[clu]['opt_x'])
                            ycoords.append(curr_dic[clu]['opt_y'])
            else:
                for clu in curr_dic:
                    if redshift[0] < curr_dic[clu]['REDSHIFT'] < redshift[1]:
                        if cent_type == 'tSZ':
                            xcoords.append(curr_dic[clu]['x'])
                            ycoords.append(curr_dic[clu]['y'])
                        else:
                            xcoords.append(curr_dic[clu]['opt_x'])
                            ycoords.append(curr_dic[clu]['opt_y'])

            for freq in final_maps:
                full_cutouts[redshift_bins[i]][freq] = cutouts[freq]
                med_imgs[redshift_bins[i]][freq] = med_img[freq]
            
    return full_cutouts,med_imgs
    
def add_redshift_bins(cat):
    for key in cat:
        curr_redshift = cat[key]['REDSHIFT']
        
        if 0 < curr_redshift < 0.5:
            cat[key]['redshift_bin'] = '0-0.5'
        elif 0.5 < curr_redshift < 1:
            cat[key]['redshift_bin'] = '0.5-1'
        elif curr_redshift > 1:
            cat[key]['redshift_bin'] = '1+'
        else:
            cat[key]['redshift_bin'] = 'None'
    return cat

def load_cf_cat_into_dic(map_obj,path,file_name):
    table = table_to_dic(clu.pkl(path+file_name))
    table = add_rounded_xy(cf,table)
    table = change_dic_key(table, ['xvals','yvals'],['x','y'])
    return table

def add_rounded_xy(map_obj,cat):
    for key in cat:
        ra = cat[key]['RA']*core.G3Units.deg
        dec = cat[key]['DEC']*core.G3Units.deg
        x,y = spt3g.maps.FlatSkyMap.angle_to_xy(map_obj, ra,dec)
        cat[key]['x'] = round(x)
        cat[key]['y'] = round(y)
    return cat
        
def import_cluster_catalog(what_cat):
    """
    imports cluster catalogs. Arguements include:
    'pol_100d', 'pol_500d', 'megadeep', 'josh3g','act'
    """
    
    path = '/home/kaylank/imports/catalogs/cluster_catalogs/'
    cat_dir = '/Users/lbleem/Desktop/python_notebooks/sptpol_500d_plots/cluster_catalogs/'
    
    if what_cat == 'pol_500d':
        sptpol500d_file = path+'sptpol500d_March2223-Copy1.sav'
        spt500d_data = readsav(sptpol500d_file, python_dict=True)['spt']
        dic = recarray_to_nested_dict(spt500d_data)
        
    elif what_cat == 'pol_100d':
        hdu = fits.open(path+'sptpol100d_catalog_huang19.fits')[1]
        dic = fitstable_to_dict(hdu)
        
        
    elif what_cat =='josh3g':
        file_path = '/sptlocal/user/joshuasobrin/clusters/2022-06_thesis_analysis/cluster_outputs/merged_cluster_list_June0622.sav'
        spt3g_data = readsav('/sptlocal/user/joshuasobrin/clusters/2022-06_thesis_analysis/cluster_outputs/merged_cluster_list_June0622.sav')['spt']
        dic = recarray_to_nested_dict(spt3g_data)
        
    elif what_cat =='megadeep':
#         file_path = path+'Min_Var_List_9_22_full_redshifts_v1.txt'
#         cluster_redshift_list = pd.read_csv(file_path,delim_whitespace=True)
        dic = clu.pkl('/home/kaylank/imports/catalogs/cluster_catalogs/megadeep_w_mass.pkl')
        dic = add_tcs(dic)
        
    elif what_cat == 'act':
        hdu = fits.open(path+ 'hilton_DR5_cluster-catalog_v1.1.fits')[1]
        dic = fitstable_to_dict(hdu)
    elif what_cat == 'sptsz':
        hdu = fits.open(path+ '2500d_cluster_sample_Bocquet19.fits')[1]
        dic = fitstable_to_dict(hdu)
    elif what_cat == 'sptecs':
        hdu = fits.open(path+ 'sptecs_catalog_oct919.fits')[1]
        dic = fitstable_to_dict(hdu)
    elif what_cat == 'planck':
        hdu = fits.open(path+ 'HFI_PCCS_SZ-union_R2.08.fits')[1]
        dic = fitstable_to_dict(hdu)
        
    dic_keys = list(dic['0'].keys())
    for key in dic_keys:
        if 'redshift' == key:
            dic = change_dic_key(dic, [key], ['REDSHIFT'])
        if key == 'M500c' or key == 'MSZ' or key == 'SPT3G_M500_SZ':
            dic = change_dic_key(dic, [key], ['M500'])
        if key == 'Dec' or key == 'SPT_DEC' or key == 'decDeg':
            dic = change_dic_key(dic, [key], ['DEC'])
        if key == 'SPT_RA' or key == 'RADeg':
            dic = change_dic_key(dic, [key], ['RA'])
            
            
            
#         if 'SPT_RA' in key or 'RADeg' in key:
#             dic = change_dic_key(dic, [key], ['RA'])
#         if 'DEC' in key or 'dec' in key or 'Dec' in key:
#             dic = change_dic_key(dic, [key], ['DEC'])
#         if 'MSZ' in key or 'M500c' in key or 'SPT3G_M500_SZ' in key:
#             dic = change_dic_key(dic, [key], ['M500'])
#         if 'redshift' == key:
#             dic = change_dic_key(dic, [key], ['REDSHIFT'])
            
            
            
    return dic
    
def import_source_catalog(what_cat):
    path = '/home/kaylank/imports/catalogs/pt_src_catalogs/'
    if what_cat =='PSW':
        curr_list = pd.read_csv(path+'spsc_standard_250_v2.csv')
        print('Filtering values with 0 > Dec > -30  and RA > 330 Change if not wanted')
        condition = (curr_list['DEC'] < -30) & (curr_list['DEC'] < 0) & (curr_list['RA'] > 330)
        filtered_list = curr_list[condition]
        
        filtered_list.reset_index(drop=True, inplace=True)
        dic = table_to_dic(filtered_list)
        
        return dic
    elif what_cat =='mel':
        pts_file = '/home/ndhuang/spt_code/spt3g_software/sources/python/main_lists/spt3g_1500d_source_list_ma_19sept23_v4.txt'
        pts_list = pd.read_csv(pts_file, comment='#', delim_whitespace=True)
        for i in range(len(pts_list)):
            if pts_list['RA(deg)'][i] < 0:
                pts_list.loc[i, 'RA(deg)'] = pts_list.loc[i, 'RA(deg)'] + 360
        dic = table_to_dic(pts_list)
        
    dic_keys = list(dic['0'].keys())
    for key in dic_keys:
        if 'RA(deg)' in key:
            dic = change_dic_key(dic, [key], ['RA'])
        if 'DEC' in key or 'dec' in key or 'Dec' in key:
            dic = change_dic_key(dic, [key], ['DEC'])
            
    return dic
            
def table_to_dic(curr_list):
    keys = list(curr_list.keys())
    length = len(curr_list[keys[0]])
    print(length)
    dic = {}
    for i in range(length):
        dic[str(i)] = {}
        for j,key in enumerate(keys):
            if key!= 'comments' and key!='tcs' and key!='core_size':
                dic[str(i)][key] = curr_list[key][i]
    return dic
          
def recarray_to_nested_dict(recarray):
    def convert_row(row):
        if isinstance(row, np.void):  # Check if it's a nested record array
            return {key: convert_row(row[key]) for key in row.dtype.names}
        else:
            return row
    temp =  [convert_row(row) for row in recarray]
    nested_dic = {}
    for i in range(len(temp)):
        nested_dic[str(i)] = temp[i]
    return nested_dic
        
def import_single_cluster_runs():
    spt_220_only = clu.pkl(path+'spt220GHz_only_full_cluster_list.pkl')
    spt_150_only = clu.pkl(path+'spt150GHz_only_full_cluster_list.pkl')
    spt_90_only = clu.pkl(path+'spt90GHz_only_full_cluster_list.pkl')
    spt_90_220 = clu.pkl(path+'spt90GHz_220GHz_full_cluster_list.pkl')
    spt_150_220 = clu.pkl(path+'spt150GHz_220GHz_full_cluster_list.pkl')
    
    return 

def clean_cat(input_cat):
    cleaned_cat = {}
    matches = []
    mega_matches = []
    for cluster in cat['megadeep']:
        megax = cat['megadeep'][cluster]['x']
        megay = cat['megadeep'][cluster]['y']
        
        for i in range(len(input_cat['xvals'])):
            catx = input_cat['xvals'][i]
            caty = input_cat['yvals'][i]
            
            if abs(megax - catx) < 3 and abs(megay - caty) < 3:
                cleaned_cat[str(i)] = {}
                matches.append(i)
                mega_matches.append(cluster)
                
    for i in matches:
        cleaned_cat[str(i)]['x'] = input_cat['xvals'][i]
        cleaned_cat[str(i)]['y'] = input_cat['yvals'][i]
        cleaned_cat[str(i)]['RA'] = input_cat['RA'][i]
        cleaned_cat[str(i)]['DEC'] = input_cat['DEC'][i]
        cleaned_cat[str(i)]['CAND_NAME'] = cat['megadeep'][cluster]['CAND_NAME']
        
    return cleaned_cat  

def create_cutout_map(map_obj,ra,dec,ext,psize=0.25):
    cutouts = []
    for i in range(len(ra)):
        cutout_template = maps.FlatSkyMap((ext*4), #nx
                                          (ext*4), #ny
                                          (psize/4)*core.G3Units.arcmin, 
                                          proj=maps.MapProjection.Proj0,
                                          alpha_center=ra[i]*core.G3Units.deg,
                                          delta_center=dec[i]*core.G3Units.deg)

        maps.reproj_map(map_obj,cutout_template, rebin=4)
        cutout_template = np.asarray(cutout_template)
        cutouts.append(cutout_template)

    return cutouts

def cutout_and_med_stack(keys,what_maps,xcoords=None,ycoords=None,ra_list=None,dec_list=None,ext=10,normalize=False,psize=0.25,ra=352.5,dec=-55,complex_cutout=False):
    if complex_cutout:
        cutouts = {}
        for freq in keys:
            temp_obj = maps.FlatSkyMap(np.asarray(what_maps[freq],order='C'),
                                          (0.25)*core.G3Units.arcmin, 
                                          proj=maps.MapProjection.Proj0,
                                          alpha_center=ra*core.G3Units.deg,
                                          delta_center=dec*core.G3Units.deg)

            cutouts[freq] = create_cutout_map(temp_obj,ra_list,dec_list,ext,psize)
        
        med_img = {}
        for freq in keys:
            med_img[freq] = make_med_stack_img(cutouts[freq],normalize=normalize)
    else:
        cutouts = {}
        med_img = {}
        for freq in keys:
            cutouts[freq] = get_cutouts(what_maps[freq], xcoords,ycoords,ext=20)
            med_img[freq] = make_med_stack_img(cutouts[freq])
    
    return cutouts, med_img

def filter_bounds(dic,ra_bounds, dec_bounds):
    filtered_dic = {}
    pop_indicies = []
    for key in dic:
        if (ra_bounds[0] < dic[key]['RA'] < ra_bounds[1]) and (dec_bounds[0] < dic[key]['DEC'] < dec_bounds[1]):
            filtered_dic[key] = dic[key]
    return filtered_dic
        
def get_cutouts(input_map,xcoords,ycoords,ext):
    input_map = np.asarray(input_map)
    cutouts = []
    for i in range(len(xcoords)):
        x = xcoords[i]
        y = ycoords[i]
        cutouts.append(input_map[y-ext:y+ext, x-ext:x+ext])
    return cutouts

def fitstable_to_dict(bintable):
    # Convert the FITS header to a dictionary
    print(':D')
    header_dict = dict(bintable.header)

    # Use dictionary comprehension to convert column data to lists
    data_dict = {colname: bintable.data[colname].tolist() for colname in bintable.columns.names}

    # Initialize the final dictionary
    final_dic = {}

    # Assuming all columns have the same length, use the first column to determine the loop range
    num_rows = len(data_dict[next(iter(data_dict))])

    # Build the final dictionary using dictionary comprehension within a loop
    for i in range(num_rows):
        final_dic[str(i)] = {key: data_dict[key][i] for key in data_dict}

    # Return a dictionary containing both the header and the data
    return {'header': header_dict, 'data': final_dic}

            
def calc_distance(ra1,ra2,dec1,dec2,units='deg'):
    distance_deg = np.sqrt( (ra1-ra2)**2 + (dec1-dec2)**2)
    
    if units =='deg' or units =='pixels':
        return distance_deg
    elif units =='arcmin':
        return distance_deg*60
    elif units == 'arcsecs':
        return distance_deg*3600
    
def change_dic_key(nested_dict, old_keys, new_keys):
    if len(old_keys) != len(new_keys):
        raise ValueError("The number of old keys must match the number of new keys.")

    for outer_key, inner_dict in nested_dict.items():
        if isinstance(inner_dict, dict):
            for old_key, new_key in zip(old_keys, new_keys):
                if old_key in inner_dict:
                    inner_dict[new_key] = inner_dict.pop(old_key)
    return nested_dict

def create_image_gallery(image_list,title,cutout_titles='None'):
    num_images = len(image_list)

    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    axes = axes.flatten() if num_images > 1 else [axes]
    for i, img in enumerate(image_list):
        axes[i].imshow(img, cmap=cmap)
        axes[i].scatter(20,20,s=2,color='white' )
        axes[i].axis('off')  # Hide the axis
        if type(cutout_titles)!=str:
            axes[i].set_title(cutout_titles[i])

    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    fig.suptitle(title,fontsize=20)
    plt.tight_layout()
    plt.show()
    return
 
def make_med_stack_img(array,normalize=False):
    med_img = np.median(np.dstack(array), -1)
    if normalize:
        med_img/=np.max(med_img)
    return med_img

def import_atca_dic():
    atca_file = readsav('/home/kaylank/imports/catalogs/pt_src_catalogs/at20g.sav')['at20g']
    at20g = {}
    for i in range(len(atca_file)):
        at20g[str(i)] = {}

    keys = ['RA','DEC','FLUX','DFLUX']

    for i in range(len(atca_file)):
        for j in range(len(atca_file[0])):
            at20g[str(i)][keys[j]] = atca_file[i][j]
    return at20g

def plot_stacks(med_img):
    n_cols = len(med_img)
    if n_cols < 5:
        fig, axs = plt.subplots(2, n_cols, figsize=(12, 9), gridspec_kw={'height_ratios': (14, 1)},sharex=True)
    else:
        fig, axs = plt.subplots(2, n_cols, figsize=(15, 9), gridspec_kw={'height_ratios': (14, 1)},sharex=True)
        
    for i, freq in enumerate(med_img):
        # Get the image for the current frequency
        image = med_img[freq]

        # Display the image in the first row
        axs[0, i].imshow(image, cmap=cmap)
        axs[0, i].set_yticks([])

        # Extract a horizontal slice at x=20 (or any other fixed x-coordinate)
        x_med = round(image.shape[1] / 2)
        slice_x = image[x_med, :]

        # Plot the horizontal slice in the third row
        axs[1, i].plot(slice_x, color='slategrey')
        axs[1, i].vlines(x_med, np.min(slice_x), np.max(slice_x), color='red', linestyle='--')
        axs[1, i].set_yticks([])
        axs[1,0].set_ylabel('xslice',fontsize=15)
        # Set the title for each column
        axs[0, i].set_title(freq)
        axs[0,i].scatter(x_med,x_med, s=20,edgecolor='white',facecolor='None',linewidth=2)
    axs[0, 0].set_ylabel("<-- 5' (arcmin) -->", fontsize=12)
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=-0.98)
    plt.tight_layout()
    plt.show()

    return

def array_to_map_obj(input_array,ra_cent=352.5,dec_cent=-55,psize=0.25):
    
    
    map_obj = maps.FlatSkyMap(input_array,
                      res=psize*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(ra_cent)*core.G3Units.deg,
                      delta_center=(dec_cent)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)
    
    return map_obj
    
    
def plot_cross_match(dic1,dic2,match1,match2,units='deg',title='None'):
    plt.rcParams['axes.linewidth'] = 1.5
    fig, axs = plt.subplots(2,2,figsize=(15,7))
 
    ra = []
    ra_diff = []
    dec = []
    dec_diff = []
    
    for key1, key2 in zip(match1, match2):
        ra_diff.append( (dic1[key1]['RA'] - dic2[key2]['RA']) * 3600)
        dec_diff.append( (dic1[key1]['DEC'] - dic2[key2]['DEC']) * 3600)
        ra.append(dic1[key1]['RA'])
        dec.append(dic1[key1]['DEC'])

        
    fig.suptitle(title,fontsize=17)
    axs[0,0].scatter(ra,ra_diff,color='slategrey')
    axs[0,0].hlines(0,345,360,color='red',linestyle='--')
    axs[0,0].set_ylim(-40,40)

    axs[0,1].scatter(dec,dec_diff,color='slategrey')
    axs[0,1].hlines(0,-48,-60,color='red',linestyle='--')

    axs[1,0].hist(ra_diff,color='slategrey')
    axs[1,1].hist(dec_diff,color='slategrey')
    axs[1,0].vlines(np.median(ra_diff),0,20,color='red',linestyle='--',label=str(round(np.median(ra_diff),2))+'"')
    axs[1,1].vlines(np.median(dec_diff),0,20,color='red',linestyle='--',label=str(round(np.median(dec_diff),2))+'"')
    axs[0,0].set_ylabel('RA Offset (")',fontsize=15)
    axs[0,1].set_ylabel('DEC Offset (")',fontsize=15)


    

    axs[1,0].legend(fontsize=15)
    axs[1,1].legend(fontsize=15)
    
    axs[0,0].set_xlabel('RA')
    
    axs[0,1].set_xlabel('DEC')

    ra_data = [['Median: '+str(round(np.median(ra_diff),2))],
            ['Mean: '+str(round(np.mean(ra_diff),2))],
            ['RMS: ' +str(round(rms(ra_diff),2))],
            ['Num Matches: '+str(len(ra_diff))]]
    
    ra_table = axs[0,0].table(cellText=ra_data, colLabels=['RA (")'],
                     cellLoc='center', loc='top')
    ra_table.set_fontsize(13)
    ra_table.scale(0.4, 1.2) 

    dec_data = [['Median: '+str(round(np.median(dec_diff),2))],
            ['Mean: '+str(round(np.mean(dec_diff),2))],
            ['RMS: ' +str(round(rms(dec_diff),2))]]

    dec_table = axs[0,1].table(cellText=dec_data, colLabels=['DEC (")'], cellLoc='center', loc='top')
    dec_table.set_fontsize(13)
    dec_table.scale(0.4, 1.2) 

    return 


def cross_match(dic1,dic2,match_radius,units='deg',ra_bounds=None,dec_bounds=None):
    final_matches = []
    for key in dic1:
        ra1 = dic1[key]['RA']
        dec1 = dic1[key]['DEC']
        curr_match_dist = 10000
        
        matches = []
        for key2 in dic2:
            ra2 = dic2[key2]['RA']
            dec2 = dic2[key2]['DEC']
            try:
                if ra_bounds[0] < ra2 < ra_bounds[1] and dec_bounds[0] < ra2 < dec_bounds[1]:
                    dist = calc_distance(ra1,ra2,dec1,dec2,units=units)
                    if dist < match_radius:
                        curr_match_dist = dist
                        final_matches.append((key,key2))
            except:
                dist = calc_distance(ra1,ra2,dec1,dec2,units=units)
                if dist < match_radius:
                    curr_match_dist = dist
                    final_matches.append((key,key2))
                
    
    return final_matches
    
def rms(values):
    values = np.array(values)
    
    # Square all values
    squared_values = values ** 2
    
    # Calculate the mean of the squared values
    mean_of_squares = squared_values.mean()
    
    # Take the square root of the mean
    rms = np.sqrt(mean_of_squares)
    
    return rms

def distance(ra_1,dec_1,ra_2,dec_2):
    return np.sqrt( (ra_1-ra_2)**2 + (dec_1 - dec_2)**2 )