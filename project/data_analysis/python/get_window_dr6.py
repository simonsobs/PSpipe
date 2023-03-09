"""
This script create the window functions used in the PS computation
They consist of a point source mask, a galactic mask and a mask based on the amount of cross linking in the data, we also produce a window that include the pixel weighting.
The different masks are apodized.
We also produce a binary mask that will later be used for the kspace filtering operation, in order to remove the edges and avoid nasty pixels before
this not so well defined Fourier operation.
"""

import sys

import numpy as np
from pspy import pspy_utils, so_dict, so_map, so_mpi, so_window
from pspipe_utils import pspipe_list
from pixell import enmap

import scipy
from scipy.ndimage import morphology

def create_crosslink_mask(xlink_map, cross_link_threshold):
    # remove pixels with very little amount of cross linking
    xlink = so_map.read_map(xlink_map)
    xlink_lowres = xlink.downgrade(32)
    with np.errstate(invalid="ignore"):
        x_mask = (np.sqrt(xlink_lowres.data[1] ** 2 + xlink_lowres.data[2] ** 2) / xlink_lowres.data[0])
    x_mask[np.isnan(x_mask)] = 1
    x_mask[x_mask >= cross_link_threshold] = 1
    x_mask[x_mask < cross_link_threshold] = 0
    x_mask = 1 - x_mask
    xlink_lowres.data[0] = x_mask
    xlink = so_map.car2car(xlink_lowres, xlink)
    x_mask = xlink.data[0].copy()
    id = np.where(x_mask > 0.9)
    x_mask[:] = 0
    x_mask[id] = 1
    return x_mask

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# the apodisation lenght of the point source mask in degree
apod_pts_source_degree = d["apod_pts_source_degree"]
# the apodisation lenght of the survey x gal x cross linking mask
apod_survey_degree = d["apod_survey_degree"]
# we will skip the edges of the survey where the noise is very difficult to model
skip_from_edges_degree = d["skip_from_edges_degree"]
# the threshold on the amount of cross linking to keep the data in
apply_cross_link_threshold = d["apply_cross_link_threshold"]

apply_dec_cut = d["apply_dec_cut"]

# pixel weight with inverse variance above n_ivar * median are set to ivar
# this ensure that the window is not totally dominated by few pixels with too much weight
n_med_ivar = d["n_med_ivar"]

window_dir = "windows"
surveys = d["surveys"]

pspy_utils.create_directory(window_dir)

patch = None
if "patch" in d:
    patch = so_map.read_map(d["patch"])

# here we list the different windows that need to be computed, we will then do a MPI loops over this list
n_wins, sv_list, ar_list = pspipe_list.get_arrays_list(d)

print(f"number of windows to compute : {n_wins}")
so_mpi.init(True)

subtasks = so_mpi.taskrange(imin=0, imax=n_wins - 1)
print(subtasks)
for task in subtasks:

    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
    
    # using same gal mask for everything
    #gal_mask = so_map.read_map(d[f"gal_mask_{sv}_{ar}"])
    gal_mask = so_map.read_map(d[f"gal_mask"])

    survey_mask = gal_mask.copy()
    survey_mask.data[:] = 1

    maps = d[f"maps_{sv}_{ar}"]
    
    ivar_all = gal_mask.copy()
    ivar_all.data[:] = 0

    for k, map in enumerate(maps):

        if d[f"src_free_maps_{sv}"] == True:
            index = map.find("map_srcfree.fits")
        else:
            index = map.find("map.fits")

        ivar_map = map[:index] + "ivar.fits"
        #print(ivar_map)
        ivar_map = so_map.read_map(ivar_map)
        survey_mask.data[ivar_map.data[:] == 0.0] = 0.0
        ivar_all.data[:] += ivar_map.data[:]

    ivar_all.data[:] /= np.max(ivar_all.data[:])

    for k, map in enumerate(maps):

        if d[f"src_free_maps_{sv}"] == True:
            index = map.find("map_srcfree.fits")
        else:
            index = map.find("map.fits")
        
        if d[f"apply_cross_link_threshold"] == True:
            cross_link_threshold = d["cross_link_threshold"]
            xlink_map = map[:index] + "xlink.fits"
            print(xlink_map)
            x_mask = create_crosslink_mask(xlink_map, cross_link_threshold)
            survey_mask.data *= x_mask
        
        if d[f"apply_dec_cut"] == True:
            skip_dec_high = d["skip_dec_high"]
            skip_dec_low = d["skip_dec_low"]
            dec_mask = gal_mask.copy()
            dec_mask.data[:] = 1
            dec_mask.data[((skip_dec_high*np.pi/180.)>dec_mask.data.posmap()[0]) &(dec_mask.data.posmap()[0]>(skip_dec_low*np.pi/180.))] = 0
            survey_mask.data *= dec_mask.data
            
    survey_mask.data *= gal_mask.data

    if patch is not None:
        survey_mask.data *= patch.data
    
    if sv[-5:] == 'north':
        survey_mask.data[(survey_mask.data.posmap()[0]<(survey_mask.data.posmap()[1]*(-23./18.)+(124.*np.pi/180.))) & (survey_mask.data.posmap()[0]<(survey_mask.data.posmap()[1]*(23./18.)+(103.5*np.pi/180.)))] = 0
    elif sv[-5:] == 'south':
        survey_mask.data[(survey_mask.data.posmap()[0]>(survey_mask.data.posmap()[1]*(-23./18.)+(124.*np.pi/180.))) | (survey_mask.data.posmap()[0]>(survey_mask.data.posmap()[1]*(23./18.)+(103.5*np.pi/180.)))] = 0
        
    # so here we create a binary mask this will only be used in order to skip the edges before applying the kspace filter
    # this step is a bit arbitrary and preliminary, more work to be done here
    
    if d[f"fill_holes_in_windows"] == True or d[f"fill_holes_in_maps"] == True:
        
        binary_with_holes = survey_mask.copy()
        binary_without_holes = survey_mask.copy()
        # get intial filled binary, where we fill the holes then skip some distance from the edges
        binary_fill = survey_mask.copy()
        binary_fill.data[:] = morphology.binary_fill_holes(binary_fill.data[:])
        
        dist_fill = so_window.get_distance(binary_fill, rmax=apod_survey_degree * np.pi / 180)
        
        binary_fill.data[dist_fill.data < skip_from_edges_degree / 2] = 0
        
        # we want to keep the outer edge from the non-filled binary 
        # this accomplishes that (even though it maybe looks a little weird)
        all_ones = survey_mask.copy()
        all_ones.data[:] = 1
        binary_without_holes.data *= (all_ones.data - binary_fill.data)
        binary_without_holes.data += binary_fill.data
        if d[f"fill_holes_in_windows"] == True:
            binary = binary_without_holes.copy()
        else:
            binary = survey_mask.copy()
        
        # these are the holes we have effectively filled (if True above) in the final window:
        coords_holes = np.where((binary_without_holes.data[:] == 1.0) & (binary_with_holes.data[:]==0.0))
        num_pix_hole = len(coords_holes[0])
        coords_holes_list = [[coords_holes[0][i],coords_holes[1][i]] for i in range(num_pix_hole)]
        
        print(f'window {sv} {ar}')
        print('number of holes: %s' % num_pix_hole)
        
        if d[f"output_hole_coords"] == True:
            output_hole_dir = d[f"output_hole_dir"]
            pspy_utils.create_directory(output_hole_dir)
            np.save(f"{output_hole_dir}/holes_window_{sv}_{ar}.npy", coords_holes_list)
        
        if d[f"fill_holes_in_maps"] == True:
            filled_map_dir = d[f"filled_map_dir"]
            pspy_utils.create_directory(filled_map_dir)
            
            for k, map in enumerate(maps):
                
                dir_index = map.rfind('/')
                map_name = map[dir_index+1:]
                
                map_map = so_map.read_map(map)
                
                map_map_filled = map_map.copy()
                
                # these are holes in the maps (so missing data in temperature and/or polarization) but not in the windows:
                coords_tholes = np.where((binary_without_holes.data[:] == 1.0) & (binary_with_holes.data[:]!=0.0) & (map_map.data[0,:,:]==0.0))
                coords_pholes = np.where((binary_without_holes.data[:] == 1.0) & (binary_with_holes.data[:]!=0.0) & (map_map.data[1,:,:]==0.0) & (map_map.data[2,:,:]==0.0))
                
                num_pix_thole = len(coords_tholes[0])
                num_pix_phole = len(coords_pholes[0])
                
                coords_tholes_list = [[coords_tholes[0][i],coords_tholes[1][i]] for i in range(num_pix_thole)]
                coords_pholes_list = [[coords_pholes[0][i],coords_pholes[1][i]] for i in range(num_pix_phole)]
                
                print(f'{map_name}')
                print('number of T holes: %s' % num_pix_thole)
                print('number of P holes: %s' % num_pix_phole)
                
                if d[f"output_hole_coords"] == True:
                    np.save(f"{output_hole_dir}/temp_holes_map_{sv}_{ar}_{k}.npy", coords_tholes_list)
                    np.save(f"{output_hole_dir}/pol_holes_map_{sv}_{ar}_{k}.npy", coords_pholes_list)
                
                # fill in holes, using only nearest 4 neighboring pixels 
                # (if all neighbors are 0, the pixel remains a hole, but this shouldn't happen for the maps we're interested in)
                for ch in coords_holes_list:
                    neighbor_pixs = [[ch[0]-1,ch[1]], [ch[0],ch[1]-1], [ch[0]+1,ch[1]], [ch[0],ch[1]+1]]
                    neighbor_sum = np.asarray([0.,0.,0.])
                    neighbor_num = np.asarray([0.,0.,0.])
                    for pp in neighbor_pixs:
                        if pp not in coords_holes_list:
                            if pp not in coords_tholes_list:
                                 neighbor_num[0] += 1.
                                 neighbor_sum[0] += map_map.data[0,pp[0],pp[1]]
                            if pp not in coords_pholes_list:
                                 neighbor_num[1] += 1.
                                 neighbor_sum[1] += map_map.data[1,pp[0],pp[1]]
                                 neighbor_num[2] += 1.
                                 neighbor_sum[2] += map_map.data[2,pp[0],pp[1]]
                    neighbor_avg = neighbor_sum/neighbor_num
                    map_map_filled.data[:,ch[0],ch[1]] = neighbor_avg
                
                for ch in coords_tholes_list:
                    neighbor_pixs = [[ch[0]-1,ch[1]], [ch[0],ch[1]-1], [ch[0]+1,ch[1]], [ch[0],ch[1]+1]]
                    neighbor_sum = 0.
                    neighbor_num = 0.
                    for pp in neighbor_pixs:
                        if pp not in coords_holes_list:
                            if pp not in coords_tholes_list:
                                 neighbor_num += 1.
                                 neighbor_sum += map_map.data[0,pp[0],pp[1]]
                    neighbor_avg = neighbor_sum/neighbor_num
                    map_map_filled.data[0,ch[0],ch[1]] = neighbor_avg
                    
                for ch in coords_pholes_list:
                    neighbor_pixs = [[ch[0]-1,ch[1]], [ch[0],ch[1]-1], [ch[0]+1,ch[1]], [ch[0],ch[1]+1]]
                    neighbor_sum = np.asarray([0.,0.])
                    neighbor_num = 0.
                    for pp in neighbor_pixs:
                        if pp not in coords_holes_list:
                            if pp not in coords_pholes_list:
                                 neighbor_num += 1.
                                 neighbor_sum += map_map.data[1:,pp[0],pp[1]]
                    neighbor_avg = neighbor_sum/neighbor_num
                    map_map_filled.data[1:,ch[0],ch[1]] = neighbor_avg
                
                map_map_filled.write_map(filled_map_dir + map_name)
        
    else:
        binary = survey_mask.copy()
    
    dist = so_window.get_distance(binary, rmax=apod_survey_degree * np.pi / 180)
    # Note that we don't skip the edges as much for the binary mask
    # compared to what we will do with the final window, this should prevent some aliasing from the kspace filter to enter the data
    binary.data[dist.data < skip_from_edges_degree / 2] = 0
    
    binary.data = binary.data.astype(np.float32)
    binary.write_map(f"{window_dir}/binary_{sv}_{ar}.fits")

    # Now we create the final window function that will be used in the analysis
    survey_mask.data[dist.data < skip_from_edges_degree] = 0
    survey_mask = so_window.create_apodization(survey_mask, "C1", apod_survey_degree, use_rmax=True)
    # using same ps mask for everything
    #ps_mask = so_map.read_map(d[f"ps_mask_{sv}_{ar}"])
    ps_mask = so_map.read_map(d[f"ps_mask"])
    ps_mask = so_window.create_apodization(ps_mask, "C1", apod_pts_source_degree, use_rmax=True)
    survey_mask.data *= ps_mask.data
    
    survey_mask.data = survey_mask.data.astype(np.float32)
    survey_mask.write_map(f"{window_dir}/window_{sv}_{ar}.fits")
    
    # We also create an optional window which also include pixel weighting
    # Note that with use the threshold n_ivar * med so that pixels with very high
    # hit count do not dominate
    
    survey_mask_weighted = survey_mask.copy()
    id = np.where(ivar_all.data[:] * survey_mask.data[:] != 0)
    med = np.median(ivar_all.data[id])
    ivar_all.data[ivar_all.data[:] > n_med_ivar * med] = n_med_ivar * med
    
    if d[f"apply_ivar_smoothing"] == True:
        ivar_smoothing_arcmin = d[f"ivar_smoothing_arcmin"]
        ivar_all.data[:] = enmap.smooth_gauss(ivar_all.data[:], ivar_smoothing_arcmin*(1./60.)*(np.pi/180.))
    
    # does the ordering matter here? so smooth then sqrt vs sqrt them smooth? I don't think so...
    if d[f"take_sqrt_ivar"] == True:
        ivar_all.data[:] = np.sqrt(ivar_all.data[:])
    
    survey_mask_weighted.data[:] *= ivar_all.data[:]
    
    survey_mask_weighted.data = survey_mask_weighted.data.astype(np.float32)
    survey_mask_weighted.write_map(f"{window_dir}/window_w_{sv}_{ar}.fits")

    # plot
    binary = binary.downgrade(4)
    binary.plot(file_name=f"{window_dir}/binary_{sv}_{ar}")
    
    survey_mask = survey_mask.downgrade(4)
    survey_mask.plot(file_name=f"{window_dir}/window_{sv}_{ar}")

    survey_mask_weighted = survey_mask_weighted.downgrade(4)
    survey_mask_weighted.plot(file_name=f"{window_dir}/window_w_{sv}_{ar}")
