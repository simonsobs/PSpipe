from pspy import so_map, so_window, so_dict, pspy_utils
import numpy as np
import sys

def mask_based_on_crosslink(xlink_map, cross_link_threshold):

    xlink = so_map.read_map(xlink_map)
    xlink_lowres = xlink.downgrade(32)
    x_mask = np.sqrt(xlink_lowres.data[1]**2 + xlink_lowres.data[2]**2) / xlink_lowres.data[0]
    x_mask[x_mask >= cross_link_threshold] = 1
    x_mask[x_mask < cross_link_threshold] = 0
    x_mask = 1 - x_mask
    xlink_lowres.data[0] = x_mask
    xlink = so_map.car2car(xlink_lowres, xlink)
    x_mask = xlink.data[0].copy()
    id = np.where(x_mask > 0.9)
    x_mask[:] = 0
    x_mask[id] = 1
    print(x_mask.shape)
    return x_mask



d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

apod_pts_degree = 0.3
apod_survey_degree = 2
skip_from_edges_degree = 1
cross_link_threshold = 0.97

data_dir = d["data_dir"]
window_dir = data_dir + "windows"
surveys = d["surveys"]

pspy_utils.create_directory(window_dir)
mask = so_map.read_map("%s/masks/act_dr4.01_mask_s13s16_0.100mJy_8.0arcmin.fits" % data_dir)
mask = so_window.create_apodization(mask, "C1", apod_pts_degree, use_rmax=True)
mask_gal= so_map.read_map("%s/masks/mask_galactic_equatorial_car_halfarcmin_pixelmatch.fits" % data_dir)

for sv in surveys:

    arrays = d["arrays_%s" % sv]
    
    for ar in arrays:
    
        survey_mask = mask_gal.copy()
        survey_mask.data[:] = 1
        
        maps = d["maps_%s_%s" % (sv, ar)]
        
        for k, map in enumerate(maps):
            print(map)
            map = so_map.read_map(map)
            survey_mask.data[map.data[0] == 0.] = 0.
            map.info()
        
        
        for k, map in enumerate(maps):
            index = map.find("map.fits")
            xlink_map = map[:index] + "xlink.fits"
            print(xlink_map)
            x_mask = mask_based_on_crosslink(xlink_map, cross_link_threshold)
            survey_mask.data *= x_mask
        
        survey_mask.data *= mask_gal.data
        dist = so_window.get_distance(survey_mask, rmax = apod_survey_degree*np.pi/180)
        survey_mask.data[dist.data < skip_from_edges_degree] = 0

        survey_mask = so_window.create_apodization(survey_mask, "C1", apod_survey_degree, use_rmax=True)
        survey_mask.data *= mask.data

        survey_mask.write_map("%s/window_%s_%s.fits" % (window_dir, sv, ar))
        survey_mask = survey_mask.downgrade(4)
        survey_mask.plot(file_name="%s/window_%s_%s" % (window_dir, sv, ar))

