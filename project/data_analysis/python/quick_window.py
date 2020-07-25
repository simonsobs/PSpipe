from pspy import so_map, so_window, so_dict, pspy_utils
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


apod_pts_degree = 0.3
apod_survey_degree = 2
skip_from_edges_degree = 1

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
            
        survey_mask.data *= mask_gal.data
        dist = so_window.get_distance(survey_mask, rmax = apod_survey_degree*np.pi/180)
        survey_mask.data[dist.data < skip_from_edges_degree] = 0

        survey_mask = so_window.create_apodization(survey_mask, "C1", apod_survey_degree, use_rmax=True)
        survey_mask.data *= mask.data

        survey_mask.write_map("%s/window_%s_%s.fits" % (window_dir, sv, ar))
        survey_mask = survey_mask.downgrade(4)
        survey_mask.plot(file_name="%s/window_%s_%s" % (window_dir, sv, ar))

