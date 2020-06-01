"""
This script generate the window functions for the different surveys and arrays
"""
from pspy import so_map, so_window, pspy_utils, so_dict
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir = "toy_windows"

pspy_utils.create_directory(window_dir)

surveys = d["surveys"]

print("Generating window functions")

for sv in surveys:

    arrays = d["arrays_%s" % sv]

    if d["pixel_%s" % sv] == "CAR":
        ra0, ra1, dec0, dec1 = d["ra0_%s" % sv], d["ra1_%s" % sv], d["dec0_%s" % sv], d["dec1_%s" % sv]
        car_box =  [[dec0, ra0], [dec1, ra1]]
        binary = so_map.read_map(d["template_%s" % sv], car_box=car_box)
                                     
        binary.data[:] = 1
        if d["binary_is_survey_mask_%s" % sv] == True:
            binary.data[:] = 0
            binary.data[1:-1, 1:-1] = 1
    
    elif d["pixel_%s" % sv] == "HEALPIX":
        binary=so_map.healpix_template(ncomp=1, nside=d["nside_%s" % sv])
        binary.data[:] = 1

    for ar in arrays:
        
        window = binary.copy()
        
        if d["galactic_mask_%s" % sv] == True:
            gal_mask = so_map.read_map(d["galactic_mask_%s_file_%s" % (sv, ar)], car_box=car_box)
            gal_mask.plot(file_name="%s/gal_mask_%s_%s" % (plot_dir, sv, ar))
            window.data[:] *= gal_mask.data[:]
        
        if d["survey_mask_%s" % sv] == True:
            survey_mask = so_map.read_map(d["survey_mask_%s_file_%s" % (sv, ar)], car_box=car_box)
            survey_mask.plot(file_name="%s/survey_mask_mask_%s_%s" % (plot_dir, sv, ar))
            window.data[:] *= survey_mask.data[:]

        apo_radius_degree = (d["apo_radius_survey_%s" % sv])
        window = so_window.create_apodization(window,
                                              apo_type=d["apo_type_survey_%s" % sv],
                                              apo_radius_degree=apo_radius_degree)

        if d["pts_source_mask_T_%s" % sv] == True:
            mask = so_map.read_map(d["pts_source_mask_file_%s_%s" % (sv, ar)], car_box=car_box)
            mask = so_window.create_apodization(mask,
                                                apo_type=d["apo_type_mask_%s" % sv],
                                                apo_radius_degree=d["apo_radius_mask_%s_%s" % (sv, ar)])
            win_T, win_pol = window.copy(), window.copy()
            win_T.data[:] *= mask.data[:]
            if d["also_mask_pol_%s" % sv]:
                win_pol.data[:] *= mask.data[:]

                                                
            win_T.write_map("%s/window_T_%s_%s.fits" % (window_dir, sv, ar))
            win_pol.write_map("%s/window_pol_%s_%s.fits" % (window_dir, sv, ar))

