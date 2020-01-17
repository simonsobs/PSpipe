from pspy import so_map

mask_gal = so_map.read_map('mask_equatorial_temperature_fsky0p70_freq145_ns4096.fits')
mask_survey = so_map.read_map('survey_mask_la_093.fits')

mask_gal_down = mask_gal.downgrade(4)
mask_survey_down = mask_survey.downgrade(4)

mask_gal_down.data[mask_gal_down.data != 1.]=0
mask_survey_down.data[mask_survey_down.data != 1.]=0

mask_gal_down.info()
mask_gal_down.plot()

mask_survey_down.info()
mask_survey_down.plot()

mask_gal_down.write_map('mask_equatorial_512.fits')
mask_survey_down.write_map('survey_mask_512.fits')
