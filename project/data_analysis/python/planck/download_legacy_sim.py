import wget
import os

freq = [100, 143, 217]
hms = ["hm1", "hm2"]
nsims = 300

for iii in range(nsims):
    for f in freq:
        for hm in hms:
            os.system(f"wget -O legacy_noise_sim_{f}_{hm}_{iii:05d}.fits http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_{f}_{hm}_map_mc_{iii:05d}.fits")

