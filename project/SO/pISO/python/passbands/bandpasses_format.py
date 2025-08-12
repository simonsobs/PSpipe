"""
Goes from LAT_MF_bands.csv to tube-wide bandpasses by averaging over the UFMs. The bandpasses measurements are from JackOS in #lat-analysis.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

cut_LF_noise = False
cut_edges = True

save_path = "passbands"

df_MF = pd.read_csv('passbands/LAT_MF_bands.csv')
frequency = df_MF['frequency']
# df_UHF = pd.read_csv('/pscratch/sd/m/merrydup/pipe0004_BN/bandpasses/LAT_UHF_bands.csv')

# https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/210894849/Instrument+Diagrams+for+Data+Reduction
UFMs_mapping = {
    'i1': ['mv24', 'mv28'],     # 'mv21' not in LAT_MF_bands.csv
    'i3': ['mv13', 'mv20', 'mv34'],
    'i4': ['mv14', 'mv32', 'mv49'],
    # 'i6': ['mv11', 'mv25', 'mv26'],   no bandpasses for i6 for now
}
all_ufms = ['mv24', 'mv28', 'mv13', 'mv20', 'mv34', 'mv14', 'mv32', 'mv49']

BANDS = ['090', '150']
band_edges = {
    '090': [70, 115],
    '150': [118, 178]
}

bandpasses = {}
bandpasses_mean = {}

for band in BANDS:
    for tube, ufms in UFMs_mapping.items():
        bandpasses[f'{tube}_f{band}'] = np.mean([df_MF[f'{ufm}_f{band}'] for ufm in ufms], axis=0)
        if cut_LF_noise:
            bandpasses[f'{tube}_f{band}'][:50] *= 0.
            bandpasses[f'{tube}_f{band}'][bandpasses[f'{tube}_f{band}'] < 0.015] *= 0.
        if cut_edges:
            band_mask = (frequency > band_edges[band][0]) & (frequency < band_edges[band][1])
            bandpasses[f'{tube}_f{band}'][~band_mask] *= 0.
            
    bandpasses_mean[f'mean_f{band}'] = np.mean([df_MF[f'{ufm}_f{band}'] for ufm in all_ufms], axis=0)
    if cut_edges:
        band_mask = (frequency > band_edges[band][0]) & (frequency < band_edges[band][1])
        bandpasses_mean[f'mean_f{band}'][~band_mask] *= 0.



fig, ax = plt.subplots()
for name, bp in bandpasses.items():
    np.savetxt(f'{save_path}/bandpass_{name}_ufm_average.dat', np.array([frequency, bp]).T, header="nu_ghz                   passband")
    ax.plot(frequency, bp, label=name)
for name, bp in bandpasses_mean.items():
    np.savetxt(f'{save_path}/bandpass_{name}.dat', np.array([frequency, bp]).T, header="nu_ghz                   passband")
    ax.plot(frequency, bp, label=name, color='black', linewidth=1.8)
ax.legend()
ax.set_xlabel('Frequency (GHz)', fontsize=15)
plt.savefig(f'{save_path}/bandpasses_ufm_average.png')
plt.close()

