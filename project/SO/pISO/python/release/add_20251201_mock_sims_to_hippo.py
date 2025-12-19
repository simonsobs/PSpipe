from os.path import join as opj
from henry import Henry
import hippometa as hm 

client = Henry()
sim_dir = '/scratch/gpfs/SIMONSOBS/users/zatkins/projects/lat-iso/piso_old/tiger_deep56_sim/sim_factory/sims/maps'

# collection layers:
# Mock LAT ISO-SV1 Sims
# - Mock LAT ISO-SV1 (1000x) Sims
#   - Mock LAT ISO-SV1 (1000x) tube freq Sims
#     - Mock LAT Deep56 Signal Map (1000x) tube freq {(Tags)}
#     - Mock LAT Deep56 Noise Map (1000x) tube freq (Set y)

top_collection = client.new_collection(
    name='Mock LAT ISO-SV1 Sims',
    description='Mock simulations of the LAT-ISO SV1 data. Each simulation ' \
    'includes the four MF tubes i1, i3, i4, and i6, and the two UHF tubes c1 ' \
    'and i5. Each tube and frequency band includes variations of signal maps ' \
    '(with/without random systematics, with/without map-level lensing) and ' \
    'four-way split noise maps.\n\n' \
    'See [this page](https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/1452802049/LAT+ISO+Mock+Simulations) ' \
    'for more technical information.',
    products=[],
)

for sim_num in range(10):
    sim_collection = client.new_collection(
        name=f'Mock LAT ISO-SV1 (1000{sim_num}) Sims',
        description='Mock simulations of the LAT-ISO SV1 data for sim index ' \
        f'1000{sim_num}. Each simulation includes the four MF tubes i1, i3, ' \
        'i4, and i6, and the two UHF tubes c1 and i5. Each tube and frequency ' \
        'band includes variations of signal maps (with/without random ' \
        'systematics, with/without map-level lensing) and four-way split noise' \
        'maps.\n\n' \
        'See [this page](https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/1452802049/LAT+ISO+Mock+Simulations) ' \
        'for more technical information.',
        products=[],
    )

    for mapname in ['i1_f090', 'i1_f150', 'i3_f090', 'i3_f150', 'i4_f090', 
                    'i4_f150', 'i6_f090', 'i6_f150', 'c1_f220', 'c1_f280', 
                    'i5_f220', 'i5_f280']:
        tube, freq = mapname.split('_')
        tube_freq_collection = client.new_collection(
            name=f'Mock LAT ISO-SV1 (1000{sim_num}) {tube} {freq} Sims',
            description='Mock simulations of the LAT-ISO SV1 data for sim index ' \
            f'1000{sim_num}, {tube} {freq} only. Includes variations of signal ' \
            'maps (with/without random systematics, with/without map-level ' \
            'lensing) and four-way split noise maps.\n\n' \
            'See [this page](https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/1452802049/LAT+ISO+Mock+Simulations) ' \
            'for more technical information.',
            products=[],
        )

        # signal maps
        for tag in ['_', '_syst_', '_lens_', '_syst_lens_']:
            map_file = f'signal_sim_map{tag}LAT_{tube}_{freq}_1000{sim_num}.fits'

            if map_file == 'signal_sim_map_syst_LAT_i1_f150_10009.fits':
                signal_map_set = client.pull_product('69387f74f22a066989fc54e3', realize_sources=False)
            else:
                nametag = {'_': '',
                        '_syst_': ' (Systematics)',
                        '_lens_': ' (Map-level Lensing)',
                        '_syst_lens_': ' (Systematics, Map-level Lensing)'
                        }[tag]
                
                desctag = {'_': '',
                        '_syst_': ' Includes a random systematics realization.',
                        '_lens_': ' Includes map-level lensing with lenspyx.',
                        '_syst_lens_': ' Includes a random systematics realization and map-level lensing with lenspyx.'
                        }[tag]

                name = f'Mock LAT Deep56 Signal Map (1000{sim_num}) {tube} {freq}{nametag}'
                desc = f'Mock signal simulation of the LAT-ISO SV1 data, for ' \
                f'{tube} {freq}. The index of this simulation is 1000{sim_num}. ' \
                f'Same map for each of the four-way splits.{desctag} See ' \
                '[this page](https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/1452802049/LAT+ISO+Mock+Simulations) ' \
                'for more technical information.'

                signal_map_set = client.new_product(
                    name=name,
                    description=desc,
                    metadata=hm.MapSet(telescope='lat', instrument=tube,
                                    frequency=freq, release='lat-iso-sv1',
                                    patch='deep56',
                                    polarization_convention='COSMO',
                                    pixelisation='cartesian'),
                    sources={'map': {'path': opj(sim_dir, map_file),
                                    'description': 'the simulated map'}}
                                    )

            tube_freq_collection.append(signal_map_set)

        # noise maps
        for k in range(4):
            map_file = f'noise_sim_map_LAT_{tube}_{freq}_set{k}_1000{sim_num}.fits'

            if map_file == 'noise_sim_map_LAT_i1_f150_set0_10009.fits':
                noise_map_set = client.pull_product('69387ef3f22a066989fc54dd', realize_sources=False)
            else:
                desctag = {0: ' first',
                        1: ' second',
                        2: ' third',
                        3: ' fourth'
                        }[k]

                name = f'Mock LAT Deep56 Noise Map (1000{sim_num}) {tube} {freq} (Set {k})'
                desc = f'Mock noise simulation of the LAT-ISO SV1 data, for ' \
                f'{tube} {freq},{desctag} split of the four-way split. The index ' \
                f'of this simulation is 1000{sim_num}. See ' \
                '[this page](https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/1452802049/LAT+ISO+Mock+Simulations) ' \
                'for more technical information.'

                noise_map_set = client.new_product(
                    name=name,
                    description=desc,
                    metadata=hm.MapSet(telescope='lat', instrument=tube,
                                    frequency=freq, release='lat-iso-sv1',
                                    patch='deep56',
                                    polarization_convention='COSMO',
                                    pixelisation='cartesian', split=k),
                    sources={'map': {'path': opj(sim_dir, map_file),
                                    'description': 'the simulated map'}}
                                    )

            tube_freq_collection.append(noise_map_set)
        
        sim_collection.append(tube_freq_collection)
    
    top_collection.append(sim_collection)

client.push(top_collection)