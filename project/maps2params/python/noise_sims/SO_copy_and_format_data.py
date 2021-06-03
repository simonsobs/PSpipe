from pspy import so_dict, pspy_utils
import os, sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

split_list = d["split_list"]
scan_list = d["scan_list"]

repo = {}
repo["split0"] = "/global/cfs/cdirs/sobs/sims/scan-s0001"
repo["split1"] = "/global/cfs/cdirs/sobs/sims/scan-s0001/00000001"

pspy_utils.create_directory("hits_map")

for split in split_list:
    pspy_utils.create_directory(split)
    for scan in scan_list:
        for file in ["map", "bmap"]:
            original = "%s/%s_telescope_all_time_all_%s.fits" % (repo[split], scan, file)
            copy = "%s/%s_telescope_all_time_all_%s.fits" % (split, scan, file)
            
            print(original)
            print(copy)
            
            os.system("cp %s %s" % (original, copy))

        original = "%s/%s_telescope_all_time_all_hmap.fits" % (repo["split0"], scan)
        copy = "%s/%s_telescope_all_time_all_hmap.fits" % ("hits_map", scan)
        
        print(original)
        print(copy)

        os.system("cp %s %s" % (original, copy))
