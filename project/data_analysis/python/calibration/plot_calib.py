import pickle
import pylab as plt
import numpy as np

name = {}

name["AxA-AxP"] = ["dr6_pa4_f220xdr6_pa4_f220-dr6_pa4_f220xPlanck_f217",
                   "dr6_pa5_f090xdr6_pa5_f090-dr6_pa5_f090xPlanck_f100",
                   "dr6_pa5_f150xdr6_pa5_f150-dr6_pa5_f150xPlanck_f143",
                   "dr6_pa6_f090xdr6_pa6_f090-dr6_pa6_f090xPlanck_f100",
                   "dr6_pa6_f150xdr6_pa6_f150-dr6_pa6_f150xPlanck_f143"]

name["AxA-PxP"] = ["dr6_pa4_f220xdr6_pa4_f220-Planck_f217xPlanck_f217",
                   "dr6_pa5_f090xdr6_pa5_f090-Planck_f100xPlanck_f100",
                   "dr6_pa5_f150xdr6_pa5_f150-Planck_f143xPlanck_f143",
                   "dr6_pa6_f090xdr6_pa6_f090-Planck_f100xPlanck_f100",
                   "dr6_pa6_f150xdr6_pa6_f150-Planck_f143xPlanck_f143"]

name["PxP-AxP"] = ["Planck_f217xPlanck_f217-dr6_pa4_f220xPlanck_f217",
                   "Planck_f100xPlanck_f100-dr6_pa5_f090xPlanck_f100",
                   "Planck_f143xPlanck_f143-dr6_pa5_f150xPlanck_f143",
                   "Planck_f100xPlanck_f100-dr6_pa6_f090xPlanck_f100",
                   "Planck_f143xPlanck_f143-dr6_pa6_f150xPlanck_f143"]





dir_list = ["calibration_results", "calibration_results_npipe_bias_corrected", "calibration_results_npipe_bias_corrected_fg_sub", "calibration_results_npipe_bias_corrected_new_beams_fg_sub", "calibration_results_npipe_bias_corrected_16_arcmin_fg_sub", "calibration_results_npipe_bias_corrected_inpainted_fg_sub"]
arrays = ["pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
xticks = ["AxA-AxP", "AxA-PxP", "PxP-AxP", "AxA-AxP", "AxA-PxP", "PxP-AxP", "AxA-AxP", "AxA-PxP", "PxP-AxP", "AxA-AxP", "AxA-PxP", "PxP-AxP", "AxA-AxP", "AxA-PxP", "PxP-AxP", "AxA-AxP", "AxA-PxP", "PxP-AxP"  ]
# open a file, where you stored the pickled data

for i, ar in enumerate(arrays):

    cal_list, std_list = [], []
    AxA_AxP = name["AxA-AxP"][i]
    AxA_PxP = name["AxA-PxP"][i]
    PxP_AxP = name["PxP-AxP"][i]
    for dir in dir_list:
        file = open(f"{dir}/calibs_dict.pkl", 'rb')
        data = pickle.load(file)

        cal_AxA_AxP, std_AxA_AxP = data[AxA_AxP]["calibs"]
        cal_AxA_PxP, std_AxA_PxP = data[AxA_PxP]["calibs"]
        cal_PxP_AxP, std_PxP_AxP = data[PxP_AxP]["calibs"]
        print(dir, ar)
#        print("AxA_AxP", cal_AxA_AxP)
        if dir == "calibration_results_npipe_bias_corrected_new_beams_fg_sub":
            if ar == "pa4_f220":
                print("PxP-AxP", cal_PxP_AxP)

            else:
                print("AxA_PxP", cal_AxA_PxP)
 #       print("PxP_AxP", cal_PxP_AxP)
        print("")
        cal_list += [cal_AxA_AxP]
        cal_list += [cal_AxA_PxP]
        cal_list += [cal_PxP_AxP]

        std_list += [std_AxA_AxP]
        std_list += [std_AxA_PxP]
        std_list += [std_PxP_AxP]
    
    f, ax = plt.subplots(figsize=(15,8))

    plt.title(f"calib {ar}")
    plt.errorbar(np.arange(18), cal_list,  std_list, fmt=".", color= "black", ecolor=['red','blue','green','red','blue','green', 'red','blue','green',  'red','blue','green', 'red','blue','green', 'red','blue','green'  ])
    plt.axvline(2.5, color="black", linestyle="--", alpha=0.5)
    plt.axvline(5.5, color="black", linestyle="--", alpha=0.5)
    plt.axvline(8.5, color="black", linestyle="--", alpha=0.5)
    plt.axvline(11.5, color="black", linestyle="--", alpha=0.5)
    plt.axvline(14.5, color="black", linestyle="--", alpha=0.5)
    plt.text(.01, .99, "no correction", ha='left', va='top', transform=ax.transAxes)
    plt.text(.18, .99, "+NPipe syst correct", ha='left', va='top', transform=ax.transAxes)
    plt.text(.35, .99, "+bandpass correction", ha='left', va='top', transform=ax.transAxes)
    plt.text(.50, .99, "+Planck beam in ACT win", ha='left', va='top', transform=ax.transAxes)
    plt.text(.68, .99, "(3)+mask 16'", ha='left', va='top', transform=ax.transAxes)
    plt.text(.82, .99, "(3)+inpainting >150mJy sources", ha='left', va='top', transform=ax.transAxes)
    plt.text(.01, .04, "(1)", ha='left', va='top', transform=ax.transAxes)
    plt.text(.18, .04, "(2)", ha='left', va='top', transform=ax.transAxes)
    plt.text(.35, .04, "(3)", ha='left', va='top', transform=ax.transAxes)
    plt.text(.50, .04, "(4)", ha='left', va='top', transform=ax.transAxes)
    plt.text(.68, .04, "(5)", ha='left', va='top', transform=ax.transAxes)
    plt.text(.82, .04, "(6)", ha='left', va='top', transform=ax.transAxes)

    plt.xticks(np.arange(18), xticks, rotation=90)
    plt.savefig(f"cal_{ar}.png", bbox_inches="tight" )
    plt.clf()
    plt.close()
