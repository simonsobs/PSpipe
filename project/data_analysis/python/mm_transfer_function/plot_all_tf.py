import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "18"


tf_dir = "tf_estimator_AxP_PxP_fg_sub"

_, end = tf_dir.split('estimator_')
n1, n2, n3, n4 = end[0], end[2],  end[4], end[6]


plt.figure(figsize=(16, 8))

colors = ["red", "orange", "green", "blue", "gray"]
arrays = ["pa5_f090", "pa6_f090", "pa5_f150", "pa6_f150", "pa4_f220"]


count = 0
for col, ar in zip(colors, arrays):
    lb, tf, tferr = np.loadtxt(f"{tf_dir}/tf_estimator_dr6_{ar}.dat", unpack=True)
  #  lth, tf_fit = np.loadtxt(f"{tf_dir}/tf_fit_dr6_{ar}_logistic_fixed_amp.dat", unpack=True)

    plt.errorbar(lb-10 +count*5, tf, tferr, fmt="o", color=col, label=ar) #, color="blue")
    plt.plot(lb, lb*0+1, "--", color="black", alpha=0.5)#, color="blue")
    count += 1
    
plt.ylabel(r"$T_{\ell} = C_{\ell, \rm %s  x  %s}^{\rm TT} / C_{\ell, \rm %s  x %s}^{\rm TT}$" % (n1, n2, n3, n4), fontsize=25)
plt.xlabel(r"$\ell$", fontsize=25)

plt.xlim(0, 1800)
plt.ylim(0.9,1.05)
plt.legend()
plt.savefig(f"{tf_dir}.png", bbox_inches="tight")
plt.clf()
plt.close()
