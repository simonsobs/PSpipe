import numpy as np 
import matplotlib.pyplot as plt 

def plot_pseudo_spec(scenario, ax=None, i=0, label_suffix=None):
    mask_name, ps_name, filt_name = scenario

    res_dict = np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy', allow_pickle=True).item()

    xmin = res_dict['lmin2_fit']
    
    if ax is None:
        ax = plt.gca()

    den = res_dict['pseudo_den']
    ydata = np.divide(res_dict['pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))

    best_fit = res_dict['pseudo_best_fit']
    err = res_dict['pseudo_err']
    stderr = res_dict['pseudo_stderr']

    alpha = res_dict['pseudo_alpha']
    alpha_err = res_dict['pseudo_alpha_err']

    ax.plot(ydata, color=f'C{i}', alpha=0.3)
    label = rf'$\alpha_s={alpha:.2f} \pm {alpha_err:.2f}, \%\mathrm{{rms}}={100 * np.mean((err[xmin:] / best_fit[xmin:])**2)**0.5:.2f}$'
    if label_suffix:
        label = f'{label} ({label_suffix})'
    ax.plot(best_fit, label=label, color=f'C{i}', alpha=0.6)
    ax.axvline(xmin, color=f'C{i}', alpha=0.6)

def plot_binned_pseudo_spec(scenario, bin_cent, ax=None, i=0, label_suffix=None):
    mask_name, ps_name, filt_name = scenario

    res_dict = np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy', allow_pickle=True).item()

    xmin = res_dict['bmin2_fit']
    
    if ax is None:
        ax = plt.gca()

    den = res_dict['binned_pseudo_den']
    ydata = np.divide(res_dict['binned_pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))

    best_fit = res_dict['binned_pseudo_best_fit']
    err = res_dict['binned_pseudo_err']
    stderr = res_dict['binned_pseudo_stderr']

    alpha = res_dict['binned_pseudo_alpha']
    alpha_err = res_dict['binned_pseudo_alpha_err']

    ax.plot(bin_cent, ydata, color=f'C{i}', alpha=0.3)
    label = rf'$\alpha_s={alpha:.2f} \pm {alpha_err:.2f}, \%\mathrm{{rms}}={100 * np.mean((err[xmin:] / best_fit[xmin:])**2)**0.5:.2f}$'
    if label_suffix:
        label = f'{label} ({label_suffix})'
    ax.plot(bin_cent, best_fit, label=label, color=f'C{i}', alpha=0.6)
    ax.axvline(bin_cent[xmin], color=f'C{i}', alpha=0.6)

def plot_pseudo_cov_diag(scenario, which, ax=None, i=0, label_suffix=None):
    mask_name, ps_name, filt_name = scenario

    res_dict = np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy', allow_pickle=True).item()

    xmin = res_dict['lmin4_fit']
    
    if ax is None:
        ax = plt.gca()

    den = res_dict['pseudo_cov_diag_den']
    ydata = np.divide(res_dict['pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    alpha_s = res_dict['pseudo_alpha']

    best_fit = res_dict[f'pseudo_cov_diag_best_fit{which}']
    err = res_dict[f'pseudo_cov_diag_err{which}']
    stderr = res_dict[f'pseudo_cov_diag_stderr{which}']

    alpha = res_dict[f'pseudo_cov_diag_alpha{which}']
    alpha_err = res_dict[f'pseudo_cov_diag_alpha{which}_err']

    ax.plot(ydata, color=f'C{i}', alpha=0.3)
    label = rf'$\alpha_{which}/\alpha_s={alpha/alpha_s:.2f} \pm {alpha_err/alpha_s:.2f}, \%\mathrm{{rms}}={100 * np.mean((err[xmin:] / best_fit[xmin:])**2)**0.5:.2f}$'
    if label_suffix:
        label = f'{label} ({label_suffix})'

    ax.plot(best_fit, label=label, color=f'C{i}', alpha=0.6)
    ax.axvline(xmin, color=f'C{i}', alpha=0.6)

def plot_binned_pseudo_cov_diag(scenario, bin_cent, which, ax=None, i=0, label_suffix=None):
    mask_name, ps_name, filt_name = scenario

    res_dict = np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy', allow_pickle=True).item()

    xmin = res_dict['bmin4_fit']
    
    if ax is None:
        ax = plt.gca()

    den = res_dict['binned_pseudo_cov_diag_den']
    ydata = np.divide(res_dict['binned_pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    alpha_s = res_dict['pseudo_alpha']

    best_fit = res_dict[f'binned_pseudo_cov_diag_best_fit{which}']
    err = res_dict[f'binned_pseudo_cov_diag_err{which}']
    stderr = res_dict[f'binned_pseudo_cov_diag_stderr{which}']

    alpha = res_dict[f'binned_pseudo_cov_diag_alpha{which}']
    alpha_err = res_dict[f'binned_pseudo_cov_diag_alpha{which}_err']

    ax.plot(bin_cent, ydata, color=f'C{i}', alpha=0.3)
    label = rf'$\alpha_{which}/\alpha_s={alpha/alpha_s:.2f} \pm {alpha_err/alpha_s:.2f}, \%\mathrm{{rms}}={100 * np.mean((err[xmin:] / best_fit[xmin:])**2)**0.5:.2f}$'
    if label_suffix:
        label = f'{label} ({label_suffix})'

    ax.plot(bin_cent, best_fit, label=label, color=f'C{i}', alpha=0.6)
    ax.axvline(bin_cent[xmin], color=f'C{i}', alpha=0.6)

def plot_spec_cov_diag(scenario, which, ax=None, i=0, label_suffix=None):
    mask_name, ps_name, filt_name = scenario

    res_dict = np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy', allow_pickle=True).item()

    xmin = res_dict['lmin4_fit']
    
    if ax is None:
        ax = plt.gca()

    den = res_dict['spec_cov_diag_den']
    ydata = np.divide(res_dict['spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    alpha_s = res_dict['pseudo_alpha']

    best_fit = res_dict[f'spec_cov_diag_best_fit{which}']
    err = res_dict[f'spec_cov_diag_err{which}']
    stderr = res_dict[f'spec_cov_diag_stderr{which}']

    alpha = res_dict[f'spec_cov_diag_alpha{which}']
    alpha_err = res_dict[f'spec_cov_diag_alpha{which}_err']

    ax.plot(ydata, color=f'C{i}', alpha=0.3)
    label = rf'$\alpha_{which}/\alpha_s={alpha/alpha_s:.2f} \pm {alpha_err/alpha_s:.2f}, \%\mathrm{{rms}}={100 * np.mean((err[xmin:] / best_fit[xmin:])**2)**0.5:.2f}$'
    if label_suffix:
        label = f'{label} ({label_suffix})'

    ax.plot(best_fit, label=label, color=f'C{i}', alpha=0.6)
    ax.axvline(xmin, color=f'C{i}', alpha=0.6)

def plot_binned_spec_cov_diag(scenario, bin_cent, which, ax=None, i=0, label_suffix=None):
    mask_name, ps_name, filt_name = scenario

    res_dict = np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy', allow_pickle=True).item()

    xmin = res_dict['bmin4_fit']
    
    if ax is None:
        ax = plt.gca()

    den = res_dict['binned_spec_cov_diag_den']
    ydata = np.divide(res_dict['binned_spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    alpha_s = res_dict['pseudo_alpha']

    best_fit = res_dict[f'binned_spec_cov_diag_best_fit{which}']
    err = res_dict[f'binned_spec_cov_diag_err{which}']
    stderr = res_dict[f'binned_spec_cov_diag_stderr{which}']

    alpha = res_dict[f'binned_spec_cov_diag_alpha{which}']
    alpha_err = res_dict[f'binned_spec_cov_diag_alpha{which}_err']

    ax.plot(bin_cent, ydata, color=f'C{i}', alpha=0.3)
    label = rf'$\alpha_{which}/\alpha_s={alpha/alpha_s:.2f} \pm {alpha_err/alpha_s:.2f}, \%\mathrm{{rms}}={100 * np.mean((err[xmin:] / best_fit[xmin:])**2)**0.5:.2f}$'
    if label_suffix:
        label = f'{label} ({label_suffix})'

    ax.plot(bin_cent, best_fit, label=label, color=f'C{i}', alpha=0.6)
    ax.axvline(bin_cent[xmin], color=f'C{i}', alpha=0.6)

def detailed_single_scenario_plot(scenario, bin_cent):
    mask_name, ps_name, filt_name = scenario

    res_dict = np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy', allow_pickle=True).item()

    def f(x, alpha, func, xmin, den):
        return np.divide(func(alpha)[xmin:], den[xmin:], where=den[xmin:]!=0, out=np.zeros_like(den[xmin:]))
    
    lmin2_fit = res_dict['lmin2_fit']
    bmin2_fit = res_dict['bmin2_fit']
    lmin4_fit = res_dict['lmin4_fit']
    bmin4_fit = res_dict['bmin4_fit']

    # fit pseudo
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 6), height_ratios=(2, 1), sharey='row', sharex='col')

    den = res_dict['pseudo_den']
    ydata = np.divide(res_dict['pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))

    pseudo_best_fit = res_dict['pseudo_best_fit']
    pseudo_err = res_dict['pseudo_err']
    pseudo_stderr = res_dict['pseudo_stderr']

    pseudo_alpha = res_dict['pseudo_alpha']
    pseudo_alpha_err = res_dict['pseudo_alpha_err']

    axs[0, 0].plot(ydata)
    axs[0, 0].plot(pseudo_best_fit, label=rf'$\alpha_s={pseudo_alpha:.3f} \pm {pseudo_alpha_err:.3f}$')
    axs[0, 0].axvspan(0, lmin2_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid()
    axs[0, 0].set_ylabel(r'$\tilde{\mathcal{C}_{\ell}}(\alpha) / \tilde{\mathcal{C}_{\ell}}(0)$')
    axs[0, 0].set_title('unbinned')

    axs[1, 0].plot(pseudo_err / pseudo_best_fit, label=f'$\chi^2={np.mean(pseudo_stderr[lmin2_fit:]**2):.3f}$')
    axs[1, 0].axvspan(0, lmin2_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[1, 0].set_ylim(-.05, .05)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid()
    axs[1, 0].set_xlabel('$\ell$')
    axs[1, 0].set_ylabel('$\Delta y_s / y_s$')

    den = res_dict['binned_pseudo_den']
    ydata = np.divide(res_dict['binned_pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['binned_pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    
    binned_pseudo_best_fit = res_dict['binned_pseudo_best_fit']
    binned_pseudo_err = res_dict['binned_pseudo_err']
    binned_pseudo_stderr = res_dict['binned_pseudo_stderr']

    binned_pseudo_alpha = res_dict['binned_pseudo_alpha']
    binned_pseudo_alpha_err = res_dict['binned_pseudo_alpha_err']

    axs[0, 1].plot(bin_cent, ydata)
    axs[0, 1].plot(bin_cent, binned_pseudo_best_fit, label=rf'$\alpha_s={binned_pseudo_alpha:.3f} \pm {binned_pseudo_alpha_err:.3f}$')
    axs[0, 1].axvspan(0, bin_cent[bmin2_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].grid()
    axs[0, 1].set_title('binned')

    axs[1, 1].plot(bin_cent, binned_pseudo_err / binned_pseudo_best_fit, label=f'$\chi^2={np.mean(binned_pseudo_stderr[bmin2_fit:]**2):.3f}$')
    axs[1, 1].axvspan(0, bin_cent[bmin2_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[1, 1].set_ylim(-.05, .05)
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].grid()
    axs[1, 1].set_xlabel('$\ell$')

    fig.suptitle('pseudo specs')
    fig.tight_layout()
    plt.show()

    # fit pseudo_cov_diag
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 8), height_ratios=(2, 1, 1), sharey='row', sharex='col')

    den = res_dict['pseudo_cov_diag_den']
    ydata = np.divide(res_dict['pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    
    pseudo_cov_diag_best_fit2 = res_dict['pseudo_cov_diag_best_fit2']
    pseudo_cov_diag_err2 = res_dict['pseudo_cov_diag_err2']
    pseudo_cov_diag_stderr2 = res_dict['pseudo_cov_diag_stderr2']

    pseudo_cov_diag_alpha2 = res_dict['pseudo_cov_diag_alpha2']
    pseudo_cov_diag_alpha2_err = res_dict['pseudo_cov_diag_alpha2_err']

    pseudo_cov_diag_best_fit4 = res_dict['pseudo_cov_diag_best_fit4']
    pseudo_cov_diag_err4 = res_dict['pseudo_cov_diag_err4']
    pseudo_cov_diag_stderr4 = res_dict['pseudo_cov_diag_stderr4']

    pseudo_cov_diag_alpha4 = res_dict['pseudo_cov_diag_alpha4']
    pseudo_cov_diag_alpha4_err = res_dict['pseudo_cov_diag_alpha4_err']

    axs[0, 0].plot(ydata)
    axs[0, 0].plot(pseudo_cov_diag_best_fit2, label=rf'$\alpha_2/\alpha_s={pseudo_cov_diag_alpha2/pseudo_alpha:.3f} \pm {pseudo_cov_diag_alpha2_err/pseudo_alpha:.3f}$', color='C1', linestyle='--')
    axs[0, 0].plot(pseudo_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={pseudo_cov_diag_alpha4/pseudo_alpha:.3f} \pm {pseudo_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
    axs[0, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid()
    axs[0, 0].set_ylabel(r'$\tilde{\Sigma_{\ell,\ell}}(\alpha) / \tilde{\Sigma_{\ell,\ell}}(0)$')
    axs[0, 0].set_title('unbinned')

    axs[1, 0].plot(pseudo_cov_diag_err2 / pseudo_cov_diag_best_fit2, label=f'$\chi_2^2={np.mean(pseudo_cov_diag_stderr2[lmin4_fit:]**2):.3f}$')
    axs[1, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[1, 0].set_ylim(-.1, .1)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid()
    axs[1, 0].set_ylabel('$\Delta y_2 / y_2$')

    axs[2, 0].plot(pseudo_cov_diag_err4 / pseudo_cov_diag_best_fit4, label=f'$\chi_4^2={np.mean(pseudo_cov_diag_stderr4[lmin4_fit:]**2):.3f}$')
    axs[2, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[2, 0].set_ylim(-.1, .1)
    axs[2, 0].legend(loc='lower right')
    axs[2, 0].grid()
    axs[2, 0].set_xlabel('$\ell$')
    axs[2, 0].set_ylabel('$\Delta y_4 / y_4$')

    den = res_dict['binned_pseudo_cov_diag_den']
    ydata = np.divide(res_dict['binned_pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['binned_pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    
    binned_pseudo_cov_diag_best_fit2 = res_dict['binned_pseudo_cov_diag_best_fit2']
    binned_pseudo_cov_diag_err2 = res_dict['binned_pseudo_cov_diag_err2']
    binned_pseudo_cov_diag_stderr2 = res_dict['binned_pseudo_cov_diag_stderr2']

    binned_pseudo_cov_diag_alpha2 = res_dict['binned_pseudo_cov_diag_alpha2']
    binned_pseudo_cov_diag_alpha2_err = res_dict['binned_pseudo_cov_diag_alpha2_err']

    binned_pseudo_cov_diag_best_fit4 = res_dict['binned_pseudo_cov_diag_best_fit4']
    binned_pseudo_cov_diag_err4 = res_dict['binned_pseudo_cov_diag_err4']
    binned_pseudo_cov_diag_stderr4 = res_dict['binned_pseudo_cov_diag_stderr4']

    binned_pseudo_cov_diag_alpha4 = res_dict['binned_pseudo_cov_diag_alpha4']
    binned_pseudo_cov_diag_alpha4_err = res_dict['binned_pseudo_cov_diag_alpha4_err']

    axs[0, 1].errorbar(bin_cent, ydata, yerr, linestyle='none')
    axs[0, 1].plot(bin_cent, binned_pseudo_cov_diag_best_fit2, label=rf'$\alpha_2/\alpha_s={binned_pseudo_cov_diag_alpha2/pseudo_alpha:.3f} \pm {binned_pseudo_cov_diag_alpha2_err/pseudo_alpha:.3f}$', color='C1', linestyle='--')
    axs[0, 1].plot(bin_cent, binned_pseudo_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={binned_pseudo_cov_diag_alpha4/pseudo_alpha:.3f} \pm {binned_pseudo_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
    axs[0, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].grid()
    axs[0, 1].set_title('binned')

    axs[1, 1].errorbar(bin_cent, binned_pseudo_cov_diag_err2 / binned_pseudo_cov_diag_best_fit2, yerr / binned_pseudo_cov_diag_best_fit2, linestyle='none', label=f'$\chi_2^2={np.mean(binned_pseudo_cov_diag_stderr2[bmin4_fit:]**2):.3f}$')
    axs[1, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[1, 1].set_ylim(-.1, .1)
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].grid()

    axs[2, 1].errorbar(bin_cent, binned_pseudo_cov_diag_err4 / binned_pseudo_cov_diag_best_fit4, yerr / binned_pseudo_cov_diag_best_fit4, linestyle='none', label=f'$\chi_4^2={np.mean(binned_pseudo_cov_diag_stderr4[bmin4_fit:]**2):.3f}$')
    axs[2, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[2, 1].set_ylim(-.1, .1)
    axs[2, 1].legend(loc='lower right')
    axs[2, 1].grid()
    axs[2, 1].set_xlabel('$\ell$')

    fig.suptitle('pseudo cov diags')
    fig.tight_layout()
    plt.show()

    # fit spec_cov_diag
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 8), height_ratios=(2, 1, 1), sharey='row', sharex='col')

    den = res_dict['spec_cov_diag_den']
    ydata = np.divide(res_dict['spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    
    spec_cov_diag_best_fit2 = res_dict['spec_cov_diag_best_fit2']
    spec_cov_diag_err2 = res_dict['spec_cov_diag_err2']
    spec_cov_diag_stderr2 = res_dict['spec_cov_diag_stderr2']

    spec_cov_diag_alpha2 = res_dict['spec_cov_diag_alpha2']
    spec_cov_diag_alpha2_err = res_dict['spec_cov_diag_alpha2_err']

    spec_cov_diag_best_fit4 = res_dict['spec_cov_diag_best_fit4']
    spec_cov_diag_err4 = res_dict['spec_cov_diag_err4']
    spec_cov_diag_stderr4 = res_dict['spec_cov_diag_stderr4']

    spec_cov_diag_alpha4 = res_dict['spec_cov_diag_alpha4']
    spec_cov_diag_alpha4_err = res_dict['spec_cov_diag_alpha4_err']

    axs[0, 0].plot(ydata)
    axs[0, 0].plot(spec_cov_diag_best_fit2, label=rf'$\alpha_2/\alpha_s={spec_cov_diag_alpha2/pseudo_alpha:.3f} \pm {spec_cov_diag_alpha2_err/pseudo_alpha:.3f}$', color='C1', linestyle='--')
    axs[0, 0].plot(spec_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={spec_cov_diag_alpha4/pseudo_alpha:.3f} \pm {spec_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
    axs[0, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid()
    axs[0, 0].set_ylabel(r'$\tilde{\Sigma_{\ell,\ell}}(\alpha) / \tilde{\Sigma_{\ell,\ell}}(0)$')
    axs[0, 0].set_title('unbinned')

    axs[1, 0].plot(spec_cov_diag_err2 / spec_cov_diag_best_fit2, label=f'$\chi_2^2={np.mean(spec_cov_diag_stderr2[lmin4_fit:]**2):.3f}$')
    axs[1, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[1, 0].set_ylim(-.1, .1)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid()
    axs[1, 0].set_ylabel('$\Delta y_2 / y_2$')

    axs[2, 0].plot(spec_cov_diag_err4 / spec_cov_diag_best_fit4, label=f'$\chi_4^2={np.mean(spec_cov_diag_stderr4[lmin4_fit:]**2):.3f}$')
    axs[2, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
    axs[2, 0].set_ylim(-.1, .1)
    axs[2, 0].legend(loc='lower right')
    axs[2, 0].grid()
    axs[2, 0].set_xlabel('$\ell$')
    axs[2, 0].set_ylabel('$\Delta y_4 / y_4$')

    den = res_dict['binned_spec_cov_diag_den']
    ydata = np.divide(res_dict['binned_spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['binned_spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    
    binned_spec_cov_diag_best_fit2 = res_dict['binned_spec_cov_diag_best_fit2']
    binned_spec_cov_diag_err2 = res_dict['binned_spec_cov_diag_err2']
    binned_spec_cov_diag_stderr2 = res_dict['binned_spec_cov_diag_stderr2']

    binned_spec_cov_diag_alpha2 = res_dict['binned_spec_cov_diag_alpha2']
    binned_spec_cov_diag_alpha2_err = res_dict['binned_spec_cov_diag_alpha2_err']

    binned_spec_cov_diag_best_fit4 = res_dict['binned_spec_cov_diag_best_fit4']
    binned_spec_cov_diag_err4 = res_dict['binned_spec_cov_diag_err4']
    binned_spec_cov_diag_stderr4 = res_dict['binned_spec_cov_diag_stderr4']

    binned_spec_cov_diag_alpha4 = res_dict['binned_spec_cov_diag_alpha4']
    binned_spec_cov_diag_alpha4_err = res_dict['binned_spec_cov_diag_alpha4_err']

    axs[0, 1].errorbar(bin_cent, ydata, yerr, linestyle='none')
    axs[0, 1].plot(bin_cent, binned_spec_cov_diag_best_fit2, label=rf'$\alpha_2/\alpha_s={binned_spec_cov_diag_alpha2/pseudo_alpha:.3f} \pm {binned_spec_cov_diag_alpha2_err/pseudo_alpha:.3f}$', color='C1', linestyle='--')
    axs[0, 1].plot(bin_cent, binned_spec_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={binned_spec_cov_diag_alpha4/pseudo_alpha:.3f} \pm {binned_spec_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
    axs[0, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].grid()
    axs[0, 1].set_title('binned')

    axs[1, 1].errorbar(bin_cent, binned_spec_cov_diag_err2 / binned_spec_cov_diag_best_fit2, yerr / binned_spec_cov_diag_best_fit2, linestyle='none', label=f'$\chi_2^2={np.mean(binned_spec_cov_diag_stderr2[bmin4_fit:]**2):.3f}$')
    axs[1, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[1, 1].set_ylim(-.1, .1)
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].grid()

    axs[2, 1].errorbar(bin_cent, binned_spec_cov_diag_err4 / binned_spec_cov_diag_best_fit4, yerr / binned_spec_cov_diag_best_fit4, linestyle='none', label=f'$\chi_4^2={np.mean(binned_spec_cov_diag_stderr4[bmin4_fit:]**2):.3f}$')
    axs[2, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
    axs[2, 1].set_ylim(-.1, .1)
    axs[2, 1].legend(loc='lower right')
    axs[2, 1].grid()
    axs[2, 1].set_xlabel('$\ell$')

    fig.suptitle('cov diags')
    fig.tight_layout()
    plt.show()

masks = ['none', 'mnms', 'pt_src', 'pt_src_sig_sqrt_pixar']
pss = ['white', 'noise_EE_l100_cap', 'noise_TT', 'signal_EE']
filts = ['m_exact', 'l_ish', 'l_ish_m_exact', 'binary_cross']

bin_cent = np.loadtxt('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/binning/BIN_ACTPOL_50_4_SC_large_bin_at_low_ell', usecols=2)
bin_cent = bin_cent[bin_cent < 5400][:-1]

aliases = dict(pt_src_sig_sqrt_pixar='pt_nse', noise_EE_l100_cap='noise_EE', m_exact='m', l_ish_m_exact='l_ish_m')
for i in masks + pss + filts:
    if i not in aliases:
        aliases[i] = i

detailed_single_scenario_plot(('pt_src', 'noise_TT', 'binary_cross'), bin_cent)

listr = masks
listc = filts
listi = pss

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(22, 12), sharey='row', sharex='col')
for r, relem in enumerate(listr):
    for c, celem in enumerate(listc):
        ax = axs[r, c]
        if r == len(listr) - 1:
            ax.set_xlabel(r'$\ell$', fontsize=10)
        if c == 0:
            ax.set_ylabel(r'$\tilde{\mathcal{C}_{\ell}}(\alpha) / \tilde{\mathcal{C}_{\ell}}(0)$', fontsize=10)
        for i, ielem in enumerate(listi):
            scenario = [0, 0, 0]
            tags = [0, 0, 0]
            for e, elem in enumerate((relem, celem, ielem)):
                if elem in masks:
                    assert elem not in pss + filts
                    scenario[0] = elem
                    tags[e] = 'mask'
                if elem in pss:
                    assert elem not in masks + filts
                    scenario[1] = elem
                    tags[e] = 'ps'
                if elem in filts:
                    assert elem not in masks + pss
                    scenario[2] = elem
                    tags[e] = 'filt'
            scenario = tuple(scenario)

            plot_pseudo_spec(scenario, ax=ax, i=i, label_suffix=f'{tags[2]}: {aliases[ielem]}')
            if i == len(listi) - 1:
                ax.set_title(f'{tags[0]}: {aliases[relem]}, {tags[1]}: {aliases[celem]}')
                ax.grid()
                ax.legend(fontsize=10)
fig.tight_layout(h_pad=0.15, w_pad=0)

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(22, 12), sharey='row', sharex='col')
for r, relem in enumerate(listr):
    for c, celem in enumerate(listc):
        ax = axs[r, c]
        if r == len(listr) - 1:
            ax.set_xlabel(r'$\ell$', fontsize=10)
        if c == 0:
            ax.set_ylabel(r'$\tilde{\Sigma_{\ell,\ell}}(\alpha) / \tilde{\Sigma_{\ell,\ell}}(0)$', fontsize=10)
        for i, ielem in enumerate(listi):
            scenario = [0, 0, 0]
            tags = [0, 0, 0]
            for e, elem in enumerate((relem, celem, ielem)):
                if elem in masks:
                    assert elem not in pss + filts
                    scenario[0] = elem
                    tags[e] = 'mask'
                if elem in pss:
                    assert elem not in masks + filts
                    scenario[1] = elem
                    tags[e] = 'ps'
                if elem in filts:
                    assert elem not in masks + pss
                    scenario[2] = elem
                    tags[e] = 'filt'
            scenario = tuple(scenario)

            plot_binned_spec_cov_diag(scenario, bin_cent, 4, ax=ax, i=i, label_suffix=f'{tags[2]}: {aliases[ielem]}')
            if i == len(listi) - 1:
                ax.set_title(f'{tags[0]}: {aliases[relem]}, {tags[1]}: {aliases[celem]}')
                ax.grid()
                ax.legend(fontsize=10)
fig.tight_layout(h_pad=0.15, w_pad=0)