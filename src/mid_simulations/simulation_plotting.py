import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.plotting import plot_design_matrix


def plot_proportion_sig(results_df, hue_order, axs_barplot):
    # The way I am changing alpha of the bars is a little hacky
    # It should be stable, I rely on the x-location of the bars (rounded) to match
    # up with the axis label index.  I also control the model order, so that part
    # should be correct.
    alpha_values = results_df.copy()
    alpha_values = alpha_values.drop(
        ['mean', 'tval', 'pval', 'sigp'], axis=1
    ).drop_duplicates()

    sns.barplot(
        data=results_df,
        x='contrast',
        y='sigp',
        hue='model',
        ax=axs_barplot,
        hue_order=hue_order,
        order=results_df['contrast'],
    )
    x_tick_label_location = {
        text_obj.get_position()[0]: text_obj.get_text()
        for text_obj in axs_barplot.get_xticklabels()
    }
    axs_barplot.set_ylim(0, 1)
    axs_barplot.axhline(0.05, color='red', linestyle='--')
    axs_barplot.set_xlabel('Contrast')
    axs_barplot.set_ylabel('Proportion significant \n (solid=power, opaque=error rate)')
    axs_barplot.tick_params(axis='x', rotation=90)
    # Change alpha of bars
    # axs.containers is a list (possibly of lists), where the
    # outer list refers to model and list within list is contrast
    for model_num, container in enumerate(
        axs_barplot.containers
    ):  # this will be the models
        model_loop = hue_order[model_num]
        for subcontainer in container:  # this will be the contrasts
            # use the x position of bar to get contrast name
            contrast_name = x_tick_label_location[int(np.round(subcontainer.get_x()))]
            subcontainer.set_alpha(
                alpha_values[
                    (alpha_values['model'] == model_loop)
                    & (alpha_values['contrast'] == contrast_name)
                ]['plot_alpha_val_power_error'].values[0]
            )


def plot_contrast_estimates(results_df, hue_order, axs):
    alpha_values = results_df.copy()
    alpha_values = alpha_values.drop(
        ['mean', 'tval', 'pval', 'sigp'], axis=1
    ).drop_duplicates()

    sns.violinplot(
        data=results_df,
        x='contrast',
        y='tval',
        hue='model',
        ax=axs,
        inner=None,
        hue_order=hue_order,
        legend=False,
    )
    axs.axhline(0.0, color='red', linestyle='--')
    axs.set_xlabel('Contrast')
    axs.set_ylabel('Group-level T-stats \n (opaque should have T-stats=0)')
    axs.tick_params(axis='x', rotation=90)
    x_labels_in_order = [text_obj.get_text() for text_obj in axs.get_xticklabels()]
    alpha_vec = []
    for contrast_name in x_labels_in_order:
        for model in hue_order:
            subset_df = alpha_values[
                (alpha_values['model'] == model)
                & (alpha_values['contrast'] == contrast_name)
            ]['plot_alpha_val_power_error']
            if subset_df.shape[0] > 0:
                alpha_vec.append(subset_df.values[0])
    for subcollection, alpha in zip(axs.collections, alpha_vec):
        subcollection.set_alpha(alpha)


def plot_results(results_df, analysis_label, stacked=False):
    # my method for changing alpha of the bars is a little hacky
    # and it may not be stable.
    models_included = results_df['model'].unique()
    hue_order = ['Saturated', 'CueYesDeriv', 'CueNoDeriv']
    hue_order = [model for model in hue_order if model in models_included]
    if stacked:
        fig, axs = plt.subplots(2, 1, figsize=(25, 15))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(analysis_label)
    plot_proportion_sig(results_df, hue_order, axs[0])
    plot_contrast_estimates(results_df, hue_order, axs[1])
    plt.tight_layout()
    plt.show()


def plot_dict_of_results(results_dict, contrasts=True, fig_path=None):
    num_plot_pairs = len(results_dict.keys())
    nrows = int(np.ceil(num_plot_pairs / 2))
    ncols = int(num_plot_pairs / nrows)

    if contrasts:
        n_subrow = 1
        n_subcol = 2
        figsize1 = 12 * ncols
        figsize2 = 5 * nrows
    if not contrasts:
        n_subrow = 2
        n_subcol = 1
        figsize1 = 30
        figsize2 = 6 * nrows

    fig = plt.figure(figsize=(figsize1, figsize2))
    outer = fig.subfigures(nrows, ncols, wspace=-0.1, hspace=0.6)
    if nrows > 1 and ncols > 1:
        outer_flat = outer.flatten()
    else:
        outer_flat = [outer]

    for i, fig_label in enumerate(results_dict.keys()):
        inner = outer_flat[i].subplots(n_subrow, n_subcol)

        outer_flat[i].suptitle(fig_label, fontsize=15, y=1.05)
        results_df_loop = results_dict[fig_label]
        contrast_rows = results_df_loop['contrast'].str.contains('-')
        if contrasts:
            keep = contrast_rows
        if not contrasts:
            keep = not contrast_rows
        results_df_loop = results_df_loop[keep]
        models_included = results_df_loop['model'].unique()
        hue_order = ['Saturated', 'CueYesDeriv', 'CueNoDeriv']
        hue_order = [model for model in hue_order if model in models_included]
        plot_proportion_sig(results_df_loop, hue_order, inner[0])
        plot_contrast_estimates(results_df_loop, hue_order, inner[1])
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_design_ordered_regressors(desmat, desname, ax):
    """
    Plots design matrix with regressors ordered by trial type and stimulus type (within trial type)
    (regressors are now ordered when the design matrix is created)
    """
    plot_design_matrix(desmat, ax=ax, rescale=False)
    max_val = 1.2 * desmat.drop(columns=['constant']).max().max()
    im = ax.images[-1]
    im.set_clim(vmin=0, vmax=max_val)
    # im.set_cmap('BuPu')
    im = ax.images[0]
    plt.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    return ax


def plot_bias_contrastf(
    results,
    contrasts_only=False,
    omit_noderiv=False,
    jitter=False,
    fig_path=None,
    newname_cue_yes_deriv=None,
    nsubs=500,
):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap.set_bad('lightgrey')

    if jitter:
        title = 'Bias (jittered ITI)'
    else:
        title = 'Bias'

    bigfont = 14
    smallfont1 = 10
    smallfont2 = 10
    fig_width = 8
    rotate_xlab = 20
    if not contrasts_only:
        smallfont2 = 5
        fig_width = 8
        rotate_xlab = 90

    f, axs = plt.subplots(
        len(results.keys()),
        1,  # gridspec_kw={'hspace': 0.5},
        figsize=(fig_width, 10),
        sharex=True,
    )
    cbar_ax = f.add_axes([0.91, 0.2, 0.03, 0.5])
    f.suptitle(
        f"{title} \nAverage of group Cohen's Ds across simulations \nBias occurs when values are nonzero",
        fontsize=bigfont,
    )
    if contrasts_only:
        omit_string = '^((?!-).)*$'
    else:
        omit_string = None

    for idx, (setting, data) in enumerate(results.items()):
        data = data.copy()
        if newname_cue_yes_deriv is not None:
            data.loc[data['model'] == 'CueYesDeriv', 'model'] = newname_cue_yes_deriv
        data.loc[data['plot_alpha_val_power_error'] == 1, 'tval'] = np.nan
        setting = setting.replace(', ', '\n')
        # add extra blank lines when needed for aesthetics
        if len(setting.split('\n')) < 3:
            setting += '\n   ' * (3 - len(setting.split('\n')))
        dat_plot = (
            data.groupby(['contrast', 'model'])[['tval']]
            .mean()
            .reset_index()
            .pivot(index='contrast', columns='model', values='tval')
            .transpose()
        )
        dat_plot = dat_plot / np.sqrt(nsubs)
        dat_plot = dat_plot[dat_plot.columns.drop(list(dat_plot.filter(regex='Deriv')))]
        if omit_string:
            dat_plot = dat_plot[
                dat_plot.columns.drop(list(dat_plot.filter(regex=omit_string)))
            ]
        if omit_noderiv:
            dat_plot = dat_plot.loc[
                ~dat_plot.index.str.contains('NoDeriv|SaturatedDeriv')
            ]
        else:
            dat_plot = dat_plot.loc[~dat_plot.index.str.contains('SaturatedDeriv')]
        dat_plot = dat_plot[sorted(dat_plot.columns, key=lambda x: ('-' in x, x))]
        g = sns.heatmap(
            dat_plot,
            vmin=-0.01,
            vmax=0.05,
            center=0,
            cbar=idx == 0,
            cmap=cmap,
            ax=axs[idx],
            cbar_ax=None if idx else cbar_ax,
            annot=True,
            fmt='.2f',
            annot_kws={'fontsize': smallfont2},
        )
        g.set_yticklabels(g.get_yticklabels(), size=smallfont1, rotation=0)
        g.set_xticklabels(g.get_xticklabels(), size=smallfont1, rotation=rotate_xlab)
        # if idx < num_plots - 1:
        axs[idx].set_xlabel('')
        axs[idx].set_ylabel(
            setting, rotation=0, labelpad=170, loc='bottom', fontsize=smallfont1
        )
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_bias(
    results,
    contrasts_only=False,
    omit_noderiv=False,
    jitter=False,
    fig_path=None,
    newname_cue_yes_deriv=None,
    nsubs=500,
):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap.set_bad('lightgrey')

    if jitter:
        title = 'Bias (jittered ITI)'
    else:
        title = 'Bias'

    bigfont = 14
    smallfont1 = 10
    smallfont2 = 10
    fig_width = 8
    rotate_xlab = 20
    if not contrasts_only:
        smallfont2 = 5
        fig_width = 12
        rotate_xlab = 90

    f, axs = plt.subplots(
        len(results.keys()),
        1,  # gridspec_kw={'hspace': 0.5},
        figsize=(fig_width, 10),
        sharex=True,
    )
    cbar_ax = f.add_axes([0.91, 0.2, 0.03, 0.5])
    f.suptitle(
        f"{title} \nAverage of group Cohen's Ds across simulations \nBias occurs when values are nonzero",
        fontsize=bigfont,
    )
    if contrasts_only:
        omit_string = '^((?!-).)*$'
    else:
        omit_string = None

    for idx, (setting, data) in enumerate(results.items()):
        data = data.copy()
        if newname_cue_yes_deriv is not None:
            data.loc[data['model'] == 'CueYesDeriv', 'model'] = newname_cue_yes_deriv
        data.loc[data['plot_alpha_val_power_error'] == 1, 'tval'] = np.nan
        setting = setting.replace(', ', '\n')
        # add extra blank lines when needed for aesthetics
        if len(setting.split('\n')) < 3:
            setting += '\n   ' * (3 - len(setting.split('\n')))
        dat_plot = (
            data.groupby(['contrast', 'model'])[['tval']]
            .mean()
            .reset_index()
            .pivot(index='contrast', columns='model', values='tval')
            .transpose()
        )
        dat_plot = dat_plot / np.sqrt(nsubs)
        dat_plot = dat_plot[dat_plot.columns.drop(list(dat_plot.filter(regex='Deriv')))]
        if omit_string:
            dat_plot = dat_plot[
                dat_plot.columns.drop(list(dat_plot.filter(regex=omit_string)))
            ]
        if omit_noderiv:
            dat_plot = dat_plot.loc[
                ~dat_plot.index.str.contains('NoDeriv|SaturatedDeriv')
            ]
        else:
            dat_plot = dat_plot.loc[~dat_plot.index.str.contains('SaturatedDeriv')]
        cue_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('Cue')], key=sort_key
        )
        fb_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('FB')], key=sort_key
        )

        dat_plot = dat_plot[cue_cols + fb_cols]
        g = sns.heatmap(
            dat_plot,
            vmin=-0.01,
            vmax=0.05,
            center=0,
            cbar=idx == 0,
            cmap=cmap,
            ax=axs[idx],
            cbar_ax=None if idx else cbar_ax,
            annot=True,
            fmt='.2f',
            annot_kws={'fontsize': smallfont2},
        )
        g.set_yticklabels(g.get_yticklabels(), size=smallfont1, rotation=0)
        g.set_xticklabels(g.get_xticklabels(), size=smallfont1, rotation=rotate_xlab)
        # if idx < num_plots - 1:
        axs[idx].set_xlabel('')
        axs[idx].set_ylabel(
            setting, rotation=0, labelpad=170, loc='bottom', fontsize=smallfont1
        )
    plt.figtext(0.9, 0.01, '*Contrast distributed by ABCD', ha='left', fontsize=10)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_error_grid(
    results,
    contrasts_only=False,
    omit_noderiv=False,
    jitter=False,
    fig_path=None,
    newname_cue_yes_deriv=None,
):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap.set_bad('lightgrey')
    if jitter:
        title = 'Type I error (jittered ITI)'
    else:
        title = 'Type I error'
    num_plots = len(results.keys())
    bigfont = 14
    smallfont1 = 10
    smallfont2 = 10
    fig_width = 8
    rotate_xlab = 20
    if not contrasts_only:
        smallfont2 = 5
        fig_width = 8
        rotate_xlab = 90

    f, axs = plt.subplots(
        num_plots,
        1,  # gridspec_kw={'hspace': 0.5},
        figsize=(fig_width, 10),
        sharex=True,
    )
    cbar_ax = f.add_axes([0.91, 0.2, 0.03, 0.5])
    f.suptitle(f'{title}', fontsize=16)
    if contrasts_only:
        omit_string = '^((?!-).)*$'
    else:
        omit_string = None
    for idx, (setting, data) in enumerate(results.items()):
        data = data.copy()
        if newname_cue_yes_deriv is not None:
            data.loc[data['model'] == 'CueYesDeriv', 'model'] = newname_cue_yes_deriv
        data['sigp'] = data['sigp'].astype(float)
        data.loc[data['plot_alpha_val_power_error'] == 1, 'sigp'] = pd.NA
        setting = setting.replace(', ', '\n')
        # add extra blank lines when needed for aesthetics
        if len(setting.split('\n')) < 3:
            setting += '\n   ' * (3 - len(setting.split('\n')))
        dat_plot = (
            data.groupby(['contrast', 'model'])[['sigp']]
            .mean()
            .reset_index()
            .pivot(index='contrast', columns='model', values='sigp')
            .transpose()
        )
        dat_plot = dat_plot[dat_plot.columns.drop(list(dat_plot.filter(regex='Deriv')))]
        if omit_string:
            dat_plot = dat_plot[
                dat_plot.columns.drop(list(dat_plot.filter(regex=omit_string)))
            ]
        if omit_noderiv:
            dat_plot = dat_plot.loc[
                ~dat_plot.index.str.contains('NoDeriv|SaturatedDeriv')
            ]
        else:
            dat_plot = dat_plot.loc[~dat_plot.index.str.contains('SaturatedDeriv')]
        cue_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('Cue')], key=sort_key
        )
        fb_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('FB')], key=sort_key
        )
        dat_plot = dat_plot[cue_cols + fb_cols]
        g = sns.heatmap(
            dat_plot,
            vmin=0,
            vmax=1,
            center=0.05,
            cbar=idx == 0,
            cmap=cmap,
            ax=axs[idx],
            cbar_ax=None if idx else cbar_ax,
            annot=True,
            fmt='.2f',
            annot_kws={'fontsize': smallfont2},
        )
        g.set_yticklabels(g.get_yticklabels(), size=smallfont1, rotation=0)
        g.set_xticklabels(g.get_xticklabels(), size=smallfont1, rotation=rotate_xlab)
        # if idx < num_plots - 1:
        axs[idx].set_xlabel('')
        axs[idx].set_ylabel(
            setting, rotation=0, labelpad=170, loc='bottom', fontsize=smallfont1
        )

    plt.figtext(0.9, 0.01, '*Contrast distributed by ABCD', ha='left', fontsize=10)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def get_sim_set_con_significant(data, omit_string, omit_noderiv):
    # Get number of simulations from data
    first_con = data['contrast'].unique()[0]
    first_model = data['model'].unique()[0]
    nsims = data[
        (data['contrast'] == first_con) & (data['model'] == first_model)
    ].shape[0]
    sig_prop_cutoff = 0.066
    dat_sig = (
        data.groupby(['contrast', 'model'])[['sigp']]
        .mean()
        .reset_index()
        .pivot(index='contrast', columns='model', values='sigp')
        .transpose()
    )
    dat_sig = dat_sig[dat_sig.columns.drop(list(dat_sig.filter(regex=omit_string)))]
    if omit_noderiv:
        dat_sig = dat_sig.loc[~dat_sig.index.str.contains('NoDeriv|SaturatedDeriv')]
    else:
        dat_sig = dat_sig.loc[~dat_sig.index.str.contains('SaturatedDeriv')]
    cue_cols = sorted(
        [col for col in dat_sig.columns if col.startswith('Cue')], key=sort_key
    )
    fb_cols = sorted(
        [col for col in dat_sig.columns if col.startswith('FB')], key=sort_key
    )
    dat_sig = dat_sig[cue_cols + fb_cols]
    rows, cols = np.where(dat_sig.values > sig_prop_cutoff)
    dat_sig_loc_tuples = [(cols[i], rows[i]) for i in range(len(rows))]
    return dat_sig_loc_tuples


def sort_key(x):
    has_dash = '-' in x
    has_star = '*' in x
    if not has_dash and not has_star:
        return (0, x)
    elif has_dash and not has_star:
        return (1, x)
    else:  # has both '-' and '*'
        return (2, x)


def plot_bias_significance(
    results,
    contrasts_only=False,
    omit_noderiv=False,
    jitter=False,
    fig_path=None,
    newname_cue_yes_deriv=None,
    nsubs=500,
):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap.set_bad('lightgrey')

    if jitter:
        title = 'Bias (jittered ITI)'
    else:
        title = 'Bias'

    bigfont = 14
    smallfont1 = 10
    smallfont2 = 10
    fig_width = 8
    rotate_xlab = 20
    if not contrasts_only:
        smallfont2 = 5
        fig_width = 8
        rotate_xlab = 90

    num_plots = len(results.keys())
    f, axs = plt.subplots(
        len(results.keys()),
        1,  # gridspec_kw={'hspace': 0.5},
        figsize=(fig_width, 10),
        sharex=True,
    )
    cbar_ax = f.add_axes([0.91, 0.2, 0.03, 0.5])
    f.suptitle(
        f"{title} \nAverage of group Cohen's Ds across simulations \nBias occurs when values are nonzero",
        fontsize=bigfont,
    )
    if contrasts_only:
        omit_string = '^((?!-).)*$'
    else:
        omit_string = None
    for idx, (setting, data) in enumerate(results.items()):
        data = data.copy()
        if newname_cue_yes_deriv is not None:
            data.loc[data['model'] == 'CueYesDeriv', 'model'] = newname_cue_yes_deriv
        data.loc[data['plot_alpha_val_power_error'] == 1, 'tval'] = np.nan
        data['sigp'] = data['sigp'].astype(float)
        data.loc[data['plot_alpha_val_power_error'] == 1, 'sigp'] = pd.NA

        setting = setting.replace(', ', '\n')
        # add extra blank lines when needed for aesthetics
        if len(setting.split('\n')) < 3:
            setting += '\n   ' * (3 - len(setting.split('\n')))
        dat_plot = (
            data.groupby(['contrast', 'model'])[['tval']]
            .mean()
            .reset_index()
            .pivot(index='contrast', columns='model', values='tval')
            .transpose()
        )
        dat_plot = dat_plot / np.sqrt(nsubs)
        dat_plot = dat_plot[dat_plot.columns.drop(list(dat_plot.filter(regex='Deriv')))]
        if omit_string:
            dat_plot = dat_plot[
                dat_plot.columns.drop(list(dat_plot.filter(regex=omit_string)))
            ]
        if omit_noderiv:
            dat_plot = dat_plot.loc[
                ~dat_plot.index.str.contains('NoDeriv|SaturatedDeriv')
            ]
        else:
            dat_plot = dat_plot.loc[~dat_plot.index.str.contains('SaturatedDeriv')]
        cue_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('Cue')], key=sort_key
        )
        fb_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('FB')], key=sort_key
        )

        dat_plot = dat_plot[cue_cols + fb_cols]
        sig_cells = get_sim_set_con_significant(data, omit_string, omit_noderiv)
        g = sns.heatmap(
            dat_plot,
            vmin=-0.01,
            vmax=0.05,
            center=0,
            cbar=idx == 0,
            cmap=cmap,
            ax=axs[idx],
            cbar_ax=None if idx else cbar_ax,
            annot=True,
            fmt='.2f',
            annot_kws={'fontsize': smallfont2},
        )
        for sig_cell in sig_cells:
            fix_val = 0.02
            sig_cell = (sig_cell[0] + 0.015, sig_cell[1] + fix_val)
            g.add_patch(
                plt.Rectangle(
                    sig_cell,
                    1 - 0.028,
                    1 - fix_val / 2,
                    ec='#433e3d',
                    fc='none',
                    lw=4,
                )
            )
        g.set_yticklabels(g.get_yticklabels(), size=smallfont1, rotation=0)
        g.set_xticklabels(g.get_xticklabels(), size=smallfont1, rotation=rotate_xlab)

        axs[idx].set_xlabel('')
        axs[idx].set_ylabel(
            setting,
            rotation=0,
            labelpad=170,
            loc='bottom',
            fontsize=smallfont1,
        )
    plt.figtext(0.9, 0.01, '*Contrast distributed by ABCD', ha='left', fontsize=10)

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_bias_significance_poster(
    results,
    contrasts_only=False,
    omit_noderiv=False,
    jitter=False,
    fig_path=None,
    newname_cue_yes_deriv=None,
    nsubs=500,
):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap.set_bad('lightgrey')
    # cmap = LinearSegmentedColormap.from_list(
    #     'orange_white_gray',
    #     ['#888888', '#FFFFFF', 'orange'],  # left, center, right
    #     N=256,
    # )

    if jitter:
        title = 'Bias (jittered ITI)'
    else:
        title = 'Bias'

    bigfont = 55
    smallfont1 = 45
    smallfont2 = 45
    fig_width = 8
    rotate_xlab = 10
    if not contrasts_only:
        smallfont2 = 5
        fig_width = 8
        rotate_xlab = 90

    num_plots = len(results.keys())
    f, axs = plt.subplots(
        len(results.keys()),
        1,  # gridspec_kw={'hspace': 0.5},
        figsize=(25, 8),
        sharex=True,
    )
    cbar_ax = f.add_axes([0.91, 0.1, 0.03, 0.75])
    f.suptitle(
        f"{title} in CueFeedback Model \nAverage of group Cohen's Ds across simulations",
        fontsize=bigfont,
        y=1.09,
    )
    if contrasts_only:
        omit_string = '^((?!-).)*$'
    else:
        omit_string = None
    for idx, (setting, data) in enumerate(results.items()):
        data = data.copy()
        if newname_cue_yes_deriv is not None:
            data.loc[data['model'] == 'CueYesDeriv', 'model'] = newname_cue_yes_deriv
        data.loc[data['plot_alpha_val_power_error'] == 1, 'tval'] = np.nan
        data['sigp'] = data['sigp'].astype(float)
        data.loc[data['plot_alpha_val_power_error'] == 1, 'sigp'] = pd.NA
        setting = setting.replace(', ', '\n')
        # add extra blank lines when needed for aesthetics
        if len(setting.split('\n')) < 3:
            setting += '\n   ' * (3 - len(setting.split('\n')))
        dat_plot = (
            data.groupby(['contrast', 'model'])[['tval']]
            .mean()
            .reset_index()
            .pivot(index='contrast', columns='model', values='tval')
            .transpose()
        )
        dat_plot = dat_plot / np.sqrt(nsubs)
        dat_plot = dat_plot[dat_plot.columns.drop(list(dat_plot.filter(regex='Deriv')))]

        if omit_string:
            dat_plot = dat_plot[
                dat_plot.columns.drop(list(dat_plot.filter(regex=omit_string)))
            ]
        if omit_noderiv:
            dat_plot = dat_plot.loc[
                ~dat_plot.index.str.contains('NoDeriv|SaturatedDeriv|Saturated')
            ]
        else:
            dat_plot = dat_plot.loc[~dat_plot.index.str.contains('SaturatedDeriv')]

        cue_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('Cue')], key=sort_key
        )
        fb_cols = sorted(
            [col for col in dat_plot.columns if col.startswith('FB')], key=sort_key
        )

        dat_plot = dat_plot[cue_cols + fb_cols]
        sig_cells = get_sim_set_con_significant(data, omit_string, omit_noderiv)
        g = sns.heatmap(
            dat_plot,
            vmin=-0.01,
            vmax=0.05,
            center=0,
            cbar=idx == 0,
            cmap=cmap,
            ax=axs[idx],
            cbar_ax=None if idx else cbar_ax,
            annot=True,
            fmt='.2f',
            annot_kws={'fontsize': smallfont2},
        )
        if cbar_ax is not None:
            cbar_ax.tick_params(labelsize=20)
        for sig_cell in sig_cells:
            fix_val = 0.02
            sig_cell = (sig_cell[0] + 0.015, sig_cell[1] + fix_val)
            g.add_patch(
                plt.Rectangle(
                    sig_cell,
                    1 - 0.028,
                    1 - fix_val / 0.7,
                    ec='#433e3d',
                    fc='none',
                    lw=4,
                )
            )
        # g.set_yticklabels(g.get_yticklabels(), size=smallfont1, rotation=0)
        g.set_yticklabels('', size=smallfont1, rotation=0)
        g.set_xticklabels(g.get_xticklabels(), size=smallfont1, rotation=rotate_xlab)

        axs[idx].set_xlabel('')
        # axs[idx].set_ylabel(
        #     setting,
        #     rotation=0,
        #     labelpad=450,
        #     loc='bottom',
        #     fontsize=smallfont1,
        # )
        axs[idx].set_ylabel(
            setting,
            rotation=0,
            fontsize=smallfont1,
        )
        axs[idx].yaxis.set_label_coords(-0.4, -0.6)
        axs[idx].yaxis.label.set_ha('left')
        if (idx + 1) < num_plots:
            axs[idx].tick_params(
                axis='x', which='both', bottom=False, top=False, labelbottom=False
            )
        else:
            axs[idx].tick_params(axis='x', length=25, width=4)

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()
