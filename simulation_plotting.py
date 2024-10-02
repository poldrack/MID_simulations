import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from scipy.stats import ttest_1samp
import pickle
from collections import OrderedDict
from nilearn.plotting import plot_design_matrix


def plot_proportion_sig(results_df, hue_order, axs_barplot):
    # The way I am changing alpha of the bars is a little hacky
    # It should be stable, I rely on the x-location of the bars (rounded) to match
    # up with the axis label index.  I also control the model order, so that part
    # should be correct.
    alpha_values = results_df.copy()
    alpha_values = alpha_values.drop(
        ["mean", "tval", "pval", "sigp"], axis=1
    ).drop_duplicates()

    sns.barplot(
        data=results_df,
        x="contrast",
        y="sigp",
        hue="model",
        ax=axs_barplot,
        hue_order=hue_order,
        order=results_df["contrast"],
    )
    x_tick_label_location = {
        text_obj.get_position()[0]: text_obj.get_text()
        for text_obj in axs_barplot.get_xticklabels()
    }
    axs_barplot.set_ylim(0, 1)
    axs_barplot.axhline(0.05, color="red", linestyle="--")
    axs_barplot.set_xlabel("Contrast")
    axs_barplot.set_ylabel("Proportion significant \n (solid=power, opaque=error rate)")
    axs_barplot.tick_params(axis="x", rotation=90)
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
                    (alpha_values["model"] == model_loop)
                    & (alpha_values["contrast"] == contrast_name)
                ]["plot_alpha_val_power_error"].values[0]
            )


def plot_contrast_estimates(results_df, hue_order, axs):
    alpha_values = results_df.copy()
    alpha_values = alpha_values.drop(
        ["mean", "tval", "pval", "sigp"], axis=1
    ).drop_duplicates()

    sns.violinplot(
        data=results_df,
        x="contrast",
        y="tval",
        hue="model",
        ax=axs,
        inner=None,
        hue_order=hue_order,
        legend=False,
    )
    axs.axhline(0.0, color="red", linestyle="--")
    axs.set_xlabel("Contrast")
    axs.set_ylabel("Group-level T-stats \n (opaque should have T-stats=0)")
    axs.tick_params(axis="x", rotation=90)
    x_tick_label_location = {
        text_obj.get_position()[0]: text_obj.get_text()
        for text_obj in axs.get_xticklabels()
    }
    x_labels_in_order = [text_obj.get_text() for text_obj in axs.get_xticklabels()]
    alpha_vec = []
    for contrast_name in x_labels_in_order:
        for model in hue_order:
            subset_df = alpha_values[
                (alpha_values["model"] == model)
                & (alpha_values["contrast"] == contrast_name)
            ]["plot_alpha_val_power_error"]
            if subset_df.shape[0] > 0:
                alpha_vec.append(subset_df.values[0])
    for subcollection, alpha in zip(axs.collections, alpha_vec):
        subcollection.set_alpha(alpha)


def plot_results(results_df, analysis_label, stacked=False):
    # my method for changing alpha of the bars is a little hacky
    # and it may not be stable.
    models_included = results_df["model"].unique()
    hue_order = ["Saturated", "CueYesDeriv", "CueNoDeriv"]
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


def plot_dict_of_results(results_dict, contrasts=True):
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
        contrast_rows = results_df_loop["contrast"].str.contains("-")
        if contrasts:
            keep = contrast_rows == True
        if not contrasts:
            keep = contrast_rows == False
        results_df_loop = results_df_loop[(keep == True)]
        models_included = results_df_loop["model"].unique()
        hue_order = ["Saturated", "CueYesDeriv", "CueNoDeriv"]
        hue_order = [model for model in hue_order if model in models_included]
        plot_proportion_sig(results_df_loop, hue_order, inner[0])
        plot_contrast_estimates(results_df_loop, hue_order, inner[1])
    plt.show()


def plot_design_ordered_regressors(desmat, desname, ax):
    """
    Plots design matrix with regressors ordered by trial type and stimulus type (within trial type)
    (regressors are now ordered when the design matrix is created)
    """
    plot_design_matrix(desmat, ax=ax)
    ax.set_title(desname)
    return ax


def plot_bias(results):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap.set_bad("lightgrey")

    num_plots = len(results.keys())
    f, axs = plt.subplots(
        len(results.keys()),
        1,  # gridspec_kw={'hspace': 0.5},
        figsize=(20, 10),
        sharex=True,
    )
    cbar_ax = f.add_axes([0.91, 0.4, 0.03, 0.5])
    f.suptitle(
        "Bias \nAverage of group T-statistics across simulations \nBias occurs when values are nonzero",
        fontsize=16,
    )
    for idx, (setting, data) in enumerate(results.items()):
        data = data.copy()
        data.loc[data["plot_alpha_val_power_error"] == 1, "tval"] = np.nan
        setting = setting.replace(", ", "\n")
        dat_plot = (
            data.groupby(["contrast", "model"])[["tval"]]
            .mean()
            .reset_index()
            .pivot(index="contrast", columns="model", values="tval")
            .transpose()
        )
        # dat_plot = dat_plot[dat_plot.columns.drop(list(dat_plot.filter(regex="-")))]
        dat_plot = dat_plot[
            dat_plot.columns.drop(list(dat_plot.filter(regex="Derivative")))
        ]
        # dat_plot = dat_plot[sorted(dat_plot.columns)]
        dat_plot = dat_plot[sorted(dat_plot.columns, key=lambda x: ("-" in x, x))]
        g = sns.heatmap(
            dat_plot,
            vmin=-0.5,
            vmax=1,
            center=0,
            cbar=idx == 0,
            cmap=cmap,
            ax=axs[idx],
            cbar_ax=None if idx else cbar_ax,
            annot=True,
            fmt=".2f",
        )
        if idx < num_plots - 1:
            axs[idx].set_xlabel("")
        axs[idx].set_ylabel(setting, rotation=0, labelpad=200, loc="bottom")
    plt.show()
