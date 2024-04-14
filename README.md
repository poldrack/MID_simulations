This code contains simulations to assess the ability of various models for the Monetary Incentive Delay (MID) task for fMRI data.

The following notebooks are of greatest interest:

- [MID_sim_groupmodel.ipynb](MID_sim_groupmodel.ipynb) - This is the primary set of simulations in the paper.
- [MID_sim_groupmodel_jittered.ipynb](MID_sim_groupmodel_jittered.ipynb) - These are the simulations that include jittered intertrial intervals.
- [MID_AHRB_demographics.ipynb](MID_AHRB_demographics.ipynb) - This includes some analyses of demographics and behavior in the AHRB dataset.
- [MID_vif.ipynb](MID_vif.ipynb) - This notebook computes and plots the variance inflation factors for the jittered ITI models.

Older notebooks are included in the `older_notebooks` directory; these were the initial notebooks that did not use a group model and are included only for the purposes of seeing how the analyses developed.