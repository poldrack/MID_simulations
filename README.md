## Code for simulations presented in the manuscript: "Unintended bias in the pursuit of collinearity solutions in fMRI analysis" 

This code contains simulations to assess the ability of various models for the Monetary Incentive Delay (MID) task for fMRI data.

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to generate and/or sync the dependencies to the local virtual environment.  The following will create the virtual environment, `.venv`, in the root directory. 

```
git clone https://github.com/poldrack/MID_simulations.git
cd MID_simulations
uv sync
```

The code that generates results and figures from simulation portion of the manuscript is contained in the notebooks in `manuscript_code`.  It relies on timing files from the ABCD data set, which cannot be shared here.  If you would like to run the code and do not have the ABCD data, you can instead use the AHRB data files, which are included.  Simply change the variable `dataset` to `dataset = 'AHRB'` within the notebook.

If using Jupyter, you can start a Jupyter server with access to this project's virtual environment using:
```
uv run --with jupyter jupyter lab
```
Otherwise, if using VSCode you can simply select `.venv` as the kernel.

The following files contain the main results

- [manuscript_code/manuscript_simulations_figures.ipynb](manuscript_code/manuscript_simulations_figures.ipynb) - This is the primary set of simulations in the paper.
- [manuscript_code/duration_rt_by_condition_outcome.ipynb](manuscript_code/duration_rt_by_condition_outcome.ipynb) - The exploration of how the stimulus durations and response times differ according to cue type and trial outome (hit/miss as well as Correct/Too Slow/Too Soon).
- [manuscript_code/vif_explore_dev.ipynb](manuscript_code/vif_explore_dev.ipynb) - This code generates the figures related to the appendices about the variance inflation factor, including the new VIF algorithm developed in this project.
- [manuscript_code/modeling_mistakees_fig1_generation.ipynb](manuscript_code/modeling_mistakees_fig1_generation.ipynb) - This code generates the plots used in panels of Figure 1 from the manuscript.

Older notebooks are included in the `old` directory; these were the initial notebooks that did not use a group model and are included only for the purposes of seeing how the analyses developed.