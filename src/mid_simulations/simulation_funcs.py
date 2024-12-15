# functions for simulations

import importlib.resources
import multiprocessing
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm import expression_to_contrast_vector
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative
from scipy.stats import gamma, ttest_1samp

data_path = importlib.resources.files('mid_simulations') / 'Data'


def get_beta_dicts(dataset='AHRB'):
    if (dataset == 'AHRB') or (dataset == 'testdata'):
        beta_dicts = [
            {},
            {'Cue: LargeWin': 0.35, 'Cue: SmallWin': 0.35},
            {'Fixation: LargeWin': 0.35, 'Fixation: SmallWin': 0.35},
            {'Probe': 1.1},
            {'Feedback: LargeWinHit': 0.4, 'Feedback: SmallWinHit': 0.4},
        ]
    elif dataset == 'ABCD':
        beta_dicts = [
            {},
            {'Cue: LargeWin': 0.15, 'Cue: SmallWin': 0.15},
            {'Fixation: LargeWin': 0.15, 'Fixation: SmallWin': 0.15},
            # {
            #     'Cue: LargeWin': 0.15,
            #     'Cue: SmallWin': 0.15,
            #     'Fixation: LargeWin': 0.15,
            #     'Fixation: SmallWin': 0.15,
            # },
            {'Probe': 0.55},
            {'Probe: RT': 0.3},
            # {
            #     'Cue: LargeWin': 0.15,
            #     'Cue: SmallWin': 0.15,
            #     'Fixation: LargeWin': 0.15,
            #     'Fixation: SmallWin': 0.15,
            #     'Probe': 0.5,
            #     'Probe: RT': 0.3,
            # },
            {'Feedback: LargeWinHit': 0.18, 'Feedback: SmallWinHit': 0.18},
            # {'Feedback: LargeWinHit': 0.09, 'Feedback: LargeWinMiss': -0.09},
        ]
    return beta_dicts


desmat_column_rename = {
    'CUE_LargeGain': 'Cue: LargeWin',
    'CUE_LargeGain_derivative': 'Cue: LargeWin Deriv',
    'CUE_LargeLoss': 'Cue: LargeLoss',
    'CUE_LargeLoss_derivative': 'Cue: LargeLoss Deriv',
    'CUE_NoMoneyStake': 'Cue: Neutral',
    'CUE_NoMoneyStake_derivative': 'Cue: Neutral Deriv',
    'CUE_SmallGain': 'Cue: SmallWin',
    'CUE_SmallGain_derivative': 'Cue: SmallWin Deriv',
    'CUE_SmallLoss': 'Cue: SmallLoss',
    'CUE_SmallLoss_derivative': 'Cue: SmallLoss Deriv',
    'FIXATION_LargeGain': 'Fixation: LargeWin',
    'FIXATION_LargeLoss': 'Fixation: LargeLoss',
    'FIXATION_NoMoneyStake': 'Fixation: Neutral',
    'FIXATION_SmallGain': 'Fixation: SmallWin',
    'FIXATION_SmallLoss': 'Fixation: SmallLoss',
    'PROBE': 'Probe',
    'PROBE_RT': 'Probe: RT',
    'FEEDBACK_HIT_LargeGain': 'Feedback: LargeWinHit',
    'FEEDBACK_HIT_LargeLoss': 'Feedback: LargeLossHit',
    'FEEDBACK_HIT_NoMoneyStake': 'Feedback: NeutralHit',
    'FEEDBACK_HIT_SmallGain': 'Feedback: SmallWinHit',
    'FEEDBACK_HIT_SmallLoss': 'Feedback: SmallLossHit',
    'FEEDBACK_MISS_LargeGain': 'Feedback: LargeWinMiss',
    'FEEDBACK_MISS_LargeLoss': 'Feedback: LargeLossMiss',
    'FEEDBACK_MISS_NoMoneyStake': 'Feedback: NeutralMiss',
    'FEEDBACK_MISS_SmallGain': 'Feedback: SmallWinMiss',
    'FEEDBACK_MISS_SmallLoss': 'Feedback: SmallLossMiss',
    'FEEDBACK_HIT_LargeGain_derivative': 'Feedback: LargeWinHit Deriv',
    'FEEDBACK_HIT_LargeLoss_derivative': 'Feedback: LargeLossHit Deriv',
    'FEEDBACK_HIT_NoMoneyStake_derivative': 'Feedback: NeutralHit Deriv',
    'FEEDBACK_HIT_SmallGain_derivative': 'Feedback: SmallWinHit Deriv',
    'FEEDBACK_HIT_SmallLoss_derivative': 'Feedback: SmallLossHit Deriv',
    'FEEDBACK_MISS_LargeGain_derivative': 'Feedback: LargeWinMiss Deriv',
    'FEEDBACK_MISS_LargeLoss_derivative': 'Feedback: LargeLossMiss Deriv',
    'FEEDBACK_MISS_NoMoneyStake_derivative': 'Feedback: NeutralMiss Deriv',
    'FEEDBACK_MISS_SmallGain_derivative': 'Feedback: SmallWinMiss Deriv',
    'FEEDBACK_MISS_SmallLoss_derivative': 'Feedback: SmallLossMiss Deriv',
}


def load_event_file(eventfile: str, verbose=False, sep='\t'):
    assert sep in ['\t', ','], 'invalid separator'
    df = pd.read_csv(eventfile, sep=sep)
    if verbose:
        print(f'Loading {eventfile}')
        print(df.shape)
    assert df.shape[1] > 2, 'bad parsing'
    return df


def check_events_df_long(events_long, dataset='AHRB'):
    required_columns = ['onset', 'duration', 'trial_type']
    num_trial_types = len(events_long['trial_type'].unique())
    assert all(
        [col in events_long.columns for col in required_columns]
    ), 'events_long missing required columns'
    assert events_long.shape[0] > 0, 'events_df has no rows'
    assert events_long.shape[1] == 3, 'events_df has extra columns'
    if dataset == 'ABCD':
        assert num_trial_types == 22, 'ABCD should have 22 trial types'
    elif (dataset == 'AHRB') or (dataset == 'testdata'):
        assert num_trial_types == 21, 'AHRB should have 21 trial types'


def get_events_df_for_subject(sub, dataset='AHRB', verbose=False):
    sub = str(sub).zfill(2)
    events_df = None
    maxtime = None
    for run in [1, 2]:
        eventfile = f'{data_path}/{dataset}/sub-{sub}/ses-1/func/sub-{sub}_ses-1_task-mid_run-0{run}_events.tsv'
        df = load_event_file(eventfile, verbose)
        if events_df is None:
            events_df = df
            maxtime = np.ceil(
                events_df['FEEDBACK_ONSET'].max()
                + events_df['FEEDBACK_DURATION'].values[-1]
            )
        else:
            onset_cols = [col for col in df.columns if 'ONSET' in col]
            for col in onset_cols:
                df[col] += maxtime
            events_df = pd.concat([events_df, df])
    events_df.reset_index(inplace=True)
    events_df['trial_number'] = events_df.index
    return events_df


def get_subdata_long(sub, dataset='AHRB', verbose=False):
    # reformat from wide to long format
    if dataset not in ['AHRB', 'ABCD', 'testdata']:
        raise ValueError('Invalid dataset, must be AHRB, ABCD, or testdata')
    events_df = get_events_df_for_subject(sub, dataset=dataset, verbose=verbose)

    if (dataset == 'AHRB') or (dataset == 'testdata'):
        event_names = ['CUE', 'FIXATION', 'PROBE', 'FEEDBACK']
    elif dataset == 'ABCD':
        events_df['PROBE_RT_ONSET'] = events_df['PROBE_ONSET']
        events_df['PROBE_RT_DURATION'] = events_df['RT_corrected'] / 1000
        event_names = ['CUE', 'FIXATION', 'PROBE', 'PROBE_RT', 'FEEDBACK']

    events_long_onsets = events_df.melt(
        id_vars=['TRIAL_TYPE', 'PROBE_HIT', 'trial_number'],
        value_vars=[f'{val}_ONSET' for val in event_names],
        var_name='event',
        value_name='onset',
    )
    events_long_onsets['event'] = events_long_onsets['event'].str.replace('_ONSET', '')
    events_long_durations = events_df.melt(
        id_vars=['TRIAL_TYPE', 'PROBE_HIT', 'trial_number'],
        value_vars=[f'{val}_DURATION' for val in event_names],
        var_name='event',
        value_name='duration',
    )
    events_long_durations['event'] = events_long_durations['event'].str.replace(
        '_DURATION', ''
    )

    events_long = pd.merge(
        events_long_onsets,
        events_long_durations,
        on=['TRIAL_TYPE', 'PROBE_HIT', 'trial_number', 'event'],
    )
    events_long.sort_values('onset', inplace=True)
    events_long['FEEDBACK_HIT_MISS'] = events_long['PROBE_HIT'].map(
        {1: 'FEEDBACK_HIT', 0: 'FEEDBACK_MISS'}
    )
    events_long.loc[events_long['event'] == 'FEEDBACK', 'event'] = events_long.loc[
        events_long['event'] == 'FEEDBACK', 'FEEDBACK_HIT_MISS'
    ]

    # if (dataset == 'AHRB') or (dataset == 'testdata'):
    #    events_long['event'] = events_long['event'] + '_'
    # if dataset == 'ABCD':
    events_long.loc[events_long['event'].str.contains('PROBE'), 'TRIAL_TYPE'] = ''
    events_long.loc[~events_long['event'].str.contains('PROBE'), 'event'] = (
        events_long.loc[~events_long['event'].str.contains('PROBE'), 'event'] + '_'
    )

    events_long['trial_type'] = events_long['event'] + events_long['TRIAL_TYPE']
    events_long.drop(
        ['TRIAL_TYPE', 'PROBE_HIT', 'FEEDBACK_HIT_MISS', 'trial_number', 'event'],
        axis=1,
        inplace=True,
    )
    events_long.reset_index(drop=True, inplace=True)
    check_events_df_long(events_long, dataset=dataset)
    return events_long


def insert_jitter(events_in, min_iti=2, max_iti=6):
    events = events_in.copy()
    events.sort_values('onset', inplace=True)
    # Check number of stimuli between Cue stimuli
    num_stim_between_cue_trials = (
        events.index[events['trial_type'].str.contains('CUE')].diff().dropna()
    )
    assert (
        len(num_stim_between_cue_trials.unique()) == 1
    ), 'different number of stimuli between CUE events'
    num_jitter = int(events.shape[0] / num_stim_between_cue_trials[0])
    jitter = np.round(np.random.uniform(min_iti, max_iti, num_jitter), 2)
    jitter[0] = 0
    cumulative_jitter = np.cumsum(jitter)
    cumulative_jitter_repeat = np.repeat(
        cumulative_jitter, num_stim_between_cue_trials[0]
    )
    events['onset'] += cumulative_jitter_repeat
    return events


# I no longer use this
def spm_hrf_russ(TR, p=[6, 16, 1, 1, 6, 0, 32]):
    """An implementation of spm_hrf.m from the SPM distribution
    Arguments:
    Required:
    TR: repetition time at which to generate the HRF (in seconds)
    Optional:
    p: list with parameters of the two gamma functions:
                                                        defaults
                                                        (seconds)
    p[0] - delay of response (relative to onset)         6
    p[1] - delay of undershoot (relative to onset)      16
    p[2] - dispersion of response                        1
    p[3] - dispersion of undershoot                      1
    p[4] - ratio of response to undershoot               6
    p[5] - onset (seconds)                               0
    p[6] - length of kernel (seconds)                   32

    """
    p = [float(x) for x in p]
    fMRI_T = 16.0
    TR = float(TR)
    dt = TR / fMRI_T
    u = np.arange(p[6] / dt + 1) - p[5] / dt
    hrf = (
        gamma.pdf(u, p[0] / p[2], scale=1.0 / (dt / p[2]))
        - gamma.pdf(u, p[1] / p[3], scale=1.0 / (dt / p[3])) / p[4]
    )
    good_pts = np.array(range(int(p[6] / TR))) * fMRI_T
    hrf = hrf[list(good_pts.astype('int'))]
    hrf = hrf / np.sum(hrf)
    return hrf


def make_stick_function(onsets, durations, length, resolution=0.2):
    """
    Create a stick function with onsets and durations

    Parameters
    ----------
    onsets : list
        List of onset times
    durations : list
        List of duration times
    length : float
        Length of the stick function (in seconds)
    resolution : float
        Resolution of the stick function (in seconds)
        0.2 secs by default

    Returns
    -------
    sf : np.array
        Timepoints of the stick function
    """
    timepoints = np.arange(0, length, resolution)
    sf = np.zeros_like(timepoints)
    for onset, duration in zip(onsets, durations):
        sf[(timepoints >= onset) & (timepoints < onset + duration)] = 1
    sf_df = pd.DataFrame({'sf': sf})
    sf_df.index = timepoints
    return sf_df


def create_design_matrix(
    events_df_long,
    oversampling=50,
    tr=1,
    verbose=False,
    add_deriv=False,
    desmat_column_rename=desmat_column_rename,
):
    conv_resolution = tr / oversampling
    maxtime = np.ceil(np.max(events_df_long['onset']) + 10)
    timepoints_conv = np.round(np.arange(0, maxtime, conv_resolution), 3)
    timepoints_data = np.round(np.arange(0, maxtime, tr), 3)
    hrf_func = spm_hrf(tr, oversampling=oversampling)
    hrf_deriv_func = spm_time_derivative(tr, oversampling=oversampling)
    if verbose:
        print(f'Maxtime: {maxtime}')
        print(f'Timepoints convolution: {timepoints_conv.shape}')
        print(f'Timepoints data: {timepoints_data.shape}')
    trial_types = events_df_long['trial_type'].unique()
    desmtx_microtime = pd.DataFrame()
    desmtx_conv_microtime = pd.DataFrame()

    for trial_type in trial_types:
        trial_type_onsets = events_df_long[events_df_long['trial_type'] == trial_type][
            'onset'
        ].values
        trial_type_durations = events_df_long[
            events_df_long['trial_type'] == trial_type
        ]['duration'].values
        sf_df = make_stick_function(
            trial_type_onsets, trial_type_durations, maxtime, resolution=conv_resolution
        )
        desmtx_microtime[trial_type] = sf_df.sf.values
        desmtx_conv_microtime[trial_type] = np.convolve(sf_df.sf.values, hrf_func)[
            : sf_df.shape[0]
        ]
        if add_deriv:
            desmtx_conv_microtime[f'{trial_type}_derivative'] = np.convolve(
                sf_df.sf.values, hrf_deriv_func
            )[: sf_df.shape[0]]
    desmtx_conv_microtime.index = timepoints_conv
    desmtx_conv = desmtx_conv_microtime.loc[timepoints_data]
    desmtx_conv = desmtx_conv.rename(columns=desmat_column_rename)
    desmtx_conv = desmtx_conv[sorted(desmtx_conv.columns)]
    desmtx_conv['constant'] = 1
    return desmtx_conv


def create_design_matrices(
    events_df_long, oversampling=5, tr=1, gen_saturated_deriv=False, verbose=False
):
    all_designs = {}
    all_designs['Saturated'] = create_design_matrix(
        events_df_long,
        oversampling=oversampling,
        tr=tr,
        verbose=verbose,
        add_deriv=False,
    )
    if gen_saturated_deriv:
        all_designs['SaturatedDeriv'] = create_design_matrix(
            events_df_long,
            oversampling=oversampling,
            tr=tr,
            verbose=verbose,
            add_deriv=True,
        )
    events_df_long_cue_imp = events_df_long[
        events_df_long['trial_type'].str.contains('CUE|FEEDBACK')
    ].copy()
    events_df_long_cue_imp['duration'] = tr / oversampling
    all_designs['CueNoDeriv'] = create_design_matrix(
        events_df_long_cue_imp,
        oversampling=oversampling,
        tr=tr,
        verbose=verbose,
        add_deriv=False,
    )
    all_designs['CueYesDeriv'] = create_design_matrix(
        events_df_long_cue_imp,
        oversampling=oversampling,
        tr=tr,
        verbose=verbose,
        add_deriv=True,
    )
    return all_designs


def generate_data_nsim(desmtx_conv, beta_dict, nsims=100, noise_sd=1, beta_sub_sd=1):
    """
    Generate data based on the design matrix and beta values

    Parameters
    ----------

    desmtx_conv : pd.DataFrame
        Design matrix with convolved regressors
    beta_dict : dict
        Dictionary of beta values for each regressor of interest
    nsims : int
        Number of simulations to return
    noise_sd : float
        Standard deviation of the noise
    beta_sub_sd : float
        Standard deviation of the betas across subjects
    """
    # check the beta dict
    betas = np.zeros((desmtx_conv.shape[1], 1))
    for key in beta_dict.keys():
        assert key in desmtx_conv.columns, f'{key} not in desmtx'
    betas = np.array(
        [
            beta_dict[key] if key in beta_dict.keys() else 0
            for key in desmtx_conv.columns
        ],
        dtype='float32',
    )
    betas = np.atleast_2d(betas).T
    betas_mat_noise = betas @ np.ones((1, nsims)) + np.random.normal(
        0, beta_sub_sd, (betas.shape[0], nsims)
    )
    #        betas += np.random.normal(0, beta_sub_sd, betas.shape)
    ntime = desmtx_conv.shape[0]

    data = desmtx_conv @ betas_mat_noise + np.random.normal(0, noise_sd, (ntime, nsims))
    return data


def create_contrasts_NOT_USING(designs):
    """
    Creates contrast matrix for each of the designs that will produce estimates for
     each of the regressors, ANT: W-Neut, ANT: LW - Neut, FB: WHit - NeutHit, FB: LWHit - LWMiss
    input:
    designs: dict
        Dictionary of design matrices
    output:
    contrasts_strings: dict
        Dictionary of contrast in string format
    contrast_matrices: dict
        Dictionary of contrast matrices, ncontrast x nregressors for each design
    c_pinv_x_mat: dict
        Dictionary of contrast matrices x pinv(design matrix), ncontrast x ntimepoints for each design.
          To be used for contrast estimation (avoids recomputing pinv)
    """
    ant_w_vs_neut = {
        'saturated': '.5 * CUE_LargeGain  + .5 * CUE_SmallGain - 1 * CUE_NoMoneyStake',
        'cue only': '.5 * CUE_LargeGain + .5 * CUE_SmallGain - 1 * CUE_NoMoneyStake',
        'fix only': '.5 * FIXATION_LargeGain + .5 * FIXATION_SmallGain - 1 * FIXATION_NoMoneyStake',
        'cue fix': '.5 * CUEFIX_LargeGain + .5 * CUEFIX_SmallGain - 1 * CUEFIX_NoMoneyStake',
    }
    ant_lw_vs_neut = {
        'saturated': '1 * CUE_LargeGain - 1 * CUE_NoMoneyStake',
        'cue only': '1 * CUE_LargeGain - 1 * CUE_NoMoneyStake',
        'fix only': '1 * FIXATION_LargeGain - 1 * FIXATION_NoMoneyStake',
        'cue fix': '1 * CUEFIX_LargeGain - 1 * CUEFIX_NoMoneyStake',
    }
    contrasts_strings = {}
    contrast_matrices = {key: [] for key in designs.keys()}
    c_pinv_xmats = {}
    for desname, desmat in designs.items():
        design_columns = np.sort(desmat.columns)
        contrasts_strings[desname] = {
            colname: colname for colname in design_columns if 'constant' not in colname
        }
        contrasts_strings[desname]['ANT: W-Neut'] = ant_w_vs_neut[desname]
        contrasts_strings[desname]['ANT: LW-Neut'] = ant_lw_vs_neut[desname]
        contrasts_strings[desname]['FB: WHit-NeutHit'] = (
            '.5 * FEEDBACK_HIT_LargeGain + .5 * FEEDBACK_HIT_SmallGain - 1 * FEEDBACK_HIT_NoMoneyStake'
        )
        contrasts_strings[desname]['FB: LWHit-LWMiss'] = (
            '1 * FEEDBACK_HIT_LargeGain - 1 * FEEDBACK_MISS_LargeGain'
        )
        contrast_matrices[desname] = np.array(
            [
                expression_to_contrast_vector(
                    contrasts_strings[desname][key], desmat.columns
                )
                for key in contrasts_strings[desname].keys()
            ]
        )
        pinv_desmat = np.linalg.pinv(desmat)
        c_pinv_xmats[desname] = contrast_matrices[desname] @ pinv_desmat
    return contrasts_strings, contrast_matrices, c_pinv_xmats


def create_contrasts(designs):
    """
    Creates contrast matrix for each of the designs that will produce estimates for
     each of the regressors, ANT: W-Neut, ANT: LW - Neut, FB: WHit - NeutHit, FB: LWHit - LWMiss
    input:
    designs: dict
        Dictionary of design matrices
    output:
    contrasts_strings: dict
        Dictionary of contrast in string format
    contrast_matrices: dict
        Dictionary of contrast matrices, ncontrast x nregressors for each design
    c_pinv_x_mat: dict
        Dictionary of contrast matrices x pinv(design matrix), ncontrast x ntimepoints for each design.
          To be used for contrast estimation (avoids recomputing pinv)
    """
    contrasts_strings = {}
    contrast_matrices = {key: [] for key in designs.keys()}
    c_pinv_xmats = {}
    for desname, desmat in designs.items():
        design_columns = np.sort(desmat.columns)
        col_names_no_space = [name.replace(' ', '') for name in desmat.columns]
        col_names_no_space_no_colon = [
            name.replace(':', '_') for name in col_names_no_space
        ]
        contrasts_strings[desname] = {
            colname: colname.replace(' ', '').replace(':', '_')
            for colname in design_columns
            if 'constant' not in colname
        }
        contrasts_strings[desname]['Cue: W-base'] = (
            '.5 * Cue_LargeWin  + .5 * Cue_SmallWin'
        )
        contrasts_strings[desname]['Cue: W-Neut'] = (
            '.5 * Cue_LargeWin  + .5 * Cue_SmallWin - 1 * Cue_Neutral'
        )
        contrasts_strings[desname]['Cue: LW-Neut'] = (
            '1 * Cue_LargeWin - 1 * Cue_Neutral'
        )
        contrasts_strings[desname]['FB: WHit-NeutHit'] = (
            '.5 * Feedback_LargeWinHit + .5 * Feedback_SmallWinHit - 1 * Feedback_NeutralHit'
        )
        contrasts_strings[desname]['FB: LWHit-LWMiss'] = (
            '1 * Feedback_LargeWinHit - 1 * Feedback_LargeWinMiss'
        )

        contrast_matrices[desname] = np.array(
            [
                expression_to_contrast_vector(
                    contrasts_strings[desname][key], col_names_no_space_no_colon
                )
                for key in contrasts_strings[desname].keys()
            ]
        )
        pinv_desmat = np.linalg.pinv(desmat)
        c_pinv_xmats[desname] = contrast_matrices[desname] @ pinv_desmat
    return contrasts_strings, contrast_matrices, c_pinv_xmats


def est_efficiencies(designs, contrast_matrices):
    """
    Estimate the efficiency of the designs based on the contrast matrices
    """
    efficiencies = {}
    for desname, desmat in designs.items():
        efficiencies[desname] = 1 / np.diag(
            contrast_matrices[desname]
            @ np.linalg.inv(desmat.transpose() @ desmat)
            @ contrast_matrices[desname].transpose()
        )
    return efficiencies


def est_des_covs(designs):
    """
    Estimate the pairwise covariances between regressors for all designs
    """
    cov_mats = {}
    for desname, desmat in designs.items():
        cov_mats[desname] = desmat.cov()
    return cov_mats


def est_vifs(designs, contrast_matrices):
    """
    Estimate the variance inflation factor (VIF) for the designs
    """
    vifs = {}
    for desname, desmat in designs.items():
        desmat_centered_scaled = desmat.copy()
        column_keep = desmat_centered_scaled.std() != 0
        desmat_centered_scaled = desmat_centered_scaled.loc[:, column_keep]
        desmat_centered_scaled = (
            desmat_centered_scaled - desmat_centered_scaled.mean()
        ) / desmat_centered_scaled.std()
        contrast_mat = contrast_matrices[desname][:, column_keep]

        worst_case_covmat = np.linalg.pinv(
            desmat_centered_scaled.transpose() @ desmat_centered_scaled
        )
        best_case_covmat = np.linalg.pinv(
            np.multiply(
                desmat_centered_scaled.transpose() @ desmat_centered_scaled,
                np.identity(desmat_centered_scaled.shape[1]),
            )
        )
        vifs[desname] = np.diag(
            contrast_mat @ worst_case_covmat @ contrast_mat.transpose()
        ) / np.diag((contrast_mat @ best_case_covmat @ contrast_mat.transpose()))
    return vifs


def est_baseline_max_range(events, oversampling=50, tr=0.8):
    avg_durations = events.groupby('trial_type')['duration'].mean().reset_index()
    avg_durations.columns = ['trial_type', 'duration']
    avg_durations['onset'] = 10
    designs = create_design_matrices(
        avg_durations, oversampling=oversampling, tr=tr, gen_saturated_deriv=True
    )
    base_max_ranges = {}
    for desname, desmat in designs.items():
        base_max_ranges[desname] = desmat.max()
    return base_max_ranges


def scale_regressors(base_max_ranges, designs):
    designs_scaled = {}
    for desname, desmat in designs.items():
        designs_scaled[desname] = desmat.copy()
        for col in desmat.columns:
            designs_scaled[desname][col] = desmat[col] / base_max_ranges[desname][col]
    return designs_scaled


def orth_deriv_regs(designs):
    """
    Orthogonalize the derivative regressors
    """
    designs_orth = {}
    for desname, desmat in designs.items():
        desmat_orth_loop = desmat.copy()
        deriv_cols = [col for col in desmat.columns if 'Deriv' in col]
        for col in deriv_cols:
            col_no_deriv_label = col.replace(' Deriv', '')
            orth_maker_desmat = np.column_stack(
                (
                    desmat_orth_loop[col_no_deriv_label],
                    np.ones(len(desmat_orth_loop[col_no_deriv_label])),
                )
            )
            desmat_orth_loop[col] -= (
                orth_maker_desmat
                @ np.linalg.pinv(orth_maker_desmat.T @ orth_maker_desmat)
                @ orth_maker_desmat.T
                @ desmat_orth_loop[col]
            )
        designs_orth[desname] = desmat_orth_loop
    return designs_orth


def est_eff_vif_all_subs(
    oversampling=50,
    tr=0.8,
    jitter=False,
    jitter_iti_min=2,
    jitter_iti_max=6,
    dataset='AHRB',
    nsubs=None,
    orth_deriv=False,
):
    """
    Estimate the efficiency and variance inflation factor (VIF) for all subjects
    """
    subids = get_subids(dataset=dataset)
    if nsubs is None:
        nsubs = len(subids)
    # Placeholder for results
    for sub in range(1, nsubs + 1):
        events = None
        try:
            events = get_subdata_long(sub, dataset=dataset)
        except Exception as e:
            print(f'Error loading sub {sub}: {e}')
            continue
        if jitter:
            events = insert_jitter(
                events, min_iti=jitter_iti_min, max_iti=jitter_iti_max
            )

        designs = create_design_matrices(
            events, oversampling=oversampling, tr=tr, gen_saturated_deriv=True
        )
        if orth_deriv:
            designs = orth_deriv_regs(designs)
        base_max_ranges = est_baseline_max_range(
            events, oversampling=oversampling, tr=tr
        )
        # should make efficiencies comparable
        designs = scale_regressors(base_max_ranges, designs)
        if sub == 1:
            efficiencies = {model: [] for model in designs.keys()}
            vifs = {model: [] for model in designs.keys()}
        contrast_strings, contrast_matrices, c_pinv_xmats = create_contrasts(designs)
        efficiencies_loop = est_efficiencies(designs, contrast_matrices)
        vifs_loop = est_vifs(designs, contrast_matrices)
        for key in efficiencies.keys():
            efficiencies[key].append(efficiencies_loop[key])
            vifs[key].append(vifs_loop[key])
    eff_output, vif_output = organize_vifs_effs(
        efficiencies, vifs, contrast_strings, designs
    )
    return {'efficiencies': eff_output, 'vifs': vif_output}


def est_des_covmats_all_subs(
    oversampling=50,
    tr=0.8,
    jitter=False,
    jitter_iti_min=2,
    jitter_iti_max=6,
    dataset='AHRB',
    nsubs=108,
):
    """
    Estimate the efficiency and variance inflation factor (VIF) for all subjects
    """
    # Placeholder for results
    for sub in range(1, nsubs + 1):
        events = None
        try:
            events = get_subdata_long(sub, dataset=dataset)
        except Exception as e:
            print(f'Error loading sub {sub}: {e}')
            continue
        if jitter:
            events = insert_jitter(
                events, min_iti=jitter_iti_min, max_iti=jitter_iti_max
            )

        designs = create_design_matrices(events, oversampling=oversampling, tr=tr)
        if sub == 1:
            covmats = {model: [] for model in designs.keys()}
        covmats_loop = est_des_covs(designs)
        for key in covmats.keys():
            covmats[key].append(covmats_loop[key])
        covmat_avg = {}
        for key in covmats.keys():
            covmat_avg[key] = sum(covmats[key]) / len(covmats[key])
    return covmat_avg


def organize_vifs_effs(efficiencies, vifs, contrast_strings, designs):
    """
    Organize the efficiencies and VIFs into a DataFrame for easier analysis
    """
    efficiencies = {key: np.stack(val, axis=1) for key, val in efficiencies.items()}
    vifs = {key: np.stack(val, axis=1) for key, val in vifs.items()}
    eff_output = {}
    vif_output = {}
    for key in efficiencies.keys():
        eff_output[key] = pd.DataFrame(
            efficiencies[key].transpose(), columns=contrast_strings[key].keys()
        )
        vif_output[key] = pd.DataFrame(
            vifs[key].transpose(),
            columns=contrast_strings[key].keys(),
        )
        eff_output[key] = pd.melt(
            eff_output[key], var_name='contrast', value_name='efficiency'
        )
        eff_output[key]['model'] = key
        vif_output[key] = pd.melt(
            vif_output[key], var_name='regressor', value_name='vif'
        )
        vif_output[key]['model'] = key
    list_eff_dfs = [eff_output[key] for key in eff_output.keys()]
    eff_df = pd.concat(list_eff_dfs, axis=0)
    list_vif_dfs = [vif_output[key] for key in vif_output.keys()]
    vif_df = pd.concat(list_vif_dfs, axis=0)
    return eff_df, vif_df


def make_analysis_label(beta_dict, jitter=False, jitter_iti_min=2, jitter_iti_max=6):
    if len(beta_dict) > 0:
        analysis_label = ', '.join([f'{key}={val}' for key, val in beta_dict.items()])
    else:
        analysis_label = 'Null model'
    if jitter:
        analysis_label += f'\n jittered ITIs ({jitter_iti_min}-{jitter_iti_max})'
    return analysis_label


def get_subids(dataset='AHRB'):
    subdirs = glob(f'{data_path}/{dataset}/sub*')
    subids = [string_val.split('-', 1)[1] for string_val in subdirs]
    return subids


def sim_group_models_parallel(
    beta_dict,
    noise_sd,
    beta_sub_sd,
    nsims=100,
    oversampling=5,
    tr=1,
    jitter=False,
    jitter_iti_min=2,
    jitter_iti_max=6,
    verbose=False,
    n_jobs=None,
    dataset='AHRB',
    nsubs=None,
):
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1  # save one core for the OS
    subids = get_subids(
        dataset=dataset
    )  # not using this yet since there are 4383 ABCD subjects
    if nsubs is None:
        nsubs = len(subids)
    results = Parallel(n_jobs=n_jobs)(
        delayed(sim_data_est_cons_sub)(
            sub,
            beta_dict,
            noise_sd,
            beta_sub_sd,
            nsims=nsims,
            oversampling=oversampling,
            tr=tr,
            jitter=jitter,
            jitter_iti_min=jitter_iti_min,
            jitter_iti_max=jitter_iti_max,
            verbose=verbose,
            dataset=dataset,
        )
        for sub in range(1, nsubs + 1)
    )
    group_model_output_df, contrast_strings = organize_sim_data_results(results)
    desmats_example = results[0]['designs']
    # used in plotting to indicate when rejection rate reflects power (1) or error rate (.5)
    group_model_output_df = make_pow_error_column(
        group_model_output_df, beta_dict, contrast_strings
    )

    return group_model_output_df, contrast_strings, desmats_example


def sim_data_est_cons_sub(
    sub,
    beta_dict,
    noise_sd,
    beta_sub_sd,
    nsims=100,
    oversampling=5,
    tr=1,
    jitter=False,
    jitter_iti_min=2,
    jitter_iti_max=6,
    verbose=False,
    dataset='AHRB',
):
    try:
        events = get_subdata_long(sub, dataset=dataset)
    except Exception as e:
        # this is bad practice in general, but we need to do it here
        # because this is wrapped in a delayed call in joblib
        print(f'Error loading sub {sub}')
        print(e)
        return None
    if jitter:
        events = insert_jitter(events, min_iti=jitter_iti_min, max_iti=jitter_iti_max)
    designs = create_design_matrices(events, oversampling=oversampling, tr=tr)
    contrast_strings, contrasts_matrices, c_pinv_xmats = create_contrasts(designs)
    data = generate_data_nsim(
        designs['Saturated'],
        beta_dict,
        nsims=nsims,
        noise_sd=noise_sd,
        beta_sub_sd=beta_sub_sd,
    )
    contrast_ests = {}
    for desname, x_pinv_mat in c_pinv_xmats.items():
        contrast_ests[desname] = x_pinv_mat @ data
    output = {
        'contrast_strings': contrast_strings,
        'contrast_ests': contrast_ests,
        'designs': designs,
    }
    return output


def organize_sim_data_results(results):
    contrast_strings = results[0]['contrast_strings']
    all_contrast_ests_lists = {key: [] for key in contrast_strings.keys()}
    for result in results:
        if result is not None:
            contrast_ests = result['contrast_ests']
            for desname in contrast_strings.keys():
                all_contrast_ests_lists[desname].append(contrast_ests[desname])

    all_contrast_ests = {
        key: np.stack(val, axis=2) for key, val in all_contrast_ests_lists.items()
    }
    nsims = all_contrast_ests['Saturated'].shape[1]
    ouput_names = ['model', 'contrast', 'mean', 'tval', 'pval', 'sigp']
    group_model_output = {output_name: [] for output_name in ouput_names}
    for model in contrast_strings.keys():
        for contrast_number, contrast_name in enumerate(contrast_strings[model].keys()):
            contrast_mean = np.mean(
                all_contrast_ests[model][contrast_number, :, :], axis=1
            )
            contrast_tval, contrast_pval = ttest_1samp(
                all_contrast_ests[model][contrast_number, :, :],
                0,
                axis=1,
                alternative='two-sided',
            )
            contrast_sigp = contrast_pval < 0.05
            group_model_output['model'].extend([model] * nsims)
            group_model_output['contrast'].extend([contrast_name] * nsims)
            group_model_output['mean'].extend(contrast_mean)
            group_model_output['tval'].extend(contrast_tval)
            group_model_output['pval'].extend(contrast_pval)
            group_model_output['sigp'].extend(contrast_sigp)
    group_model_output_df = pd.DataFrame(group_model_output)
    return group_model_output_df, contrast_strings


def make_pow_error_column(results_df, beta_dict, contrast_strings):
    beta_dict_keys = list(beta_dict.keys())
    beta_dict_keys = [val.replace(': ', '_') for val in beta_dict_keys]
    models = results_df['model'].unique()

    results_df['plot_alpha_val_power_error'] = 0.5
    for model in models:
        contrasts_model = results_df.loc[
            results_df['model'] == model, 'contrast'
        ].unique()
        if model == 'cue fix':
            # for cue fix, both CUE_ and FIXATION_ betas add signal to CUEFIX_ beta
            beta_dict_keys = [val.replace('CUE_', 'CUEFIX_') for val in beta_dict_keys]
            beta_dict_keys = [
                val.replace('FIXATION_', 'CUEFIX_') for val in beta_dict_keys
            ]
        for contrast_model in contrasts_model:
            contrast_loop = contrast_strings[model][contrast_model]
            beta_keys_in_contrast = []
            for key in beta_dict_keys:
                if key in contrast_loop:
                    beta_keys_in_contrast.append(key)
            if len(beta_keys_in_contrast) > 0:
                results_df.loc[
                    (results_df['model'] == model)
                    & (results_df['contrast'] == contrast_model),
                    'plot_alpha_val_power_error',
                ] = 1
    return results_df


if __name__ == '__main__':  # pragma: no cover
    SUB = 1
    DATASET = 'testdata'

    events_df = get_events_df_for_subject(SUB, dataset=DATASET)

    events_long = get_subdata_long(SUB, dataset=DATASET, verbose=True)
    design_matrices = create_design_matrices(events_long, verbose=True)

    simdata = generate_data_nsim(design_matrices['Saturated'], {})

    contrast_strings, contrast_matrices, c_pinv_xmats = create_contrasts(
        design_matrices
    )

    nsims = 100
    beta_dicts = get_beta_dicts()
    for beta_dict in beta_dicts:
        data = generate_data_nsim(
            design_matrices['Saturated'],
            beta_dict,
            nsims=nsims,
            noise_sd=1,
            beta_sub_sd=1,
        )
        assert data is not None
        assert data.shape[1] == nsims
        assert all(np.var(data) > 0)
