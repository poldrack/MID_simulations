# tests for simulation functions
# at this point they are mostly just smoke tests

import pytest
import pandas as pd
import warnings
import numpy as np
from simulation_funcs import (
    load_event_file,
    get_events_df_for_subject,
    get_subdata_long,
    insert_jitter,
    spm_hrf,
    make_stick_function,
    create_design_matrices,
    generate_data_nsim,
    get_beta_dicts,
    create_contrasts,
    make_analysis_label,
    sim_group_models_parallel,
    sim_data_est_cons_sub
)

SUB = 1 
DATASET = 'testdata' 


@pytest.fixture
def events_df():
    return get_events_df_for_subject(SUB, DATASET)


@pytest.fixture
def events_long():
    return get_subdata_long(SUB, DATASET, verbose=True)


@pytest.fixture
def design_matrices(events_long):
    return create_design_matrices(events_long, verbose=True)


@pytest.fixture
def beta_dicts():
    return get_beta_dicts()


def test_load_events_file():
    testfile = 'testdata/sub-01/ses-1/func/sub-01_ses-1_task-mid_run-01_events.tsv'
    df = load_event_file(testfile, verbose=True)
    assert df is not None
    assert df.shape[0] == 50
    assert df.shape[1] == 13


def test_load_events_file_raise_exception_on_invalid_sep():
    testfile = 'testdata/sub-01/ses-1/func/sub-01_ses-1_task-mid_run-01_events.tsv'
    with pytest.raises(AssertionError):
        df = load_event_file(testfile, sep=None, verbose=True)



def test_load_events_file_raise_exception_on_sep_mismatch():
    testfile = 'testdata/sub-01/ses-1/func/sub-01_ses-1_task-mid_run-01_events.tsv'
    with pytest.raises(AssertionError):
        df = load_event_file(testfile, sep=',', verbose=True)



def test_get_events_df_for_subject(events_df):
    assert events_df is not None
    assert isinstance(events_df, pd.core.frame.DataFrame)
    assert events_df.shape[0] == 100


def test_get_subdata_long(events_long):
    assert events_long is not None
    assert events_long.shape[0] == 400
    assert events_long.shape[1] == 3


def test_insert_jitter(events_long):
    jittered_events_df = insert_jitter(events_long)
    assert jittered_events_df is not None


def test_spm_hrf():
    hrf = spm_hrf(2)
    assert hrf is not None
    assert len(hrf) == 16


def test_make_stick_function():
    onsets = [0, 20]
    durations = [2, 2]
    length = 40
    sf = make_stick_function(onsets, durations, length, resolution=1)
    assert sf is not None
    assert sf.shape[0] == 40
    assert sf.sf.sum() == 4


def test_create_design_matrices(design_matrices):
    testvar_dict = {
        'saturated': ['CUE_NoMoneyStake', 'FIXATION_NoMoneyStake', 'PROBE_NoMoneyStake',
            'FEEDBACK_MISS_NoMoneyStake', 'CUE_LargeLoss', 'FIXATION_LargeLoss',
            'PROBE_LargeLoss', 'FEEDBACK_HIT_LargeLoss',
            'FEEDBACK_HIT_NoMoneyStake', 'CUE_LargeGain', 'FIXATION_LargeGain',
            'PROBE_LargeGain', 'FEEDBACK_MISS_LargeGain', 'CUE_SmallLoss',
            'FIXATION_SmallLoss', 'PROBE_SmallLoss', 'FEEDBACK_MISS_SmallLoss',
            'FEEDBACK_MISS_LargeLoss', 'CUE_SmallGain', 'FIXATION_SmallGain',
            'PROBE_SmallGain', 'FEEDBACK_MISS_SmallGain', 'FEEDBACK_HIT_SmallLoss',
            'FEEDBACK_HIT_SmallGain', 'FEEDBACK_HIT_LargeGain', 'constant'],
        'cue only': ['CUE_NoMoneyStake', 'FEEDBACK_MISS_NoMoneyStake', 'CUE_LargeLoss',
            'FEEDBACK_HIT_LargeLoss', 'FEEDBACK_HIT_NoMoneyStake', 'CUE_LargeGain',
            'FEEDBACK_MISS_LargeGain', 'CUE_SmallLoss', 'FEEDBACK_MISS_SmallLoss',
            'FEEDBACK_MISS_LargeLoss', 'CUE_SmallGain', 'FEEDBACK_MISS_SmallGain',
            'FEEDBACK_HIT_SmallLoss', 'FEEDBACK_HIT_SmallGain',
            'FEEDBACK_HIT_LargeGain', 'constant'],
        'cue fix': ['CUEFIX_SmallGain', 'CUEFIX_SmallLoss', 'CUEFIX_LargeGain',
            'CUEFIX_LargeLoss', 'CUEFIX_NoMoneyStake', 'FEEDBACK_MISS_NoMoneyStake',
            'FEEDBACK_HIT_LargeLoss', 'FEEDBACK_HIT_NoMoneyStake',
            'FEEDBACK_MISS_LargeGain', 'FEEDBACK_MISS_SmallLoss',
            'FEEDBACK_MISS_LargeLoss', 'FEEDBACK_MISS_SmallGain',
            'FEEDBACK_HIT_SmallLoss', 'FEEDBACK_HIT_SmallGain',
            'FEEDBACK_HIT_LargeGain', 'constant']
    }
    for model, testvars in testvar_dict.items():
        assert model in design_matrices
        assert all([i in design_matrices[model].columns for i in testvars])
    

def test_get_beta_dicts(beta_dicts):
    assert beta_dicts is not None


def test_generate_data_nsim(design_matrices, beta_dicts):
    nsims = 100
    for beta_dict in beta_dicts:
        data = generate_data_nsim(design_matrices['saturated'], beta_dict, nsims=nsims, noise_sd=1, beta_sub_sd=1)
        assert data is not None
        assert data.shape[1] == nsims
        assert all(np.var(data) > 0)


def test_create_contrasts(design_matrices):
    contrast_strings, contrast_matrices, c_pinv_xmats = create_contrasts(design_matrices)

    for model in design_matrices.keys():
        assert model in contrast_matrices
        assert model in contrast_strings
        assert model in c_pinv_xmats
        if model == 'saturated':
            assert contrast_matrices[model].shape == (29, 26)
        else:
            assert contrast_matrices[model].shape == (19, 16)
            

def test_make_analysis_label(beta_dicts):
    for beta_dict in beta_dicts:
        figure_label = make_analysis_label(beta_dict)
        assert figure_label is not None

    assert 'Null model' in make_analysis_label({})


def test_make_analysis_label_jitter(beta_dicts):
    for beta_dict in beta_dicts:
        figure_label = make_analysis_label(beta_dict, jitter=True)
        assert figure_label is not None

    assert 'jittered' in make_analysis_label({}, jitter=True)


def test_sim_group_models_parallel(beta_dicts):
    nsims = 10
    noise_sd=1
    beta_sub_sd=1
    for beta_dict in beta_dicts:
        results, _ = sim_group_models_parallel(beta_dict, noise_sd, beta_sub_sd, 
                                               nsims=nsims, conv_resolution=.2, tr=1)
        assert results is not None

def test_sim_data_est_con_sub():
    nsims = 10
    noise_sd=1
    beta_sub_sd=1
    sub = 1    
    contrast_strings, contrast_ests  = sim_data_est_cons_sub(sub, {}, noise_sd, beta_sub_sd, 
                          nsims=nsims, conv_resolution=.2, tr=1)
    assert contrast_strings is not None
    assert contrast_ests is not None


def test_sim_data_est_con_sub_jitter():
    nsims = 10
    noise_sd=1
    beta_sub_sd=1
    sub = 1    
    contrast_strings, contrast_ests  = sim_data_est_cons_sub(sub, {}, noise_sd, beta_sub_sd, 
                          nsims=nsims, conv_resolution=.2, tr=1, jitter=True)
    assert contrast_strings is not None
    assert contrast_ests is not None


def test_sim_data_est_con_sub_returns_none_on_exception():
    nsims = 10
    noise_sd=1
    beta_sub_sd=1
    sub = 1
    contrast_strings, contrast_ests = sim_data_est_cons_sub(sub, {}, noise_sd, beta_sub_sd, 
                          nsims=nsims, conv_resolution=.2, tr=1)
    assert contrast_strings is None
    assert contrast_ests is None