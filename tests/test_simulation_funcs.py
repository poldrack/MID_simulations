import importlib.resources

import pandas as pd
import pytest
from mid_simulations.simulation_funcs import (
    create_design_matrices,
    get_beta_dicts,
    get_events_df_for_subject,
    get_subdata_long,
    insert_jitter,
    load_event_file,
    make_stick_function,
)


@pytest.fixture
def SUB():
    return 1


@pytest.fixture
def DATASET():
    return 'AHRB'


@pytest.fixture
def beta_dicts():
    return get_beta_dicts()


@pytest.fixture
def data_path():
    return importlib.resources.files('mid_simulations') / 'Data'


@pytest.fixture
def testfile(data_path):
    return f'{data_path}/AHRB/sub-01/ses-1/func/sub-01_ses-1_task-mid_run-01_events.tsv'


@pytest.fixture
def events_df(SUB, DATASET):
    return get_events_df_for_subject(SUB, DATASET)


@pytest.fixture
def events_long(SUB, DATASET):
    return get_subdata_long(SUB, dataset=DATASET, verbose=True)


@pytest.fixture
def design_matrices(events_long):
    return create_design_matrices(events_long)


def test_get_beta_dicts(beta_dicts):
    assert beta_dicts is not None


def test_load_events_file(testfile):
    df = load_event_file(testfile, verbose=True)
    assert df is not None
    assert df.shape[0] == 50
    assert df.shape[1] == 13


def test_load_events_file_raise_exception_on_invalid_sep(testfile):
    with pytest.raises(AssertionError):
        df = load_event_file(testfile, sep=None, verbose=True)


def test_load_events_file_raise_exception_on_sep_mismatch(testfile):
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


def test_make_stick_function():
    onsets = [0, 20]
    durations = [2, 2]
    length = 40
    sf = make_stick_function(onsets, durations, length, resolution=1)
    assert sf is not None
    assert sf.shape[0] == 40
    assert sf.sf.sum() == 4


def test_create_design_matrices(design_matrices):
    assert design_matrices is not None
