# Test NCPT
import pytest
import numpy as np

from lumos_ncpt_tools.ncpt import NCPT
from lumos_ncpt_tools.utils import load_test_data


def test_ncpt():
    # Hard-coded parameters
    test_id = 2190554
    subtest_id = 29
    
    # Load test data
    df = load_test_data()
    ncpt = NCPT(df)
    
    # Remove all incomplete tests
    filt_df, _ = ncpt.filter_by_completeness()
    
    # Remove one subtest from one test
    df_removed = filt_df.drop(filt_df[(filt_df['test_run_id'] == test_id) 
                                      & (filt_df['specific_subtest_id'] == subtest_id)].index)
    ncpt_removed = NCPT(df_removed)
    
    df_to_test, _ = ncpt_removed.filter_by_completeness()
    assert test_id not in df_to_test['test_run_id']