# Test OutliersMixin
import pytest
import numpy as np

from lumos_ncpt_tools.ncpt import NCPT
from lumos_ncpt_tools.utils import load_test_data


test_conditions = [
    (10e8),
    (-10e8)
]

@pytest.mark.parametrize(
    'outlier_val', 
    test_conditions
)
def test_outliers(outlier_val):
    # Hard-coded parameters
    thresh = 5
    outlier_id = -1
    check_subtest = 29
    
    # Load test data
    df = load_test_data()
    ncpt = NCPT(df)
    
    # Add row with an outlier score
    outlier_df = {key: np.nan for key in ncpt.df.columns}
    outlier_df['raw_score'] = outlier_val
    outlier_df['test_run_id'] = outlier_id
    outlier_df['specific_subtest_id'] = check_subtest
    ncpt.df = ncpt.df.append(outlier_df, ignore_index=True)
    
    # Filter outliers
    check_subtests = ncpt.df['specific_subtest_id'].unique()
    filt_df = ncpt.filter_outliers_by_subtest('raw_score', thresh, [check_subtest])
    assert outlier_id not in filt_df['test_run_id']