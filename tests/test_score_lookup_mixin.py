# Test ScoreLookupMixin
import pytest
import numpy as np

from lumos_ncpt_tools.ncpt import NCPT
from lumos_ncpt_tools.utils import load_test_data


test_conditions = [
    ('rank_INT'),
    ('census_rank_INT')
]

@pytest.mark.parametrize(
    'norm_method', 
    test_conditions
)
def test_score_lookup(norm_method):
    subtests_to_negate = {39, 40, 48, 49}    
    df = load_test_data()
    ncpt = NCPT(df)
    
    # Add normalized scores
    ncpt.lookup_normed_scores(norm_method)
    score_col = f'{norm_method}_normed_score'
    assert score_col in ncpt.df.columns
    assert ncpt.df[score_col].isna().sum() == 0
    
    # Check that the relative ordering of the raw score + normalized scores is the same
    subtests = ncpt.df['specific_subtest_id'].unique()
    for sub in subtests:
        subtest_df = ncpt.df.query('specific_subtest_id == @sub')   
        if sub in subtests_to_negate:
            neg_scores = -1 * subtest_df['raw_score']
            subtest_df = subtest_df.assign(raw_score=neg_scores)
        subtest_filt = subtest_df.drop_duplicates(subset=['raw_score'])
        subtest_filt = subtest_filt.drop_duplicates(subset=[score_col])
        idx_sort_raw = subtest_filt.sort_values(by=['raw_score']).index.to_numpy()
        idx_sort_norm = subtest_filt.sort_values(by=[score_col]).index.to_numpy()
        assert np.array_equal(idx_sort_raw, idx_sort_norm)
    
    # Add Grand Index
    # Tests that GI is not calculated for incomplete assessments
    ncpt.add_grand_index()
    _, exclude_df = ncpt.filter_by_completeness() 
    assert 'grand_index' in ncpt.df.columns
    assert ncpt.df['grand_index'].isna().sum() == len(exclude_df)