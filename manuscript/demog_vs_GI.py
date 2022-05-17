import pkgutil
import os

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lumos_ncpt_tools.ncpt import NCPT
from lumos_ncpt_tools.utils import load_data

class Figure1():
    """Plot age vs. GI for each test battery"""
    
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    age_bin_edges = [18, 29, 39, 49, 59, 69, 99]
    age_bin_labels = ['18-29', '30-39', '40-49', '50-59', '60-69',
                      '70+']
    edu_bins = {'HS': [1, 2], 
                'college': [3, 4, 8], 
                'higher': [5, 6, 7]}
    edu_bin_order = ['HS', 'college', 'higher']
    edu_bin_labels = ["High school \n Some high school",
                      "College \n Some college \n Associate's degree",
                      "Master's degree \n Professional degree \n Ph.D."]
    ylim = [80, 115]
    
    def __init__(self, data_dir, save_dir, figsize):
        self.data_dir = data_dir
        self.png_path = os.path.join(save_dir, 'figure1.png')
        self.svg_path = os.path.join(save_dir, 'figure1.svg')        
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.batteries = self.config['batteries'].keys()
        self.figsize = figsize
        
    def make_figure(self):
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        all_dfs = []
        
        # Age
        for bat_id in self.batteries:
            bat_df = self._get_battery_data(bat_id)  
            sns.lineplot(x='binned_age', y='grand_index', data=bat_df, ax=axes[0], legend=False, lw=0.5,
                         **{'label': bat_id})
            df_new = bat_df.assign(battery=bat_id)
            all_dfs.append(df_new)
        
        axes[0].set_xlabel('Age (years)')
        axes[0].set_ylabel('Grand Index')
        axes[0].set_ylim(self.ylim)
        
        # Education
        combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        sns.barplot(x='edu_bin', y='grand_index', hue='battery', 
                    data=combined_df, errwidth=0.5, errcolor='k', ax=axes[1])
        axes[1].set_xticklabels(self.edu_bin_labels)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Grand Index')
        axes[1].set_ylim(self.ylim)
        axes[1].legend(loc='upper left', title='Battery', frameon=False,
                       ncol=1)
        
        plt.tight_layout()
        plt.show()        
        fig.savefig(self.png_path, bbox_inches='tight')
        fig.savefig(self.svg_path, transparent=True, bbox_inches='tight')   
        
    def _get_battery_data(self, bat_id):
        data_fn = f'battery{bat_id}_df.csv'
        bat_df = load_data(self.data_dir, data_fn) 
        bat_ncpt = NCPT(bat_df)
        del bat_df
        GI_df = bat_ncpt.df.drop_duplicates(subset=['test_run_id'])
        GI_df = GI_df.assign(binned_age=np.nan)
        GI_df = GI_df.assign(edu_bin=np.nan)
        
        # Age
        cuts = pd.cut(GI_df['age'], self.age_bin_edges,
                      labels=self.age_bin_labels,
                      include_lowest=True)       
        GI_df.loc[:, 'binned_age'] = pd.Categorical(cuts, categories=self.age_bin_labels,
                                                    ordered=True)
        
        # Education       
        for ed_key, ed_bin in self.edu_bins.items():
            ed_inds = GI_df.query('education_level in @ed_bin').index
            GI_df.loc[ed_inds, 'edu_bin'] = ed_key
        bin_inds = GI_df.loc[:, 'edu_bin']
        GI_df.loc[:, 'edu_bin'] = pd.Categorical(bin_inds, categories=self.edu_bin_order,
                                                 ordered=True)
                   
        return GI_df
