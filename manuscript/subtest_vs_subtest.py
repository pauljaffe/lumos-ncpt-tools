import pkgutil
import os

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lumos_ncpt_tools.ncpt import NCPT
from lumos_ncpt_tools.utils import load_data

class Figure2():
    """Subtest score correlation matrices for each battery."""
    # Only users who completed the entire test battery
    # are included in the correlation analyses.
    
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    
    def __init__(self, data_dir, save_dir, figsize):
        self.data_dir = data_dir
        self.png_path = os.path.join(save_dir, 'figure2.png')
        self.svg_path = os.path.join(save_dir, 'figure2.svg')        
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.batteries = self.config['batteries'].keys()
        self.figsize = figsize
        self.palette = 'viridis'
        self.vmin = 0 # For defining the color bar axis 
        self.vmax = 0.7
        # Customize order that subtests are arranged on heatmap
        self.subtest_order = {14: [29, 30, 28, 27, 26],
                              17: [29, 30, 28, 27, 32, 31],
                              25: [29, 30, 28, 33, 27, 32, 31],
                              26: [29, 30, 28, 33, 27, 32, 38, 39, 40, 36, 37],
                              32: [29, 30, 28, 33, 27, 31, 38, 39, 40],
                              39: [29, 30, 28, 33, 31, 38, 39, 40],
                              50: [29, 30, 43, 44, 31, 45, 39, 40],
                              60: [55, 51, 54, 53, 52]}
       
    def make_figure(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize)
        gs = fig.add_gridspec(14, 31)
        gs_plots = [gs[0:5, 0:5], gs[0:5, 8:13], gs[0:5, 16:21], gs[0:5, 24:29],
                    gs[8:13, 0:5], gs[8:13, 8:13], gs[8:13, 16:21], gs[8:13, 24:29]]
        cbar_ax = fig.add_subplot(gs[0:13, 30])
        
        for ax_ind, bat_id in enumerate(self.batteries):
            ax = fig.add_subplot(gs_plots[ax_ind])
            bat_corrs = self._get_battery_data(bat_id)  
            if ax_ind == 3:  
                sns.heatmap(bat_corrs, ax=ax, square=True,
                            cmap=self.palette, cbar_ax=cbar_ax,
                            vmin=self.vmin, vmax=self.vmax, 
                            cbar_kws={"label": "Pearson's r"})
            else:
                sns.heatmap(bat_corrs, ax=ax, square=True,
                            cmap=self.palette, cbar=False,
                            vmin=self.vmin, vmax=self.vmax)
                
            ax.set_title(f'Battery {bat_id}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                               rotation_mode='anchor')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right',
                               rotation_mode='anchor')
            
        plt.show()        
        fig.savefig(self.png_path, bbox_inches='tight')
        fig.savefig(self.svg_path, transparent=True, bbox_inches='tight')   
        
    def _get_battery_data(self, bat_id):
        print(f'Battery ID {bat_id}')
        data_fn = f'battery{bat_id}_df.csv'
        bat_df = load_data(self.data_dir, data_fn) 
        bat_ncpt = NCPT(bat_df)
        del bat_df
        filt_df, _ = bat_ncpt.filter_by_completeness()        
        subtests = self.subtest_order[bat_id]
        scores = {}
        
        for sub in subtests:
            name = self.config['subtests'][sub][1]
            sub_df = filt_df.query('specific_subtest_id == @sub')
            scores[name] = sub_df['rank_INT_normed_score'].values
        score_df = pd.DataFrame(scores)
        corrs = score_df.corr(method='pearson')      
        np.fill_diagonal(corrs.values, np.nan)
        
        min_r = corrs.min().min()
        max_r = corrs.max().max()
        print(f'Min r: {min_r}, max r: {max_r}')
        if bat_id != 60:
            ar_gr_r = corrs.loc['Arithmetic', 'Grammar']
            print(f'GR/AR correlation: {ar_gr_r}')
        if bat_id in [26, 32, 39, 50]:
            digit_trailsA_r = corrs.loc['Digit symbol', 'Trails A']
            digit_trailsB_r = corrs.loc['Digit symbol', 'Trails B']
            print(f'Digit symbol/Trails A correlation: {digit_trailsA_r}')
            print(f'Digit symbol/Trails B correlation: {digit_trailsB_r}')
        print('--------------------------')           
        return corrs  