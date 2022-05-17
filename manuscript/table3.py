import pkgutil
import os

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data
from .manuscript_utils import Table

class Table3():

    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    
    def __init__(self, save_dir, figsize):
        self.png_path = os.path.join(save_dir, 'table3.png')
        self.svg_path = os.path.join(save_dir, 'table3.svg')        
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.batteries = self.config['batteries'].keys()
        self.figsize = figsize
        
    def make_table(self):
        cols = [f'Battery {bat_id}' for bat_id in self.batteries]
        rows = ['Subtest IDs']
    
        table_data = []
        for bat_id in self.batteries:
            bat_subtests = self.config['batteries'][bat_id][1]
            if len(bat_subtests) > 8:
                s1 = ', '.join(str(s) for s in bat_subtests[:4])
                s2 = ', '.join(str(s) for s in bat_subtests[4:8])
                s3 = ', '.join(str(s) for s in bat_subtests[8:])
                subtest_str = s1 + ',\n' + s2 + ',\n' + s3
            elif len(bat_subtests) > 4:
                s1 = ', '.join(str(s) for s in bat_subtests[:4])
                s2 = ', '.join(str(s) for s in bat_subtests[4:])
                subtest_str = s1 + ',\n' + s2
            else:
                subtest_str = ', '.join(str(s) for s in bat_subtests)
            table_data.append([subtest_str])
        
        data_array = np.array(table_data).T
        table = Table(self.figsize, scale=(1, 2))
        table_fig = table.plot_table(cols, rows, data_array)
        table_fig.savefig(self.png_path, bbox_inches='tight')
        table_fig.savefig(self.svg_path, transparent=True, bbox_inches='tight')    