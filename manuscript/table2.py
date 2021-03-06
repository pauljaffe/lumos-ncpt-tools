import pkgutil
import os

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data
from .manuscript_utils import Table

class Table2():
    
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    
    def __init__(self, save_dir, figsize):
        self.png_path = os.path.join(save_dir, 'table2.png')
        self.svg_path = os.path.join(save_dir, 'table2.svg')        
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.subtests = self.config['subtests'].keys()
        self.figsize = figsize
        
    def make_table(self):
        rows = [f'Subtest ID {sub_id}' for sub_id in self.subtests]
        cols = ['Task', 'Short name', 'Version', 'Other versions']
    
        table_data = []
        for sub_id in self.subtests:
            this_data = self.config['subtests'][sub_id]
            table_data.append(this_data)
        
        data_array = np.array(table_data)
        table = Table(self.figsize, scale=(0.8, 1))
        table_fig = table.plot_table(cols, rows, data_array)
        table_fig.savefig(self.png_path, bbox_inches='tight')
        table_fig.savefig(self.svg_path, transparent=True, bbox_inches='tight')   