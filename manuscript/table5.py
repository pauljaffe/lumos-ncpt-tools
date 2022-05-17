import pkgutil
import os

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data
from .manuscript_utils import Table

class Table5():
    
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    
    def __init__(self, save_dir, figsize):
        self.png_path = os.path.join(save_dir, 'table5.png')
        self.svg_path = os.path.join(save_dir, 'table5.svg')   
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.figsize = figsize
        
    def make_table(self):
        cols = ['Education level', 'Description'] 
        
        table_data = []
        for key, val in self.config['education'].items():
            table_data.append([key, val])
        
        data_array = np.array(table_data)
        table = Table(self.figsize, scale=(1, 1))
        table_fig = table.plot_table_no_rows(cols, data_array)
        table_fig.savefig(self.png_path, bbox_inches='tight')
        table_fig.savefig(self.svg_path, transparent=True, bbox_inches='tight')   