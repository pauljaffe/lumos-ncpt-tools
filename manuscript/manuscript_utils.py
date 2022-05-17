import numpy as np
import matplotlib.pyplot as plt


class Table():
    
    def __init__(self, figsize, scale=None, fontsize=10):
        self.figsize = figsize
        self.fontsize = fontsize
        self.scale = scale
    
    def plot_table(self, col_labels, row_labels, data):
        row_colors = plt.cm.BuPu(np.full(len(row_labels), 0.1))
        col_colors = plt.cm.BuPu(np.full(len(col_labels), 0.1))

        plt.figure(linewidth=2, figsize=self.figsize)
        table = plt.table(cellText=data,
                          rowLabels=row_labels,
                          rowColours=row_colors,
                          rowLoc='right',
                          colColours=col_colors,
                          colLabels=col_labels,
                          loc='center',
                          cellLoc='center')
        
        self._adjust(table)
        table_fig = plt.gcf()
        plt.draw()

        return table_fig
    
    def plot_table_no_rows(self, col_labels, data):
        # Plots table without labeled rows
        col_colors = plt.cm.BuPu(np.full(len(col_labels), 0.1))

        plt.figure(linewidth=2, figsize=self.figsize)
        table = plt.table(cellText=data,
                          rowLoc='right',
                          colColours=col_colors,
                          colLabels=col_labels,
                          loc='center',
                          cellLoc='center')
        
        self._adjust(table)
        table_fig = plt.gcf()
        plt.draw()

        return table_fig
    
    def _adjust(self, table):
        if self.scale is not None:
            table.scale(*self.scale)
        table.set_fontsize(self.fontsize)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.box(on=None)