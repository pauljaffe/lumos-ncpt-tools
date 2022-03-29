import numpy as np
import matplotlib.pyplot as plt


class Table():
    
    def __init__(self, figsize):
        self.figsize = figsize
    
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

        table.scale(1, 2)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.box(on=None)
        table_fig = plt.gcf()
        plt.draw()

        return table_fig