import matplotlib.pyplot as plt

from manuscript.table1 import Table1
from manuscript.table2 import Table2
from manuscript.table3 import Table3
from manuscript.table4 import Table4
from manuscript.table5 import Table5
from manuscript.summary_stats import SummaryStats
from manuscript.subtest_vs_subtest import Figure1
from manuscript.norm_tables import NormTables


# This script generates the tables/figures and reports summary stats for the
# data descriptor manuscript. 

#data_directory = 'CHANGE/TO/DATA/DIRECTORY' 
#save_directory = 'CHANGE/TO/SAVE/DIRECTORY' 
#norm_save_directory = 'CHANGE/TO/NORM/SAVE/DIRECTORY'
t1_figsize = (7.5, 5.5)
t2_figsize = (7.5, 5.5)
t3_figsize = (7.5, 1)
t4_figsize = (7.5, 5.5)
t5_figsize = (7.5, 5.5)
f1_figsize = (7.2, 4)

# Plotting params
fontsize = 6
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': fontsize})

# Summary stats
ss = SummaryStats(data_directory)
ss.get_summary_stats()

# Make tables / figures for paper
t1 = Table1(data_directory, save_directory, t1_figsize)
t1.make_table()

t2 = Table2(save_directory, t2_figsize)
t2.make_table()

t3 = Table3(save_directory, t3_figsize)
t3.make_table()

t4 = Table4(save_directory, t4_figsize)
t4.make_table()

t5 = Table5(save_directory, t5_figsize)
t5.make_table()

plt.rcParams.update({'font.size': 5})
f1 = Figure1(data_directory, save_directory, f1_figsize)
f1.make_figure()

norm_tables = NormTables(data_directory, norm_save_directory) 
norm_tables.make_tables()
