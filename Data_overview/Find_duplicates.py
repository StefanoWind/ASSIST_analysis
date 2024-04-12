# -*- coding: utf-8 -*-
"""
Find duplicated filenames on the DAP
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('dap-py')
from doe_dap_dl import DAP
from datetime import datetime

#%% Inputs
channel_wrong='awaken/sb.assist.z01.00'#channel containing duplicates
channel_right='awaken/sc1.assist.z01.00'#channel being duplicated
data_format='cdf'

username='sletizia'
password='pass_DAP1506@'

t_start='2022-10-07 00:00'#start date
t_end='2023-11-30 00:00'#end date

#%% Initialization
a2e = DAP('a2e.energy.gov')
a2e.setup_cert_auth(username=username, password=password)

time_range_dap = [datetime.strftime(datetime.strptime(t_start, '%Y-%m-%d %H:%M'), '%Y%m%d%H%M%S'),
                  datetime.strftime(datetime.strptime(t_end, '%Y-%m-%d %H:%M'), '%Y%m%d%H%M%S')]

#%% Main
filter_wrong = {'Dataset': channel_wrong,
          'date_time': {'between': time_range_dap},
          'file_type':data_format}

file_list_wrong=[f['Filename'] for f in a2e.search(filter_wrong, table='Inventory')]

filter_right = {'Dataset': channel_right,
          'date_time': {'between': time_range_dap},
          'file_type':data_format}

file_list_right=[('.').join(f['Filename'].split('.')[1:]) for f in a2e.search(filter_right, table='Inventory')]

dupl=[]
for f in file_list_wrong:
    if ('.').join(f.split('.')[1:]) in file_list_right:
        dupl.append(f)

#%% Output
file_save=os.path.join('data',channel_wrong.replace('/','.')+'_'+channel_right.replace('/','.')+'_'+data_format+'_dupl.txt')
with open(file_save, "w") as file:
    # Iterate over each string in the list
    for string in dupl:
        # Write the string to the file
        file.write(string + "\n")
