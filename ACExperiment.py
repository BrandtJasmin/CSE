import numpy as np
import re
import csv

# read running times
def read_running_times():
    running_times_file = 'Data_saps_swgsp\\cpu_times_inst_param.csv'
    running_times = []
    with open(running_times_file, newline='') as csvfile:
        running_times_data = list(csv.reader(csvfile))
    for i in range(1,len(running_times_data)):
        next_line = running_times_data[i][0]
        next_rt_vector = [float(s) for s in re.findall(r'-?\d+\.?\d*', next_line)][2:]
        running_times.append(next_rt_vector)
    running_times = np.asarray(running_times)
    lambda_ = 100
    #running_times_alt = np.exp(-lambda_*running_times)
    running_times_alt = (-1) * running_times
    return running_times_alt
