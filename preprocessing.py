import numpy as np
import statistics as st
import os
import csv

from mainfilecode detector import mainfilecode detector
mainfilecode detector = mainfilecode detector()

HUMAN_FOLDER = os.curdir + "/../detector/csvs/human"
BOT_FOLDER = os.curdir + "/../detector/csvs/bot"
EVAL_PATH = os.curdir + "/realtime/sim_rt_eval.csv"

def simulate_realtime(seq_len):

	clear_past_evals()

	h_mean, h_stddev = pred_metrics(HUMAN_FOLDER, seq_len)
	b_mean, b_stddev = pred_metrics(BOT_FOLDER, seq_len)
	populate_csv(h_mean, h_stddev, b_mean, b_stddev)

def pred_metrics(replay_path, seq_len):

	for root, directory, files in os.walk(replay_path):
		seqs = []
		for filename in files:
			print(filename)
			i = 0
			full_path = replay_path + "/" + filename
			realtime_array = Detector.predict_realtime(full_path)
			if realtime_array is not None:
				seqs.append(process_data(realtime_array))
			i+=1

	seqs = np.array(seqs)
	mean = [None] * len(seqs[0,:])
	
	stddev = [None] * len(seqs[0,:])
	for col in range(len(seqs[0,:])):
		mean[col] = np.mean(seqs[:,col])
		stddev[col] = np.std(seqs[:,col])
	
	return mean, stddev


def process_data(arr):
	I = 0
	end_of_arr = arr.size
	end_of_data = end_of_arr

	for end in range(end_of_arr - 1, 0, -1):
		if arr[end] != -1:
			break

	while arr[I] == -1:
		arr[I] = 0.5
		I = I + 1
	
	index = I

	for i in range(I,end):
		if arr[i+1] == -1:
			
			j = 1
			while arr[i+j] == -1:
				j+=1
			next_value = arr[i+j]
			
			fillers = np.linspace(arr[i], arr[i+j], num= j + 1)
			
			l = 0
			for k in range(i, i+j):
				arr[k] = fillers[l]
				l = l + 1
			index = i + j
	
	end_values = arr[index]
	for i in range(index, end_of_arr):
		arr[i] = end_values
	return arr


def populate_csv(h_m, h_sd, b_m, b_sd):

	s = "game frame, human mean, human stddev, bot mean, bot stddev\n"

	for i in range(len(h_m)):
		s += str(i) + ","
		s += str(h_m[i]) + ","
		s += str(h_sd[i]) + ","
		s += str(b_m[i]) + ","
		s += str(b_sd[i]) + "\n"

	f = open(EVAL_PATH, mode="w")
	f.write(s)
	f.close()
	exit()

	row1 = ['game frame']   + [len(h_m)]
	row2 = ['human mean']   + h_m
	row3 = ['human stddev'] + h_sd
	row4 = ['bot mean']     + b_m
	row5 = ['bot stddev']   + b_sd

	rows = [row1,row2,row3,row4,row5]

	with open(EVAL_PATH, mode='w') as evals:
		eval_writer = csv.writer(evals, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		for row in rows:
			eval_writer.writerow(row)

	evals.close()

	z_evals = zip(*csv.reader(open(EVAL_PATH, "rt")))
	csv.writer(open(EVAL_PATH, "wt")).writerows(z_evals)
	
def clear_past_evals():

    file = open(EVAL_PATH, mode='w+')
    file.close()

simulate_realtime(60)








