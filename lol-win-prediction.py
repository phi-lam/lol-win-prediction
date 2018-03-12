#PHI LAM
#COEN 140 Final Project
#lol-win-prediction.py

################################# IMPORTS ###################################
import numpy as np
import pandas as pd
import math
import sys
from operator import itemgetter
from sklearn import svm
from sklearn.model_selection import train_test_split

########################## VARIABLE DECLARATIONS ###########################

# Instances: ~7100 professional games
# NALCS: rows 2 - 1115
# EULCS: rows 1116 - 2055
# LCK: rows 2056 - 3374

#FILE PATHS
gold_inputfile = "gold.csv"
leagueoflegends_inputfile = "LeagueofLegends.csv"


########################## FUNCTION DEFINITIONS ############################

def parse_goldfile(in_file):
	df = pd.read_csv(in_file, header = 0, nrows=7620)
	original_headers = list(df.columns.values)
	df = df._get_numeric_data()
	numeric_headers = list(df.columns.values)
	out_array = df.as_matrix()

	df.to_csv('parsed_gold.csv')
	return out_array

def parse_leagueoflegends(in_file):
	df = pd.read_csv(in_file, header = 0)
	original_headers = list(df.columns.values)
	df = df._get_numeric_data()
	numeric_headers = list(df.columns.values)

	#return match outcomes only
	out_array = df.as_matrix()[:,1]
	out_array = out_array.reshape(-1, 1)
	# print(out_array)

	df.to_csv('parsed_leagueoflegends.csv')
	return out_array

#-------------------
#	FUNCTION: create_training_data()
#
#	Associates each metric with a specific match ID

#-----------------------------------------------------------------
#
#	FUNCTION: team_winrates()
#
#	Input: LeagueofLegends.csv
#	Procedure:
#		1) Iterate through list,
#

#-----------------------------------------------------------------
#	FUNCTION: populate()
#   Input: training-data.txt
#	Procedure:
#		1) Entire data set goes into source_training_data
#		2) First column becomes 'y', the rest is 'x'
#

def populate_training(in_file, source_array, data_y, data_x):
	global training_y, training_x, training_data
	for sampleid, line in enumerate(in_file):
		features = line.split()
		for featureid, feature in enumerate(features):
			feature = feature.strip()
			source_array[sampleid][featureid] = feature

	#---Convert to numpy arrays
	#--- Import the source data as numpy array, delete strings, convert to floats
	full_list = np.asarray(source_array)
	full_list = np.delete(full_list, 0, axis=0)
	full_list = full_list.astype(float)
	temp = full_list
	training_data = full_list
	# print(full_list)
	# training_data = np.delete(training_data, 4, 1)

	#--- Separate y and x
	data_y = full_list[:,[0]]
	data_x = np.delete(temp, 0, axis=1)

	training_y = data_y
	training_x = data_x
	#-----Debugging-------
	#print("==== data_y ====\n", np.shape(data_y))
	# print(data_y)
	#print("==== data_x ====\n", np.shape(data_x))
	# print(data_x)

#-----------------------------------------------------------------
#	FUNCTION: populate_test()
#   Input: test-data.txt
#	Procedure:
#		1) Entire data set goes into source_training_data
#		2) First column becomes 'y', the rest is 'x'
#

def populate_test(in_file, source_array, data_y, data_x):
	global test_y, test_x
	for sampleid, line in enumerate(in_file):
		features = line.split()
		for featureid, feature in enumerate(features):
			feature = feature.strip()
			source_array[sampleid][featureid] = feature

	#---Convert to numpy arrays
	#--- Import the source data as floats (also skips first row of text)
	full_list = np.asarray(source_array)
	full_list = np.delete(full_list, 0, axis=0)
	full_list = full_list.astype(float)
	temp = full_list

	# print(full_list)
	# training_data = np.delete(training_data, 4, 1)

	#--- Separate y and x
	data_y = full_list[:,[0]]
	data_x = np.delete(temp, 0, axis=1)

	test_y = data_y
	test_x = data_x
	#-----Debugging-------
	#print("==== data_y ====\n", np.shape(data_y))
	# print(data_y)
	#print("==== data_x ====\n", np.shape(data_x))
	# print(data_x)


################################# main ###################################

# ---- Opening files/correcting user input ----
# if len(sys.argv) == 3:
# 	try:
# 		f1 = open(sys.argv[1])
# 		f2 = open(sys.argv[2])
# 	except:
# 		print("Usage: arguments must be text files")
# 		exit()
# elif len(sys.argv) == 4:
# 	try:
# 		f1 = open(sys.argv[1])
# 		f2 = open(sys.argv[2])
# 		sys.stdout = open(sys.argv[3], "w")
# 	except:
# 		print("Usage: arguments must be text files")
# 		exit()
# else:
# 	print("Usage: linear-regression.py training.txt test.txt out_file.txt")
# 	print("Note: if out_file.txt missing, will print to stdout")
# 	exit()

# ---- Take input data ----
print("Parsing gold.csv...")
gold_nparray = parse_goldfile(gold_inputfile)

d = {}
d[0] = gold_nparray[:,4].reshape(-1,1) #gold_5min
d[1] = gold_nparray[:,9].reshape(-1,1) #gold_10min
d[2] = gold_nparray[:,14].reshape(-1,1) #gold_15min
d[3] = gold_nparray[:,20].reshape(-1,1) #gold_20min
d[4] = gold_nparray[:,25].reshape(-1,1) #gold_25min
d[5] = gold_nparray[:,29].reshape(-1,1) #gold_30min
print("Done.")

print("Parsing LeagueofLegends.csv...")
matchoutcomes_nparray = parse_leagueoflegends(leagueoflegends_inputfile)
print("Done.")

print("=== SVM Classification === ")

for i in range(6):
	print(d[i])
	gold_train, gold_test, outcome_train, outcome_test = train_test_split(d[i], matchoutcomes_nparray, test_size=0.2)
	if np.isnan(gold_train.any()):
		break

	clf = svm.LinearSVC()
	clf.fit(gold_train, outcome_train.ravel())
	print("Prediction error at time: ", 5*(i + 1), ":00")
	print(clf.score(gold_test, outcome_test))

#end
