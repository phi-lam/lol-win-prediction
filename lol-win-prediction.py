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
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn.model_selection import train_test_split


########################## VARIABLE DECLARATIONS ###########################

# Instances: ~7100 professional games
# NALCS: rows 2 - 1115
# EULCS: rows 1116 - 2055
# LCK: rows 2056 - 3374
# TODO: if at least 3-4k instances, is ok
# clean data > just many instances
# normalize gold values
# 2 col. per character (each team), 1 flag if character chosen

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
	# print(out_array) #

	df.to_csv('parsed_leagueoflegends.csv')
	return out_array

#
#	FUNCTION: delete_nanrows(nparray_)
#
#	Description: Call this function after compiling all columns for the test.
#
def delete_nanrows(np_data, np_labels):
	np_data = np.asarray(np_data)
	np_labels = np.asarray(np_labels)
	if len(np_data) != len(np_labels):
		raise Exception("np_data and np_labels must have identical lengths")
	if len(np_data) != num_instances:
		raise Exception("delete_nanrows needs np_data with original num_instances")

	#print("array lengths with nan")
	#print("np_data: ", len(np_data), " | np_labels: ", len(np_labels))

	np_labels = np_labels[~np.isnan(np_data).any(axis=1)]
	np_data = np_data[~np.isnan(np_data).any(axis=1)]

	#print("array lengths w/o nan")
	#print("np_data: ", len(np_data), " | np_labels: ", len(np_labels))

	np.savetxt("np_data.csv", np_data)
	np.savetxt("np_labels.csv", np_labels)

	#print("### np_data: ", np_data, " \n ### np_labels: ", np_labels)

	return {'data': np_data, 'labels': np_labels}

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

# ==== Workflow:
#	1) Parse all files
#		- make a dict with columns from relevant timestamps
#	2) For each timestamp (5_min, 10_min, 15_min, etc):
#		- Assemble data set with columns from relevant timestamps (delete nan arrays)
#		- Separate training and test data (train_test_split)
#		- Train each model
#		- Test each model

print("Parsing gold.csv...")
gold_nparray = parse_goldfile(gold_inputfile)
num_instances = len(gold_nparray)

gold_col = {}
gold_col[0] = gold_nparray[:,4].reshape(-1,1) #gold_5min
gold_col[1] = gold_nparray[:,9].reshape(-1,1) #gold_10min
gold_col[2] = gold_nparray[:,14].reshape(-1,1) #gold_15min
gold_col[3] = gold_nparray[:,20].reshape(-1,1) #gold_20min
gold_col[4] = gold_nparray[:,25].reshape(-1,1) #gold_25min
gold_col[5] = gold_nparray[:,29].reshape(-1,1) #gold_30min

print("Done.")

print("Parsing LeagueofLegends.csv...")
matchoutcomes_nparray = parse_leagueoflegends(leagueoflegends_inputfile)
print("Done.")

#crossvalidation

for i in range(6):

	# Assemble data array
	#temp fix
	input_array = gold_col[i]

	if np.isnan(input_array).any():
		result = delete_nanrows(input_array, matchoutcomes_nparray)
		data = result['data']
		outcomes = result['labels']
	else:
		data = input_array
		outcomes = matchoutcomes_nparray

	#print("array lengths: ", len(data), len(outcomes))
	print("\n===== Prediction error at time: ", 5*(i + 1), ":00 =====", sep='')
	data_train, data_test, outcome_train, outcome_test = train_test_split(data, outcomes, test_size=0.2)

	print("\n - LDA, Gold Test ")
	lda = discriminant_analysis.LinearDiscriminantAnalysis()
	lda.fit(data_train, outcome_train.ravel())
	pred = lda.predict(data_test)
	print("\t", lda.score(data_test, outcome_test), "\n")

	print("\n - Logistic Regression, Gold Test")
	log_reg = linear_model.LogisticRegression()
	log_reg.fit(data_train, outcome_train.ravel())
	pred = log_reg.predict(data_test)
	print("\t", log_reg.score(data_test, outcome_test), "\n")
	#print("\t", accuracy_score(outcome_test, pred), "\n")

	print("\n - SVM Classification, Gold Test")
	svm_lin = svm.LinearSVC()
	svm_lin.fit(data_train, outcome_train.ravel())
	pred = svm_lin.predict(data_test)
	print("\t", svm_lin.score(data_test, outcome_test), "\n")

#end
