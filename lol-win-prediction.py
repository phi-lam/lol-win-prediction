#PHI LAM
#COEN 140 Final Project
#lol-win-prediction.py

# TODO:
	# - finish parsing each column
	# - convert each column into a feature:
	#		- for each instance:
	#			- have a column for each minute where the index is the
	#			  number of the objective claimed (arr[5] is # taken by min 5)
	#			- keep running counter of objective taken (init = 0)
	#			- when timestamp is parsed, insert(++counter) into that column
	#		- during the feature compilation stage, grab the column desired,
	#			which describes "amount of objectives taken by this time"
	#
	#
	#	Future work: split objective by category:
	#		- elemental drags have different values
	#		- different towers have different influences on the next objectives

# TODO:
	# - include Stochastic Gradient Descent
	# - include PCA
	# - if PCA doesn't work as desired, then implement individual tests

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

# Instances: 7620 professional games
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

def parse_labels(in_file):
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

def parse_leagueoflegends(in_file):
	df = pd.read_csv(in_file, header = 0)
	original_headers = list(df.columns.values)
	blue_kills_list = df['bKills'].tolist()
	blue_towers_list = df['bTowers']
	blue_dragons_list = df['bDragons']
	blue_barons_list = df['bBarons']
	blue_heralds_list = df['bHeralds']
	red_kills_list = df['rKills']
	red_towers_list = df['rTowers']
	red_dragons_list = df['rDragons']
	red_barons_list = df['rBarons']
	red_heralds_list = df['rHeralds']
	# print(blue_kills_list)
	# print(blue_barons_list)
	# print(blue_dragons_list)
	# print(blue_barons_list)
	# print(blue_heralds_list)

# def parse_kills(blue_kills_list):
	## Parse kills
	# print(len(blue_kills_list))
	bKills = np.zeros((7620,100))		# 100 is max game length in minutes
	for i, row in enumerate(blue_kills_list):
		print(row) # Print entire row
		for item in row.split('['):
			for elt in item.split(','):
				try:
					if (float(elt) < 100): # By experimentation, this captures latest game
						kill_time = int(float(elt))
						print(kill_time) # Check that each kill from row is present
						bKills[i][kill_time] += 1
				except:
					continue
		print("------------------")
	np.savetxt("parsed_kills.csv", bKills, delimiter=",")


##############################################################################
#	FUNCTION: delete_nanrows(nparray_)
#
#	Description: After compiling all desired columns into the data set
#			(as a numpy array), call this function to remove all instances (rows)
#			with NaN values. Then, two new numpy array (data and labels)
#			will be passed to train_test_split.
#
def delete_nanrows(np_data, np_labels):

	print("Removing rows with NaN values...")

	np_data = np.asarray(np_data)
	np_labels = np.asarray(np_labels)
	if len(np_data) != len(np_labels):
		raise Exception("np_data and np_labels must have identical lengths")
	if len(np_data) != num_instances:
		raise Exception("delete_nanrows needs np_data with original num_instances")

	#print("Num. instances before removal: ")
	#print("np_data: ", len(np_data), " | np_labels: ", len(np_labels))

	np_labels = np_labels[~np.isnan(np_data).any(axis=1)]
	np_data = np_data[~np.isnan(np_data).any(axis=1)]

	#print("Num. instances after removal: ")
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

parse_leagueoflegends(leagueoflegends_inputfile)

# print("Parsing gold.csv...")
# gold_nparray = parse_goldfile(gold_inputfile)
# num_instances = len(gold_nparray)
#
# gold_col = {}
# gold_col[0] = gold_nparray[:,4].reshape(-1,1) #gold_5min
# gold_col[1] = gold_nparray[:,9].reshape(-1,1) #gold_10min
# gold_col[2] = gold_nparray[:,14].reshape(-1,1) #gold_15min
# gold_col[3] = gold_nparray[:,20].reshape(-1,1) #gold_20min
# gold_col[4] = gold_nparray[:,25].reshape(-1,1) #gold_25min
# gold_col[5] = gold_nparray[:,29].reshape(-1,1) #gold_30min
#
# print("Done.")
#
# print("Parsing LeagueofLegends.csv...")
# matchoutcomes_nparray = parse_labels(leagueoflegends_inputfile)
# print("Done.")
#
# #crossvalidation
#
# for i in range(6):

# 	print("\n===== Prediction error at time: ", 5*(i + 1), ":00 =====", sep='')
# 	# Assemble data array
# 	#temp fix
# 	input_array = gold_col[i]
#
# 	if np.isnan(input_array).any():
# 		result = delete_nanrows(input_array, matchoutcomes_nparray)
# 		data = result['data']
# 		outcomes = result['labels']
# 	else:
# 		data = input_array
# 		outcomes = matchoutcomes_nparray
#
# 	#print("array lengths: ", len(data), len(outcomes))
#
# 	data_train, data_test, outcome_train, outcome_test = train_test_split(data, outcomes, test_size=0.2)
#
#	########## GOLD TEST ############
# 	print("\n - LDA, Gold Test ")
# 	lda = discriminant_analysis.LinearDiscriminantAnalysis()
# 	lda.fit(data_train, outcome_train.ravel())
# 	#pred = lda.predict(data_test)
# 	print("\t", lda.score(data_test, outcome_test), "\n")
#
# 	print("\n - QDA, Gold Test ")
# 	qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
# 	qda.fit(data_train, outcome_train.ravel())
# 	print("\t", qda.score(data_test, outcome_test), "\n")
#
# 	print("\n - Logistic Regression, Gold Test")
# 	log_reg = linear_model.LogisticRegression()
# 	log_reg.fit(data_train, outcome_train.ravel())
# 	#pred = log_reg.predict(data_test)
# 	print("\t", log_reg.score(data_test, outcome_test), "\n")
# 	#print("\t", accuracy_score(outcome_test, pred), "\n")
#
# 	print("\n - SVM Classification, Gold Test")
# 	svm_lin = svm.LinearSVC()
# 	svm_lin.fit(data_train, outcome_train.ravel())
# 	#pred = svm_lin.predict(data_test)
# 	print("\t", svm_lin.score(data_test, outcome_test), "\n")
#

################ KILLS TEST ############
# for i in range(6):
#
# #end
