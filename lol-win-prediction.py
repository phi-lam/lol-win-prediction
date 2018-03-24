#PHI LAM
#COEN 140 Final Project
#lol-win-prediction.py

# TODO:
#	- clean up code

# TODO (complete):
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

##############################################################################
#	FUNCTION: parse_kills
#

def parse_kills(kills_list):
	debug = False
	## Parse kills
	# print(len(bKills_list))
	kills = np.zeros((7620,100))		# 100 is max game length in minutes
	for i, row in enumerate(kills_list):
		if debug: print(row) # Print entire row
		for item in row.split('['):
			for elt in item.split(','):
				try:
					if (float(elt) < 100): # By experimentation, this captures latest game
						kill_time = int(float(elt))
						if debug: print(kill_time) # Check that each kill from row is present
						kills[i][kill_time] += 1
				except:
					continue
		if debug: print("------------------")
	np.savetxt("parsed_kills.csv", kills, delimiter=",")
	return kills # returns a numpy array

########################################################################
#
#	FUNCTION: parse_towers
#
#	Description: Currently serves as parser for all objectives.
#				Once dragons and towers are differentiated, may need to specialize
#				functions.
#

def parse_towers(towers_list):
	debug = False
	towers = np.zeros((7620,100))		# 100 is max game length in minutes
	for i, row in enumerate(towers_list):
		if debug: print(row) # Print entire row
		for item in row.split('['):
			for elt in item.split(','):
				try:
					if (float(elt) < 100): # By experimentation, this captures latest game
						kill_time = int(float(elt))
						if debug: print(kill_time) # Check that each kill from row is present
						towers[i][kill_time] += 1
				except:
					continue
		if debug: print("------------------")
	np.savetxt("parsed_towers.csv", towers, delimiter=",")
	return towers # returns a numpy array

########################################################################
#
#	FUNCTION: parse_barons
#
#	Description: Currently serves as parser for all objectives.
#				Once dragons and barons are differentiated, may need to specialize
#				functions.
#

def parse_barons(barons_list):
	debug = True
	barons = np.zeros((7620,100))		# 100 is max game length in minutes
	for i, row in enumerate(barons_list):
		if debug: print(row) # Print entire row
		for item in row.split('['):
			for elt in item.split(','):
				try:
					if (float(elt) < 100): # By experimentation, this captures latest game
						kill_time = int(float(elt))
						if debug: print(kill_time) # Check that each kill from row is present
						barons[i][kill_time] += 1
				except:
					continue
		if debug: print("------------------")
	np.savetxt("parsed_barons.csv", barons, delimiter=",")
	return barons # returns a numpy array

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

# ==== Workflow:
#	1) Parse all files
#		- make a dict with columns from relevant timestamps
#	2) For each timestamp (5_min, 10_min, 15_min, etc):
#		- Assemble data set with columns from relevant timestamps (delete nan arrays)
#		- Separate training and test data (train_test_split)
#		- Train each model
#		- Test each model

# parse_leagueoflegends(leagueoflegends_inputfile)

# Read columns from LeagueofLegends.csv
df = pd.read_csv(leagueoflegends_inputfile, header = 0)
original_headers = list(df.columns.values)

blue_towers_list = df['bTowers']
blue_dragons_list = df['bDragons']
blue_barons_list = df['bBarons']
blue_heralds_list = df['bHeralds']
red_kills_list = df['rKills']
red_towers_list = df['rTowers']
red_dragons_list = df['rDragons']
red_barons_list = df['rBarons']
red_heralds_list = df['rHeralds']

print("Parsing barons...")
bbaron_nparray = parse_barons(df['bBarons'].tolist())
bbaron_col = {}
bbaron_col[0] = bbaron_nparray[:,4].reshape(-1,1) #blue barons @ 5 min
bbaron_col[1] = bbaron_nparray[:,9].reshape(-1,1) #blue barons @ 10 min
bbaron_col[2] = bbaron_nparray[:,14].reshape(-1,1) #blue barons @ 15 min
bbaron_col[3] = bbaron_nparray[:,19].reshape(-1,1) #blue barons @ 20 min
bbaron_col[4] = bbaron_nparray[:,24].reshape(-1,1) #blue barons @ 25 min
bbaron_col[5] = bbaron_nparray[:,29].reshape(-1,1) #blue barons @ 30 min


rbaron_nparray = parse_towers(df['rBarons'].tolist())
rbaron_col = {}
rbaron_col[0] = rbaron_nparray[:,4].reshape(-1,1) #red barons @ 5 min
rbaron_col[1] = rbaron_nparray[:,9].reshape(-1,1) #red barons @ 10 min
rbaron_col[2] = rbaron_nparray[:,14].reshape(-1,1) #red barons @ 15 min
rbaron_col[3] = rbaron_nparray[:,19].reshape(-1,1) #red barons @ 20 min
rbaron_col[4] = rbaron_nparray[:,24].reshape(-1,1) #red barons @ 25 min
rbaron_col[5] = rbaron_nparray[:,29].reshape(-1,1) #red barons @ 30 min
exit()
print("Done.")


print("Parsing gold.csv...")
# For gold, the leftmost column in .csv isn't counted as an index. min_5 is col[4]
gold_nparray = parse_goldfile(gold_inputfile)
num_instances = len(gold_nparray)
gold_col = {}
gold_col[0] = gold_nparray[:,4].reshape(-1,1) #gold_5min
gold_col[1] = gold_nparray[:,9].reshape(-1,1) #gold_10min
gold_col[2] = gold_nparray[:,14].reshape(-1,1) #gold_15min
gold_col[3] = gold_nparray[:,19].reshape(-1,1) #gold_20min
gold_col[4] = gold_nparray[:,24].reshape(-1,1) #gold_25min
gold_col[5] = gold_nparray[:,29].reshape(-1,1) #gold_30min
print("Done.")


print("Parsing LeagueofLegends.csv...")
print("Parsing game outcomes...")
matchoutcomes_nparray = parse_labels(leagueoflegends_inputfile)


print("Parsing kills...")
bkill_nparray = parse_kills(df['bKills'].tolist())
bkill_col = {}
bkill_col[0] = bkill_nparray[:,4].reshape(-1,1) #blue kills @ 5 min
bkill_col[1] = bkill_nparray[:,9].reshape(-1,1) #blue kills @ 10 min
bkill_col[2] = bkill_nparray[:,14].reshape(-1,1) #blue kills @ 15 min
bkill_col[3] = bkill_nparray[:,19].reshape(-1,1) #blue kills @ 20 min
bkill_col[4] = bkill_nparray[:,24].reshape(-1,1) #blue kills @ 25 min
bkill_col[5] = bkill_nparray[:,29].reshape(-1,1) #blue kills @ 30 min


rkill_nparray = parse_kills(df['rKills'].tolist())
rkill_col = {}
rkill_col[0] = rkill_nparray[:,4].reshape(-1,1) #red kills @ 5 min
rkill_col[1] = rkill_nparray[:,9].reshape(-1,1) #red kills @ 10 min
rkill_col[2] = rkill_nparray[:,14].reshape(-1,1) #red kills @ 15 min
rkill_col[3] = rkill_nparray[:,19].reshape(-1,1) #red kills @ 20 min
rkill_col[4] = rkill_nparray[:,24].reshape(-1,1) #red kills @ 25 min
rkill_col[5] = rkill_nparray[:,29].reshape(-1,1) #red kills @ 30 min
# print(rkill_col[5])
# exit()
print("Done.")

print("Parsing towers...")
btower_nparray = parse_towers(df['bTowers'].tolist())
btower_col = {}
btower_col[0] = btower_nparray[:,4].reshape(-1,1) #blue towers @ 5 min
btower_col[1] = btower_nparray[:,9].reshape(-1,1) #blue towers @ 10 min
btower_col[2] = btower_nparray[:,14].reshape(-1,1) #blue towers @ 15 min
btower_col[3] = btower_nparray[:,19].reshape(-1,1) #blue towers @ 20 min
btower_col[4] = btower_nparray[:,24].reshape(-1,1) #blue towers @ 25 min
btower_col[5] = btower_nparray[:,29].reshape(-1,1) #blue towers @ 30 min

rtower_nparray = parse_towers(df['rTowers'].tolist())
rtower_col = {}
rtower_col[0] = rtower_nparray[:,4].reshape(-1,1) #red towers @ 5 min
rtower_col[1] = rtower_nparray[:,9].reshape(-1,1) #red towers @ 10 min
rtower_col[2] = rtower_nparray[:,14].reshape(-1,1) #red towers @ 15 min
rtower_col[3] = rtower_nparray[:,19].reshape(-1,1) #red towers @ 20 min
rtower_col[4] = rtower_nparray[:,24].reshape(-1,1) #red towers @ 25 min
rtower_col[5] = rtower_nparray[:,29].reshape(-1,1) #red towers @ 30 min
print("Done.")

# TODO: See why inhibitors cause division by zero @ 5 min
print("Parsing inhibitors...")
binhib_nparray = parse_towers(df['bInhibs'].tolist())
binhib_col = {}
binhib_col[0] = binhib_nparray[:,4].reshape(-1,1) #blue inhibs @ 5 min
binhib_col[1] = binhib_nparray[:,9].reshape(-1,1) #blue inhibs @ 10 min
binhib_col[2] = binhib_nparray[:,14].reshape(-1,1) #blue inhibs @ 15 min
binhib_col[3] = binhib_nparray[:,19].reshape(-1,1) #blue inhibs @ 20 min
binhib_col[4] = binhib_nparray[:,24].reshape(-1,1) #blue inhibs @ 25 min
binhib_col[5] = binhib_nparray[:,29].reshape(-1,1) #blue inhibs @ 30 min

rinhib_nparray = parse_towers(df['rInhibs'].tolist())
rinhib_col = {}
rinhib_col[0] = rinhib_nparray[:,4].reshape(-1,1) #red inhibs @ 5 min
rinhib_col[1] = rinhib_nparray[:,9].reshape(-1,1) #red inhibs @ 10 min
rinhib_col[2] = rinhib_nparray[:,14].reshape(-1,1) #red inhibs @ 15 min
rinhib_col[3] = rinhib_nparray[:,19].reshape(-1,1) #red inhibs @ 20 min
rinhib_col[4] = rinhib_nparray[:,24].reshape(-1,1) #red inhibs @ 25 min
rinhib_col[5] = rinhib_nparray[:,29].reshape(-1,1) #red inhibs @ 30 min
# exit()
print("Done.")

print("Parsing dragons...")
bdragon_nparray = parse_towers(df['bDragons'].tolist())
bdragon_col = {}
bdragon_col[0] = bdragon_nparray[:,4].reshape(-1,1) #bdragon_5min
bdragon_col[1] = bdragon_nparray[:,9].reshape(-1,1) #bdragon_10min
bdragon_col[2] = bdragon_nparray[:,14].reshape(-1,1) #bdragon_15min
bdragon_col[3] = bdragon_nparray[:,19].reshape(-1,1) #bdragon_20min
bdragon_col[4] = bdragon_nparray[:,24].reshape(-1,1) #bdragon_25min
bdragon_col[5] = bdragon_nparray[:,29].reshape(-1,1) #bdragon_30min

rdragon_nparray = parse_towers(df['rDragons'].tolist())
rdragon_col = {}
rdragon_col[0] = rdragon_nparray[:,4].reshape(-1,1) #rdragon_5min
rdragon_col[1] = rdragon_nparray[:,9].reshape(-1,1) #rdragon_10min
rdragon_col[2] = rdragon_nparray[:,14].reshape(-1,1) #rdragon_15min
rdragon_col[3] = rdragon_nparray[:,19].reshape(-1,1) #rdragon_20min
rdragon_col[4] = rdragon_nparray[:,24].reshape(-1,1) #rdragon_25min
rdragon_col[5] = rdragon_nparray[:,29].reshape(-1,1) #rdragon_30min
print("Done.")



#crossvalidation

for i in range(6):

	print("\n===== Prediction error at time: ", 5*(i + 1), ":00 =====", sep='')
	# Assemble data array
	#temp fix
	# print(np.shape(gold_col[i]))
	# print(np.shape(bkill_col[i]))
	input_array = np.concatenate((gold_col[i], bkill_col[i], rkill_col[i], btower_col[i], rtower_col[i], bdragon_col[i], rdragon_col[i], binhib_col[i], rinhib_col[i]), axis=1)
	print(np.shape(input_array))

	if np.isnan(input_array).any():
		result = delete_nanrows(input_array, matchoutcomes_nparray)
		data = result['data']
		outcomes = result['labels']
	else:
		data = input_array
		outcomes = matchoutcomes_nparray

	#print("array lengths: ", len(data), len(outcomes))

	data_train, data_test, outcome_train, outcome_test = train_test_split(data, outcomes, test_size=0.2)

	########## GOLD TEST ############
	print("\n - LDA, Gold Test ")
	lda = discriminant_analysis.LinearDiscriminantAnalysis()
	lda.fit(data_train, outcome_train.ravel())
	#pred = lda.predict(data_test)
	print("\t", lda.score(data_test, outcome_test), "\n")

	print("\n - QDA, Gold Test ")
	qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
	qda.fit(data_train, outcome_train.ravel())
	print("\t", qda.score(data_test, outcome_test), "\n")

	print("\n - Logistic Regression, Gold Test")
	log_reg = linear_model.LogisticRegression()
	log_reg.fit(data_train, outcome_train.ravel())
	#pred = log_reg.predict(data_test)
	print("\t", log_reg.score(data_test, outcome_test), "\n")
	#print("\t", accuracy_score(outcome_test, pred), "\n")

	print("\n - SVM Classification, Gold Test")
	svm_lin = svm.LinearSVC()
	svm_lin.fit(data_train, outcome_train.ravel())
	#pred = svm_lin.predict(data_test)
	print("\t", svm_lin.score(data_test, outcome_test), "\n")


################ KILLS TEST ############
# for i in range(6):
#
# #end
