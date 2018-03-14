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

print("gold@20 with nan")
print(d[3])
print("gold@30 without nan")
d[3] = d[3].dropna()
print(d[3])

print("Parsing LeagueofLegends.csv...")
matchoutcomes_nparray = parse_leagueoflegends(leagueoflegends_inputfile)
print("Done.")

#crossvalidation

print("\n=== LDA, Gold Test === ")
for i in range(6):
	if np.isnan(d[i]).any():
		continue
	print(d[i])
	gold_train, gold_test, outcome_train, outcome_test = train_test_split(d[i], matchoutcomes_nparray, test_size=0.2)

	clf1 = discriminant_analysis.LinearDiscriminantAnalysis()
	clf1.fit(gold_train, outcome_train.ravel())
	pred = clf1.predict(gold_test)
	print("Prediction error at time: ", 5*(i + 1), ":00")

	print("\t", clf1.score(gold_test, outcome_test), "\n")

print("\n=== Logistic Regression, Gold Test === ")
for i in range(6):
	if np.isnan(d[i]).any():
		continue
	print(d[i])
	gold_train, gold_test, outcome_train, outcome_test = train_test_split(d[i], matchoutcomes_nparray, test_size=0.2)
	if np.isnan(gold_train).any():
		continue

	clf2 = linear_model.LogisticRegression()
	clf2.fit(gold_train, outcome_train.ravel())
	pred = clf2.predict(gold_test)

	print("Prediction error at time: ", 5*(i + 1), ":00")
	print("\t", clf2.score(gold_test, outcome_test), "\n")


print("\n=== SVM Classification, Gold Test === ")

for i in range(6):
	print(d[i])
	gold_train, gold_test, outcome_train, outcome_test = train_test_split(d[i], matchoutcomes_nparray, test_size=0.2)
	if np.isnan(gold_train).any():
		continue

	clf3 = svm.LinearSVC()
	clf3.fit(gold_train, outcome_train.ravel())
	pred = clf3.predict(gold_test)
	print("Prediction error at time: ", 5*(i + 1), ":00")
	print("\t", clf3.score(gold_test, outcome_test), "\n")

#end
