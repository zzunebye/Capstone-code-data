#!/usr/bin/env python3
import json
import os.path
import sys

# as per the metadata file, input and output directories are the arguments
[_, reference_file, submission_file] = sys.argv

truth_values = json.load(open(reference_file, 'r'))
submission = json.load(open(submission_file, 'r'))


observed = 0
correct = 0
total = len(truth_values.keys())

print(len(truth_values), 'entries in reference file')

for reference_id in truth_values.keys():
	if reference_id in submission.keys():
		print('matching entry:', reference_id)
		observed += 1
		if submission[reference_id] == truth_values[reference_id]:
			correct += 1
	else:
		print('unmatched entry:', reference_id, '-- no reference value for this document')

score = correct / total

print(observed, 'matched entries in submission')
print(total, 'entries in reference file')

print('sdqc accuracy:', score)
