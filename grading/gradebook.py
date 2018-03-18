import numpy, pandas
import os,glob

def print_zeros_with_submission(gbook_df,subdir):
	zsub_ctr = 0
	for index,row in gbook_df[gbook_df['grade']==0].iterrows():
		lname = row['Last Name']
		fname = row['First Name']
		globpath = os.path.join(subdir,"{}*{}*".format(lname,fname),"Submission*","*")
		submitted_files = glob.glob(globpath)
		#print len(submitted_files)
		#print globpath
		if len(submitted_files)>0:
			print lname,fname,row['Display ID']
			for fname in map(os.path.basename,submitted_files):
				print "\t",fname
			zsub_ctr += 1
	print "Zeros w/ submission:",zsub_ctr
