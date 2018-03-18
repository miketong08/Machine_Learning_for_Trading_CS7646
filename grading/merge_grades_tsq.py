"""Merge grades into T-Square format."""

import argparse
import pandas as pd

# Columns to pull from respective CSV files
# Note: Must be mututally exclusive, except for join_col
tsq_grades_cols = ['Display ID', 'ID', 'Last Name', 'First Name']
grades_cols = ['ID', 'grade']
join_col = 'ID'


def merge_grades(tsq_grades_file, grades_file, out_file, keep_tsq_header=True):
    """Merge grades from grades_file into (blank) tsq_grades_file, save as out_file."""
    #print "merge_grades(): Merging {} and {}".format(tsq_grades_file, grades_file)  # [debug]
    
    # Read CSV files
    tsq_grades = pd.read_csv(tsq_grades_file, skiprows=1)
    grades = pd.read_csv(grades_file)
    if grades.shape[1] == 2 or any([col not in grades.columns for col in grades_cols]):
        print "merge_grades(): grades_file has wicked (T-Square-like?) header! Trying to skip..."  # [debug]
        print "merge_grades(): grades.columns: {}".format(grades.columns)  # [debug]
        grades = pd.read_csv(grades_file, skiprows=1)
        if grades.shape[1] == 2 or any([col not in grades.columns for col in grades_cols]):
            print "merge_grades(): Nope, still bad. Aborting..."  # [debug]
            raise Exception("Bad header in grades_file: {}".format(grades_file))  # [debug]
        else:
            print "merge_grades(): Worked!"  # [debug]
            print "merge_grades(): grades.columns: {}".format(grades.columns)  # [debug]

    # Merge them
    grades_joined = pd.merge(tsq_grades[tsq_grades_cols], grades[grades_cols], left_on=join_col, right_on=join_col, how='left')

    # Write output CSV
    #print "merge_grades(): Saving as {} (keep_tsq_header? {})".format(out_file, keep_tsq_header)  # [debug]

    # Copy T-Square header, if desired
    if keep_tsq_header:
        tsq_header = ""
        with open(tsq_grades_file, 'r') as f:
            tsq_header = f.readline() + f.readline()  # two lines, second one is blank
        #print "merge_grades(): Copying T-Square header:-\n{}".format(tsq_header)  # [debug]
        with open(out_file, 'w') as f:
            f.write(tsq_header)

    grades_joined.to_csv(out_file, index=False, mode=('a' if keep_tsq_header else 'w'))
    print "merge_grades(): Done!"  # [debug]

    print "merge_grades(): {} zeros".format((grades['grade'] == 0.0).sum())


def main():
    parser = argparse.ArgumentParser(description=merge_grades.__doc__)
    parser.add_argument('--tsq_grades_file', required=True, help='blank T-Square grades.csv file')
    parser.add_argument('--grades_file', required=True, help='CSV file with scores in it')
    parser.add_argument('--out_file', required=True, help='output CSV file (can overwrite)')
    parser.add_argument('--drop_tsq_header', dest='keep_tsq_header', action='store_false', help='drop the 2-line header on top of T-Square grades.csv file')
    args = parser.parse_args()
    merge_grades(args.tsq_grades_file, args.grades_file, args.out_file, args.keep_tsq_header)


if __name__ == '__main__':
    main()
