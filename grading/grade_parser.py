import csv

d_grade = {}
i=0
with open('CS 7646 Fall 2015 Gradebook - Sheet1.csv', 'rb') as read_file:
    reader = csv.reader(read_file, delimiter=',', quotechar='"')
    for row in reader:
        if i>0:
            d_grade[row[0]] = row[13]
        i += 1

ls_full_rows = []
with open('grades.csv', 'rb') as read_file:
    reader = csv.reader(read_file,delimiter=',', quotechar='"')
    for row in reader:
        if len(row)>0 and row[0] in d_grade:
            row[4] = d_grade[row[0]]
        ls_full_rows.append(row)

with open('new_gradebook.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"')
    for row in ls_full_rows:
        print row
        writer.writerow(row)
