import csv

with open("users_ai_with_cat.csv", "r") as csvfile:
    rowCount =  len(csvfile.readlines())

try:
    with open("users_ai_with_ca.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([rowCount + 2, 'Marlon', 'web developer', 'He is professional.', 'Irvine', '20', 'Web Expert',
                         'full stack    Javascript    software development    web development'])

except:
        print "hhh"


    # [id, name, title, description, location, salary, category, skills]

