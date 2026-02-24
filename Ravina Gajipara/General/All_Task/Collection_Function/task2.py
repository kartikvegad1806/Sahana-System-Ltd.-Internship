student = {}

roll = input("Enter Roll No: ")
name = input("Enter Name: ")
marks = int(input("Enter Marks: "))

student[roll] = {
    "Roll": roll,
    "name": name,
    "marks": marks,
    "grade":None
}

for key, value in student.items():
    marks = value['marks']
    if marks >= 90:
        value['grade'] = 'A'
    elif marks >= 80 and marks < 90:
        value['grade'] = 'B'
    elif marks >= 60 and marks < 80:
        value['grade'] = 'C'
    elif marks >= 40 and marks < 60:
        value['grade'] = 'D'
    elif marks < 40:
        value['grade'] = 'Fail'
    else:
        print("A not a valid mark")

print(student)