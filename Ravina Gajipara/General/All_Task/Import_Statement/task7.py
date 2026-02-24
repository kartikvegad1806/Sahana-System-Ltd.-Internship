def write_data(n):
    
    students = []

    # Open files for writing
    info_file = open("D:\Desktop Of D\CorePython\All_Task\Import_Statement\studentInfo.txt", "w")
    marks_file = open("D:\Desktop Of D\CorePython\All_Task\Import_Statement\studentMarks.txt", "w")

    for i in range(n):
        print(f"\nStudent {i+1}")

        roll = input("Enter Roll No: ")
        name = input("Enter Name: ")

        m1 = int(input("Enter Marks Subject 1: "))
        m2 = int(input("Enter Marks Subject 2: "))
        m3 = int(input("Enter Marks Subject 3: "))

        # Write student info
        info_file.write(f"{roll}-{name}\n")

        # Write student marks
        marks_file.write(f"{roll}-{m1}-{m2}-{m3}\n")

        # Calculate average
        avg = (m1 + m2 + m3) / 3

        students.append((roll, name, avg))

    info_file.close()
    marks_file.close()

    # Sort by average (descending)
    students.sort(key=lambda x: x[2], reverse=True)

    # Open grade files
    A_file = open("Agrade.txt", "w")
    B_file = open("Bgrade.txt", "w")
    C_file = open("Cgrade.txt", "w")

    # Store according to grade
    for roll, name, avg in students:
        record = f"{roll}-{name}-{avg:.2f}\n"

        if 80 <= avg <= 100:
            A_file.write(record)
        elif 60 <= avg < 80:
            B_file.write(record)
        elif avg < 60:
            C_file.write(record)

    A_file.close()
    B_file.close()
    C_file.close()

    print("\nData written successfully with grading!")


# Main
num = int(input("Enter number of students: "))
write_data(num)