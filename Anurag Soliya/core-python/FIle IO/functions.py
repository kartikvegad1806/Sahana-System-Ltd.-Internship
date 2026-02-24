def inp():
    main_list = []
    number = int(input("Enter number of lines: "))
    while number:
        str1 = input("Enter line: ")
        main_list.append(str1)
        number -= 1
    print(main_list)
    return main_list

def write_data(main_list):
    file_name = input("Enter file name: ")
    file = open(file_name , "w")
    for i in main_list:
        file.write(f"{i}\n")
    file.close()
    return file_name

def read_data(file_name):
    file = open(file_name , "r")
    content = file.read()
    print(content)

def countlines(file_name):
    count = 0
    file = open(file_name , "r")
    data = file.read()
    count = data.count("\n")
    print(count)

def countwords(file_name):
    file = open(file_name , "r")
    data = file.read()
    count = len(data.split())
    print(count)

def countchar(file_name):
    file = open(file_name , "r")
    content = file.read()
    count = len(content)
    print(count)

def countcharwithspace(file_name):
    file = open(file_name , "r")
    count_i = 0
    for i in file.read():
        if i == "\n" or i == " ":
            pass
        else:
            count_i += 1
    print(count_i)