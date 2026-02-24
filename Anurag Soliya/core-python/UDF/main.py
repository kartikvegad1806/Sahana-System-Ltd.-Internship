num_lists = int(input("Enter number of lists you want to create: "))

all_lists = []  

for i in range(num_lists):
    print(f"\nCreating List {i + 1}")
    
    size = int(input("How many elements in this list? "))
    
    current_list = []
    
    for j in range(size):
        element = input(f"Enter element {j + 1}: ")
        current_list.append(element)
    
    all_lists.append(current_list)

def func(*a):
    if len(a) == 1:
        print(a)
    elif len(a) == 2:
        ls1 , ls2 = a
        for i in ls2:
            ls1.append(i)
        print(ls1)
        print(f"maximum element is: {max(ls1)}")
        print(f"minimum element is: {min(ls1)}")
    elif len(a) == 3:
        ls1, ls2, ls3 = a
        for i in ls3:
            ls2.append(i)
        for i in ls2:
            ls1.append(i)
        print(ls1)
        print(f"sum of ls1 is {sum(ls1)}")
    else:
        combined = []
        for lst in a:
            combined += lst
        print(combined)
        squares = map(lambda x: x ** 2, combined)
        print(*squares)
        odd = filter(lambda x: (x%2!=0) , combined)
        print(*odd)

func(*all_lists)

