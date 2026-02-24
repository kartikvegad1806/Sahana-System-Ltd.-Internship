def calculate(*lists):

    # Case 1 → Only one list
    if len(lists) == 1:
        print("List:", lists[0])

    # Case 2 → Two lists
    elif len(lists) == 2:
        combined = lists[0] + lists[1]
        print("Concatenated List:", combined)
        print("Maximum:", max(combined))
        print("Minimum:", min(combined))

    # Case 3 → Three lists
    elif len(lists) == 3:
        combined = []
        for lst in lists:
            combined += lst

        print("Concatenated List:", combined)
        print("Total Sum:", sum(combined))

    # Case 4 → More than 3 lists
    else:
        combined = []
        for lst in lists:
            combined += lst

        print("Concatenated List:", combined)

        square_list = list(map(lambda x: x*x, combined))
        print("Square List:", square_list)

        odd_list = list(filter(lambda x: x % 2 != 0, combined))
        print("Odd List:", odd_list)


# 🔹 User Input Section

num_lists = int(input("Enter number of lists: "))

all_lists = []

for i in range(1, num_lists + 1):
    n = int(input(f"Enter number of elements for list {i}: "))
    
    temp_list = []
    for j in range(n):
        element = int(input(f"Enter element {j+1}: "))
        temp_list.append(element)
    
    all_lists.append(temp_list)

# Call function
calculate(*all_lists)