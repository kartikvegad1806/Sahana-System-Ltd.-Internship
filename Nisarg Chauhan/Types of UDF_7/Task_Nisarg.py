def calculate(*lists):
    """Function that handles variable number of lists"""
    
    if len(lists) == 1:
        # If Single list then print the list
        print(lists[0])
    
    elif len(lists) == 2:
        # If Two lists then concatenate and print max-min
        concatenated = lists[0] + lists[1]
        min_val = min(concatenated)
        max_val = max(concatenated)
        print(f"Min = {min_val} & Max = {max_val}")
    
    elif len(lists) == 3:
        # If Three lists then concatenate and print sum of all elements
        concatenated = lists[0] + lists[1] + lists[2]
        total_sum = sum(concatenated)
        print(total_sum)
    
    else:
        # If N lists then concatenate, square elements, and find odd numbers
        concatenated = []
        for lst in lists:
           concatenated.extend(lst)
        print("Concatenated list:", concatenated)
        # Lambda function to square the elements
        squared = list(map(lambda x: x**2, concatenated))
        print("Squared elements:", squared)
        
        # Lambda function to find the odd numbers
        odd_numbers = list(filter(lambda x: x % 2 != 0, concatenated))
        print("Odd numbers:", odd_numbers)


# Main execution of the code
num_lists = int(input("Enter number of lists: "))
lists = []

for i in range(num_lists):
    n = int(input(f"Enter number of elements for list {i+1}: "))
    # elements = list(map(int, input(f"Enter elements for list {i+1}: ").split()))
    elements = []
    for j in range(n):
        element = int(input(f"Enter element {j+1}: "))
        elements.append(element)
    lists.append(elements)

calculate(*lists)