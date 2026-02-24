n = int(input("Enter number of elements: "))

main_list = []

for i in range(n):
    num = int(input("Enter element: "))
    main_list.append(num)

print("Main List:", main_list)

total = sum(main_list)

if total % 2 != 0:
    print("Cannot divide list into two equal sum sublists")
else:
    target = total // 2
    sub1 = []
    sub2 = main_list.copy()
    current_sum = 0

    for num in main_list:
        if current_sum + num <= target:
            sub1.append(num)
            sub2.remove(num)
            current_sum += num

    if current_sum == target:
        print("Sublist 1:", sub1)
        print("Sublist 2:", sub2)
    else:
        print("Equal division not possible")