def file_process(demo_path, dummy_path):

    # Step 1: Take number of lines
    n = int(input("Enter number of lines: "))

    # Step 2: Write data in demo.txt
    with open(demo_path, "w") as f:
        for i in range(n):
            line = input(f"Enter line {i+1}: ")
            f.write(line + "\n")

    print("\nData written to demo.txt")

    # Step 3: Read data from demo.txt
    with open(demo_path, "r") as f:
        lines = f.readlines()

    print("\nContent of demo.txt:")
    for line in lines:
        print(line.strip())

    # Step 4: Reverse line order and write in dummy.txt
    reversed_lines = lines[::-1]

    with open(dummy_path, "w") as f:
        f.writelines(reversed_lines)

    print("\nLine-wise reversed data written to dummy.txt")

    # Step 5: Replace word in dummy.txt
    old_word = input("\nEnter word to replace: ")
    new_word = input("Enter replacement word: ")

    with open(dummy_path, "r") as f:
        content = f.read()

    updated_content = content.replace(old_word, new_word)

    with open(dummy_path, "w") as f:
        f.write(updated_content)

    print("\nWord replaced successfully in dummy.txt")


# User passes file paths
demo_file = input("Enter full path for demo.txt: ")
dummy_file = input("Enter full path for dummy.txt: ")

file_process(demo_file, dummy_file)