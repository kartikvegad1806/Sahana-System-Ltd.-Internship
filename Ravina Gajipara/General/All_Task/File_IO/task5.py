def file_operation(filepath):
    
    # Step 1: Take number of lines
    n = int(input("Enter number of lines: "))
    
    # Step 2: Write data to file
    with open(filepath, "a") as f:
        for i in range(n):
            line = input(f"Enter line {i+1}: ")
            f.write(line + "\n")
    
    print("\nData written successfully!\n")
    
    # Step 3: Read data from file
    with open(filepath, "r") as f:
        content = f.read()
    
    print("File Content:\n")
    print(content)
    
    # Step 4: Counting
    lines = content.splitlines()
    words = content.split()
    chars_with_space = len(content)
    chars_without_space = len(content.replace(" ", "").replace("\n", ""))
    
    print("Total Lines:", len(lines))
    print("Total Words:", len(words))
    print("Characters (with spaces):", chars_with_space)
    print("Characters (without spaces):", chars_without_space)


# 🔹 User passes filename + location
path = input("Enter full file path (example: D:\\test.txt): ")

file_operation(path)