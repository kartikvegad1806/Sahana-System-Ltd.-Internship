def write_data(filepath):
        # Step 1: Take number of lines- use 3 dif fun 
        n = int(input("Enter number of lines: "))

        # Step 2: Write data to file
        with open(filepath, "a") as f:
            for i in range(n):
                line = input(f"Enter line {i+1}: ")
                f.write(line + "\n")
        print("\nData written successfully!\n")

def reverse_lines(demo_file, dummy_path):
        # Step 3: Reverse line order and write in dummy.txt
        with open(demo_file, "r") as f:
            lines = f.readlines()

        reversed_lines = lines[::-1]

        with open(dummy_path, "w") as f:
            f.writelines(reversed_lines)

        print("\nLine-wise reversed data written to dummy.txt")

def replace_word(dummy_path):
        # Step 4: Replace word in dummy.txt
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

write_data(demo_file)
reverse_lines(demo_file, dummy_file)
replace_word(dummy_file)
