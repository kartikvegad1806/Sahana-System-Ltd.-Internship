# a=[1,'a',2,'b',3,4,55,'asit','nimesh']
a = input("Enter a value: ").split(",")
str_list=[]
int_list=[]

for item in a:
    if item.isdigit():      
        int_list.append(int(item))
    else:
        str_list.append(item)

print("String List:", str_list)
print("Integer List:", int_list)
       
print(min(int_list))
print(max(int_list))


# reverse = str_list.reverse()
# print(reverse)
str_list.reverse()
print(str_list)
