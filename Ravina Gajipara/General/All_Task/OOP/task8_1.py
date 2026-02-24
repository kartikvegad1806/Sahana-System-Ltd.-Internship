class MaxLimitExceeded(Exception):
    pass

class Bank:
    def __init__(self, trans_limit, amount_limit):
        self.__trans_limit = trans_limit      
        self.__amount_limit = amount_limit    

    def _withdraw_process(self, amount, bank_name):

        if amount > self.__amount_limit:
            raise MaxLimitExceeded("Amount limit exceeded!")

        if self.__trans_limit <= 0:
            raise MaxLimitExceeded("Transaction limit exceeded!")

        self.__amount_limit -= amount
        self.__trans_limit -= 1

        print(f"{bank_name} Transaction Successful ")
        print("Remaining Amount Limit:", self.__amount_limit)
        print("Remaining Transaction Limit:", self.__trans_limit)


    def withdraw(self, amount):
        pass  


class HDFCBank(Bank):
    def __init__(self):
        super().__init__(3, 20000)

    def withdraw(self, amount):   
        self._withdraw_process(amount, "HDFC")


class AXISBank(Bank):
    def __init__(self):
        super().__init__(5, 30000)

    def withdraw(self, amount):   
        self._withdraw_process(amount, "AXIS")


bank_name = input("Enter Bank Name (HDFCBank / AXISBank): ")

if bank_name == "HDFCBank":
    bank = HDFCBank()
elif bank_name == "AXISBank":
    bank = AXISBank()
else:
    print("Invalid Bank Name")
    exit()

while True:
    try:
        amount = int(input("Enter amount to withdraw: "))
        bank.withdraw(amount)

    except MaxLimitExceeded as e:
        print("Transaction Failed :", e)
        print("Process Terminated.")
        break

    next_trans = input("Do you want next transaction? (yes/no): ")
    if next_trans.lower() != "yes":
        print("Thank you! Process Terminated.")
        break