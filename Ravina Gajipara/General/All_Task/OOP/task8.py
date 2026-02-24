class MaxLimitExceeded(Exception):
    pass


def transaction(bank):

    if bank == "HDFCBank":
        trans_limit = 3
        amount_limit = 20000

    elif bank == "AXISBank":
        trans_limit = 5
        amount_limit = 30000

    else:
        print("Invalid Bank Name")
        return

    while True:
        try:
            withdraw = int(input("Enter amount to withdraw: "))

            if withdraw > amount_limit:
                raise MaxLimitExceeded("Max amount limit exceeded!")

            if trans_limit <= 0:
                raise MaxLimitExceeded("Max transaction limit exceeded!")

            amount_limit -= withdraw
            trans_limit -= 1

            print(f"Transaction Successful ✅")
            print(f"Remaining Amount Limit: {amount_limit}")
            print(f"Remaining Transactions: {trans_limit}")

        except MaxLimitExceeded as e:
            print("Transaction Failed ❌:", e)

        next_trans = input("Do you want next transaction? (yes/no): ")

        if next_trans.lower() != "yes":
            print("Thank you! Process terminated.")
            break


bank = input("Enter bank name: ")
transaction(bank)