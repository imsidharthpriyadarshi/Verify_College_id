libray_token=["library", "librarian","member","books"]
reciept_token=["invoice" ,"receipt", "fees" , "fee" , "transaction" , "tuition" , "cash" , "cheque" , "money" , "charges" , "payment" , "accounts" , "paid" , "bank" , "charges"]




def isLibraryCard(text:str):
    for token in libray_token:
        if token in text.lower():
            print(token)
            return True
    return False


def isFeeReciept(text:str):
    for token in reciept_token:
        if token in text.lower():
            print(token)
            return True
    return False
