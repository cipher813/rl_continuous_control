import os

folder_list = ["data","results","notebooks","scripts","charts","archive"]

for folder in folder_list:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f"Creating {folder}.")
    else:
        print(f"{folder} already exists - skipping.")
