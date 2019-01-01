import os
import zipfile

path = input("Specify full path to root repo directory: ")
while not os.path.exists(path):
    print("The path you input is not correct - please fix.")
    path = input("Specify full path to root repo directory: ")

folder_list = ["data","results","notebooks","scripts","charts","archive"]

for folder in folder_list:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f"Creating {folder}.")
    else:
        print(f"{folder} already exists - skipping.")

data_path = path + "data/"
de = input("Do you need to download an environment? (1) Yes (2) No: ")
print(f"You input {de} which is type {type(de)}")
if int(de)==1:
    os.chdir(data_path)
    agents = input("(1) Single or (2) Multi-Agent implementation?")
    server = input("Do you need a cloud (headless) version (Linux only)? (1) Yes, (2) No: ")

    if int(server)==1:
        if int(agents)==1:
            os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip")
        else: # agents==2
            os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip")
    else: # server==2
        operating_system = input("What is your operating system: (1) Linux, (2) Mac, (3)  Windows?: ")
        if int(operating_system)==1:
            if int(agents)==1:
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip")
            else:
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip")
        elif int(operating_system)==2:
            if int(agents)==1:
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip")
            else:
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip")

        else: # operating_system==3
            version = input("What Windows version? (1) 32-bit, (2) 64-bit: ")
            if int(agents)==1 and int(version)==1:
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip")
            elif int(agents)==1 and int(version)==2:
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip")
            elif int(agents)==2 and int(version)==1:
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip")
            else: #agents==2 version==2
                os.system("wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip")

    for file in os.listdir(data_path):
        if file.split('.')[-1]=="zip":
            fp = data_path + file
            with zipfile.ZipFile(fp,'r') as zip_ref:
                zip_ref.extractall(data_path)
            print("Environment downloaded and unzipped.")

print("Setup complete.")
