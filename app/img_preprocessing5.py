import os

path="/home/sidharth/Documents/datasets/doubts"
del_path="/home/sidharth/Documents/verify_id_project_datas/model_orientation_college_id/portrait"
for file in os.listdir(path):
    for del_file in os.listdir(del_path):
        if file ==del_file:
            os.remove(os.path.join(del_path, del_file))
            print(del_file,"deleted")
