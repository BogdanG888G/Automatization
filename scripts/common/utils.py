import os

def get_files_from_directory(folder):
    files_list = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.csv', '.xlsx', '.xlsb')):
                files_list.append(os.path.join(root, f))
    return files_list
