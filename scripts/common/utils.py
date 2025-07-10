import os

def get_files_from_directory(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(('.csv', '.xlsx', '.xlsb'))
    ]
