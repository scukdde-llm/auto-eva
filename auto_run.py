import os

from typing import List


def get_file_list(dir: str) -> List[str]:
    file_list = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.toml'):
                file_list.append(os.path.join(root, file))
    return file_list


file_list = get_file_list("./config")

for f in file_list:
    os.system("rm auto_eva.toml")
    os.system(
        f"ln -s {f} auto_eva.toml")
    os.system("python3 auto_eva.py")
