import os

file_list = ["1024_top_3", "2000_top_2"]

for f in file_list:
    os.system("rm auto_eva.toml")
    os.system(
        f"ln -s ./config/auto_eva_t_0_7_ir_vec_chunk_{f}.toml auto_eva.toml")
    os.system("python3 auto_eva.py")
