#!/bin/python3

import os

top_p = ["-2.0", "-1.9", "-1.8", "-1.7", "-1.6", "-1.5", "-1.4", "-1.3", "-1.2", "-1.1",
         "-1.0", "-0.9", "-0.8", "-0.7", "-0.6", "-0.5", "-0.4", "-0.3", "-0.2", "-0.1"]

if __name__ == "__main__":
    for p in top_p:
        os.system(
            f"sed \'s/frequency_penalty = 0.1/frequency_penalty = {p}/g;s/name = \"auto_eva_t_0_7_f_0_1\"/name = \"auto_eva_t_0_7_f_{p[1:].replace('.', '_')}_n\"/g\' ./config/frequency/auto_eva_t_0_7_f_0_1.toml > auto_eva_t_0_7_f_{p[1:].replace('.', '_')}_n.toml")
