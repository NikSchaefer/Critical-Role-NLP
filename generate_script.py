import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

FILE_PATH = "data/C2E040.json"

f = open(FILE_PATH, "r")
data = json.load(f)

meta_data = data["METADATA"]

turns = data["TURNS"]

w = open("script.txt", "w")

for turn in turns:
    name = turn["NAMES"][0]
    utterances = "\n".join(turn["UTTERANCES"])
    
    w.write(name + ": " + utterances + "\n\n")



