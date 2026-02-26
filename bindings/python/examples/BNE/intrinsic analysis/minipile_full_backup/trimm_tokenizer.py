import json
import os

tokens = [2**17, 2**16, 2**15, 2**14, 2**13]

data = {}
for file in os.listdir("data/BNE"):
    if file.endswith("vocab.json") or file.endswith("txt"):
        continue
    with open("data/BNE/" + file, "r") as file:
        data = json.load(file)
    print(len(data["model"]["vocab"]))
    print(len(data["model"]["merges"]))

"""
for token in tokens:
    with open("data/BNE/bne_byteLevel_minipile_full_tokens_262144.json", "r") as file:
        data = json.load(file)
    vocab = len(data["model"]["vocab"])
    merges = len(data["model"]["merges"])
    data_file = data
    data_file["model"]["vocab"] = list(data["model"]["vocab"])[:token]
    data_file["model"]["merges"] = list(data["model"]["merges"])[:(merges-2**18+token)]
    with open("data/BNE/bne_byteLevel_minipile_full_tokens_262144.json".replace(str(2**18), str(token)), "w") as file:
        file.write(json.dumps(data_file))
"""


# Rename files for BNE_old_algo
"""
for file in os.listdir("data/BNE_old_algo"):
    os.rename("data/BNE_old_algo/" + file, "data/BNE_old_algo/" + file.replace("bne", "bne_old_algo"))
"""