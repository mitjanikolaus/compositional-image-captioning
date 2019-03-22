import os
import json
import pickle

data_folder = "../datasets/refcoco+/"

path = os.path.join(data_folder, "instances.json")
with open(path, "r") as json_file:
    instances = json.load(json_file)

print(len(instances["images"]))
for instance in instances:
    print(instance)

for annotation in instances["annotations"][:10]:
    print(annotation)

ref_file = os.path.join(data_folder, "refs(unc).p")
with open(ref_file, "rb") as pickle_file:
    refs = pickle.load(pickle_file)

print(len(refs))
