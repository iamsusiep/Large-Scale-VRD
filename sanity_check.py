# pip install json-lines
import json_lines
_object_categories = set()
with open('/home/suji/spring20/Large-Scale-VRD/datasets/large_scale_VRD/Visual_Genome/object_categories_spo_joined_and_merged.txt') as obj_categories:
    for line in obj_categories:
        _object_categories.add(line[:-1])

val_fn= '/home/suji/spring20/vilbert_beta/data/VCR/orig/val.jsonl'
entries = []
with json_lines.open(val_fn) as reader:
    for obj in reader:
        entries.append(obj)
vcr_obj = set()
for entry in entries:
    for x in entry["objects"]:
	    vcr_obj.append(x)
print("not in category")
for x in in vcr_obj:
    if x not in _object_categories:
        print(x)
