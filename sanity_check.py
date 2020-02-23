# pip install json-lines
import json_lines
_object_categories = set()
with open('/home/suji/spring20/Large-Scale-VRD/datasets/large_scale_VRD/Visual_Genome/object_categories_spo_joined_and_merged.txt') as obj_categories:
    for line in obj_categories:
        _object_categories.add(line[:-1])

train_fn = '/home/suji/spring20/vilbert_beta/data/VCR/orig/train.jsonl'
val_fn= '/home/suji/spring20/vilbert_beta/data/VCR/orig/val.jsonl'
entries = []
with json_lines.open(val_fn) as reader:
    for obj in reader:
        entries.append(obj)
with json_lines.open(train_fn) as reader:
    for obj in reader:
        entries.append(obj)


vcr_obj = set()
for entry in entries:
    for x in entry["objects"]:
	    vcr_obj.add(x)
m = {'hairdrier': 'hair drier', 'pottedplant': 'potted plant', 'trafficlight': 'traffic light', 'teddybear':'teddy bear', 'baseballbat': 'baseball bat', 'baseballglove': 'baseball glove', 'tennisracket': 'tennis racket', 'diningtable': 'dining table', 'parkingmeter': 'parking meter', 'sportsball': 'sports ball', 'wineglass': 'wine glass', 'hotdog': 'hot dog', 'stopsign': 'stop sign', 'firehydrant': 'fire hydrant'}
print("total categories in vcr:{}\nnot in category".format(len(vcr_obj)))
for x in vcr_obj:
    if x not in _object_categories:
        if x in m.keys():
            x = m[x]
            continue
        print(x)
