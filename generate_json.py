import itertools
import json_lines
import json
import os

m = {'hairdrier': 'hair drier', 'pottedplant': 'potted plant', 'trafficlight': 'traffic light', 'teddybear':'teddy bear', 'baseballbat': 'baseball bat', 'baseballglove': 'baseball glove', 'tennisracket': 'tennis racket', 'diningtable': 'dining table', 'parkingmeter': 'parking meter', 'sportsball': 'ball', 'wineglass': 'wine glass', 'hotdog': 'hot dog', 'stopsign': 'stop sign', 'firehydrant': 'fire hydrant'}

d = []
val_fn= '/home/suji/spring20/vilbert_beta/data/VCR/orig/val.jsonl'
entries = []
with json_lines.open(val_fn) as reader:
    for obj in reader:
        entries.append(obj)
visited = set()
fn_list = []
for entry in entries:
    imgid = entry['img_id']
    if imgid in visited:
        continue
    meta_fn, img_fn = entry['metadata_fn'], entry['img_fn']
    fn_list.append((meta_fn, img_fn, imgid))
    visited.add(imgid)

for metafn, img_fn, imgid in fn_list:
    with open(os.path.join("/home/suji/spring20/vilbert_beta/data/VCR/vcr1images", metafn)) as f:
        data = json.load(f)
    w, h=data['width'], data['height']
    bb_list =data['boxes']
    obj_list = data['names']
    
    pairings_idx = [(x, y) for x, y in list(itertools.product(range(len(bb_list)), repeat=2)) if x != y]
    mydict = {}
    mydict['relationships'] = []
    for x, y in pairings_idx:
        bb1, bb2 = bb_list[x], bb_list[y] # bb1 is subject bb2 is object
        lb1, lb2 = obj_list[x], obj_list[y]
        if lb1 in m.keys():
            lb1 = m[lb1]
        if lb2 in m.keys():
            lb2 = m[lb2]
    
        mydict['relationships'].append({"predicate": "", "object":{"name": lb1, "h": bb1[3] - bb1[1], "synsets": [], "object_id": 0, "w": bb1[2] - bb1[0], "y": bb1[1], "x": bb1[0]}, "relationship_id": "", "synsets": [], "subject": {"name": lb2, "h": bb2[3] - bb2[1], "synsets": [], "object_id": 0, "w": bb2[2] - bb2[0], "y": bb2[1], "x": bb2[0]}}) 
    #mydict['image_id'] = "/home/suji/spring20/vilbert_beta/data/VCR/vcr1images/" + img_fn
    mydict['image_id'] = int(imgid.split('-')[1])
    d.append(mydict)

with open('datasets/large_scale_VRD/Visual_Genome/rel_vcr.json', 'w') as outfile:
    json.dump(d, outfile)
