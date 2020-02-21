import cPickle as pkl 
#keys: [u'boxes_obj', u'labels_obj', u'labels_rel', u'scores_obj', u'image_id', u'scores_rel', u'scores_sbj', u'boxes_rel', u'labels_sbj', u'image_idx', u'boxes_sbj']
fn = 'checkpoints/vcr_wiki_and_relco/VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/1gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers/test/reldn_detections.pkl'
with open(fn, "rb") as input_file:
    e = pkl.load(input_file)
#print(len(e[u'labels_obj']))
#print(len(e.keys()))
#print(e.keys())
boxes_obj = e[u'boxes_obj']
labels_obj = e[u'labels_obj']
e['labels_rel']
e[u'scores_obj']
e[u'image_id']
e[u'scores_rel']
e[u'scores_sbj']
e[u'boxes_rel']
e[u'labels_sbj']
e[u'image_idx']
e[u'boxes_sbj']














