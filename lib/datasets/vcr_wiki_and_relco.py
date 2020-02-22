# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from datasets.imdb_rel import imdb_rel
import utils.boxes as box_utils
import numpy as np
import scipy.sparse
import json
import cPickle
from core.config_rel import cfg

import gensim
# from autocorrect import spell
from numpy import linalg as la
import PIL
import glob
import logging
import itertools

logger = logging.getLogger(__name__)


class vcr_wiki_and_relco(imdb_rel):
    def __init__(self):
        print("im here, vcr_wiki_and_relco")
        #self.filenames = sorted(glob.glob('/home/suji/spring20/vilbert_beta/data/VCR/vcr1images/lsmdc_0001_American_Beauty/*.json'))
        self.filenames = sorted(glob.glob('/home/suji/spring20/vilbert_beta/data/VCR/vcr1images/lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/*.json'))
        print("filenames length", len(self.filenames))
        self.widths, self.heights = [], []
        self.bb_lists = []
        for fn in self.filenames:
            with open(fn) as f:
                data = json.load(f)
            w, h=data['width'], data['height']
            self.widths.append(w)
            self.heights.append(h)
            bb_list =data['boxes']
            self.bb_lists.append(bb_list)
         
        imdb_rel.__init__(self, 'vcr_wiki_and_relco')
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        self.model = None
        self.relco_model = None
        self.relco_vec_mean = None

    def get_widths_and_heights(self):
        return self.widths, self.heights

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        print('Relco words converted to lowercase.')
        gt_roidb = \
            [self._load_vcr_annotation(index, fn, len(self.filenames))
             for index, fn in enumerate(self.filenames)]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb
    def image_path_at(self, i):
        return self.filenames[i].replace('.json', '.jpg')

    def _load_vcr_annotation(self, index, fn, length):
        """
        Load image and bounding boxes info.
        """
        bb_list = self.bb_lists[index]
        print("fn {} bb_list:".format(fn), bb_list)
        print("Loading image %d/%d..." % (index, length))
        pairings = [(x, y) for x, y in list(itertools.product(bb_list, repeat=2)) if x != y]
        print("pairings", pairings)
        num_rels = len(pairings)
        print("num_rels", num_rels)
        sbj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        obj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        rel_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        # "Seg" area for pascal is just the box area
        sbj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        obj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        rel_seg_areas = np.zeros((num_rels), dtype=np.float32)

        for ix, (sbj, obj) in enumerate(pairings):

            sbj_box = sbj[:-1]
            obj_box = obj[:-1]
            rel_box = box_utils.box_union(sbj_box, obj_box)
            sbj_boxes[ix, :] = sbj_box
            obj_boxes[ix, :] = obj_box
            rel_boxes[ix, :] = rel_box
            sbj_seg_areas[ix] = (sbj_box[2] - sbj_box[0] + 1) * \
                                (sbj_box[3] - sbj_box[1] + 1)
            obj_seg_areas[ix] = (obj_box[2] - obj_box[0] + 1) * \
                                (obj_box[3] - obj_box[1] + 1)
            rel_seg_areas[ix] = (rel_box[2] - rel_box[0] + 1) * \
                                (rel_box[3] - rel_box[1] + 1)

        return {'sbj_boxes': sbj_boxes,
                'obj_boxes': obj_boxes,
                'rel_boxes': rel_boxes,
                'sbj_names': None,
                'obj_names': None,
                'prd_names': None,
                'gt_sbj_classes': None,
                'gt_obj_classes': None,
                'gt_rel_classes': None,
                'gt_sbj_overlaps': None,
                'gt_obj_overlaps': None,
                'gt_rel_overlaps': None,
                'sbj_seg_areas': sbj_seg_areas,
                'obj_seg_areas': obj_seg_areas,
                'rel_seg_areas': rel_seg_areas,
                'sbj_vecs': None,
                'obj_vecs': None,
                'prd_vecs': None,
                'flipped': False}
