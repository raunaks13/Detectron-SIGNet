#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np

from caffe2.python import workspace

sys.path.append('/detectron-docker/')

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
# import detectron.utils.vis_struct2depth as vis_utils
import detectron.utils.vis_raw as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--kitti_eigen_file',
        dest='/mnt/carla_data/carla_test_files_eigen.txt',
        help='txt file containing kitti eigen list of files',
        required=True,
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    dump_root = "/mnt/carla_data/carla_train_full/"

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()


    # For training instances
    with open('/mnt/carla_data/carla_train_full/train.txt', 'r') as f:

    # # For testing instances
    # with open('/mnt/carla_data/test_files/test_files_eigen.txt', 'r') as f:
        im_list = f.read().splitlines()
        f.close()

    #im_list = im_list[23:]
    for i, im_name in enumerate(im_list):
        logger.info('Processing {}/{}'.format(i+1, len(im_list)))
        logger.info('Processing {}'.format(im_name))

        # For training instances
        im_dir = im_name.split(" ")[0]
        im_id = im_name.split(" ")[1]
        im_name = im_dir + "/" + str(im_id) + "_rgb.png"
        im = cv2.imread('/mnt/carla_data/carla_train_full/' + im_name)

        ## For testing instances
        # im = cv2.imread('/mnt/carla_data/test_files/' + im_name)

        image_width = im.shape[1]
        im_a = im[:,:image_width//3,:]
        im_b = im[:,image_width//3:2*image_width//3,:]
        im_c = im[:,2*image_width//3:3*image_width//3,:]
        images = [im_a, im_b, im_c]
        image_files = ['_a', '_b', '_c']

        timers = defaultdict(Timer)
        t = time.time()
        masks = []
        for idx, im in enumerate(images):
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, im, None, timers=timers
                )
            """
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
            """
            if i == 0:
                logger.info(
                    ' \ Note: inference on the first image will be slower than the '
                    'rest (caches and auto-tuning need to warm up)'
                )

            file_name = os.path.basename(im_name).split(".")[0]
            dir_name = os.path.dirname(im_name)
            mask_numpy = vis_utils.vis_one_image(
                # im[:, :, ::-1],  # BGR -> RGB for visualization
                im,
                file_name,
                dir_name,
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=args.thresh,
                kp_thresh=args.kp_thresh,
                ext=args.output_ext
            )
            masks.append(mask_numpy)

        f = im_name.split("/")[-1].split(".")[0]
        dump_dir = os.path.join(dump_root, os.path.dirname(im_name).split("/")[-1])
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        masks = np.concatenate((masks[0], masks[1], masks[2]), axis=1)
        print(masks.shape)

        np.save(os.path.join(dump_dir, '{}'.format(f + "_instance_new.npy")), masks)
        # masks.astype('int8').tofile(dump_dir + '/' + '{}'.format(f + "_instance_new.raw"))
        
        # cv2.imwrite(im_name.replace(".png", "-fseg.png"), masks)
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        # sys.exit(-1)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
