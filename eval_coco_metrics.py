# Coco evaluation metrics calculation
# This needs to be run with python 2.7, further a Java JRE needs to be installed
# Before the first run, go to the coco_caption submodule directory and run:
# ./get_stanford_models.sh
import sys

import argparse

from pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

ANNOTATION_FILE_PATH = "coco_caption/annotations/captions_val2014.json"


def eval_coco_metrics(results_file):
    coco = COCO(ANNOTATION_FILE_PATH)
    cocoRes = coco.loadRes(results_file)

    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params["image_id"] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-file", help="File containing generated captions")
    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    eval_coco_metrics(parsed_args.results_file)
