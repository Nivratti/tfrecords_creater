import os
import sys
import argparse

import numpy as np
import tensorflow as tf

try:
  from create_tfrecords import create
  from dataset_utils import parse_dataset_mimic_final_structure
except:
  from tfrecords_creater.create_tfrecords import create
  from tfrecords_creater.dataset_utils import parse_dataset_mimic_final_structure

 
def generate_tfrecords(
        dataset_dir, dataset_name="train", output_directory="./tfrecords_train",
        num_shards=10, num_threads=5, shuffle=False, store_images=True,
        explicit_labels=set(), store_mimicked_structure_json=True,
        mimicked_json_filepath=None, silent_on_extra_explicit_labels=False,
        save_labels=True, labels_out_filepath=None,
    ):
    if mimicked_json_filepath is None:
        mimicked_json_filepath = os.path.join(
            output_directory, f"mimicked_structure-{dataset_name}.json"
        )

    if labels_out_filepath is None:
        labels_out_filepath = os.path.join(
            output_directory, f"labels.txt"
        )

    os.makedirs(output_directory, exist_ok=True)

    # this should be your array of image data dictionaries. 
    dataset = parse_dataset_mimic_final_structure(
        dataset_dir,
        explicit_labels=explicit_labels,
        silent_on_extra_explicit_labels=silent_on_extra_explicit_labels,
        store_mimicked_structure_json=store_mimicked_structure_json,
        mimicked_json_filepath=mimicked_json_filepath,
        save_labels=save_labels,
        labels_out_filepath=labels_out_filepath,
    )

    failed_images = create(
        dataset=dataset,
        dataset_name=dataset_name,
        output_directory=output_directory,
        num_shards=num_shards,
        num_threads=num_threads,
        shuffle=shuffle,
        store_images=store_images
    )
    return failed_images


def parse_args():

    parser = argparse.ArgumentParser(description='Wrapper arround tfrecord creater')

    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='Path to the dataset json file.', type=str,
                        required=True)

    parser.add_argument('--prefix', dest='dataset_name',
                        help='Prefix for the tfrecords (e.g. `train`, `test`, `val`).', type=str,
                        required=True)

    parser.add_argument('--output_dir', dest='output_dir',
                        help='Directory for the tfrecords.', type=str,
                        required=True)

    parser.add_argument('--shards', dest='num_shards',
                        help='Number of shards to make.', type=int,
                        required=True)

    parser.add_argument('--threads', dest='num_threads',
                        help='Number of threads to make.', type=int,
                        required=True)

    parser.add_argument('--shuffle', dest='shuffle',
                        help='Shuffle the records before saving them.',
                        required=False, action='store_true', default=True)

    parser.add_argument('--store_images', dest='store_images',
                        help='Store the images in the tfrecords.',
                        required=False, action='store_true', default=True)

    # set() instead of list -- set will preserve element order
    parser.add_argument('--explicit_labels', nargs="+", dest='explicit_labels',
                        help='Labels(classes) of dataset. You can set your own class order.',
                        required=False, default=set())

    parser.add_argument('--store_mimicked_structure_json', dest='store_mimicked_structure_json',
                        help='Store parsed dataset structure(mimicked tfrecords structure).',
                        required=False, action='store_true', default=True)

    parser.add_argument('--mimicked_json_filepath', dest='mimicked_json_filepath',
                        help='Filename to store -- parsed dataset structure(mimicked tfrecords structure).', type=str,
                        required=False)

    parsed_args = parser.parse_args()

    return parsed_args


def main():
    """
    Usage:

    python wrapper_create_tfrecords.py --dataset_path="dataset_sample/train" \
        --prefix="train" \
        --output_dir="./out/sample_tfrecords" \
        --shards=8 --threads=4 --shuffle --store_images \
        --explicit_labels "live" "spoof"

    Returns:
        list: list of failed images
    """
    # dataset_train_dir = "dataset_sample/train"

    # failed_images = generate_tfrecords(dataset_train_dir)
    # print(f"failed_images: {failed_images}")
    # pass

    args = parse_args()

    errors = generate_tfrecords(
        dataset_dir=args.dataset_path,
        dataset_name=args.dataset_name, 
        output_directory=args.output_dir,
        num_shards=args.num_shards, 
        num_threads=args.num_threads, 
        shuffle=args.shuffle, 
        store_images=args.store_images,
        explicit_labels=args.explicit_labels,
        store_mimicked_structure_json=args.store_mimicked_structure_json,
        mimicked_json_filepath=args.mimicked_json_filepath
    )
    
    if errors:
        print("%d images failed." % (len(failed_images),))
        for image_data in failed_images:
            print("Image %s: %s" % (image_data['id'], image_data['error_msg']))
    return

if __name__ == "__main__":
    main()