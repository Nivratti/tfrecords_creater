import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from create_tfrecords import create
from dataset_utils import parse_dataset_mimic_final_structure

def generate_tfrecords(
    dataset_dir, dataset_name="train", output_directory="./tfrecords_train",
    num_shards=10, num_threads=5,store_images=True):
    # this should be your array of image data dictionaries. 
    dataset = parse_dataset_mimic_final_structure(
        dataset_dir,
        store_json=True
    )


    failed_images = create(
        dataset=dataset,
        dataset_name=dataset_name,
        output_directory=output_directory,
        num_shards=num_shards,
        num_threads=num_threads,
        store_images=store_images
    )
    return failed_images


def main():
    dataset_train_dir = "dataset_sample/train"

    failed_images = generate_tfrecords(dataset_train_dir)
    print(f"failed_images: {failed_images}")
    pass

if __name__ == "__main__":
    main()