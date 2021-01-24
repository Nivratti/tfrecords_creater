import os
from loguru import logger
import json

def list_files(root_dir, mindepth = 1, maxdepth = float('inf'), filter_ext=[], return_relative_path=False):
    """
    Usage:

    d = get_all_files(rootdir, mindepth = 1, maxdepth = 2)

    This returns a list of all files of a directory, including all files in
    subdirectories. Full paths are returned.

    WARNING: this may create a very large list if many files exists in the 
    directory and subdirectories. Make sure you set the maxdepth appropriately.

    rootdir  = existing directory to start
    mindepth = int: the level to start, 1 is start at root dir, 2 is start 
               at the sub directories of the root dir, and-so-on-so-forth.
    maxdepth = int: the level which to report to. Example, if you only want 
               in the files of the sub directories of the root dir, 
               set mindepth = 2 and maxdepth = 2. If you only want the files
               of the root dir itself, set mindepth = 1 and maxdepth = 1
    
    filter_ext(list, optional) :  filter files ex. [".jpg", ".jpeg", ".png"]
    return_relative_path(bool): Default false. If true return relative path else return absolute path
    """
    root_dir = os.path.normcase(root_dir)
    file_paths = []
    root_depth = root_dir.rstrip(os.path.sep).count(os.path.sep) - 1
    lowered_filter_ext = tuple([ext.lower() for ext in filter_ext])

    for abs_dir, dirs, files in sorted(os.walk(root_dir)):
        depth = abs_dir.count(os.path.sep) - root_depth
        if mindepth <= depth <= maxdepth:
            for filename in files:
                if filter_ext:
                    if not filename.lower().endswith(lowered_filter_ext):
                        continue

                if return_relative_path:
                    rel_dir = os.path.relpath(abs_dir, root_dir)
                    if rel_dir == ".":
                        file_paths.append(filename)
                    else:
                        file_paths.append(os.path.join(rel_dir, filename))
                else:
                    # append full absolute path
                    file_paths.append(os.path.join(abs_dir, filename))

        elif depth > maxdepth:
            # del dirs[:] 
            pass
    return file_paths

def _get_folder_labels(dataset_dir, skip_hidden=True):
    """
    Returns a list of current folder subdir name as class names.
    Args:
        dataset_dir: A directory containing a set of subdirectories representing
            class names. Each subdirectory should contain PNG or JPG encoded images.
        skip_hidden: Skip folder name that starts with dot
    Returns:
    A list of sorted class names.
    """
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)

        if skip_hidden and filename.startswith('.'):
            continue
        
        if os.path.isdir(path):
            class_names.append(filename)

    return sorted(class_names)

def encode_labels_sklearn(lst_classnames):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(lst_classnames) # le.fit(["dog", "cat"])
    return le

def parse_dataset_mimic_final_structure(dataset_dir, store_mimicked_structure_json=False, mimicked_json_filepath=None):
    """
    Iterate dataset and build structure for tfrecords
    Each dict represents an image and should have a structure that mimics the tfrecord structure.
    """
    labels = _get_folder_labels(dataset_dir) # classes
    logger.info(f"Labels found: {labels}")

    if len(labels) <= 1:
        logger.error(
            f"Length of labels(classes) must be at-least 2. Found labels: {labels}"
        )
        return

    lst_data_dicts = [] # holds dataset structure
    image_index = 0 # index number of image, increase after adding it in list
    for idx, label_text in enumerate(labels):
        class_folderpath = os.path.join(dataset_dir, label_text)

        # list all image files of class folder
        lst_imagefiles = list_files(
            class_folderpath, 
            filter_ext=[".jpg", ".jpeg", ".png"],
            return_relative_path=True
        )

        logger.info(f"Total {len(lst_imagefiles)} images found in {class_folderpath}")

        for imagefile in lst_imagefiles:
            image_abs_path = os.path.join(class_folderpath, imagefile)
            image_data = {
                "filename" : image_abs_path, 
                "id" : image_index,
                "class" : {
                    "label" : idx,
                    "text": label_text # optional
                }
            }
            lst_data_dicts.append(image_data)
            # increase image index
            image_index += 1

    if store_mimicked_structure_json:
        if not mimicked_json_filepath:
            mimicked_json_filepath = os.path.join(dataset_dir, "mimicked_structure.json")

        with open(mimicked_json_filepath, 'w') as fout:
            json.dump(lst_data_dicts , fout)

    return lst_data_dicts


def main():
    dataset_train_dir = "dataset_sample/train"
    # classes = _get_folder_labels(dataset_train_dir)
    # print(f"classes: {classes}")

    lst_data_dicts = parse_dataset_mimic_final_structure(
        dataset_dir=dataset_train_dir,
        store_mimicked_structure_json=True
    )
    print(f"lst_data_dicts: {lst_data_dicts}")
    pass

if __name__ == "__main__":
    main()