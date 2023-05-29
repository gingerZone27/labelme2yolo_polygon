import json
import glob
import os
import argparse
import cv2
import math
from random import sample
from tqdm import tqdm

def generate_yolo_dataset(input_dir, output_dir, val_ratio):
    all_jsons = glob.glob(os.path.join(args.input_dir, "*.json"))
    if len(all_jsons) == 0:
        print("[ERROR] no data in the input dir.")
        exit(-1)

    dir_output_image_train = os.path.join(output_dir, "images", "train")
    dir_output_image_valid = os.path.join(output_dir, "images", "val")
    dir_output_label_train = os.path.join(output_dir, "labels", "train")
    dir_output_label_valid = os.path.join(output_dir, "labels", "val")

    try:
        os.system("mkdir -p {}".format(dir_output_image_train))
        os.system("mkdir -p {}".format(dir_output_image_valid))
        os.system("mkdir -p {}".format(dir_output_label_train))
        os.system("mkdir -p {}".format(dir_output_label_valid))
    except Exception as error:
        print("[ERROR] catch error when preparing your dataset folder: {}".format(error))
        print("[ERROR] this script will automatically generate the output directory and hence you can delete the existing one")
        exit(-1)

    dataset_valid = sample(all_jsons, int(val_ratio * len(all_jsons)))
    
    input_image_size = -1
    output_image_size = -1

    pbar = tqdm(range(len(all_jsons)))
    dict_class = {}
    for item in all_jsons:
        # extract input idx_name for both image and label
        idx_name = item.split('/')[-1].split('.')[0]
        input_image_name = os.path.join(input_dir, idx_name+".png")

        image_input = cv2.imread(input_image_name)
        input_image_size = image_input.shape[0]
        output_image_size = math.floor(input_image_size / 32) * 32
        image_input = cv2.resize(image_input, (output_image_size, output_image_size))
        cv2.imwrite(output_image_name, image_input)

        output_label_name = None
        output_image_name = None
        if item in dataset_valid:
            output_label_name = os.path.join(dir_output_label_valid, idx_name+".txt")
            output_image_name = os.path.join(dir_output_image_valid, idx_name+".png")
        else:
            output_label_name = os.path.join(dir_output_label_train, idx_name+".txt")
            output_image_name = os.path.join(dir_output_image_train, idx_name+".png")

        # read json file
        with open(item) as f:
            data_json = json.load(f)
        
        all_shapes = data_json["shapes"]
        output_label = ""
        for item_shape in all_shapes:
            if item_shape["label"] not in dict_class:
                dict_class[item_shape["label"]] =  int(len(dict_class))
            
            output_label += str(dict_class[item_shape["label"]]) + " "
            for item_point in item_shape["points"]:
                pt_x, pt_y = float(item_point[0]), float(item_point[1])
                output_label += str(pt_x / input_image_size) + " " + str(pt_y / input_image_size) + " "

            output_label += "\n"
        
        with open(output_label_name, "w") as f:
            f.write(output_label)

        pbar.update(1)

    # generate the configuration yaml of this dataset
    config_yaml = os.path.join(output_dir, "dataset.yaml")
    os.system("touch {}".format(config_yaml))
    str_names = "# class names\nnames:\n"
    str_list_names = ""
    for k, v in dict_class.items():
        str_list_names += "  {}: {}\n".format(int(v), k)
    
    with open(config_yaml, "w") as f:
        str_path = "path: " + output_dir + "\n"
        str_train = "train: images/train\n"
        str_valid = "val: images/val\n"
        str_test = "test: # no test\n"
        str_nc = "# number of class\nnc: " + str(len(dict_class)) + "\n"
        
        f.write(str_path)
        f.write(str_train)
        f.write(str_valid)
        f.write(str_test)
        f.write(str_nc)
        f.write(str_names+str_list_names)

    # make a report of the generated dataset
    os.system("clear")
    print("----------  Dataset Info  ----------")
    print("    Path: {}".format(output_dir))
    print("    Image Size: {} (Input)  --->  {} (output) # image size is a multiplication of 32.".format(input_image_size, output_image_size))
    print("    Number of training images:   {}".format(len(all_jsons)-len(dataset_valid)))
    print("    Number of validation images: {}".format(len(dataset_valid)))
    print("    Available classes: \n{}".format(str_list_names))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  "-i", type=str,   required=True, help="input directory containing images and jsons")
    parser.add_argument("--output_dir", "-o", type=str,   required=True, help="output directory for the generated yolo dataset")
    parser.add_argument("--val_ratio",  "-r", type=float, default=0.2,   help="ratio of val data from the origin")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print("[ERROR] cannot find the input dir.")
        exit(-1)
    
    generate_yolo_dataset(args.input_dir, args.output_dir, args.val_ratio)