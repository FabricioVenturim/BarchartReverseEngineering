import os
import json
import shutil
import cv2

def get_bboxes(json_file):
    """
    Extract bounding box information from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries containing bounding box information.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Check if the required keys are present in the dictionary
    task6 = data.get("task6")
    if task6 is None:
        # print("Key 'task6' not found in JSON.")
        return []

    output = task6.get("output")
    if output is None:
        print("Key 'output' not found in JSON.")
        return []

    visual_elements = output.get("visual elements")
    if visual_elements is None:
        print("Key 'visual elements' not found in JSON.")
        return []

    bboxes = visual_elements.get("bars")
    if bboxes is None:
        print("Key 'bars' not found in JSON.")
        return []

    bboxes_list = []
    
    for bbox in bboxes:
        x, y = bbox["x0"], bbox["y0"]
        x0, y0 = x, y + bbox["height"]
        x1, y1 = x + bbox["width"], y
        bboxes_list.append({'bbox': [x0, y0, x1, y1], 'height': bbox["height"]})
    
    return bboxes_list


def create_train_real_barbbox_idl(source_folder):
    """
    Create a file containing bounding box information for each image in the source folder.

    Args:
        source_folder (str): Path to the folder containing JSON files.

    """
    output_dir = "dataset/train_real"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "train_real_barbbox.idl")
    with open(output_file, "a") as f:
        json_files = [file for file in os.listdir(source_folder) if file.endswith(".json")]
        for idx, json_file in enumerate(json_files):
            img_name = json_file[:-5] + ".jpg"
            bboxes = get_bboxes(os.path.join(source_folder, json_file))
            if len(bboxes) == 0:
                continue
            line = f"{img_name} -<>- {bboxes}\n"
            f.write(line)

def create_train_real_imgsize_idl_and_copy_images(source_folder, target_folder, valid_images, direction):
    """
    Create a file containing image size information for each image in the source folder and copy images to the target folder.

    Args:
        source_folder (str): Path to the folder containing images.
        target_folder (str): Path to the folder to copy images to.
        valid_images (list): List of valid images.
        direction (str): Direction of the bar chart.

    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    output_file = "dataset/train_real/train_real_imgsize.idl"
    with open(output_file, "a") as f:
        for img_name in valid_images:
            source_path = os.path.join(source_folder, img_name)
            target_path = os.path.join(target_folder, img_name)
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
            img = cv2.imread(os.path.join(source_folder, img_name))
            img_size = f"[{img.shape[1]}, {img.shape[0]}]"
            line = f"{img_name} -<>- [{img_size}, '{direction}']\n"
            f.write(line)


if __name__ == "__main__":
    # folder horizontal bars
    create_train_real_barbbox_idl("data_challenge/anno_horizontal_bar")

    valid_images = [line.split(" -<>- ")[0] for line in open("dataset/train_real/train_real_barbbox.idl", "r")]
    source_folder = "data_challenge/img_horizontal_bar"
    target_folder = "dataset/train_real/plot"

    create_train_real_imgsize_idl_and_copy_images(source_folder, target_folder, valid_images, "horizontal")

    # folder vertical bars
    create_train_real_barbbox_idl("data_challenge/anno_vertical_bar")

    valid_images = [line.split(" -<>- ")[0] for i, line in enumerate(open("dataset/train_real/train_real_barbbox.idl", "r")) if i >= len(valid_images)]
    source_folder = "data_challenge/img_vertical_bar"
    target_folder = "dataset/train_real/plot"
    create_train_real_imgsize_idl_and_copy_images(source_folder, target_folder, valid_images, "vertical")