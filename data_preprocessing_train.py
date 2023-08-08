import os
import json
import shutil
import cv2
import time

def show_progress_message(message):
    print(f"\r{message}", end="")

def animate_progress():
    for _ in range(10):
        time.sleep(0.3)
        print(".", end="", flush=True)


def get_bboxes(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Check if the required keys are present in the dictionary
    task6 = data.get("task6")
    if task6 is None:
        print("Key 'task6' not found in JSON.")
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

    print(f"Found {len(bboxes)} bounding boxes in JSON.")

    bboxes_list = []
    for bbox in bboxes:
        x, y = bbox["x0"], bbox["y0"]
        x0, y0 = x, y + bbox["height"]
        x1, y1 = x + bbox["width"], y
        bboxes_list.append({'bbox': [x0, y0, x1, y1], 'height': bbox["height"]})
    
    return bboxes_list


def create_train_real_barbbox_idl(source_folder):
    output_dir = "dataset/train_real"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "train_real_barbbox.idl")
    with open(output_file, "a") as f:
        json_files = [file for file in os.listdir(source_folder) if file.endswith(".json")]
        total_files = len(json_files)
        for idx, json_file in enumerate(json_files):
            show_progress_message(f"Processing JSON {idx + 1}/{total_files}: ")
            img_name = json_file[:-5] + ".jpg"
            bboxes = get_bboxes(os.path.join(source_folder, json_file))
            if len(bboxes) == 0:
                continue
            line = f"{img_name} -<>- {bboxes}\n"
            f.write(line)
        print("\nProcessing completed!")


def create_train_real_imgsize_idl_and_copy_images(source_folder, target_folder, valid_images, direction):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    output_file = "dataset/train_real/train_real_imgsize.idl"
    with open(output_file, "a") as f:
        total_images = len(valid_images)
        for idx, img_name in enumerate(valid_images):
            show_progress_message(f"Copying image {idx + 1}/{total_images}")
            source_path = os.path.join(source_folder, img_name)
            target_path = os.path.join(target_folder, img_name)
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
            img = cv2.imread(os.path.join(source_folder, img_name))
            img_size = f"[{img.shape[1]}, {img.shape[0]}]"
            line = f"{img_name} -<>- [{img_size}, '{direction}']\n"
            f.write(line)
        print("\nCopying completed!")


if __name__ == "__main__":
    print("Running the script...")

    # folder horizontal bars
    create_train_real_barbbox_idl("data_challenge/anno_horizontal_bar")

    valid_images = [line.split(" -<>- ")[0] for line in open("dataset/train_real/train_real_barbbox.idl", "r")]
    source_folder = "data_challenge/img_horizontal_bar"
    target_folder = "dataset/train_real/plots"

    create_train_real_imgsize_idl_and_copy_images(source_folder, target_folder, valid_images, "horizontal")

    # Simulate some progress for the second part
    print("Processing the second part...")
    animate_progress()

    # folder vertical bars
    create_train_real_barbbox_idl("data_challenge/anno_vertical_bar")

    valid_images = [line.split(" -<>- ")[0] for i, line in enumerate(open("dataset/train_real/train_real_barbbox.idl", "r")) if i >= len(valid_images)]
    source_folder = "data_challenge/img_vertical_bar"
    target_folder = "dataset/train_real/plots"
    create_train_real_imgsize_idl_and_copy_images(source_folder, target_folder, valid_images, "vertical")

    print("All tasks completed! Have a great day! 😄")