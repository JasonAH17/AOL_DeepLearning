import os
import shutil
import xml.etree.ElementTree as ET
import random
import yaml
from tqdm import tqdm

# Configuration
PROJECT_ROOT = r"C:\Users\jsnal\OneDrive - Bina Nusantara\Semester 3\Deep Learning\AOL Project"
SOURCE_DATASET = os.path.join(PROJECT_ROOT, "dataset")
SOURCE_IMAGES = os.path.join(SOURCE_DATASET, "images")
SOURCE_ANNOTATIONS = os.path.join(SOURCE_DATASET, "annotations")

DEST_DATASET = os.path.join(PROJECT_ROOT, "datasets", "helmet_dataset")
TRAIN_IMAGES_DIR = os.path.join(DEST_DATASET, "train", "images")
TRAIN_LABELS_DIR = os.path.join(DEST_DATASET, "train", "labels")
VAL_IMAGES_DIR = os.path.join(DEST_DATASET, "val", "images")
VAL_LABELS_DIR = os.path.join(DEST_DATASET, "val", "labels")

def get_all_classes(annotations_dir):
    classes = set()
    print("Scanning for classes...")
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            classes.add(name)
    
    return sorted(list(classes))

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def main():
    # 1. Detect Classes
    classes = get_all_classes(SOURCE_ANNOTATIONS)
    print(f"Found classes: {classes}")

    # 2. Prepare Directories
    for d in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # 3. Process Files
    xml_files = [f for f in os.listdir(SOURCE_ANNOTATIONS) if f.endswith('.xml')]
    random.shuffle(xml_files)
    
    split_index = int(len(xml_files) * 0.8)
    train_files = xml_files[:split_index]
    val_files = xml_files[split_index:]

    print(f"Total files: {len(xml_files)}")
    print(f"Training: {len(train_files)}, Validation: {len(val_files)}")

    def process_batch(files, img_dest, label_dest):
        for xml_file in tqdm(files):
            # Parse XML
            xml_path = os.path.join(SOURCE_ANNOTATIONS, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get Image Info
            filename = root.find('filename').text
            # Handle case where filename in XML might differ slightly or extension issues
            # We assume filename corresponds to xml name but let's verify exists
            src_img_path = os.path.join(SOURCE_IMAGES, filename)
            if not os.path.exists(src_img_path):
                # Fallback: try replacing extension
                basename = os.path.splitext(xml_file)[0]
                possible_exts = ['.jpg', '.png', '.jpeg']
                found = False
                for ext in possible_exts:
                    if os.path.exists(os.path.join(SOURCE_IMAGES, basename + ext)):
                        src_img_path = os.path.join(SOURCE_IMAGES, basename + ext)
                        filename = basename + ext
                        found = True
                        break
                if not found:
                    print(f"Warning: Image for {xml_file} not found. Skipping.")
                    continue

            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # Convert Labels
            label_content = []
            for obj in root.findall('object'):
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    continue # Skip difficult instances if desired, or keep them
                
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    continue
                cls_id = classes.index(cls_name)
                
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                label_content.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

            # Write Label File
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(label_dest, txt_filename), 'w') as f:
                f.write('\n'.join(label_content))
            
            # Copy Image
            shutil.copy(src_img_path, os.path.join(img_dest, filename))

    print("Processing Training Set...")
    process_batch(train_files, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    
    print("Processing Validation Set...")
    process_batch(val_files, VAL_IMAGES_DIR, VAL_LABELS_DIR)

    # 4. Create data.yaml
    yaml_content = {
        'path': DEST_DATASET,
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    yaml_path = os.path.join(PROJECT_ROOT, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"Dataset preparation complete. Config saved to {yaml_path}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    main()
