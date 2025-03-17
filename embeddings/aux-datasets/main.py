from utils import *
from push_to_hub import *
import argparse
import os


def get_subset(dataset_name):
    subset = []
    
    base_path = os.path.join(dataset_name, "imagesa", "a", "a")
    base_depth = base_path.count(os.sep)
    

    for root, dirs, files in os.walk(base_path):
        
        # Check if the current directory is exactly two levels below the base path
        if root.count(os.sep) - base_depth == 2:
            
            # Process TIFF files in the current directory
            for file in sorted(files):
                if file.lower().endswith(".tif"):
                    tif_path = os.path.join(root, file)
                    base_name = file.rsplit('.', 1)[0]
                    xml_file = base_name + ".xml"
                    xml_path = os.path.join(root, xml_file)
                    
                    if os.path.exists(xml_path):
                        subset.append((tif_path, xml_path))
                    else:
                        print(f"Warning: XML file not found for {tif_path}")
    
    return subset



if __name__ ==  "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset

    dataset = []

    subset = get_subset(dataset_name)
    dataset.append(("train", subset))
    push_to_hub(dataset, dataset_name)