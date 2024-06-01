import glob
import json
import os
import numpy as np
import pickle
from utils import *

def read_json_lines_and_write(file_path, out_path, var_num, is_train):
    gam_data = {}
    with open(file_path, 'r') as json_file:
        with open(out_path, 'w') as output_file:
            for line_number, line in enumerate(json_file, start=1):
                try:
                    gam_data = {}
                    data = json.loads(line)        
                    new_X = [[sublist[var_num]] for sublist in data['X']]
                    gam_data["X"] = new_X
                    gam_data["Y"] = data["Y"] 
                    gam_data["EQ"] = data["EQ"]
                    gam_data["Skeleton"] = data["Skeleton"]

                    if not is_train: 
                        new_XT = [[sublist[var_num]] for sublist in data['XT']]
                        gam_data["XT"] = new_XT
                        gam_data["YT"] = data["YT"] 
                    json.dump(gam_data, output_file)
                    output_file.write('\n')
                except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line {line_number}: {e}")

# Update the json object 
def read_json_lines_and_update_y(file_path, residuals):
    try:
        with open(file_path, 'w') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    data = json.loads(line)
                    data["Y"] = residuals
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")
    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Break down train dataset into separate vars and return a new dataset
def create_gam_datasets(is_train ,dataset_path, write_path, VarNum):
    if os.path.exists(write_path):
        print(f"The write path '{write_path}' already exists. The function will not run.")
        return
    else:
        with open(write_path, 'w') as output_file:
            json.dump({}, output_file)
            print(f"Created Empty Json at {write_path}")

    print("Creating GAM datasets")
    print(f"Reading from {dataset_path}")
    files = glob.glob(dataset_path)[:100]

    for file in files:
        print(f"Extracting from {file}")
        read_json_lines_and_write(file, write_path, VarNum, is_train)


def gam_backfitting_preprocess(is_test, is_train, json_file, blockSize, 
                               numVars, numYs, numPoints, target, addVars,
                               const_range, trainRange, decimals, train_chars):
    print(f"Processing JSON file: {json_file}")
    text = processDataFiles(json_file)
    if is_train:
        chars = sorted(list(set(text))+['_','T','<','>',':'])
    
    text = text.split('\n') # convert the raw text to a set of examples
    if is_train:
        trainText = text[:-1] if len(text[-1]) == 0 else text
        random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
    
        dataset = CharDataset(text, blockSize, chars, numVars=1, # 1 variable because we fit 1 variable at a time
                        numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals, augment=False)
    else:
        dataset = CharDataset(text, blockSize, train_chars, numVars=1, 
                        numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals, augment=False)


    if is_train:
        return trainText, chars, dataset
    elif is_test:
        return text, dataset
    else:
        return dataset


    





