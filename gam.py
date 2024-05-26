import glob
import json
import os
import numpy as np
import pickle
from utils import *

def read_json_lines_and_write(file_path, out_path, var_num):
    gam_data = {}
    try:
        with open(file_path, 'r') as file:
            with open(out_path, 'w') as output_file:
                for line_number, line in enumerate(file, start=1):
                    try:
                        gam_data = {}
                        data = json.loads(line)
                            
                        new_X = [sublist[var_num] for sublist in data['X']]
                        gam_data["X"] = new_X
                        gam_data["Y"] = data["Y"] 
                        gam_data["EQ"] = data["EQ"]
                        gam_data["Skeleton"] = data["Skeleton"]
                        json.dump(gam_data, output_file)
                        output_file.write('\n')
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line {line_number}: {e}")
    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

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
def create_gam_datasets(dataset_path, write_path, numVars):
    files = glob.glob(dataset_path)[:100]
    for file in files:
        for i in range(numVars):
            read_json_lines_and_write(file, write_path, i)


def gam_backfitting_preprocess(is_train, json_file, blockSize, 
                               numVars, numYs, numPoints, target, addVars,
                               const_range, trainRange, decimals):
    text = processDataFiles(json_file)
    chars = sorted(list(set(text))+['_','T','<','>',':'])
    text = text.split('\n') # convert the raw text to a set of examples
    if is_train:
        trainText = text[:-1] if len(text[-1]) == 0 else text
        random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
    
    dataset = CharDataset(text, blockSize, chars, numVars=numVars, 
                        numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals, augment=False) 
    
    return dataset


    





