import glob
import json
import os
import numpy as np
import pickle
from utils import *

def read_json_lines_and_write(file_path, out_path, var_num, is_train):
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
def read_json_lines_and_update_y(file_path, residuals, out_path):
    # Read the JSON file line by line
    with open(file_path, 'r') as file:
        with open(out_path, 'w') as output_file:
            for line_number, line in enumerate(file, start=1):
                updated_data = {}
                data = json.loads(line)
                updated_data["X"] = data["X"]
                updated_data["Y"] = residuals
                # if (line_number -1) in residuals.keys():
                #     updated_data["Y"] = residuals[line_number-1]
                # else:
                #     updated_data["Y"] = data["Y"]
                    #continue
                updated_data["EQ"] = data["EQ"]
                updated_data["Skeleton"] = data["Skeleton"]

                json.dump(updated_data , output_file)
                output_file.write('\n')
    

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


def update_json_line(file_path, i, update_func, update_val):
    # Step 1: Read the JSON file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Step 2: Update the i-th dictionary
    if 0 <= i < len(lines):
        data = json.loads(lines[i].strip())
        updated_data = update_func(data, update_val)
        lines[i] = json.dumps(updated_data) + '\n'
    else:
        print(f"Line {i} is out of range.")
    
    # Step 3: Write the updated dictionaries back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def update_y(data, res):
    data["Y"] = res
    return data


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
    
        dataset = CharDataset(text, blockSize, chars, numVars, # 1 variable because we fit 1 variable at a time
                        numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals, augment=False)
    else:
        dataset = CharDataset(text, blockSize, train_chars, numVars, 
                        numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals, augment=False)


    if is_train:
        return trainText, chars, dataset
    elif is_test:
        return text, dataset
    else:
        return dataset


# Write a function to write a function to store all the predicted functions 
def map_additive_functions(additive_functions):

    combined_functions = {}

    for k, v in additive_functions.items():
        for k2,v2 in v.items():
            if k2 not in combined_functions:
                combined_functions[k2] = {}
            # First key is the dataset number
            # Keys in the value are the functions
            combined_functions[k2][k] = v2
        
    return combined_functions

def print_additive_functions(mapped_dict, idx):
    data = mapped_dict[idx]
    final_function = ""

    for k, v in data.items():

        # Need to replace the x_1 with its correct variable number
        # As we are training a 1 var model each time
        fn = v[0].lower().replace("x1", f"x{k}")

        if k != len(data):
            final_function += f"{fn} + "
        else:
            final_function += fn

    return final_function

def print_actual_functions(mapped_dict, idx):
    data = mapped_dict[idx]

    actual_functions = list(data.values())
    # The subsequent functions at each step should match the first one
    all_match = all(actual_functions[0][0] in sublist for sublist in actual_functions)
    if all_match:
        return actual_functions[0][0]
    else:
        raise ValueError("Not all sublists contain the same element.")



def add_gaussian_noise(data_path, out_path):

    with open(data_path, 'r') as file:
        with open(out_path, 'w') as output_file:
            for line_number, line in enumerate(file, start=1):
                updated_data = {}
                data = json.loads(line)
                updated_data["X"] = data["X"]

                # Calculate the standard deviation of our Y vector
                Y_std = np.std(np.array(data["Y"]))

                # Typically add 5% of Y's std
                noise_level = 0.05 * Y_std
                noise = np.random.normal(0, noise_level, np.array(data["Y"]).shape)

                updated_data["Y"] = (np.array(data["Y"])+ noise).tolist()
                updated_data["EQ"] = data["EQ"]
                updated_data["Skeleton"] = data["Skeleton"]

                json.dump(updated_data , output_file)
                output_file.write('\n')
                