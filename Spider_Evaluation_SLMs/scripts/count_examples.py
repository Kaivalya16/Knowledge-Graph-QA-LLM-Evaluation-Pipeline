import json
import glob

def count_qa_pairs(filepath):
    """
    Reads a JSON file and counts the number of QA pairs.
    Assumes the main data structure is a list of dictionaries.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # If the JSON is a direct list of objects [ {...}, {...} ]
            if isinstance(data, list):
                return len(data)
            
            # If the JSON is a dictionary containing a list, e.g., {"data": [ {...}, {...} ]}
            elif isinstance(data, dict):
                # Search for the first list within the dictionary values
                for key, value in data.items():
                    if isinstance(value, list):
                        return len(value)
                return 0
            else:
                return 0
                
    except FileNotFoundError:
        return "File not found."
    except json.JSONDecodeError:
        return "Invalid JSON format."
    except Exception as e:
        return f"An error occurred: {e}"

# --- Example Usage ---

# 1. Count a single specific file
# single_file = "modified_dataset/pos_train.json"
# count = count_qa_pairs(single_file)
# print(f"'{single_file}' contains {count} QA pairs.")

# print("-" * 30)

# 2. Count all JSON files in the current directory
print("Counting all JSON files in the directory:")
json_files = glob.glob("final_datasets_checked/*.json")

for json_file in json_files:
    pair_count = count_qa_pairs(json_file)
    print(f"'{json_file}': {pair_count} examples")