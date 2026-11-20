import json

# Define your input and output file paths
input_path = 'data/final_negQA.jsonl'
output_path = 'data/final_negQA2.json'

def convert_jsonl_to_json():
    json_array = []

    # Step 1: Read the JSONL file line by line
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip any empty lines
                # Parse the JSON object and add it to our list
                json_array.append(json.loads(line))

    # Step 2: Write the entire list to a standard JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        # indent=4 formats it nicely; ensure_ascii=False protects special characters
        json.dump(json_array, f, indent=4, ensure_ascii=False)

    print(f"Successfully converted {len(json_array)} records.")
    print(f"Standard JSON file saved to: {output_path}")

if __name__ == "__main__":
    convert_jsonl_to_json()