import json

# Define the paths to your two input files and the output file
qa_dataset_path = 'data/qa_dataset.json'
negQA_path = 'data/final_negQA.jsonl'
output_path = 'data/final_posQA.json'

def extract_matching_pairs():
    # 1. Read the final_negQA.jsonl file and collect all 'id's
    valid_ids = set()
    with open(negQA_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            valid_ids.add(record['id'])
    
    # 2. Read the qa_dataset.json file
    with open(qa_dataset_path, 'r', encoding='utf-8') as f:
        qa_dataset = json.load(f)
    
    # 3. Filter the qa_dataset to only include objects with a matching id
    matched_pairs = [pair for pair in qa_dataset if pair['id'] in valid_ids]
    
    # 4. Save the new filtered list to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matched_pairs, f, indent=4, ensure_ascii=False)
        
    # Output the results summary
    print(f"Collected {len(valid_ids)} IDs from {negQA_path}")
    print(f"Found {len(matched_pairs)} matching pairs from {qa_dataset_path}")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    extract_matching_pairs()