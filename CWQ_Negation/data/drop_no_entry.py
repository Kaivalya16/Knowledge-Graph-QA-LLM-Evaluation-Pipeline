import json
input_path = 'data/filtered_negQA.jsonl'
output_path = 'data/final_negQA.jsonl'

def remove_empty_negated_answers():
    valid_records = []
    removed_count = 0

    # Step 1: Read the file line by line
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            
            # Step 2: Check if 'negated_answers' has any items
            # An empty list [] evaluates to False in Python
            if record.get('negated_answers'):
                valid_records.append(record)
            else:
                removed_count += 1

    # Step 3: Write the valid records to the new file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in valid_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Output the results
    print(f"Successfully processed {len(valid_records) + removed_count} total records.")
    print(f"Removed {removed_count} records with empty 'negated_answers'.")
    print(f"Kept {len(valid_records)} valid records.")
    print(f"Final data saved to: {output_path}")

if __name__ == "__main__":
    remove_empty_negated_answers()