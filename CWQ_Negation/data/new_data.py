import json

# Define your input and output file paths
input_path = 'data/negQA.jsonl'
output_path = 'data/filtered_negQA.jsonl'

def clean_negated_answers():
    processed_records = []
    modified_count = 0

    # Step 1: Read the file line by line
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            
            # Extract both lists (defaulting to empty lists if they don't exist)
            answers = record.get('answers', [])
            negated_answers = record.get('negated_answers', [])
            
            # Create a set of the positive answers for faster and easier comparison
            answers_set = set(answers)
            
            original_length = len(negated_answers)
            
            # Step 2: Keep only the items in negated_answers that are NOT in the answers_set
            filtered_negated = [ans for ans in negated_answers if ans not in answers_set]
            
            # Track if we actually deleted anything for our summary printout
            if len(filtered_negated) < original_length:
                modified_count += 1
                
            # Update the record with the cleaned list
            record['negated_answers'] = filtered_negated
            processed_records.append(record)

    # Step 3: Write the updated records to the new file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in processed_records:
            # ensure_ascii=False keeps special characters (like Ãº) intact
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Successfully processed {len(processed_records)} records.")
    print(f"Deleted overlapping options in {modified_count} records.")
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    clean_negated_answers()