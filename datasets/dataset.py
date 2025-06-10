import csv
import json
import io

def convert_csv_to_json_format(input_file, output_file, limit=200):
    """
    Convert IMDB Dataset to a new file where the sentiment column contains 
    a JSON object with a review field. Limits to first 'limit' rows.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output file
        limit: Maximum number of rows to process (default: 200)
    """
    with open(input_file, 'r', encoding='utf-8') as csv_file, \
         open(output_file, 'w', encoding='utf-8', newline='') as json_file:
        
        # Read the CSV file
        reader = csv.DictReader(csv_file)
        
        # Create CSV writer with proper quoting
        writer = csv.writer(json_file, quoting=csv.QUOTE_ALL)
        
        # Write header
        writer.writerow(["review", "sentiment"])
        
        # Process each row up to the limit
        for i, row in enumerate(reader):
            if i >= limit:
                break
                
            # Create JSON object with review field
            json_obj = {"sentiment": row["sentiment"]}
            json_str = json.dumps(json_obj)
            
            # Use the CSV writer to properly handle quoting
            writer.writerow([row['review'], json_str])
    
    print(f"Conversion complete. First {limit} rows saved to {output_file}")

# Example usage
convert_csv_to_json_format('datasets/IMDB_Dataset.csv', 'datasets/IMDB_200_json.csv', 200)