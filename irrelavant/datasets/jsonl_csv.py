import json
import csv
import sys

def convert_jsonl_to_csv(input_file, output_file):
    """
    Convert JSONL file to CSV with input and output columns
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output CSV file
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as jsonl_file, \
             open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            
            # Create CSV writer
            csv_writer = csv.writer(csv_file)
            
            # Write header
            csv_writer.writerow(['input', 'output'])
            
            # Process each line in JSONL file
            for line_num, line in enumerate(jsonl_file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Parse JSON from each line
                    data = json.loads(line)
                    
                    # Extract input and output
                    input_text = data.get('inputs', {}).get('input', '')
                    output_label = data.get('outputs', {}).get('label', {})
                    
                    # Convert output to JSON string
                    output_json = json.dumps(output_label)
                    
                    # Write to CSV
                    csv_writer.writerow([input_text, output_json])
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                    
        print(f"Successfully converted {input_file} to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error: {e}")

def convert_from_string(jsonl_string, output_file):
    """
    Convert JSONL string data directly to CSV
    
    Args:
        jsonl_string (str): JSONL data as string
        output_file (str): Path to output CSV file
    """
    # Sample data from your example
    sample_data = '''{"inputs": {"input": "create login app for tutor app(query)"}, "outputs": {"label": {"action": "CODE_GENERATION", "subAction": "CODING", "platform": "NOT_FOUND", "framework": "NOT_FOUND", "languageType": "NOT_FOUND"}}}
{"inputs": {"input": "create login screen for currency converter.(query)"}, "outputs": {"label": {"action": "CODE_GENERATION", "subAction": "CODING", "platform": "NOT_FOUND", "framework": "NOT_FOUND", "languageType": "NOT_FOUND"}}}
{"inputs": {"input": "create tutor web application(query)"}, "outputs": {"label": {"action": "CODE_GENERATION", "subAction": "CODING", "platform": "DYNAMIC_WEB_APPLICATION", "framework": "NOT_FOUND", "languageType": "NOT_FOUND"}}}
{"inputs": {"input": "Login scree in react for converter app(query)"}, "outputs": {"label": {"action": "CODE_GENERATION", "subAction": "CODING", "platform": "DYNAMIC_WEB_APPLICATION", "framework": "REACT", "languageType": "REACT_JAVASCRIPT"}}}'''
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write header
            csv_writer.writerow(['input', 'output'])
            
            # Process each line
            for line_num, line in enumerate(sample_data.strip().split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Parse JSON from each line
                    data = json.loads(line)
                    
                    # Extract input and output
                    input_text = data.get('inputs', {}).get('input', '')
                    output_label = data.get('outputs', {}).get('label', {})
                    
                    # Convert output to JSON string
                    output_json = json.dumps(output_label)
                    
                    # Write to CSV
                    csv_writer.writerow([input_text, output_json])
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                    
        print(f"Successfully created {output_file} with sample data")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check command line arguments
    input_file = "datasets/code_gen.jsonl"
    output_file = "datasets/code_gen_output.csv"

    convert_jsonl_to_csv(input_file, output_file)
