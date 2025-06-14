#!/usr/bin/env python3
"""
Fix CSV output column - convert string format to proper JSON format
Keeps the CSV structure but cleans up the output column
"""

import csv
import json
import ast
from typing import List, Dict, Any


def fix_csv_output_column(input_csv_path: str, output_csv_path: str) -> None:
    """
    Fix the output column in CSV by converting string format to proper JSON
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file with fixed JSON
    """
    
    fixed_rows = []
    errors = []
    
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 since header is row 1
                try:
                    input_text = row['input'].strip()
                    output_text = row['output'].strip()
                    
                    # Parse the string-formatted dictionary in output column
                    try:
                        # First try direct JSON parsing
                        output_dict = json.loads(output_text)
                    except json.JSONDecodeError:
                        try:
                            # Try using ast.literal_eval for Python dict format
                            output_dict = ast.literal_eval(output_text)
                        except (ValueError, SyntaxError):
                            # If both fail, try replacing single quotes with double quotes
                            output_text_fixed = output_text.replace("'", '"')
                            output_dict = json.loads(output_text_fixed)
                    
                    # Convert back to proper JSON string (with double quotes)
                    proper_json_output = json.dumps(output_dict, separators=(',', ':'))
                    
                    # Create fixed row
                    fixed_row = {
                        'input': input_text,
                        'output': proper_json_output
                    }
                    
                    fixed_rows.append(fixed_row)
                    print(f"✅ Fixed row {row_num}: {input_text[:50]}...")
                    
                except Exception as e:
                    error_info = {
                        'row_num': row_num,
                        'error': str(e),
                        'input': row.get('input', 'N/A')[:100],
                        'output': row.get('output', 'N/A')[:100]
                    }
                    errors.append(error_info)
                    print(f"❌ Error processing row {row_num}: {str(e)}")
                    continue
        
        # Write fixed CSV
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['input', 'output'])
            writer.writeheader()
            writer.writerows(fixed_rows)
        
        print(f"\n🎉 Successfully fixed {len(fixed_rows)} rows")
        print(f"📁 Fixed CSV saved to: {output_csv_path}")
        
        if errors:
            print(f"⚠️  {len(errors)} rows had errors:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"   Row {error['row_num']}: {error['error']}")
        
        # Show sample of fixed output
        if fixed_rows:
            print(f"\n📋 Sample fixed row:")
            print(f"Input: {fixed_rows[0]['input'][:100]}...")
            print(f"Output: {fixed_rows[0]['output']}")
            
            # Verify it's valid JSON
            try:
                parsed = json.loads(fixed_rows[0]['output'])
                print(f"✅ Output is valid JSON: {type(parsed).__name__}")
            except:
                print(f"❌ Output is not valid JSON")
                
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found: {input_csv_path}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def validate_fixed_csv(csv_file_path: str) -> None:
    """
    Validate the fixed CSV file
    
    Args:
        csv_file_path: Path to CSV file to validate
    """
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            valid_count = 0
            total_count = 0
            schema_fields = set()
            
            for row in reader:
                total_count += 1
                try:
                    # Try to parse the output as JSON
                    output_dict = json.loads(row['output'])
                    valid_count += 1
                    
                    # Collect schema fields
                    if isinstance(output_dict, dict):
                        schema_fields.update(output_dict.keys())
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in row {total_count}")
            
            print(f"\n🔍 Validation Results:")
            print(f"   Total rows: {total_count}")
            print(f"   Valid JSON rows: {valid_count}")
            print(f"   Success rate: {(valid_count/total_count)*100:.1f}%")
            
            if schema_fields:
                print(f"\n📊 Schema fields found: {sorted(list(schema_fields))}")
                
                # Show value distributions for each field
                with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    field_values = {field: set() for field in schema_fields}
                    
                    for row in reader:
                        try:
                            output_dict = json.loads(row['output'])
                            for field in schema_fields:
                                if field in output_dict:
                                    if isinstance(output_dict[field], list):
                                        field_values[field].update(output_dict[field])
                                    else:
                                        field_values[field].add(str(output_dict[field]))
                        except:
                            continue
                    
                    for field, values in field_values.items():
                        print(f"   {field}: {len(values)} unique values")
                        if len(values) <= 10:
                            print(f"      Values: {sorted(list(values))}")
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")


def show_before_after_comparison(original_csv: str, fixed_csv: str, num_samples: int = 3) -> None:
    """
    Show before/after comparison of the CSV files
    
    Args:
        original_csv: Path to original CSV
        fixed_csv: Path to fixed CSV
        num_samples: Number of samples to show
    """
    
    try:
        print(f"\n📊 Before/After Comparison (showing {num_samples} samples):")
        print("="*80)
        
        # Read original
        with open(original_csv, 'r', encoding='utf-8') as csvfile:
            original_rows = list(csv.DictReader(csvfile))
        
        # Read fixed
        with open(fixed_csv, 'r', encoding='utf-8') as csvfile:
            fixed_rows = list(csv.DictReader(csvfile))
        
        for i in range(min(num_samples, len(original_rows), len(fixed_rows))):
            print(f"\nSample {i+1}:")
            print(f"Input: {original_rows[i]['input'][:100]}...")
            print(f"BEFORE: {original_rows[i]['output']}")
            print(f"AFTER:  {fixed_rows[i]['output']}")
            
            # Verify the after is valid JSON
            try:
                json.loads(fixed_rows[i]['output'])
                print("✅ Valid JSON")
            except:
                print("❌ Invalid JSON")
            print("-" * 40)
        
    except Exception as e:
        print(f"❌ Comparison error: {str(e)}")


if __name__ == "__main__":
    # File paths
    input_csv = "datasets/Langfuse Dataset.csv"
    output_csv = "datasets/Langfuse Dataset - Fixed.csv"
    
    print("🚀 Starting CSV output column fix...")
    
    # Fix the CSV output column
    fix_csv_output_column(input_csv, output_csv)
    
    # Validate the fixed CSV
    validate_fixed_csv(output_csv)
    
    # Show before/after comparison
    show_before_after_comparison(input_csv, output_csv)
    
    print("\n✅ CSV fix complete!")
    print(f"📁 Original file: {input_csv}")
    print(f"📁 Fixed file: {output_csv}") 