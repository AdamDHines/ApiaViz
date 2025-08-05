#!/usr/bin/env python3
"""
Script to extract images from a parquet file and save them as JPEG files.
Handles various image storage formats commonly found in ML datasets.
"""

import pandas as pd
import numpy as np
from PIL import Image
import io
import os
import base64
from pathlib import Path
import argparse


def inspect_parquet_structure(parquet_path):
    """Inspect the parquet file structure to understand the data format."""
    print(f"Inspecting parquet file: {parquet_path}")
    
    # Read just the first few rows to inspect
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nColumn info:")
    print(df.info())
    
    print("\nFirst few rows sample:")
    print(df.head(2))
    
    # Look for image-like columns
    potential_image_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['image', 'img', 'picture', 'photo']):
            potential_image_cols.append(col)
        elif df[col].dtype == 'object':
            # Check if it might contain binary data
            sample_val = df[col].iloc[0]
            if isinstance(sample_val, (bytes, dict)):
                potential_image_cols.append(col)
    
    print(f"\nPotential image columns: {potential_image_cols}")
    return df, potential_image_cols


def extract_image_from_bytes(image_data):
    """Extract PIL Image from bytes data."""
    try:
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            # Try base64 decoding
            try:
                decoded = base64.b64decode(image_data)
                return Image.open(io.BytesIO(decoded))
            except:
                pass
        return None
    except Exception as e:
        print(f"Error extracting image: {e}")
        return None


def extract_image_from_dict(image_dict):
    """Extract PIL Image from dictionary format (common in HuggingFace datasets)."""
    try:
        if isinstance(image_dict, dict):
            # HuggingFace format often has 'bytes' key
            if 'bytes' in image_dict:
                return Image.open(io.BytesIO(image_dict['bytes']))
            elif 'path' in image_dict and 'bytes' in image_dict:
                return Image.open(io.BytesIO(image_dict['bytes']))
        return None
    except Exception as e:
        print(f"Error extracting image from dict: {e}")
        return None


def extract_image_from_array(image_array):
    """Extract PIL Image from numpy array."""
    try:
        if isinstance(image_array, (list, np.ndarray)):
            arr = np.array(image_array)
            
            # Handle different array shapes and types
            if arr.ndim == 3:  # Height x Width x Channels
                if arr.dtype != np.uint8:
                    # Normalize to 0-255 if needed
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                
                if arr.shape[2] == 3:  # RGB
                    return Image.fromarray(arr, 'RGB')
                elif arr.shape[2] == 1:  # Grayscale
                    return Image.fromarray(arr.squeeze(), 'L')
            elif arr.ndim == 2:  # Grayscale
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                return Image.fromarray(arr, 'L')
        return None
    except Exception as e:
        print(f"Error extracting image from array: {e}")
        return None


def convert_parquet_to_jpegs(parquet_path, output_dir="output_images", image_column=None):
    """Convert images from parquet file to JPEG files."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Inspect the file first
    df, potential_cols = inspect_parquet_structure(parquet_path)
    
    # Determine which column contains images
    if image_column is None:
        if not potential_cols:
            print("No obvious image columns found. Please specify --image-column")
            return
        image_column = potential_cols[0]
        print(f"Using column: {image_column}")
    
    if image_column not in df.columns:
        print(f"Column '{image_column}' not found in the parquet file")
        return
    
    print(f"\nExtracting images from column '{image_column}'...")
    
    successful_conversions = 0
    failed_conversions = 0
    
    for idx, row in df.iterrows():
        try:
            image_data = row[image_column]
            image = None
            
            # Try different extraction methods based on data type
            if isinstance(image_data, dict):
                image = extract_image_from_dict(image_data)
            elif isinstance(image_data, (bytes, str)):
                image = extract_image_from_bytes(image_data)
            elif isinstance(image_data, (list, np.ndarray)):
                image = extract_image_from_array(image_data)
            
            if image is not None:
                # Convert to RGB if necessary
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                # Generate filename
                filename = f"image_{idx:06d}.jpg"
                
                # Add label to filename if available
                if 'label' in df.columns:
                    label = row['label']
                    filename = f"image_{idx:06d}_label_{label}.jpg"
                elif 'class' in df.columns:
                    class_name = row['class']
                    filename = f"image_{idx:06d}_class_{class_name}.jpg"
                
                output_path = Path(output_dir) / filename
                
                # Save as JPEG
                image.save(output_path, 'JPEG', quality=95)
                successful_conversions += 1
                
                if successful_conversions % 100 == 0:
                    print(f"Processed {successful_conversions} images...")
            else:
                failed_conversions += 1
                print(f"Failed to extract image at index {idx}")
                
        except Exception as e:
            failed_conversions += 1
            print(f"Error processing row {idx}: {e}")
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {successful_conversions} images")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert images from parquet file to JPEG")
    parser.add_argument("-p", "--parquet_file", help="Path to the parquet file",
                        default='./apiaviz/dataset/tiny-imagenet/data/train-00000-of-00001-1359597a978bc4fa.parquet')
    parser.add_argument("-o", "--output-dir", default="./apiaviz/dataset/tiny-imagenet/train", 
                       help="Output directory for JPEG files (default: output_images)")
    parser.add_argument("-c", "--image-column", help="Specific column name containing images")
    parser.add_argument("--inspect-only", action="store_true", 
                       help="Only inspect the file structure without converting")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.parquet_file):
        print(f"Error: File {args.parquet_file} not found")
        return
    
    if args.inspect_only:
        inspect_parquet_structure(args.parquet_file)
    else:
        convert_parquet_to_jpegs(args.parquet_file, args.output_dir, args.image_column)


if __name__ == "__main__":
    main()