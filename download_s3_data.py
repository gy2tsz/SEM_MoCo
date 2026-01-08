#!/usr/bin/env python3
"""Download SEM datasets from AWS S3."""

import os
import boto3
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def load_s3_config(config_path):
    """Load S3 configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_s3_folder(bucket_name, s3_prefix, local_dir, profile=None):
    """Download entire S3 folder to local directory."""
    
    # Setup S3 client
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client('s3')
    
    # Create local directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # List objects in S3
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
    
    for page in pages:
        if 'Contents' not in page:
            continue
        
        for obj in tqdm(page['Contents'], desc=f"Downloading from {s3_prefix}"):
            key = obj['Key']
            
            # Skip if it's a folder
            if key.endswith('/'):
                continue
            
            # Local file path
            local_path = os.path.join(local_dir, key[len(s3_prefix):].lstrip('/'))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            s3.download_file(bucket_name, key, local_path)
            print(f"Downloaded: {local_path}")


def download_all_datasets(config):
    """Download all datasets specified in config."""
    bucket = config.get('bucket')
    profile = config.get('profile')
    datasets = config.get('datasets', {})
    
    if not bucket:
        raise ValueError("Bucket name not specified in config")
    
    for dataset_name, dataset_config in datasets.items():
        s3_prefix = dataset_config.get('s3_prefix')
        local_dir = dataset_config.get('local_dir')
        
        if not s3_prefix or not local_dir:
            print(f"⚠️  Skipping {dataset_name}: missing s3_prefix or local_dir")
            continue
        
        print(f"\n📥 Downloading {dataset_name}...")
        print(f"   S3: s3://{bucket}/{s3_prefix}")
        print(f"   Local: {local_dir}")
        
        try:
            download_s3_folder(bucket, s3_prefix, local_dir, profile)
            print(f"✓ {dataset_name} download complete")
        except Exception as e:
            print(f"✗ Error downloading {dataset_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEM datasets from S3")
    parser.add_argument(
        "--config", 
        default="./configs/aws_s3.yaml",
        help="Path to AWS S3 config file (default: ./configs/aws_s3.yaml)"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Specific dataset to download (e.g., 'train', 'eval'). If not specified, downloads all."
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_s3_config(args.config)
        print(f"✓ Loaded config from {args.config}")
    except FileNotFoundError:
        print(f"✗ Config file not found: {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"✗ Error parsing YAML: {e}")
        exit(1)
    
    # Download specific dataset or all
    if args.dataset:
        datasets = config.get('datasets', {})
        if args.dataset not in datasets:
            print(f"✗ Dataset '{args.dataset}' not found in config")
            print(f"Available datasets: {list(datasets.keys())}")
            exit(1)
        
        dataset_config = datasets[args.dataset]
        bucket = config.get('bucket')
        profile = config.get('profile')
        s3_prefix = dataset_config.get('s3_prefix')
        local_dir = dataset_config.get('local_dir')
        
        print(f"📥 Downloading {args.dataset}...")
        download_s3_folder(bucket, s3_prefix, local_dir, profile)
        print(f"✓ {args.dataset} download complete to {local_dir}")
    else:
        download_all_datasets(config)
