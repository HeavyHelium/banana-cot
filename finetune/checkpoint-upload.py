#!/usr/bin/env python3
"""
Upload model checkpoints to HuggingFace Hub.

Usage:
    python finetune/checkpoint-upload.py --checkpoint outputs/sft_direct_full --repo-id username/model-name
    python finetune/checkpoint-upload.py --checkpoint outputs/sft_direct_full/checkpoint-512 --repo-id username/model-name
    python finetune/checkpoint-upload.py --checkpoint outputs/sft_direct_full --repo-id username/model-name --all-checkpoints
"""

import argparse
import os
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError


def upload_checkpoint(
    checkpoint_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = None,
):
    """Upload a single checkpoint to HuggingFace Hub."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    print(f"Uploading checkpoint from {checkpoint_path} to {repo_id}...")
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"✓ Repository {repo_id} is ready")
    except Exception as e:
        print(f"Warning: Could not create/verify repo: {e}")
    
    # Upload files
    try:
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message or f"Upload checkpoint from {checkpoint_path.name}",
            ignore_patterns=["*.json", "*.log"],  # Skip logs if desired
        )
        print(f"✓ Successfully uploaded {checkpoint_path.name} to {repo_id}")
        return True
    except HfHubHTTPError as e:
        print(f"✗ Error uploading: {e}")
        return False


def upload_all_checkpoints(
    output_dir: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = None,
):
    """Upload final model + all checkpoints to subfolders."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"✓ Repository {repo_id} is ready")
    except Exception as e:
        print(f"Warning: Could not create/verify repo: {e}")
    
    # Upload final model (all files in output_dir except checkpoint-* folders)
    print(f"\n{'='*60}")
    print("Uploading final model...")
    final_files = [f for f in output_dir.iterdir() if f.is_file()]
    for f in final_files:
        print(f"  Uploading {f.name}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {f.name}",
        )
    print(f"✓ Final model uploaded to {repo_id}")
    
    # Find all checkpoint directories
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    
    if not checkpoints:
        print(f"No checkpoints found in {output_dir}")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    
    # Upload each checkpoint to checkpoints/ subfolder
    for checkpoint in checkpoints:
        print(f"\n{'='*60}")
        print(f"Uploading {checkpoint.name} to checkpoints/{checkpoint.name}/...")
        api.upload_folder(
            folder_path=str(checkpoint),
            path_in_repo=f"checkpoints/{checkpoint.name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message or f"Upload {checkpoint.name}",
        )
        print(f"✓ {checkpoint.name} uploaded to checkpoints/{checkpoint.name}/")


def main():
    parser = argparse.ArgumentParser(
        description="Upload model checkpoints to HuggingFace Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory or output directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Upload all checkpoints from the output directory",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN/HF_API_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message",
    )
    
    args = parser.parse_args()
    
    # Get token from args, environment, or prompt for login
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HF_API_TOKEN")
    
    if not token:
        print("No HuggingFace token found. Attempting to login...")
        print("You can also set HF_TOKEN or HF_API_TOKEN environment variable")
        try:
            login()  # This will prompt for token or use cached credentials
            print("✓ Login successful")
        except Exception as e:
            print(f"✗ Login failed: {e}")
            print("\nTo set up authentication:")
            print("1. Get your token from: https://huggingface.co/settings/tokens")
            print("2. Run: huggingface-cli login")
            print("3. Or set environment variable: export HF_TOKEN=your_token_here")
            return
    
    # Upload
    if args.all_checkpoints:
        upload_all_checkpoints(
            output_dir=args.checkpoint,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            commit_message=args.commit_message,
        )
    else:
        upload_checkpoint(
            checkpoint_path=args.checkpoint,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            commit_message=args.commit_message,
        )
    
    print(f"\n✓ Done! Check your model at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

