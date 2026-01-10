"""Download π0 pre-trained models from HuggingFace.

This script downloads the π0-Bridge checkpoint which includes:
- PaliGemma-3B VLM (frozen)
- Action head (trainable)
- Pre-trained on Bridge Data V2 (includes Pushing skill)
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login
import torch


AVAILABLE_MODELS = {
    "pi0-bridge": {
        "repo_id": "juexzz/INTACT-pi0-finetune-bridge",
        "description": "π0 fine-tuned on Bridge Data V2 (Pushing, Pick&Place, etc.)",
        "size": "~6GB"
    },
    "pi0-base": {
        "repo_id": "lerobot/pi0",
        "description": "π0 base model (Open X-Embodiment + Physical Intelligence data)",
        "size": "~6GB"
    }
}


def download_model(model_name: str, 
                   output_dir: str = "./checkpoints",
                   token: str = None):
    """Download π0 model from HuggingFace.
    
    Args:
        model_name: Name of model ('pi0-bridge' or 'pi0-base')
        output_dir: Directory to save model
        token: HuggingFace token (optional, for gated models)
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_name]
    repo_id = model_info["repo_id"]
    
    print(f"Downloading {model_name}...")
    print(f"  Repository: {repo_id}")
    print(f"  Description: {model_info['description']}")
    print(f"  Size: {model_info['size']}")
    
    # Login if token provided
    if token:
        login(token=token)
    
    # Create output directory
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_path),
            resume_download=True,
        )
        
        print(f"\n✅ Download complete!")
        print(f"   Model saved to: {local_dir}")
        
        # Verify model files
        verify_model(local_dir)
        
        return local_dir
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. If model is gated, provide HuggingFace token:")
        print("   python download_pi0_model.py --model pi0-bridge --token YOUR_TOKEN")
        print("3. Get token from: https://huggingface.co/settings/tokens")
        raise


def verify_model(model_path: str):
    """Verify downloaded model structure."""
    model_path = Path(model_path)
    
    print("\nVerifying model files...")
    
    required_files = [
        "config.json",
        "model.safetensors",  # or pytorch_model.bin
    ]
    
    missing = []
    for file in required_files:
        file_path = model_path / file
        alt_file_path = model_path / "pytorch_model.bin"
        
        if not file_path.exists() and not alt_file_path.exists():
            missing.append(file)
        else:
            size_mb = file_path.stat().st_size / (1024**2) if file_path.exists() else 0
            print(f"  ✓ {file} ({size_mb:.1f} MB)")
    
    if missing:
        print(f"\n⚠️  Warning: Missing files: {missing}")
        print("   Model may be incomplete.")
    else:
        print("\n✅ All required files present!")


def test_model_loading(model_path: str):
    """Test loading the downloaded model."""
    print(f"\nTesting model loading from {model_path}...")
    
    try:
        from transformers import AutoModel, AutoConfig
        
        # Load config
        config = AutoConfig.from_pretrained(model_path)
        print(f"  ✓ Config loaded")
        print(f"    Model type: {config.model_type}")
        
        # Load model
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Save memory
            low_cpu_mem_usage=True
        )
        print(f"  ✓ Model loaded")
        print(f"    Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        # Check components
        print(f"\n  Model components:")
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters()) / 1e6
            print(f"    - {name}: {num_params:.1f}M params")
        
        print("\n✅ Model loads successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        print("\nThis might be OK if you don't have transformers installed yet.")
        print("Install with: pip install transformers torch")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download π0 pre-trained models for Dobot E6 deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pi0-bridge",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (for gated models)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test model loading after download"
    )
    
    args = parser.parse_args()
    
    # Download
    model_path = download_model(
        model_name=args.model,
        output_dir=args.output_dir,
        token=args.token
    )
    
    # Test loading
    if args.test:
        test_model_loading(model_path)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Collect Dobot E6 demonstrations (50-100 episodes)")
    print("2. Run fine-tuning: python finetune_pi0.py")
    print("3. Deploy on robot: python deploy_robot.py")
    print("="*60)


if __name__ == "__main__":
    main()
