"""
VL Encoder 검증 스크립트
실제 PaliGemma 모델을 다운로드하여 forward pass를 테스트합니다.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motion_vla.models.vl_encoder import VisionLanguageEncoder


def main():
    print("=" * 60)
    print("VL Encoder 검증 테스트")
    print("=" * 60)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n사용 디바이스: {device}")
    
    # Model name - PaliGemma는 크기가 크므로 로컬 환경에서 주의
    # 실제 사용 시 HF_TOKEN이 필요할 수 있음
    model_name = "google/paligemma-3b-pt-224"
    
    print(f"\n모델 로딩 중: {model_name}")
    print("(첫 실행 시 다운로드에 시간이 걸릴 수 있습니다...)")
    
    try:
        # Initialize encoder
        encoder = VisionLanguageEncoder(
            model_name=model_name,
            freeze_backbone=True,
            use_lora=False,
        )
        encoder = encoder.to(device)
        encoder.eval()
        
        print(f"✓ 모델 로딩 성공")
        print(f"  - Embedding dimension: {encoder.get_embed_dim()}")
        
        # Create dummy inputs
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        text_inputs = [
            "Pick up the red cup carefully",
            "Move to the left quickly"
        ]
        
        print(f"\n더미 입력 생성:")
        print(f"  - Images shape: {images.shape}")
        print(f"  - Text inputs: {text_inputs}")
        
        # Forward pass
        print(f"\nForward pass 실행 중...")
        with torch.no_grad():
            embeddings = encoder(images, text_inputs)
        
        print(f"✓ Forward pass 성공")
        print(f"  - Output embeddings shape: {embeddings.shape}")
        print(f"  - Expected shape: ({batch_size}, {encoder.get_embed_dim()})")
        
        # Verify shape
        assert embeddings.shape == (batch_size, encoder.get_embed_dim()), \
            f"Shape mismatch! Got {embeddings.shape}"
        
        print(f"\n✅ 모든 검증 통과!")
        
        # Show sample embedding statistics
        print(f"\n임베딩 통계:")
        print(f"  - Mean: {embeddings.mean().item():.4f}")
        print(f"  - Std: {embeddings.std().item():.4f}")
        print(f"  - Min: {embeddings.min().item():.4f}")
        print(f"  - Max: {embeddings.max().item():.4f}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
