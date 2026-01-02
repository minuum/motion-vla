"""
VL Encoder 단위 테스트
"""

import torch
import pytest
from motion_vla.models.vl_encoder import VisionLanguageEncoder


def test_vl_encoder_initialization():
    """VL Encoder 초기화 테스트"""
    # 실제 모델 다운로드 방지를 위해 mock 사용 가능
    # 여기서는 실제 테스트를 위한 스켈레톤만 제공
    pass


def test_vl_encoder_forward():
    """Forward pass 테스트"""
    # Mock input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    text = ["Pick up the cup", "Move to the left"]
    
    # 실제 테스트는 모델 로딩이 필요하므로 스킵
    # encoder = VisionLanguageEncoder(model_name="google/paligemma-3b-pt-224")
    # embeddings = encoder(images, text)
    # assert embeddings.shape == (batch_size, encoder.embed_dim)
    pass


def test_vl_encoder_with_lora():
    """LoRA 적용 테스트"""
    pass


if __name__ == "__main__":
    print("VL Encoder 테스트를 실행하려면 pytest를 사용하세요:")
    print("pytest tests/test_vl_encoder.py -v")
