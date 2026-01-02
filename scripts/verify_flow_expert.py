"""
Flow-matching Action Expert 검증 스크립트
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motion_vla.models.flow_action_expert import FlowActionExpert


def main():
    print("=" * 60)
    print("Flow-matching Action Expert 검증 테스트")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n사용 디바이스: {device}")
    
    # Hyperparameters
    batch_size = 4
    embed_dim = 768
    action_dim = 7
    chunk_size = 10
    
    print(f"\n설정:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Embedding dim: {embed_dim}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Chunk size: {chunk_size}")
    
    # Initialize Flow Expert
    print(f"\nFlow-matching Expert 초기화 중...")
    flow_expert = FlowActionExpert(
        embed_dim=embed_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden_dim=512,
        num_layers=3,
    )
    flow_expert = flow_expert.to(device)
    flow_expert.eval()
    
    print(f"✓ 초기화 성공")
    
    # Create dummy VL embeddings
    vl_embed = torch.randn(batch_size, embed_dim).to(device)
    print(f"\n더미 VL Embedding 생성: {vl_embed.shape}")
    
    # Test 1: Single action generation
    print(f"\n[Test 1] Single action 생성...")
    with torch.no_grad():
        actions = flow_expert(vl_embed, num_integration_steps=10)
    
    print(f"✓ Action 생성 성공")
    print(f"  - Output shape: {actions.shape}")
    print(f"  - Expected: ({batch_size}, {action_dim})")
    assert actions.shape == (batch_size, action_dim)
    
    # Test 2: Action chunk generation
    print(f"\n[Test 2] Action chunk 생성...")
    with torch.no_grad():
        action_chunk = flow_expert.generate_action_chunk(vl_embed)
    
    print(f"✓ Action chunk 생성 성공")
    print(f"  - Output shape: {action_chunk.shape}")
    print(f"  - Expected: ({batch_size}, {chunk_size}, {action_dim})")
    assert action_chunk.shape == (batch_size, chunk_size, action_dim)
    
    # Test 3: Flow-matching loss computation
    print(f"\n[Test 3] Flow-matching Loss 계산...")
    target_actions = torch.randn(batch_size, action_dim).to(device)
    
    flow_expert.train()
    loss = flow_expert.compute_flow_matching_loss(
        vl_embed, target_actions, num_samples=5
    )
    
    print(f"✓ Loss 계산 성공")
    print(f"  - Loss value: {loss.item():.6f}")
    assert loss.item() > 0  # Loss should be positive
    
    # Test 4: Full trajectory
    print(f"\n[Test 4] Full trajectory 생성...")
    flow_expert.eval()
    with torch.no_grad():
        trajectory = flow_expert(
            vl_embed, 
            num_integration_steps=20, 
            return_trajectory=True
        )
    
    print(f"✓ Trajectory 생성 성공")
    print(f"  - Trajectory shape: {trajectory.shape}")
    print(f"  - Expected: (20, {batch_size}, {action_dim})")
    
    # Show action statistics
    print(f"\nAction 통계 (마지막 타임스텝):")
    final_action = trajectory[-1]
    print(f"  - Mean: {final_action.mean().item():.4f}")
    print(f"  - Std: {final_action.std().item():.4f}")
    print(f"  - Min: {final_action.min().item():.4f}")
    print(f"  - Max: {final_action.max().item():.4f}")
    
    print(f"\n✅ 모든 테스트 통과!")
    
    return 0


if __name__ == "__main__":
    exit(main())
