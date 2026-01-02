"""
Vision-Language Encoder
PaliGemma 또는 OpenVLA 기반 멀티모달 인코더
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from typing import Optional, Union, Dict
import logging

logger = logging.getLogger(__name__)


class VisionLanguageEncoder(nn.Module):
    """
    Vision-Language Encoder using pre-trained VLM (PaliGemma or OpenVLA)
    
    Args:
        model_name: HuggingFace 모델 이름 (예: "google/paligemma-3b-pt-224")
        freeze_backbone: VLM backbone을 freeze할지 여부
        use_lora: LoRA를 사용한 효율적 fine-tuning 여부
        lora_r: LoRA rank (default: 8)
    """
    
    def __init__(
        self,
        model_name: str = "google/paligemma-3b-pt-224",
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        
        logger.info(f"Loading Vision-Language model: {model_name}")
        
        # Load pre-trained VLM
        try:
            self.vlm = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Freeze backbone if specified
        if freeze_backbone:
            logger.info("Freezing VLM backbone parameters")
            for param in self.vlm.parameters():
                param.requires_grad = False
        
        # Apply LoRA if specified
        if use_lora:
            logger.info(f"Applying LoRA with rank={lora_r}")
            self._apply_lora(lora_r)
        
        # Determine embedding dimension
        if embed_dim is None:
            # Auto-detect from model config
            if hasattr(self.vlm.config, "hidden_size"):
                self.embed_dim = self.vlm.config.hidden_size
            elif hasattr(self.vlm.config, "d_model"):
                self.embed_dim = self.vlm.config.d_model
            else:
                # Default fallback
                self.embed_dim = 768
                logger.warning(f"Could not detect embed_dim, using default: {self.embed_dim}")
        else:
            self.embed_dim = embed_dim
        
        logger.info(f"VL Encoder initialized with embed_dim={self.embed_dim}")
    
    def _apply_lora(self, rank: int):
        """
        Apply LoRA (Low-Rank Adaptation) to the model
        """
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # Common attention modules
            )
            
            self.vlm = get_peft_model(self.vlm, lora_config)
            logger.info("LoRA applied successfully")
        except ImportError:
            logger.error("peft library not installed. Install with: pip install peft")
            raise
    
    def forward(
        self,
        images: torch.Tensor,
        text_inputs: Union[str, list[str], Dict[str, torch.Tensor]],
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through Vision-Language Encoder
        
        Args:
            images: (B, 3, H, W) RGB images
            text_inputs: Text instructions (string, list of strings, or tokenized dict)
            return_dict: Return dictionary with additional outputs
            
        Returns:
            embeddings: (B, embed_dim) multimodal embeddings
            or dict with 'embeddings' and other outputs if return_dict=True
        """
        batch_size = images.size(0)
        
        # Process inputs if they're raw strings
        if isinstance(text_inputs, (str, list)):
            # Processor handles both image and text
            processed = self.processor(
                text=text_inputs,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            # Move to same device as images
            processed = {k: v.to(images.device) for k, v in processed.items()}
        else:
            # Already processed inputs
            processed = text_inputs
            processed["pixel_values"] = images
        
        # Forward through VLM
        outputs = self.vlm(**processed, output_hidden_states=True)
        
        # Extract embedding
        # Different models have different output structures
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            # Use [CLS] token or mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :]  # (B, embed_dim)
        else:
            raise ValueError("Could not extract embeddings from model outputs")
        
        if return_dict:
            return {
                "embeddings": embeddings,
                "outputs": outputs,
            }
        
        return embeddings
    
    def get_embed_dim(self) -> int:
        """Return the embedding dimension"""
        return self.embed_dim
    
    def save_pretrained(self, save_directory: str):
        """Save model and processor"""
        logger.info(f"Saving VL Encoder to {save_directory}")
        self.vlm.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from directory"""
        logger.info(f"Loading VL Encoder from {load_directory}")
        encoder = cls(model_name=load_directory, **kwargs)
        return encoder
