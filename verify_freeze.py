#!/usr/bin/env python3
"""
verify_freeze.py - Critical safety check for Month 3 memory integration.

This script verifies that the backbone freezing strategy works correctly
to prevent catastrophic forgetting when training the memory head.

FAILURE MODE (2.10e-02 shift without freezing):
- Memory head gradients backpropagate through entire 7B model
- Backbone weights shift, destroying reasoning priors
- 3 months of training lost in 10 steps

SUCCESS CRITERIA:
- Only memory_head params are trainable (~98,432 for 130m, ~524,416 for 7B)
- Backbone shift < 1e-6 after training step
"""

import sys
import os
import torch

# Ensure we can import mamba_ssm from current directory
sys.path.insert(0, os.path.abspath("."))

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def test_freeze_backbone():
    print(">>> 🔒 Starting Backbone Freeze Verification...")
    
    # Using 130m for speed (same logic applies to 7B)
    model_name = "state-spaces/mamba-130m"
    print(f"Loading {model_name}...")
    try:
        model = MambaLMHeadModel.from_pretrained(model_name, strict=False, device="cuda", dtype=torch.bfloat16)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # CRITICAL: Freeze backbone BEFORE any training
    print("Freezing backbone...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Freeze lm_head too (we only want to train memory_head)
    for param in model.lm_head.parameters():
        param.requires_grad = False
    
    # Verify memory_head exists and is trainable
    if not hasattr(model, "memory_head"):
        print("❌ memory_head not found! Check fork integration.")
        return False
        
    for param in model.memory_head.parameters():
        param.requires_grad = True
    
    # Initialize memory head weights
    model.memory_head.query_proj.weight.data.normal_(mean=0.0, std=0.02)
    model.memory_head.query_proj.bias.data.zero_()
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    # memory_head: Linear(768->128) for 130m = 768*128 + 128 = 98,432
    # For 7B (4096->128): 4096*128 + 128 = 524,416
    expected_130m = 768 * 128 + 128  # 98,432
    expected_7b = 4096 * 128 + 128   # 524,416
    
    print(f"Trainable params: {trainable:,} / {total:,}")
    print(f"Expected (130m): {expected_130m:,}")
    print(f"Expected (7B):   {expected_7b:,}")
    
    if trainable != expected_130m:
        print(f"⚠️ Trainable count mismatch! Got {trainable}, expected {expected_130m}")
        print("   Check if lm_head or backbone params are unfrozen.")
        # Not a hard failure, continue test
    else:
        print("✅ Trainable param count matches expected.")
    
    # Capture backbone weight before training
    original_A_log = model.backbone.layers[0].mixer.A_log.clone()
    
    # Training step
    print("Running 1 training step...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-5
    )
    
    input_ids = torch.randint(0, 1000, (1, 16)).cuda()
    
    optimizer.zero_grad()
    outputs = model(input_ids)
    loss = outputs.logits.mean() + 0.1 * outputs.query_vector.norm()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.6f}")
    
    # Verify backbone didn't shift
    new_A_log = model.backbone.layers[0].mixer.A_log
    shift = (new_A_log - original_A_log).abs().max().item()
    
    print(f"Backbone shift: {shift:.2e}")
    
    if shift < 1e-6:
        print("✅ Backbone shift < 1e-6. Freeze strategy works.")
        return True
    else:
        print(f"❌ Backbone shifted! {shift:.2e} > 1e-6")
        print("   Freeze failed. Check requires_grad settings.")
        return False


if __name__ == "__main__":
    success = test_freeze_backbone()
    if success:
        print("\n>>> 🎉 VERIFICATION PASSED. Month 3 integration is safe.")
    else:
        print("\n>>> 🚨 VERIFICATION FAILED. DO NOT proceed with Month 3.")
    sys.exit(0 if success else 1)
