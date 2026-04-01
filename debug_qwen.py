#!/usr/bin/env python3
"""Debug script to check Qwen2 attention hook capture"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test with Qwen2-0.5B
model_name = "Qwen/Qwen2-0.5B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

print(f"Model type: {model.__class__.__name__}")
print(f"Device: {model.device}")

# Get config
config = model.config
print(f"\nConfig:")
print(f"  num_layers: {config.num_hidden_layers}")
print(f"  num_heads: {config.num_attention_heads}")
print(f"  num_kv_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")
print(f"  hidden_size: {config.hidden_size}")

# Find attention modules
print(f"\nAttention modules:")
for name, module in model.named_modules():
    if 'attn' in name.lower() or 'attention' in name.lower():
        print(f"  {name}: {module.__class__.__name__}")

# Test hook capture
print(f"\n--- Testing hook capture ---")

class DebugExtractor:
    def __init__(self):
        self.captures = []

    def hook_fn(self, module, input, output):
        print(f"Hook fired! Output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"  Tuple length: {len(output)}")
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    print(f"  [{i}]: Tensor shape {item.shape}")
                elif isinstance(item, tuple):
                    print(f"  [{i}]: Tuple len {len(item)}")
                    for j, subitem in enumerate(item):
                        if isinstance(subitem, torch.Tensor):
                            print(f"    [{j}]: Tensor shape {subitem.shape}")
                else:
                    print(f"  [{i}]: {type(item)}")
        self.captures.append(output)

extractor = DebugExtractor()

# Register hooks on first attention layer
attn_modules = [(n, m) for n, m in model.named_modules() if 'attn' in n.lower()]
print(f"Found {len(attn_modules)} attention modules")

if attn_modules:
    first_attn_name, first_attn_module = attn_modules[0]
    print(f"Registering hook on: {first_attn_name}")
    handle = first_attn_module.register_forward_hook(extractor.hook_fn)

    # Run a simple forward pass
    input_ids = tokenizer.encode("Hello", return_tensors='pt').to(model.device)
    print(f"\nRunning forward pass with input_ids shape: {input_ids.shape}")

    with torch.no_grad():
        output = model(input_ids, use_cache=True)

    print(f"\nOutput type: {type(output)}")
    if hasattr(output, 'past_key_values'):
        print(f"Has past_key_values: {type(output.past_key_values)}")
        if output.past_key_values:
            print(f"past_key_values length: {len(output.past_key_values)}")
            for i, kv in enumerate(output.past_key_values):
                print(f"  Layer {i}: k={kv[0].shape}, v={kv[1].shape}")

    handle.remove()

    print(f"\nCaptures: {len(extractor.captures)}")
