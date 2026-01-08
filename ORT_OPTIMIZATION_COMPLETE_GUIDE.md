# ONNX Runtime Optimization & Quantization: Complete Guide

This comprehensive guide explains how to enable graph optimization and model quantization in ONNX Runtime (ORT) for faster inference, including when to apply each optimization and practical examples.

================================================================================
OPTIMIZATION TIMELINE
================================================================================

                    TOTAL TIME SAVED

NO OPTIMIZATION     │████ 100ms/inference
                    │ (baseline)

QUANTIZATION ONLY   │█ 25ms/inference
(already done)      │ 75% faster! (no quantization cost)

GRAPH OPT ONLY      │███ 70ms/inference
(per session)       │ 30% faster (startup: 1-3s)

BOTH COMBINED ✅    │ 25ms/inference
                    │ 75% faster total!
                    │ (quantization done, graph optimized)

================================================================================
COST ANALYSIS
================================================================================

Quantization at Model Creation:
  Frequency:   Once per model
  Cost:        200ms
  ROI:         Positive after 1st inference session
  Permanent:   Yes, 4x size reduction forever
  Status:      ✅ RECOMMENDED

Graph Optimization at Inference:
  Frequency:   Once per session
  Cost:        1-5s (one-time per session)
  ROI:         Positive for any workload
  Permanent:   Per session (resets with new session)
  Status:      ✅ RECOMMENDED

Quantization at Inference (WRONG):
  Frequency:   Every single inference session
  Cost:        200ms × number of sessions
  ROI:         Negative (repeated wasted computation)
  Permanent:   No (lost benefits when session ends)
  Status:      ❌ NOT RECOMMENDED


**Table of Contents**
- [Quick Start](#quick-start)
- [Graph Optimization](#graph-optimization-levels)
- [Model Quantization](#model-quantization)
- [Optimization Timing](#optimization-timing-where-and-when)
- [Usage Examples](#usage-examples)
- [Performance Impact](#performance-impact)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#optimization-best-practices)

---

## Quick Start

### 1. Graph Optimization Levels

```bash
# Default (best optimization)
python inference_ort.py --model model.onnx --image test.jpg

# Explicit optimization levels
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization all
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization extended
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization basic
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization disabled
```

### 2. Model Quantization

```bash
# Step 1: Quantize model (one-time)
python quantize_model.py --model model.onnx --output model_quantized.onnx

# Step 2: Use quantized model with optimization
python inference_ort.py --model model.onnx --image test.jpg \
  --use-quantization --quantized-model model_quantized.onnx \
  --graph-optimization all
```

### 3. Quick Reference: Recommended Configurations

```bash
# For maximum speed (GPU):
python inference_ort.py --model model.onnx --image test.jpg \
  --graph-optimization all \
  --use-quantization --quantized-model model_quantized.onnx

# For balanced performance (GPU):
python inference_ort.py --model model.onnx --image test.jpg \
  --graph-optimization extended \
  --use-quantization --quantized-model model_quantized.onnx

# For CPU inference:
python inference_ort.py --model model.onnx --image test.jpg \
  --no-tensorrt --graph-optimization all
```

---

## Graph Optimization Levels

ONNX Runtime provides 4 optimization levels that reorganize computation for faster execution:

### Level 1: Disabled
```bash
--graph-optimization disabled
```

**What it does**:
- No optimizations applied
- Use for debugging or baseline performance comparison
- Slowest inference

**When to use**:
- Performance comparison/benchmarking
- Debugging issues
- Validating baseline

**Speed improvement**: Baseline (1.0x)

---

### Level 2: Basic
```bash
--graph-optimization basic
```

**What it does**:
- Constant folding
- Operator fusion (simple cases)
- Identity node removal
- Redundant node elimination

**Example**:
```
Before:  Input → Identity → Conv → Output
After:   Input → Conv → Output
         (Identity node removed)
```

**When to use**:
- Conservative optimization needed
- Maximum compatibility required
- Models with simple architectures

**Speed improvement**: ~5-15% faster

---

### Level 3: Extended
```bash
--graph-optimization extended
```

**What it does**:
- All basic optimizations plus:
- Layout optimization (memory access patterns)
- More aggressive operator fusion
- Graph decomposition
- Memory layout optimization

**Example**:
```
Before:  Conv → BatchNorm → ReLU → Conv
After:   Conv+BatchNorm → ReLU+Conv
         (Layers fused for faster execution)
```

**When to use**:
- Balanced approach for most models
- Good compatibility with various hardware
- Default choice for most use cases

**Speed improvement**: ~10-30% faster

---

### Level 4: All (Default)
```bash
--graph-optimization all
```

**What it does**:
- All extended optimizations plus:
- Layout optimization across multiple passes
- Cost model-based optimizations
- Semantic operators
- Advanced node fusion
- Memory optimization

**When to use**:
- Production inference
- Maximum performance needed
- Running on dedicated hardware

**Speed improvement**: ~15-50% faster (depending on model)

---

## Model Quantization

### What is Quantization?

Quantization reduces model size and memory usage by converting floating-point weights to lower precision integer representations:

| Type | Bytes | Size | Speed | Accuracy Loss |
|------|-------|------|-------|---------------|
| **FP32** (Original) | 4 | 100% | 1.0x | 0% |
| **INT8** (Quantized) | 1 | ~25% | 2-4x | 0.1-1% |
| **FP16** (Half-precision) | 2 | ~50% | 1.5-2x | <0.1% |

### Benefits

| Metric | FP32 | INT8 | FP16 |
|--------|------|------|------|
| Model Size | 100% | ~25% | ~50% |
| Memory | 100% | ~25% | ~50% |
| Speed | 1x | 2-4x | 1.5-2x |
| Accuracy Loss | 0% | 0.1-1% | <0.1% |
| Deployment Size | 100MB | 25MB | 50MB |

### Quantization Methods

#### 1. Dynamic Quantization (Recommended for Quick Start)
No calibration dataset needed:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization (simpler, no calibration)
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized_dynamic.onnx",
    optimize_model=True,
    per_channel=False,
    weight_type=QuantType.QInt8,
)
```

**Pros**:
- No calibration data needed
- Fast (just quantizes weights)
- Good accuracy (usually <0.5% loss)

**Cons**:
- Activations not quantized
- Slightly larger than static

#### 2. Static Quantization (Better Accuracy)
Requires calibration dataset:

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

# Static quantization with calibration
quantize_static(
    model_name_or_path="model.onnx",
    model_output_path="model_quantized_static.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QOperator,
    per_channel=False,
    weight_type=QuantType.QInt8,
)
```

**Pros**:
- Activations also quantized
- Smaller model (full quantization)
- Best accuracy preservation

**Cons**:
- Requires calibration data
- More complex setup
- Slower quantization process

### Quantize Your Model

Use the provided `quantize_model.py` script:

```bash
# Basic quantization
python quantize_model.py --model model.onnx --output model_quantized.onnx

# Per-channel quantization (more accurate)
python quantize_model.py --model model.onnx --output model_quantized.onnx --per-channel

# With optimization during quantization
python quantize_model.py --model model.onnx --output model_quantized.onnx --optimize
```

**Script details**:
- Uses dynamic quantization (no calibration needed)
- Reports compression ratio
- Validates output file
- Shows size comparison

**Expected output**:
```
📊 ONNX Model Quantization (Dynamic)
   Input: model.onnx
   Output: model_quantized.onnx
   Per-channel: False
   Weight type: QuantType.QInt8
   Optimize: True

⏱️  Quantizing model...
✅ Quantization complete!

📈 Size Comparison:
   Original:    100.50 MB
   Quantized:    25.10 MB
   Compression:  75.0%
   Reduction:    75.40 MB

🎯 Benefits:
   Memory reduction: 75.0%
   Typical speedup: 2-3x faster inference
   Typical accuracy: <0.1% loss (INT8)
```

---

## Optimization Timing: Where and When

### ⚡ Key Decision

**Question**: Should optimizations be applied during model creation or inference session?

**Answer**:
- **Quantization**: Apply at model creation ✅ (permanent artifact)
- **Graph Optimization**: Apply at inference session ✅ (runtime config)
- **Never**: Quantize repeatedly at inference time ❌ (wasteful)

### Why Quantization at Model Creation?

Quantization **permanently modifies model weights**:

```
FP32 Model (100MB)        Quantized Model (25MB)
┌────────────────┐        ┌────────────────┐
│ Layer 1        │        │ Layer 1        │
│ weights: float │   →    │ weights: int8  │ ← Permanently changed
│ 4 bytes each   │        │ 1 byte each    │
│ Layer 2        │        │ Layer 2        │
│ ...            │        │ ...            │
└────────────────┘        └────────────────┘
```

**Advantages**:
- ✅ One-time cost (200ms)
- ✅ Permanent 4x size reduction
- ✅ Permanent 2-3x speed improvement
- ✅ Easy to maintain (just use quantized file)
- ✅ Efficient deployment (ship small file)

**Example**:
- Quantize once: 200ms
- 1000 inference sessions benefit: saves 1000s of computation

### Why Graph Optimization at Inference Session?

Graph optimization **reorganizes computation at runtime**:

```
Original Graph            Optimized Graph
┌──────────┐              ┌──────────────────┐
│ Conv     │              │ Conv+BatchNorm   │ ← Fused operators
├──────────┤              ├──────────────────┤
│ BatchNorm│        →     │ ReLU+Conv        │ ← Different arrangement
├──────────┤              ├──────────────────┤
│ ReLU     │              │ ...              │
├──────────┤              └──────────────────┘
│ Conv     │
└──────────┘
```

**Advantages**:
- ✅ No model file changes
- ✅ Hardware-specific (TensorRT vs CUDA vs CPU)
- ✅ Can adjust per use case
- ✅ Flexible and reversible
- ✅ Fresh optimization for current hardware

**Per-session cost**: 1-5 seconds (acceptable)

### Cost-Benefit Analysis

| Operation | Frequency | Cost | Benefit | ROI | Status |
|---|---|---|---|---|---|
| **Quantize at model** | Once | 200ms | 4x smaller forever | ✅ Excellent | **✅ DO THIS** |
| **Quantize at session** | Every session | 200ms × N | Lost each session | ❌ Negative | **❌ DON'T DO** |
| **Optimize at session** | Once per session | 1-5s | 30% faster | ✅ Positive | **✅ DO THIS** |
| **Optimize at export** | Once (baked) | Baked | Fixed optimization | ❌ Inflexible | **❌ DON'T DO** |

#### ✅ CORRECT: Quantize Once, Optimize at Sessions

```
Training → Export Model.onnx (100MB)
           │
           └─ Quantize Once (200ms) ✅
              └─ Save Model_Quantized.onnx (25MB) ← Permanent!
              
           Inference Session 1
           ├─ Load Model_Quantized.onnx (FAST, 25MB)
           ├─ Graph Optimize (1-5s)
           └─ Run Inference
           
           Inference Session 2
           ├─ Load Model_Quantized.onnx (FAST, 25MB)
           ├─ Graph Optimize (1-5s)
           └─ Run Inference

✅ One-time quantization cost
✅ Permanent 4x size reduction
✅ Permanent 2-3x speed improvement
```

## Usage Examples

### Example 1: Single Image with All Optimizations

```bash
python inference_ort.py \
  --model ./moco_resnet50.onnx \
  --image ./test_image.jpg \
  --graph-optimization all \
  --use-quantization \
  --quantized-model ./moco_resnet50_quantized.onnx
```

### Example 2: Batch Inference with Extended Optimization

```bash
python inference_ort.py \
  --model ./moco_resnet50.onnx \
  --image-dir ./test_images/ \
  --graph-optimization extended
```

### Example 3: Compare Optimization Levels

```bash
# Baseline (no optimization)
echo "=== Baseline ===" 
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization disabled

# Basic optimization
echo "=== Basic ===" 
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization basic

# Extended optimization
echo "=== Extended ===" 
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization extended

# All optimizations
echo "=== All ===" 
python inference_ort.py --model model.onnx --image test.jpg --graph-optimization all
```

### Example 4: Performance Comparison

```bash
# Original + No optimization (baseline)
echo "=== Baseline ===" 
python inference_ort.py --model model.onnx --image-dir ./images/ --graph-optimization disabled

# Optimized
echo "=== With Optimization ==="
python inference_ort.py --model model.onnx --image-dir ./images/ --graph-optimization all

# Quantized + Optimized
echo "=== Quantized + Optimization ==="
python inference_ort.py --model model.onnx --image-dir ./images/ \
  --graph-optimization all --use-quantization --quantized-model model_quantized.onnx
```

### Example 5: Complete Workflow

```bash
# Step 1: Export model to ONNX
python export_to_ort.py --model ./checkpoints/best.pth
# Output: model.onnx (100MB, FP32)

# Step 2: Quantize
python quantize_model.py --model model.onnx --output model_quantized.onnx
# Output: model_quantized.onnx (25MB, INT8)
# Result: 75% size reduction!

# Step 3: Validate accuracy
python inference_ort.py --model model.onnx --image test.jpg
python inference_ort.py --model model_quantized.onnx --image test.jpg
# Compare outputs to ensure acceptable accuracy

# Step 4: Benchmark performance
echo "Original model:"
python inference_ort.py --model model.onnx --image-dir ./data/ --graph-optimization all

echo "Quantized model:"
python inference_ort.py --model model.onnx --image-dir ./data/ \
  --use-quantization --quantized-model model_quantized.onnx --graph-optimization all

# Step 5: Deploy
cp model_quantized.onnx /production/models/
```

### Example 6: Production Inference Code

```python
import onnxruntime as ort
import numpy as np

# Create session once (reuse for multiple inferences)
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model_quantized.onnx",
    sess_options=session_options,
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Process multiple images (reuse same session)
for image_batch in image_batches:
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: image_batch})
    # Process results...
```

---

## Performance Impact

### Typical Improvements on NVIDIA GPUs

| Configuration | Speed | Memory | Model Size | Time/Image |
|---|---|---|---|---|
| No optimization | 1.0x | 1.0x | 1.0x | 100ms |
| Graph optimization (all) | 1.2-1.5x | 0.95x | 1.0x | 67-83ms |
| Quantization | 1.5-2.5x | 0.25x | 0.25x | 40-67ms |
| Both | 2.0-3.5x | 0.25x | 0.25x | 28-50ms |

### Speed Breakdown

```
100% ┤                                    
     ├────────────────────────────────────
 90% ├─ Original Model (1.0x)
     │  • No quantization
     │  • No graph optimization
 80% ├────────────────────────────────────
     │
 70% ├─ Graph Optimized Only (1.3x)
 60% │  • No quantization
 50% │  • Graph optimization: +30%
 40% ├────────────────────────────────────
     │
 20% ├─ Quantized Only (2.5x)
 10% │  • Quantization: +150%
     │  • No graph optimization
  5% ├────────────────────────────────────
     │
  0% └─ Both Optimizations (3.5x) ✅
       • Quantization: +150%
       • Graph optimization: +30%
       • TOTAL: 3.5x faster!
```

---

## Advanced Configuration

### Custom Session Options

```python
import onnxruntime as ort

# Create custom session options
session_options = ort.SessionOptions()

# Graph optimization
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Execution mode
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # or ORT_PARALLEL

# Memory optimization
session_options.enable_mem_pattern = True  # Use memory pattern optimization

# Profiling (uncomment to enable)
# session_options.enable_profiling = True
# session_options.profile_file_prefix = "profile_results"

# Threading (for CPU)
session_options.inter_op_num_threads = 8  # Inter-operation parallelism
session_options.intra_op_num_threads = 8  # Intra-operation parallelism

# Logging
session_options.log_severity_level = 2  # 0=verbose, 1=info, 2=warning, 3=error

# Create session with options
session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

### Enable Profiling

To analyze which optimizations are being applied:

```python
session_options.enable_profiling = True
session_options.profile_file_prefix = "profile"

session = ort.InferenceSession("model.onnx", sess_options=session_options)

# Run inference
results = session.run(None, inputs)

# Check profiling results
# Profile files: profile_*.json
```

Then analyze with:
```bash
ls -la profile*.json
# Review to see which operations were fused and optimized
```

---

## Troubleshooting

### Issue: Graph Optimization Not Applied

**Problem**: Optimization level is set but no speed improvement

**Solutions**:
1. Check if model supports optimizations
2. Try extended or all level
3. Enable profiling to see which optimizations were applied

```python
session_options.enable_profiling = True
```

4. Verify provider is correct:
```python
print(session.get_providers())  # Check which provider is active
```

---

### Issue: Accuracy Drop After Quantization

**Problem**: Quantized model produces significantly different results

**Solutions**:
1. Use static quantization with calibration dataset (more accurate than dynamic)
2. Use per-channel quantization (preserves more accuracy):
   ```bash
   python quantize_model.py --model model.onnx --output model_quantized.onnx --per-channel
   ```
3. Validate with representative data before deployment
4. Test accuracy threshold with test dataset

---

### Issue: Model Not Loading with Quantization

**Problem**: "Invalid model" or similar error when loading quantized model

**Solutions**:
1. Verify quantized model was created successfully:
   ```bash
   python -c "import onnx; onnx.checker.check_model('model_quantized.onnx')"
   ```
2. Check ONNX Runtime version compatibility:
   ```python
   import onnxruntime as ort
   print(ort.__version__)
   ```
3. Regenerate quantized model with latest ORT version
4. Ensure original model is valid before quantizing

---

### Issue: Slow Inference Despite Optimization

**Problem**: Still slow even with all optimizations

**Solutions**:
1. Check execution provider (should be TensorRT > CUDA > CPU):
   ```python
   print(session.get_providers())
   ```

2. Ensure batch size is appropriate for your hardware

3. Profile to identify bottlenecks:
   ```bash
   python inference_ort.py --model model.onnx --image test.jpg \
     --graph-optimization all 2>&1 | grep -i "time\|ms"
   ```

4. Check for CPU fallback (if TensorRT/CUDA not available)

5. Monitor GPU memory:
   ```bash
   nvidia-smi  # Check GPU utilization
   ```

---

### Issue: High Memory Usage with Quantization

**Problem**: Memory usage doesn't decrease after quantization

**Solutions**:
1. Ensure quantized model is loaded (not original):
   ```bash
   ls -lh model.onnx model_quantized.onnx
   ```

2. Restart Python process to clear cached models

3. Check session is using quantized model:
   ```bash
   python -c "import os; print(f'Quantized: {os.path.getsize(\"model_quantized.onnx\") / (1024*1024):.1f} MB')"
   ```

---

## Optimization Best Practices

1. **Start with graph optimization (free)** 
   - Always use `--graph-optimization all` (default)
   - No model modification, all upside

2. **Profile before quantizing** 
   - Ensure graph optimization alone provides benefits
   - Validate accuracy impact

3. **Quantize if size/speed critical** 
   - Use dynamic quantization for quick start
   - Use static with calibration for maximum accuracy

4. **Validate accuracy** 
   - Test quantized models on representative data
   - Ensure accuracy loss is acceptable (<1%)

5. **Use TensorRT when available** 
   - Best performance on NVIDIA GPUs
   - Significant speedups (2-3x beyond basic optimization)

6. **Monitor memory usage** 
   - Quantization saves both disk size and memory
   - Critical for edge devices or large models

7. **Test on target hardware** 
   - Optimizations vary by device
   - Profile on deployment hardware

8. **Reuse sessions** 
   - Create session once, reuse for multiple inferences
   - Avoids re-optimization overhead

9. **Keep original model as backup** 
   - Maintain original for comparison
   - Useful for validation and debugging

10. **Document decisions** 
    - Note which optimization level was chosen
    - Record quantization parameters used
    - Track accuracy validation results
