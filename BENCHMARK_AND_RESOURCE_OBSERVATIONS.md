# System Resource & AI Inference Benchmark Observations

**Device Platform:** ASUS TUF Gaming A15 (FA506NFR)  
**Host OS:** Arch Linux (Rolling Release)  
**Compute Architecture:** AMD Ryzen 7 7435HS (8 Cores, 16 Threads) + NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6)  
**Evaluation Date:** September 2026  
**Pipeline:** Edge AI CCTV - YOLO11 Multi-Model Video Stream Analysis Engine  

---

## 1. Executive Summary

This report documents the empirical CPU, GPU, VRAM, and thermal observations recorded during real-time surveillance video ingestion and inference across different pipeline operating states.

Dynamic AI feature toggles were introduced to eliminate unnecessary GPU/CPU workload, allowing the platform to dynamically scale compute from multi-model deep learning (Object Detection + 17-Keypoint Skeletal Pose + Biomechanical Fall Analytics) down to pure video pass-through streaming with zero inference overhead.

---

## 2. Empirical Benchmark Measurements

Measurements were sampled across multiple intervals under active camera ingestion:

| Operating State | Active AI Modules | Process CPU Utilization | NVIDIA GPU Load | VRAM Allocated | Power Draw | GPU Core Temp | System Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **State 1: Full AI Active** | YOLO11n Detect + YOLO11-Pose (17 Joints) + Kinematics | **750% – 820%**<br>*(~8 threads)* | **32% – 38%** | **1,559 MiB** / 4,096 MiB | 10.5 W – 12.3 W | **65°C** | Full surveillance pipeline tracking multi-class bounding boxes, limb articulation, and rapid hip velocity. |
| **State 2: Skeletal Tracking OFF** | YOLO11n Detect + Fall Aspect-Ratio only | **~596%**<br>*(~6 threads)* | **34% – 36%** | **1,544 MiB** / 4,096 MiB | 10.2 W | **65°C** | **Saves ~180% – 220% CPU compute** (~2 full physical cores freed) by bypassing 17-keypoint skeletal joint estimation. |
| **State 3: Zero-Compute Bypass** | Pure Pass-Through Streaming (Object Detection OFF) | **~33%**<br>*(single thread I/O)* | **30% – 34%** | **1,543 MiB** / 4,096 MiB | 9.8 W | **64°C** | **Massive ~96% reduction in CPU compute**. Camera feed streams at full framerate with near-zero compute consumption. |

---

## 3. Key Observations & System Insights

### A. 96% Compute Reduction via Dynamic Bypassing
* When **Multi-Class Object Detection** is switched OFF via the Web HUD or Mobile API, the ingestion loop immediately bypasses all DNN forward passes, letterboxing convolutions, and morphological calculations.
* CPU consumption immediately drops from **820% down to 33%**, keeping the system completely responsive and preventing thermal throttling on edge mini PCs (such as Intel N100 or AMD Ryzen units).

### B. Skeletal Pose Decoupling Efficiency
* The 17-keypoint skeletal pose model (`yolo11n-pose.onnx`) is computationally heavier per person than bounding box detection.
* Decoupling skeletal tracking via an independent runtime switch allows perimeter monitoring (intrusion polygons, directional tripwires, vehicles, packages) to run continuously at full framerate while conserving ~200% thread compute until human verification is needed.

### C. Thermal & Power Stability
* The NVIDIA GeForce RTX 2050 operated well within safe thermal boundaries throughout sustained execution:
  * **Temperature**: Stable at **64°C – 65°C** (thermal throttle threshold is 87°C).
  * **Power Consumption**: **9.8 W to 12.3 W**, indicating low electrical and cooling demands.
  * **Memory Footprint**: Memory remained tightly bounded at **~1.5 GB VRAM**, leaving ample headroom for additional concurrent camera feeds.

---

## 4. Hardware Deployment Recommendations

1. **For Intel N100 Mini PCs paired with Hailo-8 / 8L**:
   * The HailoRT PCIe accelerator offloads neural inference to dedicated NPU cores (26 TOPS), which shifts the primary compute load off the host CPU.
   * Host CPU utilization is projected at $<15\%$ under Hailo-8 execution, matching the pass-through profile observed in State 3.
2. **For CPU-Only Fallback Deployments**:
   * Default cameras to **State 2 (Detection Only)** during normal monitoring.
   * Dynamically promote a camera feed to **State 1 (Skeletal Tracking)** only when motion or a tripwire crossing is confirmed, maximizing efficiency across multi-camera installations.
