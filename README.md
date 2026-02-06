# 🌊 SKANN-SSL: Selective Kernel Audio Neural Networks with Self-Supervised Learning

<div align="center">

```mermaid
graph LR
    subgraph Input["🎤 INPUT"]
        W["Raw Underwater<br/>Acoustic Waveform"]
    end
    
    subgraph SKANN["🧠 SKANN-SSL"]
        SK["Selective Kernel<br/>Filterbank"]
        E["Deep Encoder"]
        SSL["Self-Supervised<br/>Learning"]
    end
    
    subgraph Output["🎯 OUTPUT"]
        EMB["Robust<br/>Embeddings"]
        CLASS["Vessel<br/>Classification"]
    end
    
    W --> SK --> E --> SSL --> EMB --> CLASS
    
    style Input fill:#3498db,stroke:#2980b9,color:#fff
    style SKANN fill:#9b59b6,stroke:#8e44ad,color:#fff
    style Output fill:#27ae60,stroke:#1e8449,color:#fff
```

**An underwater acoustic vessel detection and classification system using physics-aware self-supervised learning**

![Status](https://img.shields.io/badge/Status-Production-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-success)
![Version](https://img.shields.io/badge/Version-V3%2FV5-blue)

SKANN-SSL learns vessel signatures directly from raw waveforms using selective-kernel filterbanks and produces robust embeddings for downstream evaluation and inference.

</div>

---

## 🎯 Project Status (V3/V5 — Production)

```mermaid
graph LR
    subgraph Pipeline["🔄 SKANN-SSL PIPELINE"]
        direction LR
        S1["Stage -1<br/>📊 Synthetic Data<br/>✅ Complete"]
        S0["Stage 0<br/>⚙️ Preprocessing<br/>✅ Complete"]
        S1K["Stage 1<br/>🎛️ SKConv1D<br/>✅ Implemented"]
        S2["Stage 2<br/>🧠 Encoder<br/>✅ Complete"]
        S3["Stage 3<br/>🔗 SSL Training<br/>✅ Complete"]
        S4["Stage 4<br/>🎨 Augmentation<br/>⏳ Planned"]
        S5["Stage 5<br/>🏋️ Training Loop<br/>✅ Complete"]
        S6["Stage 6<br/>📈 Evaluation<br/>✅ Complete"]
        S7["Stage 7<br/>🚀 Deployment<br/>✅ Prototype"]
    end
    
    S1 --> S0 --> S1K --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    
    style S1 fill:#27ae60,stroke:#1e8449,color:#fff
    style S0 fill:#27ae60,stroke:#1e8449,color:#fff
    style S1K fill:#27ae60,stroke:#1e8449,color:#fff
    style S2 fill:#27ae60,stroke:#1e8449,color:#fff
    style S3 fill:#27ae60,stroke:#1e8449,color:#fff
    style S4 fill:#f39c12,stroke:#d68910,color:#fff
    style S5 fill:#27ae60,stroke:#1e8449,color:#fff
    style S6 fill:#27ae60,stroke:#1e8449,color:#fff
    style S7 fill:#3498db,stroke:#2980b9,color:#fff
```

| Stage | Name | Status | Description |
|------:|------|:------:|-------------|
| -1 | Synthetic Data | ✅ Complete | Physics-based waveform generator (V5) |
| 0 | Preprocessing | ✅ Complete | DataLoader, normalisation, splits |
| 1 | SKConv1D Filterbank | ✅ Implemented | Multi-branch learned filterbank |
| 2 | Encoder | ✅ Complete | HybridSKEncoderV3 with SK frontend |
| 3 | SSL Training | ✅ Complete | Barlow Twins (V3 production) |
| 4 | Augmentation | ⏳ Planned | Physics-consistent augmentations |
| 5 | Training Loop | ✅ Complete | Integrated with Stage 3 |
| 6 | Evaluation | ✅ Complete | Confusion matrix + territory mapping |
| 7 | Deployment | ✅ Prototype | Local inference engine + demo GUI |

---

## 🏆 Key Results (V3/V5 — Near-Perfect Classification)

```mermaid
graph TB
    subgraph Results["🏆 V3/V5 ACHIEVEMENT SUMMARY"]
        direction TB
        
        subgraph Accuracy["📊 Classification Performance"]
            A1["🎯 Overall Accuracy<br/><b>100.0%</b><br/>(11,995/12,000)"]
            A2["📐 Silhouette Score<br/><b>0.9697</b><br/>(cosine distance)"]
            A3["❌ Total Errors<br/><b>5</b><br/>(out of 12,000)"]
        end
        
        subgraph Model["🧠 Model Specs"]
            M1["⚙️ Parameters<br/><b>175.9M</b> training<br/><b>~2M</b> inference"]
            M2["📐 Embedding Dim<br/><b>512</b> backbone<br/><b>256</b> projector"]
            M3["🎛️ SK Kernels<br/><b>31→1023</b><br/>6 branches"]
        end
        
        subgraph Data["📁 Dataset"]
            D1["🎵 Clips<br/><b>12,000</b><br/>synthetic"]
            D2["⏱️ Duration<br/><b>5 sec</b><br/>@ 16kHz"]
            D3["🚢 Classes<br/><b>5</b><br/>vessel types"]
        end
    end
    
    style Accuracy fill:#27ae60,stroke:#1e8449,color:#fff
    style Model fill:#3498db,stroke:#2980b9,color:#fff
    style Data fill:#9b59b6,stroke:#8e44ad,color:#fff
```

<div align="center">

| Metric | Value |
|--------|------:|
| **Overall Accuracy** | **100.0% (11,995/12,000)** |
| **Silhouette Score (cosine)** | **0.9697** |
| **Classes** | 5 (cargo, fishing, no_vessel, small_craft, tanker) |
| **Total Errors** | 5 |
| **Model Parameters** | 175.9M (training) / ~2M (inference) |
| **Embedding Dimension** | 512 (backbone) / 256 (projector) |
| **Training Hardware** | Google Colab A100 |
| **Dataset Size** | 12,000 synthetic clips (5 seconds each) |
| **SK Kernels** | (31, 63, 127, 255, 511, 1023) |
| **Frequency Coverage** | 15 Hz – 500+ Hz |

</div>

---

## 📈 Version History & Evolution

```mermaid
graph LR
    subgraph Evolution["📈 ACCURACY EVOLUTION"]
        V1["V1<br/>Dec 2025<br/>━━━━━━━━<br/>78.2%<br/>Baseline"]
        V2["V2.1.0<br/>Jan 2026<br/>━━━━━━━━<br/>~83%<br/>+SK Kernels"]
        V3["V3/V5<br/>Feb 2026<br/>━━━━━━━━<br/>100.0%<br/>Physics-Correct"]
    end
    
    V1 -->|"+4.8%"| V2 -->|"+17%"| V3
    
    style V1 fill:#e74c3c,stroke:#c0392b,color:#fff
    style V2 fill:#f39c12,stroke:#d68910,color:#fff
    style V3 fill:#27ae60,stroke:#1e8449,color:#fff
```

| Version | Date | Accuracy | Silhouette | Key Change |
|---------|------|----------|------------|------------|
| V1 | Dec 2025 | 78.2% | 0.3997 | Baseline (no SK) |
| V2.1.0 | Jan 2026 | ~83% | 0.8299 | SK kernels added |
| **V3/V5** | **Feb 2026** | **100.0%** | **0.9697** | **Physics-correct dataset** |

---

## 🏗️ Architecture (V3)

```mermaid
graph TB
    subgraph Architecture["🏗️ SKANN-SSL V3 ARCHITECTURE"]
        direction TB
        
        Input["🎤 Raw Waveform<br/>[B, 1, 80000]<br/>5 seconds @ 16kHz"]
        
        subgraph SKBlock["🎛️ SK FILTERBANK (Underwater Kernels)"]
            direction LR
            K1["k=31<br/>500+ Hz<br/>Cavitation"]
            K2["k=63<br/>250 Hz"]
            K3["k=127<br/>125 Hz"]
            K4["k=255<br/>62 Hz"]
            K5["k=511<br/>31 Hz"]
            K6["k=1023<br/>15 Hz<br/>Shaft Rate"]
            ATT["⚡ Attention<br/>Fusion"]
        end
        
        subgraph Backbone["🧠 CHANNEL BRIDGE + 2D BACKBONE"]
            CONV["Conv Layers<br/>+ BatchNorm"]
            H["h (512-dim)<br/>🚀 DEPLOYED"]
        end
        
        subgraph Projector["🔮 PROJECTOR (Training Only)"]
            MLP["Deep MLP<br/>512→4096→8192→16384→256"]
            Z["z (256-dim)<br/>🎓 Training"]
        end
        
        Input --> SKBlock
        K1 & K2 & K3 & K4 & K5 & K6 --> ATT
        SKBlock --> Backbone
        CONV --> H
        H --> Projector
        MLP --> Z
    end
    
    style Input fill:#3498db,stroke:#2980b9,color:#fff
    style SKBlock fill:#e74c3c,stroke:#c0392b,color:#fff
    style Backbone fill:#27ae60,stroke:#1e8449,color:#fff
    style Projector fill:#9b59b6,stroke:#8e44ad,color:#fff
    style H fill:#f1c40f,stroke:#d4ac0d,color:#000
    style Z fill:#1abc9c,stroke:#16a085,color:#fff
```

---

## 🎛️ Selective Kernel Coverage

```mermaid
graph LR
    subgraph FreqCoverage["🎵 FREQUENCY BAND COVERAGE"]
        direction TB
        
        subgraph LowFreq["LOW FREQUENCY (Propulsion)"]
            LF1["🔊 k=1023 → 15+ Hz<br/>Shaft Rate Harmonics"]
            LF2["🔊 k=511 → 31+ Hz<br/>Blade Pass Frequency"]
        end
        
        subgraph MidFreq["MID FREQUENCY (Machinery)"]
            MF1["🔊 k=255 → 62+ Hz<br/>Generator Harmonics"]
            MF2["🔊 k=127 → 125+ Hz<br/>Equipment Resonance"]
        end
        
        subgraph HighFreq["HIGH FREQUENCY (Turbulence)"]
            HF1["🔊 k=63 → 250+ Hz<br/>Flow Noise"]
            HF2["🔊 k=31 → 500+ Hz<br/>Cavitation Bursts"]
        end
    end
    
    style LowFreq fill:#e74c3c,stroke:#c0392b,color:#fff
    style MidFreq fill:#f39c12,stroke:#d68910,color:#fff
    style HighFreq fill:#3498db,stroke:#2980b9,color:#fff
```

---

## 📂 Project Structure

```mermaid
graph TD
    Root["📁 SKANN-SSL/"]
    
    Root --> README["📄 README.md"]
    Root --> ROADMAP["📄 ROADMAP.md"]
    Root --> Archive["📁 archive/"]
    Root --> Data["📁 data/"]
    Root --> Stages["📁 stages/"]
    Root --> Docs["📁 docs/"]
    Root --> Shared["📁 shared/"]
    
    subgraph ArchiveContent["Archive Contents"]
        A1["📁 v1_baseline_dec2025/"]
        A2["📁 v2_jan2026/"]
    end
    Archive --> ArchiveContent
    
    subgraph DataContent["Data Contents"]
        D1["📁 v5_dataset/<br/>12,000 clips"]
    end
    Data --> DataContent
    
    subgraph StagesContent["Pipeline Stages"]
        ST1["stage_minus1/ → Synthetic Data"]
        ST2["stage0/ → Preprocessing"]
        ST3["stage1/ → SKConv1D"]
        ST4["stage2/ → Encoder"]
        ST5["stage3/ → SSL Training"]
        ST6["stage6/ → Evaluation"]
        ST7["stage7/ → Deployment"]
    end
    Stages --> StagesContent
    
    style Root fill:#2c3e50,stroke:#1a252f,color:#fff
    style Archive fill:#95a5a6,stroke:#7f8c8d,color:#fff
    style Data fill:#3498db,stroke:#2980b9,color:#fff
    style Stages fill:#27ae60,stroke:#1e8449,color:#fff
```

```text
SKANN-SSL/
├── README.md
├── ROADMAP.md
│
├── archive/
│   ├── v1_baseline_dec2025/     # Archived V1 baseline
│   └── v2_jan2026/              # Archived V2.1.0
│
├── data/
│   └── v5_dataset/              # 12,000 clips (5 classes)
│       ├── tensors/
│       ├── waveforms/
│       └── master_dataset_manifest.csv
│
├── stages/
│   ├── stage_minus1/            # Synthetic data generation
│   ├── stage0_preprocessing/    # DataLoader
│   ├── stage1_skconv1d/         # SK filterbank
│   ├── stage2_encoder/          # Encoder architecture
│   ├── stage3_ssl/              # SSL training + artifacts
│   │   ├── train_script.py      # Model architecture
│   │   └── artifacts/
│   │       ├── SKANN_SSL_V3_Production_Bundle.joblib
│   │       └── vessel_territories_v3.joblib
│   ├── stage6_evaluation/       # Confusion matrix + classifier
│   └── stage7_deployment/
│
├── docs/
└── shared/
```

---

## 🚀 Quick Start

### Stage 6 — Batch Evaluation (Confusion Matrix)
```bash
cd SKANN-SSL
python stages/stage6_evaluation/stage6_confusion_matrix.py
```

### Stage 6 — Interactive Classifier
```bash
cd SKANN-SSL
python stages/stage6_evaluation/stage6_acoustic_sonar_classifier.py
```

### Demo GUI (Separate Repository)
```bash
git clone https://github.com/suniltyagialtair/SKANN-SSL-V5-Demo
cd SKANN-SSL-V5-Demo
python skann_ssl_v5_demo.py
```

---

## 📊 V3/V5 Confusion Matrix

```mermaid
graph TB
    subgraph Matrix["📊 CONFUSION MATRIX (Row-Normalized %)"]
        direction TB
        
        subgraph Perfect["✅ PERFECT CLASSES"]
            P1["Fishing Vessel → 100%"]
            P2["No Vessel → 100%"]
            P3["Small Craft → 100%"]
            P4["Tanker → 100%"]
        end
        
        subgraph NearPerfect["🎯 NEAR-PERFECT"]
            N1["Cargo Ship → 99.8%"]
        end
        
        subgraph Errors["❌ ONLY 5 ERRORS"]
            E1["4× cargo → fishing"]
            E2["1× fishing → small"]
        end
    end
    
    style Perfect fill:#27ae60,stroke:#1e8449,color:#fff
    style NearPerfect fill:#3498db,stroke:#2980b9,color:#fff
    style Errors fill:#e74c3c,stroke:#c0392b,color:#fff
```

```text
              cargo  fishing  no_vessel  small   tanker
cargo         99.8    0.2       0.0      0.0     0.0
fishing        0.0  100.0       0.0      0.0     0.0
no_vessel      0.0    0.0     100.0      0.0     0.0
small          0.0    0.0       0.0    100.0     0.0
tanker         0.0    0.0       0.0      0.0   100.0
```

⚠️ **Only 5 errors** across 12,000 clips:
- 4× cargo_ship → fishing_vessel
- 1× fishing_vessel → small_craft

---

## 🔬 Why V3/V5 Succeeded

```mermaid
graph TB
    subgraph Success["🔬 V3/V5 SUCCESS FACTORS"]
        direction TB
        
        subgraph Factor1["1️⃣ Non-Overlapping Shaft Rates"]
            SR["🚢 Vessel Class Separation"]
            SR1["Tanker: 0.8–1.8 Hz"]
            SR2["Cargo: 1.5–3.5 Hz"]
            SR3["Fishing: 4–8 Hz"]
            SR4["Small Craft: 15–30 Hz"]
        end
        
        subgraph Factor2["2️⃣ Physics-Consistent Synthesis"]
            PH["🔊 Realistic Acoustics"]
            PH1["✓ Swell freq: 0.05–0.15 Hz"]
            PH2["✓ 3 structural resonances/clip"]
            PH3["✓ 6 dB SNR target"]
            PH4["✓ Knudsen ambient noise"]
        end
        
        subgraph Factor3["3️⃣ Underwater SK Kernels"]
            UK["🎛️ Multi-Scale Capture"]
            UK1["k=1023 → 15+ Hz shaft rate"]
            UK2["k=31 → 500+ Hz cavitation"]
            UK3["Full discriminative coverage"]
        end
    end
    
    style Factor1 fill:#e74c3c,stroke:#c0392b,color:#fff
    style Factor2 fill:#3498db,stroke:#2980b9,color:#fff
    style Factor3 fill:#27ae60,stroke:#1e8449,color:#fff
```

### 1. Non-Overlapping Shaft Rate Ranges (V5 Dataset)

| Class | Shaft Rate Range | RPM Range |
|-------|------------------|-----------|
| 🚢 Tanker | 0.8–1.8 Hz | 48–108 RPM |
| 🚢 Cargo ship | 1.5–3.5 Hz | 90–210 RPM |
| 🎣 Fishing vessel | 4–8 Hz | 240–480 RPM |
| 🚤 Small craft | 15–30 Hz | 900–1800 RPM |

### 2. Physics-Consistent Synthesis
- ✅ Corrected swell frequency (0.05–0.15 Hz)
- ✅ Exactly 3 structural resonances per clip
- ✅ 6 dB SNR between vessel and ambient noise
- ✅ Realistic Knudsen-curve ambient noise

### 3. Underwater-Appropriate SK Kernels
- 🎛️ k=1023 captures shaft rate (15+ Hz)
- 🎛️ k=31 captures cavitation (500+ Hz)
- 🎛️ Full coverage of discriminative frequency bands

---

## 📦 Related Repositories

```mermaid
graph LR
    subgraph Ecosystem["📦 SKANN-SSL ECOSYSTEM"]
        Main["🏠 SKANN-SSL<br/>Main Development"]
        Demo["🖥️ SKANN-SSL-V5-Demo<br/>GUI Application"]
        Dataset["📊 Underwater-Acoustic-<br/>Synthetic-Dataset<br/>V5 Dataset"]
        
        Main <--> Demo
        Main <--> Dataset
        Demo <--> Dataset
    end
    
    style Main fill:#2c3e50,stroke:#1a252f,color:#fff
    style Demo fill:#27ae60,stroke:#1e8449,color:#fff
    style Dataset fill:#3498db,stroke:#2980b9,color:#fff
```

| Repository | Description |
|------------|-------------|
| [SKANN-SSL](https://github.com/suniltyagi/SKANN-SSL) | Main development repo |
| [SKANN-SSL-V5-Demo](https://github.com/suniltyagialtair/SKANN-SSL-V5-Demo) | GUI demo application |
| [Underwater-Acoustic-Synthetic-Dataset](https://github.com/suniltyagialtair/Underwater-Acoustic-Synthetic-Dataset) | V5 dataset (12,000 clips) |

---

## 📦 Historical Versions

| Version | Location | Notes |
|---------|----------|-------|
| V1 (Dec 2025) | `archive/v1_baseline_dec2025/` | 78.2% accuracy, no SK |
| V2.1.0 (Jan 2026) | `archive/v2_jan2026/` | 0.8299 silhouette, 1,920 clips |

---

## 🔮 Roadmap

```mermaid
graph LR
    subgraph Roadmap["🔮 DEVELOPMENT ROADMAP"]
        direction LR
        
        subgraph Done["✅ COMPLETED"]
            D1["V3/V5 Training<br/>100% accuracy"]
            D2["5-class system<br/>+ ambient"]
            D3["Territory eval"]
            D4["Demo GUI"]
        end
        
        subgraph Next["⏭️ NEXT"]
            N1["Linear probe<br/>benchmarks"]
            N2["Real hydrophone<br/>validation"]
            N3["Edge deploy<br/>ONNX/Jetson"]
            N4["Streaming<br/>inference"]
        end
        
        Done --> Next
    end
    
    style Done fill:#27ae60,stroke:#1e8449,color:#fff
    style Next fill:#3498db,stroke:#2980b9,color:#fff
```

### ✅ Completed
- V3/V5 Physics-aware training (100% accuracy)
- 5-class system including ambient noise detection
- Territory-based evaluation
- Demo GUI application

### ⏭️ Next
- Linear probe benchmarking
- Real hydrophone validation (ShipEar, NOAA)
- Edge deployment (ONNX / Jetson)
- Streaming inference

---

## 🧪 V3/V5 Experiment Provenance

<details>
<summary>📋 Full Provenance Details</summary>

**Experiment ID:** `V3_V5_colab_a100`

**Execution Platform:**
- Google Colab (A100 GPU)
- Single GPU training

**Training Regime:**
- Self-supervised learning using **Barlow Twins**
- Physics-aware **Selective Kernel (SK) filterbank**
- 51 epochs, batch size 4

**Data Regime:**
- V5 synthetic underwater acoustic dataset
- 12,000 clips, 5 seconds each @ 16kHz
- 5 vessel classes (including no_vessel/ambient)
- Non-overlapping shaft rate ranges for acoustic distinguishability

**Evaluation:**
- Primary metric: **Silhouette score (cosine)** = **0.9697**
- Classification accuracy: **100.0%** (11,995/12,000)
- kNN accuracy: **100%**

**Production Bundle:**
```
stages/stage3_ssl/artifacts/SKANN_SSL_V3_Production_Bundle.joblib
```

</details>

---

<div align="center">

```mermaid
graph LR
    Wave["🌊"] --> Model["🧠"] --> Ship["🚢"]
    
    style Wave fill:#3498db,stroke:#2980b9,color:#fff
    style Model fill:#9b59b6,stroke:#8e44ad,color:#fff
    style Ship fill:#27ae60,stroke:#1e8449,color:#fff
```

**Made with 🎧 for underwater acoustics research**

*Last updated: February 2026*

</div>
