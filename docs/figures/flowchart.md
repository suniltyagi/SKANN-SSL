flowchart TD
    A["Raw Audio (x)"] --> B["SKConv1D Filterbank"]
    B --> C["Learned Spectrogram"]
    C --> D["SKConv2D Encoder"]
    D --> E["Embedding h ∈ ℝᴰ"]

    E --> F["Augmentation 1"]
    E --> G["Augmentation 2"]

    F --> H["Siamese Encoder f(x)"]
    G --> H

    H --> I["Projector Head g(h)"]
    I --> J["Embeddings z₁, z₂"]
    J --> K["Barlow Twins Loss"]

    classDef box fill:#f2f7ff,stroke:#3366cc,stroke-width:2px,rx:12,ry:12;
    classDef proc fill:#e8fff2,stroke:#33aa55,stroke-width:2px,rx:12,ry:12;
    classDef loss fill:#fff2e6,stroke:#ff9933,stroke-width:2px,rx:12,ry:12;

    class A,C,E,J box
    class B,D,F,G,H,I proc
    class K loss