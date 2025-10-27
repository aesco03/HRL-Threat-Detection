# HRL-Threat-Detection
SeniorProject: Using HRL for detecting PDF and image malware obfuscation
___
```bash
pip install -r requirements.txt
```

```bash
python train_features_dqn.py
```

Edit `configs/feature_groups.yaml` to list column names from the labeled parquet by group (`meta_cols`, `pdf_cols`, `text_cols`, `image_cols`) and the `label_col` (e.g., `Class`).

```bash
python train_raw_dqn.py
```

For raw file training once `data/files/` and `data/labels.csv` are populated, aligning extractor outputs to the same schema used in the features dataset.

```bash
python train_features_dqn.py
```

Edit `configs/feature_groups.yaml` to list column names from the labeled parquet by group (`meta_cols`, `pdf_cols`, `text_cols`, `image_cols`) and the `label_col`.

```bash
python train_raw_dqn.py
```

For raw file training once `data/files/` and `data/labels.csv` are populated, aligning extractor outputs to the same schema used in the features dataset.
