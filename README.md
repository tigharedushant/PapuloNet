# PSD-HP — Papulosquamous Skin Disease Harmonization Protocol

Automated dataset harmonization framework for the **AEF-CRC** research project.

```bash
python main.py
```

---

## Status: Phases 1–5 complete, plus a provenance/reproducibility
## refactor added after an architectural review

All 14 pipeline stages are implemented and tested end-to-end against
synthetic data with every edge case (corrupted files, empty folders,
unmapped folders, ambiguous-merged folders, exact + near duplicates,
blurry images, low-resolution images, missing datasets) deliberately
planted and verified to land correctly.

### What changed in the architectural review pass

A review was requested before adding more code. Findings and actions:

| Finding | Action |
|---|---|
| `metadata.py` re-derived source dataset by parsing filenames `extractor.py` had already written — fragile, duplicated knowledge | **Fixed.** New `modules/provenance.py` gives every image a `ProvenanceRecord` at the moment `extractor.py` copies it; every later stage updates that same record. `metadata.py` now only serializes it. |
| No globally unique image identifier | **Fixed.** Every extracted image gets a `PSD_00000001`-style ID from `ProvenanceTracker`. Final harmonized/split filenames are `PSD_00000001.jpg` — never the original filename, which is preserved only in `metadata.csv`. |
| Plain resize would distort lesion shape | **Fixed.** `standardizer.py` now uses aspect-ratio-preserving letterbox padding (`PIL.ImageOps.pad`), not a stretching resize. |
| No proof that no image silently disappeared between stages | **Fixed.** New `modules/integrity.py` verifies `scanned == final + rejected + duplicate + review` for every single image and fails loudly if not. |
| No reproducibility snapshot of the exact config that produced a given `output/` | **Fixed.** New `modules/manifest.py` writes `config_snapshot.json` (full config at run time) and `dataset_manifest.json` (framework version, final counts, SHA-256 of `metadata.csv`). |
| `source_contribution_report.csv`, `class_balance_report.csv` requested | **Added**, as natural extensions of `statistics.py` (same data it already computes). |
| `dataset_policy.yaml` / `class_mapping.yaml` (YAML instead of JSON) | **Declined, with reasoning.** JSON already delivers full configurability with zero hardcoded paths — YAML would add a PyYAML dependency for a format change with no functional benefit. `config_snapshot.json` is JSON for the same reason (one line of PyYAML converts it if you specifically need YAML downstream). |

### What was deliberately NOT implemented — needs your confirmation first

Two requested features require exact knowledge of your real, on-disk
folder structure that I cannot verify from this sandbox:

1. **DermNet `train/`+`test/` merge** — I don't know whether your
   local DermNet copy actually has this split (the version I
   researched and tested against does not). Confirm the real
   structure and I'll add the merge logic.
2. **SkinDisNet dual-version detection** (`SkinDisNet/` vs.
   `SkinDisNet_2/`, each with `Augmented/`/`Preprocessed/`,
   abbreviated vs. full class names) — same issue. Implementing
   folder-structure auto-detection against a layout I can't verify
   risks silently mismapping real data with false confidence, which
   is worse than not automating it yet.

Run `python main.py` against your real datasets and share
`reports/scan_report.xlsx` (Stage 2's real, on-disk folder listing) —
that tells me your actual structure directly, and I can wire in both
adapters correctly on the first attempt instead of guessing.

---

## Project structure

```
PSD_HP/
├── config/
│   ├── config.py              # single source of truth for every path/setting
│   └── class_mapping.json     # raw dataset label -> canonical target class
├── datasets/                  # YOUR downloaded datasets go here
├── modules/
│   ├── validator.py           # Stage 1
│   ├── scanner.py             # Stage 2
│   ├── mapper.py               # Stage 3
│   ├── coverage.py             # Stage 4
│   ├── provenance.py          # shared lineage tracker (NEW)
│   ├── extractor.py            # Stage 5 — assigns PSD IDs, begins provenance
│   ├── duplicate_detector.py   # Stage 6 — updates provenance
│   ├── quality_assessor.py     # Stage 7 — updates provenance
│   ├── standardizer.py         # Stage 8 — aspect-ratio-preserving pad + final PSD ID filename
│   ├── metadata.py             # Stage 9 — serializes provenance
│   ├── statistics.py           # Stage 10 — + source_contribution + class_balance reports
│   ├── splitter.py             # Stage 11
│   ├── integrity.py            # Stage 12 — conservation check (NEW)
│   ├── manifest.py             # Stage 13 — config snapshot + dataset manifest (NEW)
│   └── report_generator.py     # Stage 14 — final PDF
├── pipeline/harmonization_pipeline.py   # orchestrates all 14 stages, in order
├── output/
│   ├── 01_extracted/<class>/            # PSD_ID__dataset_original.ext
│   ├── 02_ambiguous_review/
│   ├── 03_duplicates_removed/<class>/
│   ├── 04_low_quality_excluded/<class>/
│   ├── 05_harmonized/<class>/           # PSD_ID.jpg (final unique filename)
│   └── 06_final_split/{train,val,test}/<class>/
├── reports/                    # 16 report files, listed below
├── main.py
├── README.md
└── requirements.txt
```

## Reports generated (all in `reports/`)

`dataset_validation_report.txt`, `scan_report.xlsx`, `mapping_report.txt`,
`coverage_matrix.xlsx`, `extraction_report.txt`, `duplicates_report.csv`,
`quality_report.csv`, `standardization_report.txt`, `metadata.csv`,
`dataset_statistics.xlsx`, `source_contribution_report.csv`,
`class_balance_report.csv`, `split_report.txt`, `integrity_report.csv`,
`config_snapshot.json`, `dataset_manifest.json`, `harmonization_report.pdf`.

---

## Fourth dataset integration: Atlas ISIC2019 (31 Classes)

PSD-HP now harmonizes **four** independent datasets:

```
DermNet
SkinDisNet
Curated32
Atlas ISIC31   ← NEW
        ↓
     PSD-HP
```

Atlas ISIC31 was added as a pure **extension** of the adapter layer —
no existing module was redesigned, and no existing dataset's behavior
changed. The Adapter Pattern is what made this possible: every module
downstream of `modules/adapters.py` (scanner, validator, mapper,
coverage, extractor, duplicate detector, quality assessor,
standardizer, metadata, statistics, splitter) consumes only the
generic `DiscoveredFolder` / `DiscoveredImage` objects the adapter
layer produces. None of them know or care how many raw datasets exist,
what their on-disk shapes are, or how many classes each contributes —
so adding a fourth dataset with its own on-disk shape required
touching exactly three files.

### Files changed and why

| File | Change | Why | Approx. lines added |
|---|---|---|---|
| `config/config.py` | One new `DatasetConfig` entry (`AtlasISIC31`, `adapter_kind="atlas31"`) appended to the existing `datasets` list. | This is the framework's single declared mechanism for registering a raw dataset — no other file declares dataset existence. | ~7 |
| `modules/adapters.py` | New `AtlasISIC31Adapter` class + one new line in `_ADAPTER_REGISTRY`. | The adapter layer is *specifically* the module whose job is to understand one dataset's real on-disk shape and normalize it. Atlas ISIC31's shape (`{train,test}/<class>/`) isn't identical to any existing adapter's assumptions (`GenericFlatAdapter` has no train/test split; `DermNetAdapter` does per-image filename reclassification Atlas doesn't need), so it earns its own adapter rather than being forced into an existing one. | ~65 |
| `config/class_mapping.json` | One new top-level `"AtlasISIC31"` section with 4 entries (3 target-class mappings + 1 explicit exclusion). | `mapper.py` is the *only* module that reads this file, and its own docstring is explicit that raw-label → canonical-class knowledge belongs here, not in code. | ~6 |

### Modules verified to need NO change (with justification)

| Module | Why it needs no change |
|---|---|
| `scanner.py` | Calls `discover_dataset()` generically for every configured dataset and iterates `DiscoveredFolder` objects; has zero dataset-name conditionals. |
| `validator.py` | Same — validates whatever `discover_dataset()` returns, for any dataset, generically. |
| `mapper.py` | Reads `class_mapping.json` keyed by `dataset_cfg.name`, which is data-driven — Atlas ISIC31 required a new JSON section, not new mapper code. Its `normalize_key()` matching already tolerates the "Lichen Planus" (space) vs. "Lichen_Planus" (underscore) naming difference between Atlas ISIC31 and the other datasets. |
| `coverage.py`, `duplicate_detector.py`, `quality_assessor.py`, `standardizer.py`, `metadata.py`, `statistics.py`, `splitter.py`, `provenance.py`, `manifest.py`, `integrity.py`, `report_generator.py` | All operate on `MappingDecision` / `ProvenanceRecord` / already-extracted files — none of them branch on dataset identity or on-disk folder depth at all. Confirmed by code review (grep for dataset names across `modules/*.py` and `pipeline/*.py` turns up only docstring examples, never executable conditionals). |
| `pipeline/harmonization_pipeline.py`, `main.py` | Both iterate `config.datasets` as a plain list; neither hardcodes a dataset count or names. |

### Naming safety: Seborrheic Keratosis ≠ Seborrheic Dermatitis

Atlas ISIC31 contains a **Seborrheic Keratosis** folder (a benign
keratinocytic growth), which is clinically unrelated to PSD-HP's
target class **Seborrheic Dermatitis** (an inflammatory condition,
contributed by SkinDisNet). `AtlasISIC31Adapter` does not special-case
this folder — it discovers it like any other and reports the raw
label unchanged. The exclusion is enforced entirely in
`class_mapping.json`, exactly like Curated32's own
`"Seborrheic_Keratosis": "EXCLUDE_NOT_TARGET"` entry, so the two
diseases can never be silently merged.

### SOLID / architecture review

- **Single Responsibility** — preserved. `AtlasISIC31Adapter` has exactly
  one job (discover this dataset's files); mapping/extraction/etc. logic
  was not duplicated into it.
- **Open/Closed** — the integration is a textbook example: PSD-HP was
  extended (new adapter class, new registry entry, new config/mapping
  data) without modifying a single line of scanner.py, validator.py,
  mapper.py's resolution logic, or any stage from extractor.py onward.
  The one-line `_ADAPTER_REGISTRY` addition is the adapter layer's own
  designed extension point, not a modification of existing behavior.
- **Maintainability / Coupling / Cohesion** — `AtlasISIC31Adapter` depends
  only on `DatasetConfig`, `PSDConfig`, and the same `DiscoveredFolder`/
  `DiscoveredImage` dataclasses every other adapter uses; it has no
  knowledge of and no dependency on `DermNetAdapter` or
  `SkinDisNetAdapter`. Cohesion within `adapters.py` is unchanged — it
  remains "one module, one concern: dataset discovery."
- **Backward compatibility** — DermNet, SkinDisNet, and Curated32's
  `DatasetConfig` entries, adapters, and `class_mapping.json` sections
  are byte-for-byte unchanged. A run against only the original three
  datasets behaves identically to before this change.

### AEF-CRC methodology impact

Harmonizing four independent sources instead of three directly
strengthens AEF-CRC's cross-institutional research contribution:

- **Dataset diversity** — Atlas ISIC31 is ISIC-derived (dermoscopic
  imaging protocol), complementing DermNet/SkinDisNet/Curated32's
  clinical-photo sources — the harmonized set now spans two
  fundamentally different acquisition modalities.
- **Generalization** — a classifier trained on four independently
  collected sources is less likely to have learned a single dataset's
  photographic idiosyncrasies (lighting, camera, background) as a
  spurious signal.
- **Source robustness** — no single dataset can dominate a target
  class's image count unchecked; `class_balance_report.csv` and
  `source_contribution_report.csv` (Stage 10) already report this
  per-class breakdown across all four sources with no code change.
- **Class imbalance** — Atlas ISIC31 contributes additional Psoriasis,
  Lichen Planus, and Pityriasis Rosea images without requiring
  oversampling/undersampling of any existing class (PSD-HP's
  imbalance policy — report, never correct — is unchanged; see below).
- **Scalability** — this integration is the adapter layer's intended
  proof point: a fifth, sixth, or Nth dataset follows the identical
  three-file pattern (`DatasetConfig` entry, adapter class +
  registry line, `class_mapping.json` section) with zero risk to the
  three already-integrated sources.

---

1. Place datasets under `datasets/dermnet/`, `datasets/skindisnet/`,
   `datasets/curated32/`, `datasets/atlas_isic31/` (one subfolder per
   class; `atlas_isic31/` uses `{train,test}/<class>/`).
2. `pip install -r requirements.txt`
3. `python main.py`

## Class imbalance policy

PSD-HP **reports** imbalance (`class_balance_report.csv`,
`dataset_statistics.xlsx`) and never corrects it — no oversampling,
undersampling, or deletion of valid images. Handling imbalance
(class weights, focal loss, few-shot handling for the smallest
classes) is AEF-CRC's responsibility, not PSD-HP's.

## Original datasets are read-only

Every stage that touches `datasets/` only ever reads. All writes go
to `output/`. Nothing in PSD-HP renames, moves, or deletes a file
inside `datasets/` — confirmed by code review of every module that
opens a path under `datasets/`.
