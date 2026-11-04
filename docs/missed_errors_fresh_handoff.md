# Fresh ranking test: confirmed coordination and input handoff

**Coordination complete — confirmed by Bjørn via Otto on 2026-11-04.** No direct group message was sent by
the assistant. On 2026-11-03 Otto relayed Bjørn's instruction to use Frederik's
`g3_pool.zip` and pool config; maps are now local. Source metadata and current gaps are recorded in
`results/kits_missed_fresh/handoff_sources.json`. Otto approved the download and all 100 CTs/labels are now
local. `results/kits_missed_fresh/handoff_inputs.csv` contains their paths and the four required model maps;
byte/presence verification passed. On 2026-11-04 the two checkpoint files supplied by Frederik were
verified byte-identical to the locked references. Bjørn subsequently confirmed original pool-model
provenance, no previous joint ranking test and review of the exact locked check. Header/runtime preflight passed.
The scientific recipe is [locked separately](missed_errors_fresh_protocol.md); input paths are logistics.

## Confirmed source handoff

The November 3 follow-up audit verifies that the model, uncertainty and plausibility settings in
`configs/g3_pool.yaml` equal kits100's settings, all 600 required input paths exist, and the 20 scientific
fingerprints are unchanged. This checks the intended setup; it does not establish which weights produced
the shared maps. Latest upstream preparation metadata still contains no checkpoint hashes.

Before execution, Otto relayed Bjørn's explicit confirmation of all three facts on 2026-11-04:

- Bjørn has seen and approved the exact frozen `g1_missing_hull_mean` check.
- Frederik's pool archive uses the original 3 mm/6 mm models via `configs/g3_pool.yaml`, not fine-tuned G3 models.
- The joint ranking test had not yet run on the pool.

`results/kits_missed_fresh/coordination_confirmation.json` records the project-lead attestation and
its receipt via Otto. Original message time was not supplied, and no direct Frederik reply is claimed.
The handoff fields preserve these distinctions rather than inventing dates or communications.

The supplied files passed SHA-256 comparison on 2026-11-04:

| Supplied model | Local path | Result |
|---|---|---|
| 3 mm | `models/g3_pool_source_3mm/3mm_checkpoint_final.pth` | Byte-identical to locked task 297 reference |
| 6 mm | `models/g3_pool_source_6mm/6mm_checkpoint_final.pth` | Byte-identical to locked task 298 reference |

`results/kits_missed_fresh/source_checkpoint_verification.json` contains file sizes, hashes, timings and
machine details. The supplied-file identity and Bjørn's separate pool-provenance confirmation resolve
the model handoff. No more checkpoint upload or hash calculation is currently needed.

The earlier coordination request is resolved. Preserve the recorded source evidence and verification
alongside the one joint outcome stage.

For reference, these PowerShell commands reproduce the hash check from the original repository root
(adjust the model root if installed elsewhere). The current comparison is already complete:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "models\totalsegmentator\nnunet\results\Dataset297_TotalSegmentator_total_3mm_1559subj\nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth"
Get-FileHash -Algorithm SHA256 -LiteralPath "models\totalsegmentator\nnunet\results\Dataset298_TotalSegmentator_total_6mm_1559subj\nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth"
```

Locked reference hashes, now also verified on the supplied files:

| Original model | SHA-256 |
|---|---|
| 3 mm / task 297 / fold 0 | `1e38e40356adc2706a662e365405a97f862d62a7da65f6bf81d025aee1b979ac` |
| 6 mm / task 298 / fold 0 | `3489323a4b759506ab46b489eb78b72142f2143aea6e6d828604d754bd7a9270` |

The completed private handoff is at `data/handoffs/kits_missed_fresh.pending.json` (ignored by Git; the
filename is retained from preparation).
Its checkpoint-hash field now contains the computed hashes of the supplied files, with verification
evidence linked. Coordination fields now contain Bjørn's confirmation via Otto with explicit source/date
qualifications. The CT/model path manifest and source metadata are linked. Preflight was run with:

```bash
uv run python scripts/fresh_missed_errors.py --config configs/kits_missed_fresh.yaml --stage preflight --handoff data/handoffs/kits_missed_fresh.pending.json
```

Preflight passed all 100 cases / 600 files: canonical CT/GT shape/affine, model-map grids, probability
representation and input hashes. Its **ready: true** result is in `preflight.json` with `input_inventory.csv`.
It pins the private handoff, input manifest and source metadata; preserve them through scoring/evaluation.

## Execution status

Runtime projection (`runtime_projection.json`), with safety factor 2: **25.19 min** scoring, **4.92 min**
evaluation. Both were below the per-stage 30-minute cap. The **joint test is complete**: all 100 GT-free
scores were frozen before GT (713.23 s scoring), followed by one outcome stage (168.11 s) for both questions.
Neither candidate meets the primary improvement criterion. See `results/kits_missed_fresh/fresh_summary.md`
and `fresh_verification.json` for all results, secondary findings and the aggregate-only audit.

## Input manifest

`results/kits_missed_fresh/handoff_inputs.csv` already lists all 100 cases with existing paths. No one needs
to enter per-case scores or inspect labels. Its columns are:

```text
case_id,ct,label,mask,prob,m2_mask,m2_prob
```

Paths may be absolute or relative to the CSV's location. All model maps must already be on the canonical
CT grid. Probabilities use the project's uint8/1/255 encoding. Keep arrays and model weights outside Git.
If Frederik's outputs use another format, document/adapt the logistics before fresh outcome access.

`results/kits_missed_fresh/handoff_template.json` contains expected checkpoint hashes and pending fields
for actual coordination evidence and the input/inference-metadata locations. Copy/fill it outside Git.
Its reference hashes alone do not confirm provenance or notification. Do not fill dates/confirmations
unless the communication happened. The runner refuses fresh inputs while these facts are pending.

Use the saved tables for the report. The retained `evaluation_started.json` records the one outcome
opening and guards against a repeated fresh evaluation. Step 6 figure/summary and speaking notes are
ready in `results/kits_missed_fresh/report/`, generated from saved aggregates. Otto's group presentation remains.
