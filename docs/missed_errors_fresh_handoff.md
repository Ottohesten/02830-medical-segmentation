# Fresh ranking test: coordination draft and input handoff

**Draft only — not sent by the assistant.** On 2026-11-03 Otto relayed Bjørn's instruction to use Frederik's
`g3_pool.zip` and pool config; maps are now local. Source metadata and current gaps are recorded in
`results/kits_missed_fresh/handoff_sources.json`. Otto approved the download and all 100 CTs/labels are now
local. `results/kits_missed_fresh/handoff_inputs.csv` contains their paths and the four required model maps;
byte/presence verification passed. On 2026-11-04 the two checkpoint files supplied by Frederik were
verified byte-identical to the locked references. Pool-run linkage/coordination and header/runtime preflight remain pending.
The scientific recipe is [locked separately](missed_errors_fresh_protocol.md); input paths are logistics.

## Remaining confirmations before preflight

The November 3 follow-up audit verifies that the model, uncertainty and plausibility settings in
`configs/g3_pool.yaml` equal kits100's settings, all 600 required input paths exist, and the 20 scientific
fingerprints are unchanged. This checks the intended setup; it does not establish which weights produced
the shared maps. Latest upstream preparation metadata still contains no checkpoint hashes.

Three facts remain pending in the current `results/kits_missed_fresh/preflight.json`:

- Bjørn was informed of the exact frozen combination and its joint test.
- Frederik confirms the archive contains the original baseline predictions used for this pool.
- These two ranking comparisons have not already been evaluated on the pool.

The supplied files passed SHA-256 comparison on 2026-11-04:

| Supplied model | Local path | Result |
|---|---|---|
| 3 mm | `models/g3_pool_source_3mm/3mm_checkpoint_final.pth` | Byte-identical to locked task 297 reference |
| 6 mm | `models/g3_pool_source_6mm/6mm_checkpoint_final.pth` | Byte-identical to locked task 298 reference |

`results/kits_missed_fresh/source_checkpoint_verification.json` contains file sizes, hashes, timings and
machine details. This establishes the identity of the supplied files; confirmation that they produced
the shared pool maps is still needed. No more checkpoint upload or hash calculation is currently needed.

Otto can relay the short request below. The assistant has not sent it. Use actual reply dates/evidence;
do not replace missing confirmations with the config declaration or a branch push.

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

An explicitly pending handoff is prepared at `data/handoffs/kits_missed_fresh.pending.json` (ignored by Git).
Its checkpoint-hash field now contains the computed hashes of the supplied files, with verification
evidence linked. The three outstanding confirmation fields remain null. The CT/model path manifest and
source metadata are linked. Fill only the remaining facts supplied by the group, then run:

```bash
uv run python scripts/fresh_missed_errors.py --config configs/kits_missed_fresh.yaml --stage preflight --handoff data/handoffs/kits_missed_fresh.pending.json
```

While confirmations are missing, this must remain `ready: false` without opening fresh headers. Once
complete, preflight checks shapes/affines, probability representation and file hashes. Project full score
and evaluation runtimes before using arrays; ask Otto before any projected stage above 30 minutes.

Suggested message:

> I have frozen one missed-errors candidate: mean suspicion ranks of locked G1, missing-side score and
> convex-hull deficit. Its development result is weak (+0.030 review-AUC, CI includes zero; no top-20
> count gain and worse quality at 20). I will report these limitations.
>
> Please keep its test together with the already locked `far_disagree_per_volume` comparison, so the G3
> pool's ranking outcomes are opened once. The pool is the fixed next 100 seed-42 cases after kits100;
> `results/kits_missed_fresh/cohort.csv` lists the exact IDs. The two prior G3 smoke-test cases remain included.
>
> Frederik: the shared 3 mm/6 mm maps and downloaded CTs/labels are now local, with a 100-case path manifest.
> The two checkpoint files you supplied match our locked references. Please confirm they were used to
> generate the shared pool maps, that these are the same pool IDs,
> and that no ranking metrics for these comparisons have already been evaluated. No new inference is needed.
>
> Bjørn: the exact protocol/config and checkpoint hashes are on draft PR #2. Let us coordinate the single
> joint evaluation by November 18. I have not opened fresh model output or GT.

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

Once paths/coordination are available, follow README's preflight/score/evaluate stages. First check the
runtime projection; ask Otto before downloads or a projected stage above 30 minutes. The outcome stage
must contain both predeclared comparisons, with no new candidate selection.
