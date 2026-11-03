# Fresh ranking test: coordination draft and input handoff

**Draft only — not sent by the assistant.** On 2026-11-03 Otto relayed Bjørn's instruction to use Frederik's
`g3_pool.zip` and pool config; maps are now local. Source metadata and current gaps are recorded in
`results/kits_missed_fresh/handoff_sources.json`. The 100 CTs/labels are absent; download approval is pending.
The scientific recipe is [locked separately](missed_errors_fresh_protocol.md); input paths are logistics.

Suggested message:

> I have frozen one missed-errors candidate: mean suspicion ranks of locked G1, missing-side score and
> convex-hull deficit. Its development result is weak (+0.030 review-AUC, CI includes zero; no top-20
> count gain and worse quality at 20). I will report these limitations.
>
> Please keep its test together with the already locked `far_disagree_per_volume` comparison, so the G3
> pool's ranking outcomes are opened once. The pool is the fixed next 100 seed-42 cases after kits100;
> `results/kits_missed_fresh/cohort.csv` lists the exact IDs. The two prior G3 smoke-test cases remain included.
>
> Frederik: where are the original 3 mm and 6 mm masks/probability maps, CTs/labels and inference metadata?
> Please confirm the baseline models were used (not fine-tuned weights), that these are the same pool IDs,
> and that no ranking metrics for these comparisons have already been evaluated. No new inference is needed.
>
> Bjørn: the exact protocol/config and checkpoint hashes are on draft PR #2. Let us coordinate the single
> joint evaluation by November 18. I have not opened fresh model output or GT.

## Files to provide

The assistant can build the manifest from the shared folder paths when they are known. No one needs to
manually enter per-case scores or inspect labels. Expected columns:

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
