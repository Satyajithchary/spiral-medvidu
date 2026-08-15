# GitHub page setup

Everything to paste into the GitHub interface after the first push.

## Repository name

```
spiral-medvidu
```

## About, the short description under the repository title

Pick one. GitHub truncates around 100 characters in search results, so the first
clause carries the meaning.

**Recommended.** Leads with the finding rather than the acronym.

```
Reported grounding accuracy on MedVidBench is bounded by output token budget, not perception. ECCV 2026 MedVidU (Oral).
```

**Alternative, method-first.**

```
SPIRAL: structured supervision harvesting and self-refining inference for surgical video understanding. ECCV 2026 MedVidU (Oral).
```

**Website field.** Point at the OpenReview forum, not the Springer page.

```
https://openreview.net/forum?id=Ude69Oicir
```

## Topics

Add through the gear icon beside About. GitHub allows twenty; these fifteen
cover the searches that matter.

```
medical-video-understanding
surgical-video-analysis
vision-language-model
video-language-model
qwen3-vl
multimodal-llm
temporal-grounding
spatiotemporal-grounding
surgical-skill-assessment
osats
test-time-compute
benchmark-evaluation
eccv2026
lora
medical-ai
```

## Settings to enable

- **Issues.** Others will hit the same format problems documented in the README.
- **Discussions.** Optional, useful if the benchmark findings attract questions.
- **Releases.** Tag `v1.0-eccv2026` at the camera-ready commit so the paper
  points at a fixed state.
- Disable Wiki and Projects unless they will be used, since empty tabs look
  unfinished.

## First release

```bash
git tag -a v1.0-eccv2026 -m "ECCV 2026 MedVidU camera-ready"
git push origin v1.0-eccv2026
```

Release notes:

```
Camera-ready release accompanying the ECCV 2026 MedVidU Workshop paper.

Contains the full pipeline (data preparation, supervision harvesting, LoRA
adaptation, self-refining inference, submission builder) and every number
reported in the paper as JSON under results/.

Two benchmark findings are reproducible from this tag:
  - output token budget bounds reported spatiotemporal grounding accuracy by a
    factor of 7.35, while per-box quality remains constant
  - prediction indexing must carry question identity, since 3,975 identifiers
    cover 6,245 test rows
```

## Before making the repository public

- [ ] Replace `<user>` in the README clone command and `CHANGE-ME` in
      `CITATION.cff` with the real path.
- [ ] Insert the repository URL into the paper's release statement, then
      recompile the camera-ready.
- [ ] Confirm `configs/paths.yaml` is untracked. It contains local absolute
      paths and is listed in `.gitignore`.
- [ ] Confirm no frames, checkpoints or prediction files were committed:
      `git ls-files | grep -Ei '\.(jpg|png|safetensors|bin|pt)$'`
      should return only the four files under `assets/`.
- [ ] Do not commit the Springer PDF. Link the OpenReview version instead.

## Suggested pinned issue

Titled *Known benchmark pitfalls*, linking the four notes at the end of the
README. Anyone starting on MedVidBench meets the same four problems, and a
pinned issue reaches them sooner than a README section.
