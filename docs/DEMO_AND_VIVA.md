# WatermarkX — Demo Script, Viva Notes and Limitations

## 1. Five-minute demo flow

Before the demo: activate the venv, run `streamlit run app.py`, and press
**Generate** once so the model is already loaded (it takes ~30 s the first time).

1. **Generate** — prompt *"Explain how vaccines work in simple words."*, choose **Both**.
   Point out: the two texts read equally naturally, but only the watermarked one
   has a high z-score and a green-token fraction well above 0.5.
2. **Verify** — paste the *normal* text: not detected. Paste the *watermarked*
   text: detected. Read out the caption: *statistical evidence, not proof of authorship*.
3. **Robustness** — the watermarked text is already filled in.
   - Run **Synonym** (30%): z drops but usually stays above 3 → watermark survives.
   - Run **Paraphrase**: z falls close to 0 → watermark removed, meaning similarity still high.
4. **Analytics** — show the dataset tables and charts: detection rate on clean
   text, false-positive rate on normal text, and detection after each attack.
5. **History** — every run above is saved; download the CSV.

## 2. Final results (20 prompts, CPU run)

| Measure | Result |
|---|---|
| Normal texts wrongly flagged (false positive rate) | 0 of 20 (0%) |
| Watermarked texts detected (true positive rate) | 12 of 20 (60%) |
| Mean z: watermarked / normal | 3.45 / 0.11 |

| Attack (on the 12 detected texts) | Still detected | Mean z after (4.06 before) | Meaning similarity |
|---|---|---|---|
| Paraphrase | 0 of 12 | 0.57 | 0.89 |
| Synonym (30%) | 7 of 12 | 3.52 | 0.94 |
| Translation EN-DE-EN | 6 of 12 | 3.05 | 0.97 |
| Combined | 1 of 12 | 0.54 | 0.84 |

**Bug found and fixed:** the first translation run hallucinated EU-regulation text
for bare list numbers ("5."), doubling 12 outputs and making translation look
stronger (25% still detected). After the fix the correct figure is 50%. The old
rows are kept in `experiments/archive/`.

## 3. Likely viva questions

**What is the KGW watermark?**
At each generation step, the previous token seeds a random split of the
vocabulary into a "green" half and a "red" half (`greenlist_ratio=0.5`). A bias
(`bias=2.0`) is added to green tokens' logits, so the model picks green tokens
more often than chance. The text still reads normally.

**How is it detected?**
Re-compute each token's green list from the token before it and count green
tokens. Without a watermark about 50% are green. The z-score measures how far
the observed count is above 50% in standard deviations:
`z = (green − 0.5·T) / sqrt(T · 0.5 · 0.5)`, where T is the number of scored tokens.
We call the text watermarked when z ≥ 3.0.

**Why z ≥ 3.0?**
For unwatermarked text, z > 3 happens by chance about 0.13% of the time
(one-sided), so false positives are rare. A higher threshold means fewer false
alarms but more missed watermarks.

**Why are some watermarked texts not detected?**
The watermark can only shift choices where the model is uncertain. When the
next word is nearly certain (low entropy), the +2.0 bias rarely changes it.
Instruction-tuned models at temperature 0.7, with Qwen's default `top_k=20` and
`repetition_penalty`, produce fairly predictable text, so some samples get
z-scores just below 3. Shorter texts also have less statistical evidence.
We report these misses rather than hiding them.

**Does the attacker need the secret key?**
No. All attacks only change the text. The paraphrasing model generates with
watermarking switched off; synonym replacement and translation don't know the
green lists at all.

**Why does paraphrasing work best?**
Detection depends on *pairs* of consecutive tokens (each token's green list
depends on the one before). Paraphrasing changes most tokens and their order,
so most pairs are new and their green status is back to 50/50 chance.
Synonym replacement changes about 30% of eligible words, so most token pairs
survive and the watermark usually remains.

**How do you know the attacked text still means the same thing?**
Sentence-embedding cosine similarity (all-MiniLM-L6-v2): about 0.8 or more
means close meaning, about 0 means unrelated. An attack that removed the
watermark but produced nonsense would show a low similarity.

**What does "not comparable" mean?**
If the original watermarked text was not detected (z < 3), an attack cannot
"remove" a watermark that wasn't detected. Those samples are excluded from
attack success rates instead of being counted as successes.

## 4. Limitations (be upfront about these)

- **Small dataset.** 20 prompts, one sample each. Rates have wide uncertainty;
  for example, 12/20 = 60%, but the true rate could plausibly be 39–78%.
- **Every text hit the 200-token limit** (199 tokens scored), so length effects were not studied.
- **One model, one watermark setting.** Results are for Qwen2.5-1.5B with
  KGW (ratio 0.5, bias 2.0, lefthash). Other models or a higher bias would differ.
- **The attacker uses the same model as the generator** (Qwen, watermark off)
  for paraphrasing. A different paraphrasing model may behave differently.
- **Synonym replacement is crude.** WordNet sometimes picks rare or awkward
  words ("unwellness"), and only base-form words are replaced.
- **Translation uses only German** in the dataset run (French and Spanish are supported).
- **Similarity is approximate.** Embedding similarity measures topic and meaning
  overlap, not factual accuracy.
- **CPU-only runs** were slow (~2.5 min per text) because the laptop GPU was
  unavailable during the experiments; the results are unaffected, only the time.
- **Detection is statistical evidence, not proof.** A non-detection does not
  mean a human wrote the text.
