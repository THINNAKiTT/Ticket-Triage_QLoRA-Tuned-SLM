| System | Schema-valid | Category acc | Urgency acc | Entity F1 (all) | Entity F1 (extractable) | Full exact (all) | Full exact (extractable) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, zero-shot | 41.2 | 7.4 | 19.6 | 11.4 | 38.7 | 0.0 | 0.0 |
| Base, 8-shot | 98.6 | 33.8 | 37.2 | 8.0 | 40.7 | 0.0 | 10.8 |
| TF-IDF + LogReg (no entities) | n/a | 77.7 | 63.5 | n/a | n/a | n/a | n/a |
| Fine-tuned (QLoRA) | 100.0 | 79.7 | 64.2 | 24.8 | 91.4 | 9.5 | 43.2 |
| Fine-tuned, merged (bf16) | 98.0 | 76.4 | 64.2 | 25.2 | 92.7 | 7.4 | 38.5 |
