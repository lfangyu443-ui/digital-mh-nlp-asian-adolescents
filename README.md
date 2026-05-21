# Digital Mental Health NLP Pipeline for Asian Adolescents
### A Clinically-Grounded, Multilingual Framework for Early Mental Health Risk Detection — Deployable Across APAC School and Community Settings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Languages: ZH | EN](https://img.shields.io/badge/Languages-ZH%20%7C%20EN-red.svg)]()
[![Status: Research](https://img.shields.io/badge/Status-Research-green.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/lfangyu443-ui/digital-mh-nlp-asian-adolescents/blob/main/notebooks/01_data_exploration.ipynb)

> **Author:** Lin Fang Yu 林芳伃 | MPH Candidate, National University of Singapore
> **Affiliation:** NUS Saw Swee Hock School of Public Health
> **Contact:** el649078@u.nus.edu | lfangyu443@gmail.com
> **Last Updated:** May 2026

---

## The Problem

Across the Asia-Pacific region, adolescent mental health is a growing crisis — yet existing digital tools systematically fail to detect risk in this population:

- Most NLP tools are trained on **English-language Reddit/Twitter data** from Western populations
- Asian adolescents express distress **indirectly** — through academic pressure framing, somatic complaints, and family obligation language — that Western-trained models miss entirely
- The same dynamics apply wherever **collectivist cultural values**, high-stakes academic systems, and mental health stigma intersect

| Country | Key Pressure | Help-Seeking Barrier |
|---|---|---|
| Singapore | PSLE, O/A-Levels | Stigma, face-saving |
| Taiwan | GSAT, university pressure | Family shame avoidance |
| South Korea | Suneung exam culture | 눈치 social awareness |
| Japan | University entrance exams | 空気を読む cultural norms |
| Hong Kong | DSE, academic competition | Face preservation |

This project builds a **culturally-adapted, multilingual NLP pipeline** that detects distress signals the way Asian adolescents actually express them — online, in Mandarin and English, indirectly.

---

## Project Objectives

1. **Design** a 3-tier risk detection framework grounded in clinical evidence and applicable across APAC contexts
2. **Demonstrate** NLP preprocessing and classification using publicly available multilingual datasets
3. **Build** interpretable risk scoring with SHAP explainability — clinician-reviewable in any setting
4. **Document** a governance framework applicable across different national data protection regulations
5. **Propose** a deployment framework adaptable to local service pathways

---

## Pipeline Architecture

**Tier 1 — Screening:** Chinese MentalBERT (Ji et al., 2024) → binary distress classification (Mandarin + English, F1 = 88.39% on Chinese suicide risk dataset). Low distress exits the pipeline; high distress proceeds to Tier 2.

**Tier 2 — Risk Scoring:** XGBoost + SHAP explainability → risk score 0–100 with interpretable feature weights across linguistic, temporal, social, and cultural context features.

**Tier 3 — Temporal Forecasting:** LSTM with 30-day lookback (grounded in Eckstein et al., 2026) → 7-day risk trajectory with recommended intervention timing.

**Output / Action Layer:**
- **Low risk** → Digital self-care platform
- **Medium risk** → Trained counsellor referral + peer support
- **High risk** → School-based escalation + emergency mental health services

> **ALL high-risk outputs require human clinical review. No automated action at any tier.**

**Deployment context is configured locally.** Service pathway mapping is adapted to the institutional and national context of each deployment.

---

## Repository Structure

| Folder / File | Contents |
|---|---|
| `literature/annotated_bibliography.md` | 11 papers across 4 categories |
| `framework/pipeline_design.md` | Full 3-tier architecture + feature engineering |
| `framework/clinical_rationale.md` | Evidence base for every design decision |
| `framework/governance_framework.md` | Data governance + ethics + consent framework |
| `framework/guardian_connection.md` | Link to GUARDIAN multimodal project |
| `notebooks/01_data_exploration.ipynb` | Dataset EDA + visualisations |
| `notebooks/02_text_preprocessing.ipynb` | Multilingual NLP preprocessing pipeline |
| `notebooks/03_classification_demo.ipynb` | MentalBERT classification demo |
| `notebooks/04_risk_scoring.ipynb` | XGBoost + SHAP risk scoring |
| `policy/policy_brief.md` | Singapore deployment policy brief |

---

## Why Asian Adolescents?

| Dimension | Western Pattern | Asian Adolescent Pattern |
|---|---|---|
| **Language** | English | Mandarin, mixed code-switching |
| **Expression** | Direct emotional disclosure | Indirect, somatic, metaphorical |
| **Distress framing** | "I feel depressed" | "最近好累" / "考试又失败了" |
| **Platform** | Reddit, Twitter | Instagram, Discord, LINE, Weibo |
| **Help-seeking** | Individual, direct | Family-mediated, stigma-avoidant |

Chinese MentalBERT achieves **F1 = 88.39%** on Chinese mental health data vs ~68% for general multilingual BERT. Domain-specific training is not optional for this population.

---

## Architecture Rationale

**Three tiers — not one** — because a single model forces an unacceptable clinical tradeoff between sensitivity and specificity. The 3-tier design resolves this by applying progressively deeper analysis only to flagged cases.

**XGBoost over deep networks** — because counsellors need to understand *why* a score was generated, not just *what* it is. SHAP explainability provides feature-level clinical justification for every risk score.

**MentalBERT over general LLMs** — because indirect Chinese distress expression requires domain-specific pre-training on mental health social media text. General multilingual models systematically underperform on non-Latin scripts (Google DeepMind, 2025).

---

## Datasets

| Dataset | Language | Size | Access |
|---|---|---|---|
| CNSocialDepress | Chinese (Simplified) | 44,178 posts, 233 users | arXiv:2510.11233 |
| CLPsych 2024 | English | Reddit posts | CLPsych workshop |
| SWMH | Chinese + English | Multi-disorder | Academic request |

> No real patient data included. All notebooks use synthetic or publicly available research datasets only.

---

## Deployment Context Examples

| Risk Tier | Generic Pathway | Singapore | Taiwan |
|---|---|---|---|
| Low | Digital self-care platform | HealthHub / mindline.sg | HPA mental health resources |
| Medium | Counsellor referral | CHAT / webCHAT | School counsellor + 1925 |
| High | School escalation + emergency | REACH + IMH | School counsellor + 1925 emergency |

See `policy/policy_brief.md` for the full Singapore deployment framework.

---

## Data Standards & Interoperability

| Standard | Role in Pipeline |
|---|---|
| **HL7 FHIR** | Risk scores exported as FHIR Observation resources |
| **SNOMED CT** | Mental health condition coding |
| **ICD-10** | F32/F33 depression risk codes for clinical records |

---

## Ethical Considerations

| Principle | Implementation |
|---|---|
| **Clinical safety first** | Human review mandatory — no automated action |
| **Interpretability** | SHAP values for every risk score |
| **Cultural validity** | Validated on Asian adolescent populations |
| **Privacy by design** | Raw text deleted after feature extraction — never stored |
| **Equity** | Regular audits for representation, cultural, and stereotype bias |
| **Consent** | Explicit opt-in only — no passive collection |
| **Data protection** | PDPA (SG) · PDPO (HK) · PIPL (CN) · HIPAA (US) · GDPR (EU) · APPI (JP) |
| **Quadruple Aim** | Patient experience · Population health · Cost reduction · Provider experience |

---

## Key References

**Clinical Foundation**
- MOH Singapore (2024). *National Mental Health and Well-being Strategy.*
- Weng, J.H. et al. (2024). mindline.sg. *JMIR Mental Health.*
- Systematic Review (2026). Digital interventions for adolescent mental health. *Al-Rafidain Journal of Medical Sciences.*

**NLP Methods**
- Ji, S. et al. (2024). Chinese MentalBERT. *arXiv:2402.09151*
- CNSocialDepress (2025). *arXiv:2510.11233*
- Chim, J. et al. (2024). CLPsych 2024 Shared Task. *ACL Anthology.*
- Zhang, T. et al. (2022). NLP applied to mental illness detection. *NPJ Digital Medicine, 5*(1).

**Multilingual AI**
- Google DeepMind (2025). Prompting with Phonemes.
- Google DeepMind (2025). EmbeddingGemma.
- Google DeepMind (2024). Tx-LLM.

**Cognitive Science**
- Eckstein, M.K. et al. (2026). Hybrid neural-cognitive models. *Nature Human Behaviour.*

---

## Open in Colab

| Notebook | Description | Open |
|---|---|---|
| 01_data_exploration | Dataset EDA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/lfangyu443-ui/digital-mh-nlp-asian-adolescents/blob/main/notebooks/01_data_exploration.ipynb) |
| 02_text_preprocessing | Multilingual NLP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/lfangyu443-ui/digital-mh-nlp-asian-adolescents/blob/main/notebooks/02_text_preprocessing.ipynb) |
| 03_classification_demo | MentalBERT | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/lfangyu443-ui/digital-mh-nlp-asian-adolescents/blob/main/notebooks/03_classification_demo.ipynb) |
| 04_risk_scoring | XGBoost + SHAP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/lfangyu443-ui/digital-mh-nlp-asian-adolescents/blob/main/notebooks/04_risk_scoring.ipynb) |

---

## Related Projects

**GUARDIAN** — Multimodal Predictive Model for Adolescent Self-Harm Risk: [github.com/lfangyu443-ui/guardian-db](https://github.com/lfangyu443-ui/guardian-db)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Research project only. Clinical deployment requires IRB approval, informed consent, and institutional governance.*

