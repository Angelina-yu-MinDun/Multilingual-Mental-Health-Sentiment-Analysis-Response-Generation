# Multilingual Mental Health Sentiment Analysis & Response Generation

This project explores how generative AI and transformer-based NLP can support early mental health signal detection from multilingual user-written text. The system combines a fine-tuned XLM-RoBERTa classifier with a FLAN-T5 response generator to classify mental health-related language and produce short supportive messages.

## Project Overview

The notebook builds a two-stage NLP workflow:

1. **Mental health text classification** using XLM-RoBERTa.
2. **Supportive response generation** using FLAN-T5 Large.

The classifier predicts seven categories:

- Normal
- Depression
- Suicidal
- Anxiety
- Bipolar
- Stress
- Personality disorder

The goal is not to replace clinical judgement, but to prototype how multilingual language models can help identify emotional distress patterns and provide a compassionate first response.

## Dataset

The project uses a Kaggle mental health text dataset containing user statements labelled across seven categories. After preprocessing, the dataset contained **52,681 records**.

Class distribution:

| Category | Records |
| --- | ---: |
| Normal | 16,343 |
| Depression | 15,404 |
| Suicidal | 10,652 |
| Anxiety | 3,841 |
| Bipolar | 2,777 |
| Stress | 2,587 |
| Personality disorder | 1,077 |

Because the dataset is imbalanced, the training workflow uses class weights through a custom `WeightedTrainer`.

## Methodology

The classification pipeline includes:

- Text cleaning and label mapping
- Train, validation, and test split
- XLM-RoBERTa tokenization with a maximum sequence length of 128
- Fine-tuning `xlm-roberta-base` for seven-way classification
- Increased dropout to reduce overfitting
- Reduced transformer depth from 12 to 8 layers for efficiency
- Weighted cross-entropy loss for minority-class handling
- Evaluation with precision, recall, F1-score, accuracy, and confusion matrix

The response generation pipeline uses:

- `google/flan-t5-large`
- Few-shot prompt examples for empathetic response style
- Temperature and top-p sampling for varied but concise responses

## Results

The fine-tuned classifier achieved:

| Metric | Score |
| --- | ---: |
| Accuracy | 78% |
| Macro F1 | 75% |
| Weighted F1 | 79% |

Strongest categories:

- Normal: F1 0.91
- Anxiety: F1 0.82
- Bipolar: F1 0.82

Harder categories:

- Stress: F1 0.63
- Personality disorder: F1 0.64

The confusion matrix showed overlap between depression and suicidal language, which is expected because the two categories can share similar lexical and emotional signals. Some multilingual examples performed well, but non-English prediction consistency still needs improvement.

## Key Learnings

- Class weighting improves fairness toward underrepresented labels, but does not fully solve semantic overlap between mental health categories.
- XLM-RoBERTa is useful for **multilingual experimentation**, though fine-tuning data language coverage strongly affects real-world robustness.
- Generating empathetic responses requires additional safety design, especially for crisis-related or self-harm inputs.
- A production version would need stronger guardrails, escalation guidance, human review, and domain-specific validation.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── mental_health_sentiment_analysis_supportive_response.ipynb
└── docs/
    ├── Group6_Executive_summary.pdf
    ├── IB9LQ0-Group-Assignment-2024-2025.pdf
    └── portfolio-notion-draft.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/mental_health_sentiment_analysis_supportive_response.ipynb
```

The notebook downloads data through Kaggle tooling and Hugging Face models. You may need Kaggle credentials and sufficient compute for transformer fine-tuning.

## Ethical Note

This project is an academic prototype. It should not be used for diagnosis, crisis triage, or treatment decisions. Mental health AI systems require clinical validation, safety escalation pathways, privacy controls, bias testing, and human oversight before deployment.
