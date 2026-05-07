# Multilingual Mental Health Sentiment Analysis & Response Generation

An academic NLP prototype that combines **XLM-RoBERTa** classification with **FLAN-T5 Large** generation to detect mental-health-related signals from text and produce short supportive responses.

The project was developed for the University of Warwick MSBA module **IB9LQ0 Generative AI and AI Applications**. The implementation is contained in a single notebook and demonstrates a full workflow from Kaggle data ingestion to model fine-tuning, evaluation, multilingual testing, and response generation.

> This is a research and learning prototype, not a clinical, diagnostic, or crisis-intervention tool.

## What This Repository Contains

```text
.
├── README.md
├── requirements.txt
├── docs/
│   └── Executive_summary.pdf
└── notebooks/
    └── mental_health_sentiment_analysis_supportive_response.ipynb
```

The notebook includes:

- Dataset loading with `kagglehub`
- Text cleaning and label mapping
- Stratified train, validation, and test splitting
- XLM-RoBERTa tokenization
- A custom PyTorch `Dataset`
- Fine-tuning configuration for `xlm-roberta-base`
- A custom Hugging Face `Trainer` with weighted cross-entropy loss
- Model evaluation with classification report and confusion matrix
- Saved-model loading for inference
- English, Japanese, Mandarin, and Spanish prediction tests
- FLAN-T5 Large prompt-based supportive response generation

## Task Framing

The notebook treats mental health text analysis as a seven-class classification task:

| Label | Description in the notebook |
| --- | --- |
| Normal | Non-distress or ordinary positive/neutral text |
| Depression | Low mood, hopelessness, loss of joy |
| Suicidal | Crisis-like or self-harm-related language |
| Anxiety | Panic, fear, spiralling thoughts |
| Bipolar | Mood fluctuation patterns |
| Stress | Workload, pressure, insomnia, overwhelm |
| Personality disorder | Intense emotions and unstable interpersonal patterns |

After classification, the predicted label is passed into a FLAN-T5 prompt to generate a short supportive message.

## Dataset

The dataset is downloaded from Kaggle:

```python
kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")
```

After dropping rows with missing `statement` values, the notebook uses **52,681 records**.

| Class | Count |
| --- | ---: |
| Normal | 16,343 |
| Depression | 15,404 |
| Suicidal | 10,652 |
| Anxiety | 3,841 |
| Bipolar | 2,777 |
| Stress | 2,587 |
| Personality disorder | 1,077 |

The split is stratified:

| Split | Rows | Share |
| --- | ---: | ---: |
| Train | 31,608 | 60% |
| Validation | 10,536 | 20% |
| Test | 10,537 | 20% |

## Model Pipeline

### 1. Preprocessing and Tokenization

The notebook maps text labels to integer IDs and tokenizes user statements with:

```python
AutoTokenizer.from_pretrained("xlm-roberta-base")
```

Tokenization settings:

- `padding=True`
- `truncation=True`
- `max_length=128`
- `return_tensors="pt"`

### 2. Dataset Wrapper

A custom `SentimentDataset` converts tokenized encodings and labels into PyTorch tensors for Hugging Face training.

### 3. XLM-RoBERTa Fine-Tuning

The classifier starts from `xlm-roberta-base` with a custom sequence-classification configuration:

| Setting | Value |
| --- | --- |
| Number of labels | 7 |
| Hidden dropout | 0.3 |
| Attention dropout | 0.3 |
| Hidden layers | 8 |
| Layer norm epsilon | `1e-7` |
| Output attentions | `True` |

The notebook reduces the default layer count from 12 to 8 to improve training efficiency and increases dropout to reduce overfitting risk.

### 4. Training Configuration

Training uses Hugging Face `TrainingArguments`:

| Parameter | Value |
| --- | --- |
| Learning rate | `2e-5` |
| Train batch size | 16 |
| Eval batch size | 4 |
| Epochs | 6 |
| Weight decay | 0.01 |
| Evaluation strategy | Epoch |
| Save strategy | Epoch |
| Mixed precision | `fp16=True` |

To handle class imbalance, the notebook implements a custom `WeightedTrainer` that applies weighted cross-entropy loss using `compute_class_weight`.

## Training Run

The recorded notebook run completed **11,856 training steps across 6 epochs**.

Training runtime:

- **2,861 seconds**
- Approximately **47 minutes 39 seconds**
- GPU-backed Colab environment

Validation loss by epoch:

| Epoch | Training Loss | Validation Loss |
| ---: | ---: | ---: |
| 1 | 0.9939 | 0.8754 |
| 2 | 0.8052 | 0.7999 |
| 3 | 0.7449 | 0.8170 |
| 4 | 0.4406 | 0.6787 |
| 5 | 0.6643 | 0.6550 |
| 6 | 0.4409 | 0.6570 |

Final validation evaluation:

```text
eval_loss: 0.6570
eval_runtime: 32.08 seconds
eval_samples_per_second: 328.41
eval_steps_per_second: 82.10
```

## Test Results

On the 10,537-row test set:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.78 |
| Macro F1 | 0.75 |
| Weighted F1 | 0.79 |

Per-class performance:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Normal | 0.98 | 0.85 | 0.91 | 3,269 |
| Depression | 0.80 | 0.67 | 0.73 | 3,081 |
| Suicidal | 0.65 | 0.79 | 0.72 | 2,131 |
| Anxiety | 0.79 | 0.86 | 0.82 | 768 |
| Bipolar | 0.80 | 0.85 | 0.82 | 556 |
| Stress | 0.50 | 0.85 | 0.63 | 517 |
| Personality disorder | 0.59 | 0.69 | 0.64 | 215 |

The confusion matrix shows that the model often confuses **Depression** and **Suicidal**, and some examples expected as **Bipolar**, **Personality disorder**, or **Normal** are predicted as **Stress**, **Depression**, or **Anxiety**. This reflects a key limitation of mental health text classification: categories can overlap semantically even when labels are mutually exclusive.

## Inference Tests

The notebook includes custom prediction tests after saving and reloading the fine-tuned model.

English examples:

| Expected | Predicted | Confidence |
| --- | --- | ---: |
| Normal | Normal | 99.6% |
| Depression | Depression | 52.9% |
| Suicidal | Stress | 96.1% |
| Anxiety | Anxiety | 98.8% |
| Bipolar | Stress | 73.4% |
| Stress | Stress | 96.7% |
| Personality disorder | Depression | 70.0% |

Multilingual examples:

| Language | Expected | Predicted | Confidence |
| --- | --- | --- | ---: |
| Japanese | Normal | Normal | 99.7% |
| Japanese | Depression | Anxiety | 99.3% |
| Japanese | Suicidal | Suicidal | 88.8% |
| Mandarin | Anxiety | Anxiety | 99.2% |
| Mandarin | Bipolar | Suicidal | 42.7% |
| Mandarin | Stress | Stress | 47.5% |
| Mandarin | Personality disorder | Stress | 42.1% |
| Spanish | Normal | Normal | 48.7% |
| Spanish | Anxiety | Depression | 56.5% |
| Spanish | Depression | Depression | 73.4% |

These tests show that XLM-RoBERTa can process multilingual inputs, but reliable multilingual performance would require stronger multilingual evaluation data and additional calibration.

## Response Generation

The second part of the notebook uses:

```python
pipeline("text2text-generation", model="google/flan-t5-large")
```

The prompt provides two few-shot examples:

- Depression-style input -> supportive reassurance
- Suicidal-style input -> supportive, help-seeking language

Generation settings:

| Parameter | Value |
| --- | --- |
| Max length | 80 |
| Number of responses | 1 |
| Temperature | 0.8 |
| Top-p | 0.9 |
| Sampling | Enabled |

The notebook demonstrates that repeated generation for the same input can produce different responses, because sampling is enabled. It also shows that multilingual input can be classified, while generated responses remain English-only.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/mental_health_sentiment_analysis_supportive_response.ipynb
```

Notes:

- The notebook was run in a GPU-backed environment.
- Kaggle access may require credentials depending on the runtime.
- Hugging Face public model downloads require internet access.
- The fine-tuned model directory `fine-tuned-xlmr/` is not included in this repository.
- Large generated model weights are intentionally excluded from Git.

## Limitations

- The model is trained on labelled text data, not clinically validated mental health assessments.
- The labels simplify complex and overlapping mental health states.
- Minority classes remain harder despite weighted loss.
- Some high-confidence predictions are wrong, which is risky in sensitive domains.
- The generation component lacks crisis-specific guardrails and escalation logic.
- The response generator outputs English even when the input is multilingual.
- A real deployment would require privacy design, human review, safety filters, bias testing, and clinical evaluation.

## Next Improvements

- Add stronger safety handling for suicidal or crisis-like inputs.
- Evaluate with balanced multilingual test sets.
- Add confidence thresholds and abstention when predictions are uncertain.
- Compare XLM-RoBERTa with other multilingual encoders.
- Add model cards and data cards for responsible AI documentation.
- Fine-tune or prompt the generation model for multilingual supportive responses.
- Separate the notebook into reproducible scripts for training, evaluation, and inference.

## Ethical Disclaimer

This repository is for academic demonstration only. It should not be used to diagnose, triage, or provide treatment for mental health conditions. If used as the basis for future work, it must include expert review, escalation pathways, and safeguards for high-risk content.
