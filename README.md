# Multilingual Mental Health Sentiment Analysis & Response Generation

An NLP prototype where I built and fine-tuned a transformer-based mental health text classifier, then connected it with a generative AI response layer. The project combines **XLM-RoBERTa** for multilingual classification and **FLAN-T5 Large** for supportive response generation.

The implementation is contained in a single notebook and demonstrates hands-on AI development skills: dataset preparation, model selection, transformer fine-tuning, loss-function adjustment, evaluation, multilingual inference testing, and prompt-based generation.

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

The notebook demonstrates:

- Dataset loading with `kagglehub`
- Text cleaning and label mapping
- Stratified train, validation, and test splitting
- XLM-RoBERTa tokenization
- A custom PyTorch `Dataset`
- Model selection and fine-tuning configuration for `xlm-roberta-base`
- A custom Hugging Face `Trainer` with weighted cross-entropy loss for imbalanced labels
- Model training, validation monitoring, and test-set evaluation
- Saved-model loading for inference
- English, Japanese, Mandarin, and Spanish prediction tests
- FLAN-T5 Large prompt design for supportive response generation

## Task Framing

I treated mental health text analysis as a seven-class supervised classification task:

| Label | Description in the notebook |
| --- | --- |
| Normal | Non-distress or ordinary positive/neutral text |
| Depression | Low mood, hopelessness, loss of joy |
| Suicidal | Crisis-like or self-harm-related language |
| Anxiety | Panic, fear, spiralling thoughts |
| Bipolar | Mood fluctuation patterns |
| Stress | Workload, pressure, insomnia, overwhelm |
| Personality disorder | Intense emotions and unstable interpersonal patterns |

After classification, I passed the predicted label into a FLAN-T5 prompt to generate a short supportive message, turning the project from a pure classifier into a two-stage AI support prototype.

## Dataset

I used a Kaggle dataset as the training source:

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

I used a stratified split to preserve class representation across training, validation, and test sets:

| Split | Rows | Share |
| --- | ---: | ---: |
| Train | 31,608 | 60% |
| Validation | 10,536 | 20% |
| Test | 10,537 | 20% |

## Model Pipeline

### 1. Preprocessing and Tokenization

I mapped text labels to integer IDs and tokenized user statements with:

```python
AutoTokenizer.from_pretrained("xlm-roberta-base")
```

Tokenization settings:

- `padding=True`
- `truncation=True`
- `max_length=128`
- `return_tensors="pt"`

### 2. Dataset Wrapper

I implemented a custom `SentimentDataset` to convert tokenized encodings and labels into PyTorch tensors for Hugging Face training.

### 3. XLM-RoBERTa Fine-Tuning

I selected `xlm-roberta-base` because the project needed multilingual text understanding rather than an English-only classifier. I adapted it for sequence classification with a custom configuration:

| Setting | Value |
| --- | --- |
| Number of labels | 7 |
| Hidden dropout | 0.3 |
| Attention dropout | 0.3 |
| Hidden layers | 8 |
| Layer norm epsilon | `1e-7` |
| Output attentions | `True` |

I reduced the default layer count from 12 to 8 to improve training efficiency and increased dropout to reduce overfitting risk.

### 4. Training Configuration

I configured training with Hugging Face `TrainingArguments`:

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

To handle class imbalance, I implemented a custom `WeightedTrainer` that applies weighted cross-entropy loss using `compute_class_weight`. This was important because labels such as **Personality disorder** had far fewer examples than **Normal** or **Depression**.

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

After training, I saved and reloaded the fine-tuned model to test it as an inference workflow rather than only reporting training metrics.

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

For the generative AI layer, I used:

```python
pipeline("text2text-generation", model="google/flan-t5-large")
```

I designed a few-shot prompt with two examples:

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

I enabled sampling with `temperature=0.8` and `top_p=0.9`, then tested repeated generation for the same input. This showed that the response layer can produce varied supportive messages, but also highlighted a control issue: in sensitive mental health contexts, variation must be balanced with safety and consistency. The tests also showed that multilingual input could be classified, while generated responses remained English-only.

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
