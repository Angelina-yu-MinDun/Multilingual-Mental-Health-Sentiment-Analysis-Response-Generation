# Multilingual Mental Health Sentiment Analysis & Response Generation

## Portfolio Summary

This academic group project explored how transformer-based NLP and generative AI can support mental health text analysis. We built a two-stage prototype that first classifies user-written text into mental health-related categories, then generates a short supportive response based on the predicted emotional state.

The project combines **XLM-RoBERTa** for multilingual sentiment and mental health classification with **FLAN-T5 Large** for empathetic response generation. The final classifier achieved **78% accuracy**, **75% macro F1**, and **79% weighted F1** across seven labels.

## Problem

Online platforms contain large volumes of user-generated text that may reveal emotional distress, anxiety, depression, or crisis-related signals. However, these signals are difficult to monitor manually and can be linguistically subtle. The project asked whether modern language models could identify mental health patterns from text and provide an initial compassionate response.

The prototype was designed as a proof of concept for:

- Mental health monitoring on social media
- Automated support tools
- Counselling assistance workflows
- Multilingual emotional signal detection

## My Contribution

In a portfolio context, I would frame my contribution around the technical and analytical workflow:

- Prepared and cleaned labelled mental health text data
- Built the model training pipeline in Python
- Fine-tuned XLM-RoBERTa for seven-class classification
- Addressed class imbalance with weighted loss
- Evaluated model performance with classification metrics and confusion matrix
- Integrated FLAN-T5 Large to generate supportive messages
- Interpreted model strengths, limitations, and ethical risks

## Data

The dataset was sourced from Kaggle and contained user text labelled into seven categories:

- Normal
- Depression
- Suicidal
- Anxiety
- Bipolar
- Stress
- Personality disorder

After preprocessing, the dataset contained **52,681 rows**. The data was heavily imbalanced: Normal and Depression had the largest representation, while Personality disorder had the smallest. To reduce the impact of this imbalance, the training pipeline used class weights in the loss function.

## Technical Approach

The classification model was built with Hugging Face Transformers and PyTorch.

Main steps:

1. Data loading and preprocessing
2. Label mapping into seven numerical classes
3. Train, validation, and test split
4. Tokenization using XLM-RoBERTa tokenizer
5. Custom PyTorch dataset and dataloader setup
6. XLM-RoBERTa fine-tuning with weighted cross-entropy loss
7. Evaluation using precision, recall, F1-score, accuracy, and confusion matrix
8. Supportive response generation using FLAN-T5 Large

I configured the model with increased dropout and reduced transformer depth to improve training efficiency and reduce overfitting risk.

## Results

The fine-tuned model achieved:

- Accuracy: **78%**
- Macro F1-score: **75%**
- Weighted F1-score: **79%**

The model performed especially well on:

- Normal
- Anxiety
- Bipolar

It struggled more with:

- Stress
- Personality disorder
- Borderline cases between Depression and Suicidal

This makes sense because mental health language often overlaps across categories. For example, expressions of hopelessness may appear in both depression and suicidal ideation, making classification more complex than standard sentiment analysis.

## Generative AI Component

After classification, the predicted label was passed into a FLAN-T5 prompt to produce a brief supportive message. The prompt used a few examples to guide the model toward a gentle, compassionate tone.

Example workflow:

- User input: "My heart is racing and I can't breathe. What if something terrible happens?"
- Predicted category: Anxiety
- Generated response: A short calming message acknowledging the user's anxiety

This part of the project showed how discriminative and generative AI can be combined in one workflow: classification for signal detection, generation for human-like support.

## Limitations

This project is a prototype and should not be treated as a clinical tool.

Key limitations:

- Dataset labels may simplify complex mental health conditions
- Class imbalance still affects minority-category performance
- Some multilingual predictions were inconsistent
- Generated responses do not include robust crisis escalation
- The system has not been clinically validated
- Privacy, safety, and bias concerns would need deeper testing before deployment

## Future Improvements

Future work could include:

- Larger and more clinically grounded datasets
- Better multilingual training data
- Prompt tuning or instruction tuning for safer response generation
- Crisis-detection guardrails and escalation messaging
- Topic modelling to explain distress-related themes
- Human-in-the-loop review for high-risk predictions
- Model explainability tools such as attention visualization or SHAP

## Skills Demonstrated

- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa fine-tuning
- FLAN-T5 prompting
- NLP preprocessing
- Class imbalance handling
- Model evaluation
- Generative AI workflow design
- Responsible AI and ethical risk analysis

## Short LinkedIn / Portfolio Blurb

Built a mental health NLP prototype combining XLM-RoBERTa classification and FLAN-T5 response generation. The model classified user-written text into seven mental health-related categories and generated supportive responses based on predicted emotional state. Achieved 78% accuracy and 75% macro F1, while identifying key limitations around class imbalance, multilingual robustness, and AI safety in mental health applications.
