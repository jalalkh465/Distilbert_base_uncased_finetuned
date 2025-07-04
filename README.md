# DistilBERT (base-uncased) Fine-Tuned on SST-2 for Sentiment Analysis

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the **SST-2** dataset (Stanford Sentiment Treebank) for **binary sentiment classification (positive/negative)**.

---

## 🚀 Project Overview

This project involved:
- Loading the pre-trained `distilbert-base-uncased` model from Hugging Face.
- Fine-tuning the model on the SST-2 dataset using Hugging Face’s `Trainer` API.
- Saving and uploading the fine-tuned model to the Hugging Face Hub for easy reuse.
- Providing a Colab notebook (hosted on GitHub) for reproducibility.

---

## 🤖 Model Usage

You can directly use the model with Hugging Face's `pipeline`:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="Jalal465/DistilBERT_base_uncased_finetuned_sst2_Jalal")
result = classifier("I love working with generative AI models.")
print(result)
