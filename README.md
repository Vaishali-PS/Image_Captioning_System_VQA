# Intelligent Image Captioning and Visual Question Answering System

This project integrates object detection, image caption generation, and visual question answering (VQA) into a unified deep learning pipeline. It leverages state-of-the-art models including YOLOv8, EfficientNetB7, Transformers, and BLIP, with a user-friendly interface built using Streamlit.

---

## Features

- Object Detection using YOLOv8
- Feature Extraction with EfficientNetB7
- Image Captioning using Transformer-based Decoder
- Visual Question Answering with BLIP model
- Interactive Web UI with Streamlit

---
## Datasets

### Flickr8k Dataset

The Flickr8k dataset is a popular benchmark for image captioning tasks. It contains 8,000 images collected from Flickr, each paired with five different human-written captions describing the image content. This dataset is widely used for training and evaluating image caption generation models because of its diversity in scenes and objects.

- **Purpose:** Training and evaluating the image captioning pipeline.
- **Content:** 8,000 images with 5 captions per image.
- **Use in Project:** Used to train and validate the transformer-based caption generation model.

### DAQUAR Dataset

The DAQUAR (Dataset for Question Answering on Real-world images) dataset is designed for Visual Question Answering (VQA). It contains indoor scene images paired with natural language questions and answers, focusing on object recognition and spatial reasoning within indoor environments.

- **Purpose:** Training and evaluating the VQA component of the project.
- **Content:** Images of indoor scenes with annotated question-answer pairs.
- **Use in Project:** Used to fine-tune and evaluate the BLIP model for answering questions about images.
