# **VoiceSync AI - Emotion Detection ML System**

## **Overview**

This repository contains our end-to-end machine learning system developed for the final project for AAI-540 Machine Learning Operations course.

Our team developed a production-ready ML system using AWS SageMaker that demonstrates the full MLOps process, from data engineering and feature store to model deployment, monitoring, and CI/CD automation.

This repository serves as Deliverable #3 - the ML system codebase. 

## **Company Scenario - VoiceSync AI**

We developed this system for a hypothetical company called VoiceSync AI.

Modern call centers and conversational AI systems interact with customers through voice, but most systems ignore the caller's true emotional state when selecting the voice or response style of the virtual agent. 

VoiceSync AI aims to solve this by building a system that can detect emotions from speech in real time and enable emotion aware voice agent selection.

## **Machine Learning Problem**

We approached the project as a supervised multi-class classification problem.

**Task:**
Predict the speaker's emotions from short speech audio segments.

**Emotion Classes (7):**


*   Angry
*   Disgust
*   Fear
*   Happy
*   Sad
*   Surprised
*   Neutral

The system is designed as a real-time inference API capable of low-latency predictions.

## **End-to-End MLOps Process**

This project demonstrates the full ML process:

1.   Problem definition & dataset analysis
2.   Data preparation & preprocessing pipeline
3.   Feature engineering & Feature Store
4.   Model training & benchmarking
5.   Real-time model deployment
6.   Model monitoring
7.   CI/CD pipeline 

## **Dataset**

We used a combined emotional speech dataset from Kaggle containing 12,803 labeled audio files from four major public datasets:


*   CREMA-D
*   RAVDESS
*   SAVEE
*   TESS

This combined dataset provides a strong foundation for multi-class emotion classification.

## **Data Engineering Pipeline**

All data is stored in Amazon S3 and processed using AWS services.

Key steps:

*   Audio preprocessing and feature extraction
*   Feature extraction using Librosa
*   SQL querying using AWS Athena
*   Dataset merging
*   Stratified dataset splitting from train/validation/test 

Acoustic features extracted include:

*   MFCCs-based features
*   Spectral features
*   Pitch statistics

The final dataset contains 23 engineered features used for training.

## **Feature Store**

Features are stored in SageMaker Feature Store with:

*   Online store →  real-time inference
*   Offline store →  batch training and analysis

This allows for consistency between training and inference.

## **Model Training & Benchmarking**

Two models were trained and compared:

**Logistic Regression (Baseline)**

*   Accuracy: 40%
*   Macro F1: 0.40

**XGBoost (Production Model)**

*   Accuracy: 62%
*   Macro F1: 0.64

XGBoost improved macro F1 by 57.8%, and was selected for deployment.

## **Real-Time Deployment**

The best model was deployed as a real-time SageMaker endpoint.

The endpoint:

*   Accepts JSON feature vectors
*   Returns predicted emotion labels
*   Achieves <20 ms inference latency

This supports real-time call center use cases.

**Inference flow**

Audio → Feature Extraction → SageMaker Endpoint → Emotion Prediction

## **Monitoring**

We implemented three layers of monitoring:

1.   Infrastructure Monitoring (CloudWatch)

    - Tracks latency, errors, and endpoint health.

2.   Data Quality Monitoring

    - Detects feature drift

3.   Model Quality Monitoring

    - Compares predictions with ground truth to detect performance degradation.




All monitoring results are visualized in a CloudWatch dashboard.


## **CI/CD Pipeline**

We built an automated training pipeline with:

1.   Data preprocessing
2.   Model training
3.   Model evaluation
4.   Conditional accuracy gate
5.   Model registration

Models are only registered if accuracy ≥ 60%, preventing poor models from reaching production.





## **Limitations & Ethical Considerations**

*   Acted vs natural speech gap
*   Limited demographic diversity
*   Potential demographic bias in acoustic features
*   Ethical concerns around emotion detection


**Note:** Majority of datasets used give permission for academic use only.

These must be addressed before real-world deployment. 

## **Team Members**

*   **Joel Dievendorf:** Feature engineering, modeling, and ML pipeline
*  **Payal Patel:** EDA, model evaluation, deployment, project coordination, presentation slides
*   **Tommy Poole:** Data collection, data infrastructure, Athena setup, monitoring dashboards

## **Learning Outcomes**
This project demonstrates our ability to:

*   Design a production ready ML system
*   Apply MLOps workflows
*   Deploy real-time inference endpoints
*   Implement monitoring and CI/CD 
*   Collaborate using GitHub, Slack, and Asana
