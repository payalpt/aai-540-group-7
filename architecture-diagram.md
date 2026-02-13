# AAI-540 Emotion Classification MLOps Architecture

## Full Architecture Diagram

```mermaid
flowchart TB
    subgraph AWS["AWS Cloud"]

        subgraph DataIngestion["Data Ingestion Layer"]
            RAVDESS[("RAVDESS Dataset<br/>12,803 WAV files")]
            S3Raw[("S3 Bucket<br/>audio/Datasets/")]
            CSVFiles["CSV Files<br/>cleaned_emotion<br/>cleaned_age<br/>cleaned_gender"]
        end

        subgraph DataCatalog["Data Catalog Layer"]
            Athena[("AWS Athena<br/>audio_emotions DB")]
            GlueCatalog["AWS Glue<br/>Catalog"]
            CombinedData["Combined_Emotion_Data<br/>12,803 x 23 features"]
        end

        subgraph FeatureEngineering["Feature Engineering Layer"]
            Librosa["librosa<br/>Audio Feature Extraction"]
            Scaler["StandardScaler"]
            LabelEncoder["LabelEncoder<br/>7 emotion classes"]
            DerivedFeatures["Derived Features<br/>freq_range<br/>spectral_contrast<br/>energy_ratio"]
        end

        subgraph FeatureStore["SageMaker Feature Store"]
            FeatureGroup[("emotion-audio-features<br/>5,121 training records")]
            OnlineStore["Online Store<br/>Real-time retrieval"]
            OfflineStore["Offline Store<br/>Athena queries"]
        end

        subgraph DataSplits["Data Splits (Stratified)"]
            TrainData["Training<br/>5,121 (40%)"]
            TestData["Test<br/>1,280 (10%)"]
            ValData["Validation<br/>1,280 (10%)"]
            ProdData["Production<br/>5,121 (40%)<br/>5 batches"]
        end

        subgraph ModelTraining["Model Training Layer"]
            LogReg["Logistic Regression<br/>F1: 0.403"]
            XGBoost["XGBoost<br/>F1: 0.636"]
            ModelArtifacts[("S3<br/>models/benchmarks/")]
        end

        subgraph CICD["CI/CD Pipeline - SageMaker Pipelines"]
            Preprocess["Preprocessing<br/>SKLearnProcessor"]
            Train["Training<br/>XGBoost"]
            Evaluate["Evaluation<br/>Accuracy Check"]
            Condition{{"Accuracy >= 0.60?"}}
            ModelRegistry["Model Registry<br/>SoundEmotionModelGroup"]
        end

        subgraph Deployment["Model Deployment"]
            Endpoint["SageMaker Endpoint<br/>XGBoost 1.7-1<br/>ml.m5.large"]
            DataCapture["Data Capture<br/>100% sampling"]
        end

        subgraph Monitoring["Monitoring Layer"]
            DQMonitor["Data Quality Monitor<br/>Hourly schedule"]
            MQMonitor["Model Quality Monitor<br/>Hourly schedule"]
            GroundTruth["Ground Truth<br/>Labels"]
            Baseline["Baseline Statistics<br/>& Constraints"]
        end

        subgraph CloudWatch["CloudWatch"]
            Dashboard["Dashboard<br/>SageMaker-ML-Benchmarks"]
            Alarms["Alarms<br/>Latency, Errors<br/>4XX, 5XX"]
        end

    end

    subgraph External["Inference"]
        API["HTTP API<br/>JSON Input/Output"]
        Predictions["Predictions<br/>7 Emotion Classes"]
    end

    %% Data Flow
    RAVDESS --> S3Raw
    CSVFiles --> S3Raw
    S3Raw --> GlueCatalog
    GlueCatalog --> Athena
    S3Raw --> Librosa
    Librosa --> CombinedData
    Athena --> CombinedData

    %% Feature Engineering Flow
    CombinedData --> Scaler
    CombinedData --> LabelEncoder
    CombinedData --> DerivedFeatures
    Scaler --> TrainData
    Scaler --> TestData
    Scaler --> ValData
    Scaler --> ProdData

    %% Feature Store Flow
    TrainData --> FeatureGroup
    FeatureGroup --> OnlineStore
    FeatureGroup --> OfflineStore

    %% Model Training Flow
    TrainData --> LogReg
    TrainData --> XGBoost
    LogReg --> ModelArtifacts
    XGBoost --> ModelArtifacts

    %% CI/CD Flow
    ModelArtifacts --> Preprocess
    Preprocess --> Train
    Train --> Evaluate
    Evaluate --> Condition
    Condition -->|Yes| ModelRegistry
    Condition -->|No| Preprocess

    %% Deployment Flow
    ModelRegistry --> Endpoint
    XGBoost -.->|Best Model| Endpoint
    Endpoint --> DataCapture
    Endpoint --> API
    API --> Predictions

    %% Monitoring Flow
    DataCapture --> DQMonitor
    DataCapture --> MQMonitor
    GroundTruth --> MQMonitor
    Baseline --> DQMonitor
    Baseline --> MQMonitor
    DQMonitor --> Dashboard
    MQMonitor --> Dashboard
    Dashboard --> Alarms
```

## Simplified Architecture Diagram

```mermaid
flowchart LR
    subgraph Data["Data Layer"]
        WAV[("WAV Files<br/>12,803")]
        S3[("S3")]
        Athena[("Athena")]
    end

    subgraph Features["Feature Engineering"]
        Extract["librosa<br/>Feature Extraction"]
        Scale["StandardScaler"]
        FS[("Feature Store<br/>5,121 records")]
    end

    subgraph Training["Model Training"]
        LR["Logistic Reg<br/>F1: 0.40"]
        XGB["XGBoost<br/>F1: 0.64"]
    end

    subgraph Pipeline["CI/CD Pipeline"]
        Pre["Preprocess"]
        Tr["Train"]
        Eval["Evaluate"]
        Cond{{"Acc >= 0.6?"}}
        Reg["Registry"]
    end

    subgraph Deploy["Deployment"]
        EP["Endpoint<br/>ml.m5.large"]
        Cap["Data Capture"]
    end

    subgraph Monitor["Monitoring"]
        DQ["Data Quality"]
        MQ["Model Quality"]
        CW["CloudWatch<br/>Dashboard"]
    end

    WAV --> S3 --> Athena
    S3 --> Extract --> Scale --> FS
    FS --> LR & XGB
    XGB -->|Best| EP

    Pre --> Tr --> Eval --> Cond
    Cond -->|Pass| Reg --> EP

    EP --> Cap --> DQ & MQ --> CW
```

## Top-Down Pipeline View

```mermaid
flowchart TD
    subgraph Input["Input Data"]
        A[("RAVDESS<br/>12,803 WAV Files<br/>7 Emotion Classes")]
    end

    subgraph Storage["AWS Storage"]
        B[("S3 Bucket<br/>sagemaker-us-east-1-*")]
        C[("AWS Glue Catalog")]
        D[("AWS Athena<br/>audio_emotions")]
    end

    subgraph Processing["Feature Processing"]
        E["librosa Feature Extraction<br/>20 Acoustic Features"]
        F["Feature Engineering<br/>3 Derived Features"]
        G["StandardScaler + LabelEncoder"]
    end

    subgraph FeatureStore["SageMaker Feature Store"]
        H[("emotion-audio-features")]
        I["Online Store"]
        J["Offline Store"]
    end

    subgraph Split["Stratified Data Split"]
        K["Train: 5,121 - 40%"]
        L["Test: 1,280 - 10%"]
        M["Val: 1,280 - 10%"]
        N["Prod: 5,121 - 40%"]
    end

    subgraph Models["Benchmark Models"]
        O["Logistic Regression<br/>Accuracy: 0.40<br/>F1 Macro: 0.40"]
        P["XGBoost<br/>Accuracy: 0.62<br/>F1 Macro: 0.64"]
    end

    subgraph Pipeline["SageMaker Pipeline"]
        Q["PreprocessSoundData"]
        R["TrainXGBoostModel"]
        S["EvaluateModel"]
        T{{"CheckAccuracyThreshold<br/>Acc >= 0.60"}}
        U["RegisterModel"]
    end

    subgraph Deploy["Deployment"]
        V["SageMaker Endpoint<br/>XGBoost 1.7-1<br/>ml.m5.large"]
        W["Data Capture<br/>100%"]
    end

    subgraph Monitor["Monitoring"]
        X["Data Quality Monitor"]
        Y["Model Quality Monitor"]
        Z["CloudWatch Dashboard<br/>& Alarms"]
    end

    A --> B
    B --> C --> D
    B --> E --> F --> G
    G --> K & L & M & N
    K --> H
    H --> I & J
    K --> O & P
    P -->|Best Model| V

    Q --> R --> S --> T
    T -->|Pass| U --> V
    T -->|Fail| Q

    V --> W --> X & Y --> Z
```

## Horizontal Data Flow

```mermaid
flowchart LR
    subgraph Source["Source"]
        S1[("RAVDESS<br/>WAV Files")]
        S2["CSV Metadata"]
    end

    subgraph Ingest["Ingest"]
        I1[("S3")]
        I2[("Athena")]
    end

    subgraph Transform["Transform"]
        T1["librosa"]
        T2["Scaler"]
        T3["Encoder"]
    end

    subgraph Store["Store"]
        ST1[("Feature Store")]
        ST2["Train/Test Split"]
    end

    subgraph Train["Train"]
        TR1["LogReg"]
        TR2["XGBoost"]
    end

    subgraph Deploy["Deploy"]
        D1["Endpoint"]
        D2["API"]
    end

    subgraph Monitor["Monitor"]
        M1["Data Quality"]
        M2["Model Quality"]
        M3["CloudWatch"]
    end

    S1 & S2 --> I1 --> I2
    I1 --> T1 --> T2 --> T3 --> ST1 --> ST2
    ST2 --> TR1 & TR2
    TR2 -->|Winner| D1 --> D2
    D1 --> M1 & M2 --> M3
```

## Component Details

### Data Sources
- **RAVDESS Dataset**: 12,803 audio samples across 7 emotion classes
- **Emotion Classes**: Angry, Happy, Sad, Fearful, Disgusted, Neutral, Surprised
- **Class Imbalance**: Surprised (4.62%) vs others (14-17%)

### Feature Engineering
- **20 Acoustic Features**: meanfreq, sd, median, q25, q75, iqr, skew, kurt, sp_ent, sfm, mode, centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx
- **3 Derived Features**: freq_range, spectral_contrast, energy_ratio
- **Preprocessing**: StandardScaler (fit on train), LabelEncoder (0-6)

### Model Performance
| Model | Accuracy | F1 Macro | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | 0.403 | 0.403 | 6.54s |
| **XGBoost** | **0.621** | **0.636** | 4.68s |

### Monitoring
- **Data Quality**: Hourly checks against baseline statistics
- **Model Quality**: Hourly checks with ground truth comparison
- **CloudWatch**: Dashboard with latency, error, and violation metrics
- **Alarms**: High latency (>10s), Invocation errors (>5), 4XX/5XX errors
