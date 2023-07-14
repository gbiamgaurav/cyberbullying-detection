from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    imbalanced_data_file_path: str
    labeled_data_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str 
    X_test_path: list
    y_test_path: list    

@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool