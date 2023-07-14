from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    imbalanced_data_file_path: str
    labeled_data_file_path: str