import os
import sys
from Cyberbullying.logging import logging
from Cyberbullying.exception import CustomException
from Cyberbullying.config.cloud_syncer import GCloudSync
from Cyberbullying.entity.config_entity import ModelPusherConfig
from Cyberbullying.entity.artifact_entity import ModelPusherArtifacts


class ModelPusher:
    def __init__(self, model_pusher_config:ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
        self.gcloud = GCloudSync()

    def initiate_model_pusher(self)->ModelPusherArtifacts:
        logging.info("Entered the initiate model_pusher method of ModelTrainer class")
        try:
            self.gcloud.sync_folder_to_gcloud(self.model_pusher_config.BUCKET_NAME,
                                              self.model_pusher_config.TRAINED_MODEL_PATH,
                                              self.model_pusher_config.MODEL_NAME)
            
            logging.info("Uploaded best model to GCloud Storage")

            # Saving the model pusher artifacts
            model_pusher_artifacts = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME
            )

            logging.info("Exited the initiate model_pusher method of ModelTrainer class")

            return model_pusher_artifacts
        
        except Exception as e:
            raise CustomException(e,sys) from e