import pandas as pd
import pickle
import os
import sys
from Cyberbullying.constants import *
from Cyberbullying.logging import logging
from Cyberbullying.exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from Cyberbullying.ml.model import ModelArchitecture
from Cyberbullying.entity.config_entity import ModelTrainerConfig
from Cyberbullying.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts


class ModelTrainer:
    def __init__(self,data_transformation_artifacts:DataTransformationArtifacts,
                 model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def split_the_data(self, csv_path):
        try:
            logging.info("Entered the split_the_data function")
            logging.info("Read the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Split the data into X and y")

            X = df[TWEET]
            y = df[LABEL]

            logging.info("Applying splitting on the data")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(len(X_train), len(y_train))
            print(len(X_test), len(y_test))
            print(type(X_train), type(y_train))
            logging.info("Exited the splitting function")

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def tokenizing(self, X_train):
        try:
            logging.info("Applying tokenization")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(X_train)
            sequences = tokenizer.texts_to_sequences(X_train)
            logging.info("Converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            logging.info("The Sequence matrix is: {sequences_matrix}")
            return sequences_matrix, tokenizer
        
        except Exception as e:
            raise CustomException(e,sys) from e    
        
    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Entered the initiate_model_trainer function ")
            X_train,X_test,y_train,y_test = self.split_the_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()   

            model = model_architecture.get_model()



            logging.info(f"Xtrain size is : {X_train.shape}")

            logging.info(f"Xtest size is : {X_test.shape}")

            sequences_matrix,tokenizer =self.tokenizing(X_train)


            logging.info("Entered into model training")
            model.fit(sequences_matrix, y_train, 
                        batch_size=self.model_trainer_config.BATCH_SIZE, 
                        epochs = self.model_trainer_config.EPOCH, 
                        validation_split=self.model_trainer_config.VALIDATION_SPLIT, 
                        )
            logging.info("Model training finished")
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)



            logging.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            X_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            X_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                X_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH)
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e