import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
            This method creates and returns a data transformer object for the data including
            gender mapping, one-hot-encoding, column renaming, feature scaling and type adjustments.
        """
        logging.info(f'Entered get_data_transformer_object method of DataTransformation class')

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info(f'Transformers created: StandardScaler & MinMaxScaler')

            # Load schema configuration
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info(f'Columns loaded from the schema')

            # Creating Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("standard_scaler", numeric_transformer, num_features),
                    ("minmax_scaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps = [('Preprocessor', preprocessor)])
            logging.exception(f'Exception occured in get_data_transformer_object method of DataTransformation class')
            return final_pipeline
        
        except Exception as e:
            raise MyException(e, sys) from e
    
    def _map_gender_column(self, df):
        """ Maps gender column to {0: Female, 1: Male}"""
        logging.info(f'Mapping \'Gender\' column to binary values')
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        """ Creates one-hot encoding for categorical features."""
        logging.info("Creating one-hot encodings for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def _rename_columns(self, df):
        """ Rename specific columns and ensure appropriate dtypes"""
        logging.info(f'Renaming specific columns and typecasting necessary columns')
        df = df.rename(columns = {
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ['Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years', 'Vehicle_Damage_Yes']:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df: pd.DataFrame):
        """ Drop the \"id\" column if it exists"""
        logging.info(f'Dropping \'id\' column')
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
            This method initiates the data transformation component of the pipeline
        """
        try:
            logging.info(f'Data transformation initiated!')
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Test and train dataframes loaded")

            input_features_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info(f'Input and target columns defined for both, train and test, dataframes')

            # Apply transformations in specified sequence
            input_features_train_df = self._map_gender_column(input_features_train_df)
            input_features_train_df = self._drop_id_column(input_features_train_df)
            input_features_train_df = self._create_dummy_columns(input_features_train_df)
            input_features_train_df = self._rename_columns(input_features_train_df)

            input_features_test_df = self._map_gender_column(input_features_test_df)
            input_features_test_df = self._drop_id_column(input_features_test_df)
            input_features_test_df = self._create_dummy_columns(input_features_test_df)
            input_features_test_df = self._rename_columns(input_features_test_df)

            logging.info(f'Column transformations applied  to train and test data')
            preprocessor = self.get_data_transformer_object()
            logging.info(f'Preprocessor object recieved')

            logging.info(f'Initializing transformation for training data')
            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
            logging.info(f'Initializing transformation for test data')
            input_feature_test_arr = preprocessor.transform(input_features_test_df)
            logging.info(f'Data transformation completed')

            logging.info(f'Applying SMOTEENN for handling imbalanced dataset')
            smt = SMOTEENN(sampling_strategy = 'minority')
            input_feature_train_final, target_feature_train_final = smt.fit_resample(input_features_train_arr, target_feature_train_df)
            input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)

            logging.info('SMOTEENN applied to train and test dataframes')

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info(f'Feature-target concatenation done for test and train data')

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info(f'Saving transformation object and transformed files')

            logging.info(f'Data transformation completed successfully. Exiting!')
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        
        except Exception as e:
            raise MyException(e, sys) from e