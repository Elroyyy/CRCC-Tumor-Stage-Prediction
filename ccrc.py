import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np


class CCRCModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.stage_encoder = None
        self.features = []

    def fetch_data(self):
        """Fetch data from MongoDB"""
        client = MongoClient('mongodb://localhost:27017/CCRC')
        db = client['patients']
        collection = db['patient_data']
        data = pd.DataFrame(list(collection.find()))
        data.rename(columns={data.columns[1]: 'Patient_ID'}, inplace=True)
        client.close()
        return data

    def preprocess_data(self, df):
        """Preprocess and encode data"""
        df = df.drop(['_id', 'Patient_ID', 'NumberLeftTumors',
                      'NumberRightTumors', 'Side'], axis=1)

        for col in ['Calcification', 'Necrosis']:
            if col in df.columns:
                df[col] = df.groupby('Stage')[col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
                )

        label_encoders = {}
        for col in df.select_dtypes(include=['object', 'bool']):
            if col != 'Stage':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        stage_encoder = LabelEncoder()
        df['Stage'] = stage_encoder.fit_transform(df['Stage'].astype(str))

        return df, label_encoders, stage_encoder

    def train_model(self, df):
        """Train decision tree model"""
        X = df.drop(columns=['Stage'])
        y = df['Stage']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = DecisionTreeClassifier(max_depth=4)
        model.fit(X_train, y_train)

        return model, X

    def save_models(self):
        """Save trained models"""
        joblib.dump(self.model, "decision_tree_model.pkl")
        joblib.dump(self.label_encoders, "label_encoders.pkl")
        joblib.dump(self.stage_encoder, "stage_encoder.pkl")
        joblib.dump(self.features, "model_features.pkl")

    def load_models(self):
        """Load trained models"""
        try:
            self.model = joblib.load("decision_tree_model.pkl")
            self.label_encoders = joblib.load("label_encoders.pkl")
            self.stage_encoder = joblib.load("stage_encoder.pkl")
            self.features = joblib.load("model_features.pkl")
            return True
        except FileNotFoundError:
            return False

    def initialize(self):
        """Initialize and train models if needed"""
        if not self.load_models():
            print("Training new model...")
            patient_data = self.fetch_data()
            preprocessed_data, self.label_encoders, self.stage_encoder = self.preprocess_data(patient_data)
            self.model, X = self.train_model(preprocessed_data)
            self.features = list(X.columns)
            self.save_models()
            print("Model trained and saved successfully!")
        else:
            print("Models loaded successfully!")

    def predict(self, input_data):
        """Make prediction on input data"""
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        predicted_stage = self.stage_encoder.inverse_transform(prediction)[0]
        return predicted_stage

    def get_feature_options(self, feature):
        """Get options for a categorical feature"""
        if feature in self.label_encoders:
            return list(self.label_encoders[feature].classes_)
        return None

    def encode_input(self, feature, value):
        """Encode a categorical input value"""
        if feature in self.label_encoders:
            return int(self.label_encoders[feature].transform([value])[0])
        return float(value)

    def save_prediction(self, name, predicted_stage, input_data):
        """Save prediction to MongoDB"""
        client = MongoClient('mongodb://localhost:27017/CCRC')
        db = client['patients']
        prediction_collection = db['predicted_stage']

        flat_features = {}
        for k, v in input_data.items():
            if isinstance(v, (int, np.integer)) and (v == 0 or v == 1):
                flat_features[k] = bool(v)
            else:
                flat_features[k] = v

        prediction_document = {
            "name": name,
            "predicted_stage": predicted_stage,
            **flat_features
        }
        prediction_collection.insert_one(prediction_document)
        client.close()

    def get_saved_records(self):
        """Retrieve all saved predictions"""
        client = MongoClient('mongodb://localhost:27017/CCRC')
        db = client['patients']
        records = list(db['predicted_stage'].find({}, {"_id": 0}))
        client.close()

        if records:
            df = pd.DataFrame(records)

            for col in df.columns:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(str)
                elif df[col].dtype == 'object':
                    try:
                        pd.to_numeric(df[col])
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        df[col] = df[col].astype(str)

            for feature, le in self.label_encoders.items():
                if feature in df.columns:
                    df[feature] = df[feature].map(
                        lambda x: le.inverse_transform([int(x)])[0]
                        if pd.notna(x) and str(x).isdigit() else x
                    )

            return df.to_dict('records')
        return []