import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import streamlit as st
import numpy as np

def _fetch_data():
    #Connect mongoDB database - collections and choose them
    client = MongoClient('mongodb://localhost:27017/CCRC')
    db = client['patients']
    collection = db['patient_data']
    data = pd.DataFrame(list(collection.find()))
    data.rename(columns={data.columns[1]: 'Patient_ID'}, inplace=True)
    client.close()
    return data

def _preprocessing_data(df):

    #Remove unnecessary columns
    df = df.drop(['_id','Patient_ID', 'NumberLeftTumors', 'NumberRightTumors', 'Side'], axis=1)

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

    # Encode target variable "Stage"
    stage_encoder = LabelEncoder()
    df['Stage'] = stage_encoder.fit_transform(df['Stage'].astype(str))

    return df, label_encoders, stage_encoder

def _train_model(df):
    X = df.drop(columns=['Stage'])
    y = df['Stage']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)

    return model, X

def _save_models(model, label_encoders, stage_encoder, feature_list):
    joblib.dump(model, "decision_tree_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    joblib.dump(stage_encoder, "stage_encoder.pkl")
    joblib.dump(feature_list, "model_features.pkl")

def _load_models():
    model = joblib.load("decision_tree_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    stage_encoder = joblib.load("stage_encoder.pkl")
    features = joblib.load("model_features.pkl")
    return model, label_encoders, stage_encoder, features


def predict_interface(model, label_encoders, stage_encoder, features):
    client = MongoClient('mongodb://localhost:27017/CCRC')
    db = client['patients']

    # --- Sidebar Navigation ---
    st.set_page_config(page_title="CCRC Stage Predictor", page_icon="icon.png", layout="wide")

    with st.sidebar:
        st.image("icon.png")
        st.title("CCRC")
        page = st.radio(" ", ["Home", "Predict Stage", "Saved Records"])

    # --- Home Page ---
    if page == "Home":

        st.markdown(
            """
            <h1 style='text-align:left; color:#2E86C1;'>
                CCRC Cancer Stage Classifier
            </h1>
            """,
            unsafe_allow_html=True
        )
        st.subheader("AI-powered support for early detection and diagnosis")
        st.markdown(
            """
            ### Welcome to the CCRC Classifier  

            This application uses **Machine Learning** to help classify **Clear Cell Renal Cell Carcinoma (CCRC)**, the most common type of kidney cancer.  
            Early and accurate detection is key to better treatment outcomes.  

            ⚡ **Features**  
            - Input patient data (clinical/diagnostic).  
            - Predicts CCRC stages using a trained ML model.  
            - Offers quick, interpretable insights.  
            """
        )

        st.image("CT.png", width=800)

        st.markdown("""
                  ---
            ⚠️ **Disclaimer**: For **educational and research purposes only** —  
            not a replacement for professional medical advice or diagnosis.
            """)

    # --- Prediction Page ---
    elif page == "Predict Stage":
        st.markdown("<h2 style='color:#117A65;'>Predict Tumor Stage</h2>", unsafe_allow_html=True)
        st.write("Provide patient details below:")

        with st.form("patient_form"):
            name = st.text_input("Patient Name", placeholder="Enter patient full name")
            user_input = {}
            cols = st.columns(3)

            for idx, feature in enumerate(features):
                col = cols[idx % 3]

                # --- Categorical fields with LabelEncoder ---
                if feature in label_encoders:
                    le = label_encoders[feature]
                    options = ["Select..."] + list(le.classes_)  # Placeholder option
                    selected = col.selectbox(f"{feature}", options)

                    if selected == "Select...":
                        user_input[feature] = None
                    else:
                        user_input[feature] = int(le.transform([selected])[0])

                # --- Numerical fields ---
                else:
                    value_str = col.text_input(f"{feature}", value="", placeholder="Enter value")
                    if value_str == "":
                        user_input[feature] = None
                    else:
                        try:
                            user_input[feature] = float(value_str)
                        except ValueError:
                            st.warning(f" {feature} must be a number")
                            user_input[feature] = None

            submitted = st.form_submit_button("Predict Stage")

        if submitted:
            if name.strip() == "":
                st.warning("Please enter the patient's name.")
                return

            missing_features = [f for f, v in user_input.items() if v is None or v == ""]
            if missing_features:
                st.warning(f"Please fill in all fields. Missing: {', '.join(missing_features)}")
                return

            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            predicted_stage = stage_encoder.inverse_transform(prediction)[0]
            stage_descriptions = {
                "Stage I": (
                    "Stage I CCRC means the tumor is small (usually less than 7 cm) "
                    "and limited to the kidney. At this stage, the cancer has not "
                    "spread beyond the kidney. Treatment is often very effective, "
                    "and surgery may fully remove the tumor."
                ),
                "Stage II": (
                    "Stage II CCRC indicates that the tumor is larger than in Stage I "
                    "but is still confined within the kidney. While it is bigger in size, "
                    "it has not spread to nearby lymph nodes or distant organs. "
                    "Treatment usually focuses on removing the tumor surgically."
                ),
                "Stage III": (
                    "Stage III CCRC means the cancer may have spread into nearby major "
                    "blood vessels or lymph nodes, but it is still relatively local. "
                    "This stage shows greater progression than Stage II and often requires "
                    "a combination of treatments, such as surgery and additional therapies."
                ),
                "Stage IV": (
                    "Stage IV CCRC is the most advanced stage, where the cancer has spread "
                    "beyond the kidney to distant lymph nodes or other organs (such as lungs, "
                    "bones, or liver). Treatment at this stage may include surgery, targeted "
                    "therapy, immunotherapy, or a combination, depending on the patient’s health."
                ),
            }

            #text = stage_descriptions.get(predicted_stage, "No description available.")

            # Result Card
            st.markdown(
                f"""
                <div style="background-color:#E8F6F3; padding:20px; border-radius:12px; border:2px solid #1ABC9C;">
                    <h3 style="color:#117A65; text-align:center;">Prediction Successful</h3>
                    <p style="text-align:center; font-size:18px;">
                        <b style="color:#117A65;">Predicted Tumor Stage for {name}:</b><br>
                        <span style="font-size:22px; color:#2E4053;">{predicted_stage}</span>
                        </p>
                         <p style="margin-top:10px; text-align:center; font-size:16px; color:#1C2833;">
                         </p>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            prediction_collection = db['predicted_stage']
            flat_features = {}
            for k, v in user_input.items():
                # Convert boolean-like numeric fields to True/False
                if isinstance(v, (int, np.integer)) and (v == 0 or v == 1):
                    flat_features[k] = bool(v)
                else:
                    flat_features[k] = v

            prediction_document = {
                "name": name,
                "predicted_stage": predicted_stage,
                **flat_features  # unpack dict so each feature is a column
            }
            prediction_collection.insert_one(prediction_document)
            client.close()

            st.success(f" Patient record for **{name}** has been saved to the database.")


# --- Saved Records Page ---
    elif page == "Saved Records":
        st.markdown("<h2 style='color:#2E86C1;'>Saved Patient Records</h2>", unsafe_allow_html=True)

        records = list(db['predicted_stage'].find({}, {"_id": 0}))
        client.close()
        if records:
            df = pd.DataFrame(records)

            # Convert boolean columns
            for col in df.columns:
                if df[col].dtype == 'bool':
                    df[col] = df[col].map({True: 'True', False: 'False'})

            # Convert encoded categorical columns back to words
            for feature, le in label_encoders.items():
                if feature in df.columns:
                    df[feature] = df[feature].map(lambda x: le.inverse_transform([int(x)])[0] if pd.notna(x) and str(x).isdigit() else x)

            st.dataframe(df, use_container_width=True)
        else:
            st.info("No records found yet. Run a prediction first!")


patient_data = _fetch_data()
preprocessed_data, label_encoders, stage_encoder= _preprocessing_data(patient_data)
model, X = _train_model(preprocessed_data)
_save_models(model, label_encoders, stage_encoder, list(X.columns))
model, label_encoders, stage_encoder, features = _load_models()
predict_interface(model, label_encoders, stage_encoder, features)
