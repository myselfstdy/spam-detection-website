import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Page configuration
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)


# Load dataset
@st.cache_data
def load_data():
    try:
        # Works for comma-separated and tab-separated CSV files
        data = pd.read_csv("spam.csv", sep=None, engine="python")
    except Exception:
        data = pd.read_csv("spam.csv", encoding="latin-1")

    # Handle the common SMS Spam Collection column names
    if "v1" in data.columns and "v2" in data.columns:
        data = data[["v1", "v2"]]
        data.columns = ["label", "message"]

    elif "label" in data.columns and "message" in data.columns:
        data = data[["label", "message"]]

    else:
        st.error("Dataset must contain label and message columns.")
        st.stop()

    data = data.dropna()
    return data


# Train model
@st.cache_resource
def train_model(data):
    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


# Load and train
data = load_data()
model, accuracy = train_model(data)


# Website design
st.title("📩 Spam Message Detector")
st.write("Enter an SMS or email message to check whether it is spam or genuine.")

st.metric(
    label="Model Accuracy",
    value=f"{accuracy * 100:.2f}%"
)

st.divider()

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize. Click here to claim.",
    height=150
)

if st.button("🔍 Check Message", use_container_width=True):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        if prediction.lower() in ["spam", "junk"]:
            st.error("🚨 This message is SPAM.")
            st.write("Be careful. Do not click unknown links or share personal information.")
        else:
            st.success("✅ This message is NOT SPAM.")
            st.write("The model classified this message as genuine.")

st.divider()

st.caption("Built using Python, Scikit-learn, CountVectorizer and Multinomial Naive Bayes.")