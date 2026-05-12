import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Healthcare Predictive Analytics",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------
# LOGIN SYSTEM
# ---------------------------------------------------

st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username == "admin" and password == "admin123":

    st.success("Login Successful")

    # ---------------------------------------------------
    # TITLE
    # ---------------------------------------------------

    st.markdown(
        """
        <h1 style='text-align:center;color:#0E76A8;'>
        🏥 Healthcare Predictive Analytics System
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style='text-align:center;'>
        Diabetes Prediction using Machine Learning
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.write("---")

    # ---------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------

    data = pd.read_csv("diabetes.csv")

    # ---------------------------------------------------
    # SIDEBAR MENU
    # ---------------------------------------------------

    menu = st.sidebar.selectbox(
        "Menu",
        ["Home", "Dataset", "Prediction", "Analytics"]
    )

    # ---------------------------------------------------
    # HOME PAGE
    # ---------------------------------------------------

    if menu == "Home":

        st.subheader("Welcome")

        st.write(
            """
            This project predicts diabetes using
            Machine Learning algorithms.
            """
        )

        st.image(
            "https://cdn.pixabay.com/photo/2017/08/06/00/17/people-2583442_1280.jpg",
            use_container_width=True
        )

    # ---------------------------------------------------
    # DATASET PAGE
    # ---------------------------------------------------

    elif menu == "Dataset":

        st.subheader("Dataset Preview")

        st.dataframe(data)

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", data.shape[0])

        with col2:
            st.metric("Columns", data.shape[1])

        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())

    # ---------------------------------------------------
    # ANALYTICS PAGE
    # ---------------------------------------------------

    elif menu == "Analytics":

        st.subheader("Analytics Dashboard")

        diabetes_cases = data["Outcome"].value_counts()

        st.write(diabetes_cases)

        fig, ax = plt.subplots()

        ax.bar(
            ["No Diabetes", "Diabetes"],
            diabetes_cases
        )

        st.pyplot(fig)

        st.subheader("Glucose Distribution")

        fig2, ax2 = plt.subplots()

        ax2.hist(data["Glucose"])

        st.pyplot(fig2)

    # ---------------------------------------------------
    # PREDICTION PAGE
    # ---------------------------------------------------

    elif menu == "Prediction":

        st.subheader("Patient Details")

        # Features and target
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Algorithm selection
        algorithm = st.selectbox(
            "Select Algorithm",
            [
                "Logistic Regression",
                "Random Forest"
            ]
        )

        if algorithm == "Logistic Regression":

            model = LogisticRegression(max_iter=1000)

        else:

            model = RandomForestClassifier()

        # Train model
        model.fit(X_train, y_train)

        # Accuracy
        prediction_test = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            prediction_test
        )

        st.success(
            f"Model Accuracy: {accuracy*100:.2f}%"
        )

        # Input fields
        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                0,
                20,
                1
            )

            glucose = st.number_input(
                "Glucose",
                0,
                300,
                100
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                0,
                200,
                70
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                0,
                100,
                20
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                0,
                900,
                80
            )

            bmi = st.number_input(
                "BMI",
                0.0,
                70.0,
                25.0
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                0.0,
                3.0,
                0.5
            )

            age = st.number_input(
                "Age",
                1,
                120,
                30
            )

        # Prediction
        if st.button("Predict Disease"):

            patient_data = np.array([[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]])

            prediction = model.predict(patient_data)

            probability = model.predict_proba(
                patient_data
            )

            st.subheader("Prediction Result")

            if prediction[0] == 1:

                st.error(
                    "⚠ Patient may have Diabetes"
                )

                st.write(
                    f"Risk Probability: {probability[0][1]*100:.2f}%"
                )

            else:

                st.success(
                    "✅ Patient does not have Diabetes"
                )

                st.write(
                    f"Safety Probability: {probability[0][0]*100:.2f}%"
                )

        st.write("---")

        st.markdown(
            """
            <center>
            <h4>
            Developed using Machine Learning
            </h4>
            </center>
            """,
            unsafe_allow_html=True
        )

else:

    st.error("Invalid Username or Password")

    st.info(
        "Use Username: admin | Password: admin123"
    )