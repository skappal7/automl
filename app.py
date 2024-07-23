import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
import requests
from PIL import Image
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import os
import joblib
import shap
import time
from scipy.stats import randint, uniform

# Set page config
st.set_page_config(page_title="Advanced AutoML App", layout="wide")

# Security measures
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)

# In a real application, you would store these securely, not in the code
USERS = {
    "username": hash_password("password"),
    "admin": hash_password("admin_password")
}

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to encode image to base64
def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to add background image
def add_bg_from_url(image_url):
    img = load_image_from_url(image_url)
    encoded_img = img_to_base64(img)
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_img});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image
add_bg_from_url("https://www.scientific-computing.com/sites/default/files/styles/content_banner/public/content/news-story/lead-image/QuardiaShutterstock.png?h=9c610b38&itok=K4ik1nBn")

def login_page():
    st.markdown(
        """
        <style>
        .login-container {
            background-color: rgba(30, 61, 89, 0.7);
            padding: 30px;
            border-radius: 10px;
            backdrop-filter: blur(5px);
            color: white;
        }
        .login-title {
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #FFA500;
            color: #1E3D59;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">Login</p>', unsafe_allow_html=True)
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in USERS and verify_password(USERS[username], password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main_app():
    st.title("Advanced AutoML Application")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Data Upload", "Data Preprocessing", "Feature Engineering", "Model Selection", "Training", "Evaluation", "Model Explainability"])
    
    if page == "Data Upload":
        data_upload()
    elif page == "Data Preprocessing":
        data_preprocessing()
    elif page == "Feature Engineering":
        feature_engineering()
    elif page == "Model Selection":
        model_selection()
    elif page == "Training":
        model_training()
    elif page == "Evaluation":
        model_evaluation()
    elif page == "Model Explainability":
        model_explainability()

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()

def data_upload():
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success("File uploaded successfully!")
            st.write(data.head())
            st.write(f"Shape of the data: {data.shape}")
            
            # Basic data validation
            if data.shape[0] < 100:
                st.warning("The dataset is quite small. Results may not be reliable.")
            if data.isnull().sum().sum() > 0:
                st.warning("The dataset contains missing values. Consider handling them in the preprocessing step.")
            
            # Display data info
            buffer = BytesIO()
            data.info(buf=buffer)
            s = buffer.getvalue().decode()
            st.text(s)
            
            # Display summary statistics
            st.write(data.describe())
            
            # Display correlation heatmap
            corr = data.corr()
            fig = px.imshow(corr, title="Correlation Heatmap")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

def data_preprocessing():
    st.header("Data Preprocessing")
    if 'data' not in st.session_state:
        st.warning("Please upload data first.")
        return
    
    data = st.session_state.data
    
    # Display column information
    st.subheader("Column Information")
    st.write(data.dtypes)
    
    # Select target variable
    target = st.selectbox("Select target variable", data.columns)
    
    # Select features
    features = st.multiselect("Select features", [col for col in data.columns if col != target])
    
    if len(features) > 0:
        X = data[features]
        y = data[target]
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            impute_method = st.radio("Choose method to handle missing values:", ["Drop", "Mean", "Median", "Most Frequent"])
            if impute_method == "Drop":
                X = X.dropna()
                y = y[X.index]
            else:
                for column in X.columns:
                    if X[column].isnull().sum() > 0:
                        if impute_method == "Mean":
                            X[column].fillna(X[column].mean(), inplace=True)
                        elif impute_method == "Median":
                            X[column].fillna(X[column].median(), inplace=True)
                        else:
                            X[column].fillna(X[column].mode()[0], inplace=True)
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit and transform the data
        X_preprocessed = preprocessor.fit_transform(X)
        
        # Split the data
        test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=test_size, random_state=42)
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.target = target
        st.session_state.features = features
        st.session_state.preprocessor = preprocessor
        
        st.success("Data preprocessing completed!")

  def feature_engineering():
    st.header("Feature Engineering")
    if 'X_train' not in st.session_state:
        st.warning("Please complete data preprocessing first.")
        return
    
    # Feature selection
    st.subheader("Feature Selection")
    k = st.slider("Select number of top features", 1, len(st.session_state.features), len(st.session_state.features))
    
    if st.session_state.problem_type == "Classification":
        selector = SelectKBest(f_classif, k=k)
    else:
        selector = SelectKBest(f_regression, k=k)
    
    X_new = selector.fit_transform(st.session_state.X_train, st.session_state.y_train)
    
    # Get selected feature names
    selected_features = [st.session_state.features[i] for i in selector.get_support(indices=True)]
    
    st.write("Selected features:", selected_features)
    
    # Update session state
    st.session_state.X_train = X_new
    st.session_state.X_test = selector.transform(st.session_state.X_test)
    st.session_state.features = selected_features
    
    st.success("Feature engineering completed!")

def model_selection():
    st.header("Model Selection")
    if 'X_train' not in st.session_state:
        st.warning("Please complete data preprocessing first.")
        return
    
    problem_type = st.radio("Select problem type", ["Classification", "Regression"])
    st.session_state.problem_type = problem_type
    
    if problem_type == "Classification":
        model_options = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": xgb.XGBClassifier(),
            "LightGBM": LGBMClassifier()
        }
    else:
        model_options = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "Support Vector Machine": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": xgb.XGBRegressor(),
            "LightGBM": LGBMRegressor(),
            "Lasso": Lasso(),
            "Ridge": Ridge()
        }
    
    selected_model = st.selectbox("Select a model", list(model_options.keys()))
    st.session_state.model = model_options[selected_model]
    st.session_state.model_name = selected_model
    
    st.success(f"Selected model: {selected_model}")

def model_training():
    st.header("Model Training")
    if 'model' not in st.session_state:
        st.warning("Please select a model first.")
        return
    
    # Hyperparameter tuning
    st.subheader("Hyperparameter Tuning")
    use_tuning = st.checkbox("Perform hyperparameter tuning")
    
    if use_tuning:
        n_iter = st.slider("Number of iterations for random search", 10, 100, 20)
        
        if st.session_state.model_name == "Random Forest":
            param_dist = {
                'n_estimators': randint(10, 200),
                'max_depth': [None] + list(range(10, 31, 10)),
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5)
            }
        elif st.session_state.model_name == "Support Vector Machine":
            param_dist = {
                'C': uniform(0.1, 100),
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'] + list(uniform(0.1, 1).rvs(10))
            }
        elif st.session_state.model_name in ["Gradient Boosting", "XGBoost", "LightGBM"]:
            param_dist = {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3)
            }
        else:
            param_dist = {}  # For models without hyperparameter tuning
        
        if param_dist:
            model = RandomizedSearchCV(st.session_state.model, param_dist, n_iter=n_iter, cv=5, n_jobs=-1, random_state=42)
        else:
            model = st.session_state.model
    else:
        model = st.session_state.model
    
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            try:
                start_time = time.time()
                model.fit(st.session_state.X_train, st.session_state.y_train)
                end_time = time.time()
                training_time = end_time - start_time
                
                st.session_state.trained_model = model
                st.success(f"Model training completed in {training_time:.2f} seconds!")
                
                if use_tuning and param_dist:
                    st.write("Best parameters:", model.best_params_)
                
                # Save the model
                model_filename = f"{st.session_state.model_name}_{time.strftime('%Y%m%d-%H%M%S')}.joblib"
                joblib.dump(model, model_filename)
                st.success(f"Model saved as {model_filename}")
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")

def model_evaluation():
    st.header("Model Evaluation")
    if 'trained_model' not in st.session_state:
        st.warning("Please train the model first.")
        return
    
    model = st.session_state.trained_model
    y_pred = model.predict(st.session_state.X_test)
    
    if st.session_state.problem_type == "Classification":
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        
        # Classification report
        report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.write(df_report)
        
        # Confusion matrix
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual")
        st.plotly_chart(fig)
        
        # ROC curve (for binary classification)
        if len(np.unique(st.session_state.y_test)) == 2:
            from sklearn.metrics import roc_curve, auc
            y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})')
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig.update_xaxes(title="False Positive Rate")
            fig.update_yaxes(title="True Positive Rate")
            st.plotly_chart(fig)
        
    else:
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared Score: {r2:.2f}")
        
        # Actual vs Predicted plot
        fig = px.scatter(x=st.session_state.y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_shape(type="line", line=dict(dash="dash"), x0=st.session_state.y_test.min(), y0=st.session_state.y_test.min(),
                      x1=st.session_state.y_test.max(), y1=st.session_state.y_test.max())
        fig.update_layout(title="Actual vs Predicted")
        st.plotly_chart(fig)
        
        # Residual plot
        residuals = st.session_state.y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals)
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(title="Residual Plot", xaxis_title="Predicted", yaxis_title="Residuals")
        st.plotly_chart(fig)
    
    # Feature importance (for models that support it)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'feature': st.session_state.features, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        fig = px.bar(feature_importance, x='feature', y='importance', title="Feature Importance")
        st.plotly_chart(fig)
    elif st.session_state.model_name in ["Logistic Regression", "Linear Regression", "Lasso", "Ridge"]:
        coefficients = pd.DataFrame({'feature': st.session_state.features, 'coefficient': model.coef_[0] if st.session_state.model_name == "Logistic Regression" else model.coef_})
        coefficients = coefficients.sort_values('coefficient', key=abs, ascending=False)
        fig = px.bar(coefficients, x='feature', y='coefficient', title="Feature Coefficients")
        st.plotly_chart(fig)

def model_explainability():
    st.header("Model Explainability")
    if 'trained_model' not in st.session_state:
        st.warning("Please train the model first.")
        return
    
    model = st.session_state.trained_model
    
    # SHAP values
    st.subheader("SHAP (SHapley Additive exPlanations) Values")
    
    try:
        explainer = shap.Explainer(model, st.session_state.X_train)
        shap_values = explainer(st.session_state.X_test)
        
        st.write("SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, st.session_state.X_test, plot_type="bar", show=False)
        st.pyplot(fig)
        
        st.write("SHAP Dependence Plots")
        for feature in st.session_state.features:
            fig, ax = plt.subplots()
            shap.dependence_plot(feature, shap_values.values, st.session_state.X_test, show=False)
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred while generating SHAP values: {str(e)}")
    
    # Partial Dependence Plots
    st.subheader("Partial Dependence Plots")
    
    try:
        from sklearn.inspection import plot_partial_dependence
        
        features_to_plot = st.multiselect("Select features for Partial Dependence Plots", st.session_state.features)
        if features_to_plot:
            fig, ax = plt.subplots()
            plot_partial_dependence(model, st.session_state.X_train, features_to_plot, grid_resolution=50, ax=ax)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while generating Partial Dependence Plots: {str(e)}")

  def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
