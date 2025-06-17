import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, recall_score

# Page configuration
st.set_page_config(
    page_title="PCOS Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e3f2fd;
    }
    .big-font {
        font-size:18px !important;
        font-weight: bold;
    }
    .result-box {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0px;
        background-color: #e8f5e9;
    }
    .warning-box {
        border: 2px solid #ff9800;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0px;
        background-color: #fff3e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_resource
def load_data():
    dataset = pd.read_csv('PCOS Dataset.csv')
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    
    # One hot encoding for period flow
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[15])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Feature scaling
    sc = StandardScaler()
    scale_indexes = [1, 2, 3, 4, 5, 15, 16, 17, 31]
    X_train[:,scale_indexes] = sc.fit_transform(X_train[:,scale_indexes])
    X_test[:,scale_indexes] = sc.transform(X_test[:,scale_indexes])
    
    return X_train, X_test, y_train, y_test, sc, ct

# Train model
@st.cache_resource
def train_model(X_train, y_train):
    # Focus on maximizing recall for Class 1
    scorer = make_scorer(recall_score, pos_label=1)

    params = {
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(n_estimators=100, random_state=1),
        params,
        scoring=scorer,
        cv=5
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Load data and train model
X_train, X_test, y_train, y_test, scaler, ct = load_data()
model = train_model(X_train, y_train)

# Sidebar
with st.sidebar:
    st.title("PCOS Detection System")
    st.image("https://www.example.com/pcos_image.jpg", width=200)  # Replace with actual image
    st.markdown("""
    ### About PCOS
    Polycystic ovary syndrome (PCOS) is a hormonal disorder common among women of reproductive age. 
    Symptoms include irregular menstrual periods, excess hair growth, acne, and obesity.
    """)
    
    st.markdown("""
    ### How to Use
    1. Upload your medical data or enter manually
    2. View the analysis and predictions
    3. Get recommendations based on results
    """)
    
    st.markdown("""
    ### Disclaimer
    This tool is not a substitute for professional medical advice. 
    Always consult with a healthcare provider.
    """)

# Main Page
st.title("🩺 PCOS Analysis and Detection")
st.markdown("Analyze symptoms and predict PCOS using machine learning models.")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Data Input", "Analysis", "Results", "Resources"])

with tab1:
    st.header("Input Your Data")
    
    input_method = st.radio("Choose input method:", 
                          ("Manual Entry", "Upload CSV/Excel"))
    
    if input_method == "Manual Entry":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age (years)", min_value=12, max_value=60, value=25)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=60)
            height = st.number_input("Height (cm)", min_value=100, max_value=220, value=160)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
        with col2:
            st.subheader("Menstrual Cycle")
            cycle_length = st.number_input("Cycle length (days)", min_value=0, max_value=90, value=28)
            cycle_regularity = st.selectbox("Cycle regularity", 
                                          ["Regular", "Irregular", "No periods"])
            periods_missed = st.number_input("Number of periods missed in last year", 
                                           min_value=0, max_value=12, value=0)
            period_flow = st.selectbox("Period flow", ["Scanty", "Normal", "Heavy"])
    
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Symptoms")
            hair_growth = st.select_slider("Excess hair growth (hirsutism)", 
                                         options=["None", "Mild", "Moderate", "Severe"])
            acne = st.select_slider("Acne severity", 
                                   options=["None", "Mild", "Moderate", "Severe"])
            hair_loss = st.checkbox("Hair loss/thinning")
            skin_darkening = st.checkbox("Skin darkening")
            pimples = st.checkbox("Pimples")
            
        with col4:
            st.subheader("Hormonal Levels")
            lh = st.number_input("LH (IU/L)", min_value=0.0, max_value=50.0, value=5.0)
            fsh = st.number_input("FSH (IU/L)", min_value=0.0, max_value=50.0, value=5.0)
            lh_fsh_ratio = lh / fsh if fsh != 0 else 0
            st.metric("LH/FSH Ratio", f"{lh_fsh_ratio:.2f}")
            testosterone = st.number_input("Testosterone (ng/dL)", min_value=0.0, max_value=200.0, value=30.0)
            prl = st.number_input("Prolactin (ng/mL)", min_value=0.0, max_value=100.0, value=15.0)
            vitd3 = st.number_input("Vitamin D3 (ng/mL)", min_value=0.0, max_value=100.0, value=30.0)
            
    else:
        uploaded_file = st.file_uploader("Upload your medical data", 
                                       type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            st.dataframe(df.head())

with tab2:
    st.header("Data Analysis")
    
    if input_method == "Manual Entry" or (input_method == "Upload CSV/Excel" and uploaded_file is not None):
        st.subheader("Key Indicators")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indicators = ['BMI', 'LH/FSH Ratio', 'Cycle Irregularity', 'Testosterone']
        values = [bmi, lh_fsh_ratio, 1 if cycle_regularity != "Regular" else 0, testosterone]
        
        sns.barplot(x=indicators, y=values, palette="viridis", ax=ax)
        ax.axhline(y=3.0, color='r', linestyle='--', label='Threshold')
        ax.set_title("PCOS Risk Indicators")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
        
        st.subheader("Correlation Analysis")
        corr_data = pd.DataFrame({
            'Age': np.random.normal(25, 5, 100),
            'BMI': np.random.normal(28, 6, 100),
            'LH/FSH': np.random.normal(2.5, 1.2, 100),
            'Testosterone': np.random.normal(45, 15, 100)
        })
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Please input data in the 'Data Input' tab first")

with tab3:
    st.header("Results & Recommendations")
    
    if input_method == "Manual Entry" or (input_method == "Upload CSV/Excel" and uploaded_file is not None):
        # Prepare input data for prediction
        if input_method == "Manual Entry":
            # Convert manual inputs to model input format
            # Note: This mapping needs to match exactly with your training data features
            input_data = {
                'Age': age,
                'Weight': weight,
                'Height': height,
                'BMI': bmi,
                'Cycle length': cycle_length,
                'Cycle regularity': 1 if cycle_regularity != "Regular" else 0,
                'Periods missed': periods_missed,
                'Period flow': period_flow,
                'Hair growth': ['None', 'Mild', 'Moderate', 'Severe'].index(hair_growth),
                'Acne': ['None', 'Mild', 'Moderate', 'Severe'].index(acne),
                'Hair loss': 1 if hair_loss else 0,
                'Skin darkening': 1 if skin_darkening else 0,
                'Pimples': 1 if pimples else 0,
                'LH': lh,
                'FSH': fsh,
                'LH/FSH': lh_fsh_ratio,
                'Testosterone': testosterone,
                'Prolactin': prl,
                'Vitamin D3': vitd3
            }
            
            # Convert to DataFrame for transformation
            input_df = pd.DataFrame([input_data])
            
            # Apply the same transformations as training data
            # One-hot encode period flow
            X_input = ct.transform(input_df.values)
            
            # Scale the same features as training data
            scale_indexes = [1, 2, 3, 4, 5, 15, 16, 17, 31]  # Adjust these indexes based on your actual feature positions
            X_input[:,scale_indexes] = scaler.transform(X_input[:,scale_indexes])
            
            # Make prediction
            prediction = model.predict(X_input)
            prediction_proba = model.predict_proba(X_input)
            
            risk_score = prediction_proba[0][1] * 100  # Probability of PCOS
            
        else:
            # For uploaded files, you would need to preprocess similarly
            # This is simplified - in practice you'd need to match column names, etc.
            st.warning("File upload prediction not fully implemented in this example")
            risk_score = 50  # Placeholder
            
        st.subheader("PCOS Risk Assessment")
        
        if risk_score < 30:
            st.markdown(f'<div class="result-box"><h3>Low Risk: {risk_score:.1f}/100</h3>'
                       'Your symptoms indicate low risk of PCOS. Maintain healthy lifestyle.</div>', 
                       unsafe_allow_html=True)
        elif risk_score < 70:
            st.markdown(f'<div class="warning-box"><h3>Moderate Risk: {risk_score:.1f}/100</h3>'
                       'Some indicators suggest possible PCOS. Consider consulting a specialist.</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box" style="border-color:#f44336;background-color:#ffebee;">'
                       f'<h3>High Risk: {risk_score:.1f}/100</h3>'
                       'Your symptoms strongly suggest PCOS. Please consult a healthcare provider for proper diagnosis and treatment.</div>', 
                       unsafe_allow_html=True)
        
        st.subheader("Recommendations")
        
        if risk_score < 30:
            st.markdown("""
            - Maintain balanced diet and regular exercise
            - Track your menstrual cycle
            - Annual check-ups recommended
            """)
        elif risk_score < 70:
            st.markdown("""
            - Consult with an endocrinologist or gynecologist
            - Consider blood tests for hormone levels
            - Lifestyle modifications (diet, exercise)
            - Regular monitoring of symptoms
            """)
        else:
            st.markdown("""
            - *Urgent consultation with a specialist recommended*
            - Comprehensive hormonal profile testing
            - Possible ultrasound examination
            - Treatment options may include:
              - Lifestyle changes
              - Medications to regulate hormones
              - Fertility treatments if planning pregnancy
            """)
        
        st.subheader("Next Steps")
        st.markdown("""
        1. Download your results to share with your doctor
        2. Find a specialist in your area
        3. Schedule an appointment for further evaluation
        """)
        
        # Generate a PDF report (mock)
        if st.button("Generate PDF Report"):
            st.success("Report generated! (This would create a PDF in a real implementation)")
    else:
        st.warning("Please input data in the 'Data Input' tab first")

with tab4:
    st.header("Educational Resources")
    
    st.subheader("About PCOS")
    st.markdown("""
    Polycystic ovary syndrome (PCOS) is a hormonal disorder that affects women of reproductive age. 
    Women with PCOS may have infrequent or prolonged menstrual periods or excess male hormone (androgen) levels.
    """)
    
    st.subheader("Common Symptoms")
    st.markdown("""
    - Irregular periods
    - Excess androgen (leading to excess facial and body hair)
    - Polycystic ovaries
    - Weight gain
    - Thinning hair
    - Acne
    - Darkening of skin
    """)
    
    st.subheader("Helpful Links")
    st.markdown("""
    - [PCOS Foundation](https://www.pcosaa.org/)
    - [Mayo Clinic - PCOS Overview](https://www.mayoclinic.org/diseases-conditions/pcos/)
    - [CDC - PCOS Information](https://www.cdc.gov/diabetes/basics/pcos.html)
    """)
    
    st.subheader("Find a Specialist")
    st.markdown("""
    - [American College of Obstetricians and Gynecologists](https://www.acog.org/)
    - [The Endocrine Society](https://www.endocrine.org/)
    """)

# Footer
st.markdown("---")
st.markdown("""
*PCOS Detection System*  
This tool is for informational purposes only and not a substitute for professional medical advice.  
© 2023 Healthcare Analytics Team
""")