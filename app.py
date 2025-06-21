import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import joblib

# Load saved model and preprocessors
model = joblib.load('dt_model.pkl')           # ⬅️ Decision Tree model(best accuracy)
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('scaler.pkl')


# Page configuration
st.set_page_config(
    page_title="PCOS Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #FFD8D8;
    }
    .sidebar .sidebar-content {
        background-color: #eaddff;
    }
    .section {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .title {
        color: #6a0dad;
    }
    .tab-content {
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.title("PCOS Detection System")
    st.image('icon2.jpg', width=200)  # Replace with actual image
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

# Title
st.title("🩺 PCOS Risk Prediction")
st.markdown("Complete this form to assess your PCOS risk based on health and lifestyle factors")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Input", "Analysis", "Results", "Resources"])

# Tab 1: Data Input
with tab1:
    st.header("Patient Information Form")
    with st.form("pcos_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            age = st.number_input("Age (years)", min_value=10, max_value=60, value=25)
            height = st.number_input("Height (cm)", min_value=140, max_value=200, value=160)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=60)
            
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            sleeping_hours = st.number_input("Sleeping Hours per day", min_value=0, max_value=24, value=7)
            working_hours = st.number_input("Working Hours per day", min_value=0, max_value=24, value=8)
        
        with col2:
            st.subheader("Lifestyle Factors")
            smoke = st.radio("Do you smoke?", ["No", "Yes"])
            alcohol = st.radio("Do you consume alcohol?", ["No", "Yes"])
            junk_food = st.radio("Do you eat junk food frequently?", ["No", "Yes"])
            exercise = st.radio("Do you exercise regularly?", ["No", "Yes"])
            married = st.radio("Are you married?", ["No", "Yes"])
        
        st.subheader("Medical History")
        col3, col4 = st.columns(2)
        
        with col3:
            diabetic = st.radio("Do you have diabetes?", ["No", "Yes"])
            hypertension = st.radio("Do you have hypertension?", ["No", "Yes"])
            thyroid = st.radio("Do you have thyroid issues?", ["No", "Yes"])
            centrally_obese = st.radio("Are you centrally obese?", ["No", "Yes"])
        
        with col4:
            mood_swings = st.radio("Do you experience mood swings?", ["No", "Yes"])
            stress = st.radio("Do you experience stress/anxiety/depression?", ["No", "Yes"])
            stress_during_periods = st.radio("Do you experience stress during periods?", ["No", "Yes"])
            stress_affects_periods = st.radio("Does stress affect your periods?", ["No", "Yes"])
        
        st.subheader("Menstrual Health")
        col5, col6 = st.columns(2)
        
        with col5:
            first_period = st.number_input("Age at first period", min_value=8, max_value=20, value=12)
            period_flow = st.selectbox("Period flow", ["Light", "Normal", "Heavy"])
            cycle = st.number_input("Cycle length (days)", min_value=10, max_value=90, value=28)
        
        with col6:
            period_length = st.number_input("Period length (days)", min_value=1, max_value=15, value=5)
            period_intervals = st.number_input("Days between periods", min_value=10, max_value=90, value=28)
        
        st.subheader("Physical Symptoms")
        col7, col8 = st.columns(2)
        
        with col7:
            hair_loss = st.radio("Do you experience hair loss?", ["No", "Yes"])
            thinner_hair = st.radio("Has your hair become thinner?", ["No", "Yes"])
            unwanted_hair = st.radio("Do you have unwanted hair growth?", ["No", "Yes"])
        
        with col8:
            weight_gain = st.radio("Have you experienced weight gain?", ["No", "Yes"])
            weight_loss = st.radio("Have you experienced weight loss?", ["No", "Yes"])
            acne = st.radio("Do you have acne?", ["No", "Yes"])
            skin_darkening = st.radio("Have you experienced skin darkening?", ["No", "Yes"])
        
        # Submit button
        submitted = st.form_submit_button("Submit Information")

# Tab 2: Analysis
with tab2:
    st.header("Data Analysis")
    
    if 'submitted' in locals() and submitted:
        # Prepare the input data with all Yes/No converted to 1/0
        input_data = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Sleeping Hours': sleeping_hours,
            'Working Hours': working_hours,
            'Smoke': 1 if smoke == "Yes" else 0,
            'Alcohol': 1 if alcohol == "Yes" else 0,
            'Diabetic': 1 if diabetic == "Yes" else 0,
            'Hypertension': 1 if hypertension == "Yes" else 0,
            'Junk Food': 1 if junk_food == "Yes" else 0,
            'Mood Swings': 1 if mood_swings == "Yes" else 0,
            'Thyroid': 1 if thyroid == "Yes" else 0,
            'Stress/Anxiety/Depression': 1 if stress == "Yes" else 0,
            'Exercise': 1 if exercise == "Yes" else 0,
            'First Period': first_period,
            'Period Flow': period_flow,
            'Cycle': cycle,
            'Period Length': period_length,
            'Stress During Periods': 1 if stress_during_periods == "Yes" else 0,
            'Stress Affects Periods': 1 if stress_affects_periods == "Yes" else 0,
            'Hair Loss': 1 if hair_loss == "Yes" else 0,
            'Thinner Hair': 1 if thinner_hair == "Yes" else 0,
            'Unwanted Hair Growth': 1 if unwanted_hair == "Yes" else 0,
            'Weight Gain': 1 if weight_gain == "Yes" else 0,
            'Weight Loss': 1 if weight_loss == "Yes" else 0,
            'Acne': 1 if acne == "Yes" else 0,
            'Skin Darkening': 1 if skin_darkening == "Yes" else 0,
            'Centrally Obese': 1 if centrally_obese == "Yes" else 0,
            'Married': 1 if married == "Yes" else 0,
            'Period Intervals': period_intervals
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Display analysis
        st.subheader("Input Data Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))
        
    else:
        st.warning("Please submit your information in the 'Data Input' tab first")

# Tab 3: Results
with tab3:
  with tab3:
    st.header("PCOS Risk Prediction Results")
    
    if 'submitted' in locals() and submitted:
        import pandas as pd
        from Detection import predict_pcos

        # Get prediction
        result = predict_pcos(input_df)

        # Show result
        st.success("PCOS Detected" if result == 1 else "No PCOS Detected")

        st.markdown("""
        <div class="section">
            <h4>Key Contributing Factors:</h4>
            <ul>
                <li>Irregular menstrual cycle</li>
                <li>Elevated BMI (Overweight)</li>
                <li>Presence of unwanted hair growth</li>
                <li>History of acne</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please submit your information in the 'Data Input' tab first")

# Tab 4: Resources
with tab4:
    st.header("PCOS Resources and Information")
    
    st.subheader("About PCOS")
    st.markdown("""
    Polycystic ovary syndrome (PCOS) is a hormonal disorder common among women of reproductive age. 
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



# Add matplotlib import at the top if not already present
