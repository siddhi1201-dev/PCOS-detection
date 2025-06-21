import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

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
        background-color: #f9f5ff;
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
            period_flow = st.selectbox("Period flow", ["Mild", "Normal", "Heavy"])
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
        # Prepare the input data
        input_data = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'BMI': bmi,
            'Sleeping Hours': sleeping_hours,
            'Working Hours': working_hours,
            'Smoke': smoke,
            'Alcohol': alcohol,
            'Diabetic': diabetic,
            'Hypertension': hypertension,
            'Junk Food': junk_food,
            'Mood Swings': mood_swings,
            'Thyroid': thyroid,
            'Stress/Anxiety/Depression': stress,
            'Exercise': exercise,
            'First Period': first_period,
            'Period Flow': period_flow,
            'Cycle': cycle,
            'Period Length': period_length,
            'Stress During Periods': stress_during_periods,
            'Stress Affects Periods': stress_affects_periods,
            'Hair Loss': hair_loss,
            'Thinner Hair': thinner_hair,
            'Unwanted Hair Growth': unwanted_hair,
            'Weight Gain': weight_gain,
            'Weight Loss': weight_loss,
            'Acne': acne,
            'Skin Darkening': skin_darkening,
            'Centrally Obese': centrally_obese,
            'Married': married,
            'Period Intervals': period_intervals
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Display analysis
        st.subheader("Input Data Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))
        
        # Visualizations
        st.subheader("Key Health Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # BMI Analysis
            fig, ax = plt.subplots()
            ax.bar(['Your BMI'], [bmi], color='skyblue')
            ax.axhline(y=18.5, color='red', linestyle='--', label='Underweight')
            ax.axhline(y=24.9, color='green', linestyle='--', label='Healthy')
            ax.axhline(y=29.9, color='orange', linestyle='--', label='Overweight')
            ax.axhline(y=34.9, color='red', linestyle='--', label='Obese')
            ax.set_ylabel('BMI')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            # Lifestyle Factors
            lifestyle_data = {
                'Smoking': 1 if smoke == "Yes" else 0,
                'Alcohol': 1 if alcohol == "Yes" else 0,
                'Exercise': 1 if exercise == "Yes" else 0,
                'Junk Food': 1 if junk_food == "Yes" else 0
            }
            fig, ax = plt.subplots()
            ax.bar(lifestyle_data.keys(), lifestyle_data.values(), color=['red', 'orange', 'green', 'purple'])
            ax.set_ylabel('Yes=1, No=0')
            ax.set_title('Lifestyle Factors')
            st.pyplot(fig)
    else:
        st.warning("Please submit your information in the 'Data Input' tab first")

# Tab 3: Results
with tab3:
    st.header("PCOS Risk Prediction Results")
    
    if 'submitted' in locals() and submitted:
        # Placeholder for actual model prediction
        st.info("Model prediction would appear here after implementation")
        
        # Mock results section (to be replaced with actual model)
        st.markdown("""
        <div class="section">
            <h4>Next Steps for Implementation:</h4>
            <ol>
                <li>Preprocess input data (one-hot encoding, scaling)</li>
                <li>Load trained machine learning model</li>
                <li>Make prediction on processed data</li>
                <li>Display results with risk percentage and recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for what the results would look like
        st.subheader("Example Output (Mock Data)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("PCOS Risk Score", "72%", delta="High Risk", delta_color="inverse")
            st.progress(72)
            
        with col2:
            st.metric("Recommended Next Steps", "Consult Gynecologist")
        
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

# Sidebar with info
with st.sidebar:
    st.title("PCOS Detection System")
    st.image("D:\MachineLearning\PCOS-detection\icon2.jpg", width=200)  # Replace with actual image
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

# Add matplotlib import at the top if not already present
