import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="PulseSense - AI Health Analytics",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 PulseSense</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">AI-Powered Health Analytics Platform</h3>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Sidebar navigation
    if st.session_state.token:
        with st.sidebar:
            st.image("https://via.placeholder.com/150x50/2E86AB/FFFFFF?text=PulseSense", width=150)
            st.write(f"Welcome, {st.session_state.user.get('full_name', 'User')}!")
            
            # Navigation using buttons instead of radio
            st.subheader("Navigation")
            if st.button("🏠 Dashboard", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.experimental_rerun()

            st.divider()

            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.token = None
                st.session_state.user = None
                st.session_state.current_page = "Dashboard"
                st.experimental_rerun()
    else:
        st.session_state.current_page = "Login"
    
    # Page routing
    if st.session_state.current_page == "Login":
        show_login_page()
    elif st.session_state.current_page == "Dashboard":
        show_dashboard()
    elif st.session_state.current_page == "Medical Records":
        show_medical_records()
    elif st.session_state.current_page == "Symptom Checker":
        show_symptom_checker()
    elif st.session_state.current_page == "Profile":
        show_profile()

def show_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if login_user(username, password):
                        st.success("Login successful!")
                        st.session_state.current_page = "Dashboard"
                        st.experimental_rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            st.subheader("Create New Account")
            
            with st.form("register_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    username = st.text_input("Username*")
                    full_name = st.text_input("Full Name*")
                    email = st.text_input("Email*")
                
                with col2:
                    date_of_birth = st.date_input("Date of Birth*", min_value=datetime(1900, 1, 1))
                    gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                    phone = st.text_input("Phone Number*")
                
                password = st.text_input("Password*", type="password")
                confirm_password = st.text_input("Confirm Password*", type="password")
                
                submit = st.form_submit_button("Register")
                
                if submit:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    elif not all([username, full_name, email, password, phone]):
                        st.error("Please fill all required fields")
                    else:
                        if register_user({
                            "username": username,
                            "email": email,
                            "password": password,
                            "full_name": full_name,
                            "date_of_birth": date_of_birth.isoformat(),
                            "gender": gender,
                            "phone": phone
                        }):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Registration failed. Username or email may already exist.")

def show_dashboard():
    st.subheader("Health Dashboard")
    
    # Get user data
    user_data = get_user_profile()
    medical_records = get_medical_records()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("👤 Personal Info")
        st.write(f"**Name:** {user_data['full_name']}")
        st.write(f"**Email:** {user_data['email']}")
        st.write(f"**Phone:** {user_data['phone']}")
    
    with col2:
        st.info("📊 Health Stats")
        st.write(f"**Total Records:** {len(medical_records)}")
        st.write(f"**Age:** {calculate_age(user_data['date_of_birth'])}")
        st.write(f"**Gender:** {user_data['gender']}")
    
    with col3:
        st.info("⚡ Quick Actions")
        if st.button("➕ Add Medical Record", use_container_width=True):
            st.session_state.current_page = "Medical Records"
            st.experimental_rerun()
        if st.button("🔍 Check Symptoms", use_container_width=True):
            st.session_state.current_page = "Symptom Checker"
            st.experimental_rerun()
    
    # Recent medical records
    st.subheader("Recent Medical Records")
    if medical_records:
        for record in medical_records[:5]:
            with st.expander(f"{record['record_date']} - {record['record_type']}"):
                st.write(f"**Description:** {record['description']}")
                if record['diagnosis']:
                    st.write(f"**Diagnosis:** {record['diagnosis']}")
                if record['treatment']:
                    st.write(f"**Treatment:** {record['treatment']}")
    else:
        st.info("No medical records found. Add your first record!")

def show_medical_records():
    st.subheader("Medical Records Management")

    # Initialize session state for submission
    if 'submitting_record' not in st.session_state:
        st.session_state.submitting_record = False

    # Add new record form
    with st.form("add_medical_record"):
        col1, col2 = st.columns(2)

        with col1:
            record_type = st.selectbox("Record Type*", [
                "Diagnosis", "Treatment", "Surgery", "Medication",
                "Allergy", "Chronic Condition", "Other"
            ])
            record_date = st.date_input("Record Date*")

        with col2:
            diagnosis = st.text_input("Diagnosis (if any)")
            treatment = st.text_area("Treatment Details (if any)")

        description = st.text_area("Description*")

        submitted = st.form_submit_button("Add Medical Record", disabled=st.session_state.submitting_record)

        if submitted:
            st.session_state.submitting_record = True
            if not description:
                st.error("Description is required")
                st.session_state.submitting_record = False
            else:
                if add_medical_record({
                    "record_type": record_type,
                    "description": description,
                    "diagnosis": diagnosis,
                    "treatment": treatment,
                    "record_date": record_date.isoformat()
                }):
                    st.success("Medical record added successfully!")
                    st.session_state.submitting_record = False
                    st.experimental_rerun()
                else:
                    st.error("Failed to add medical record")
                    st.session_state.submitting_record = False
    
    # Display existing records
    st.subheader("Your Medical Records")
    records = get_medical_records()
    
    if records:
        for record in records:
            with st.expander(f"{record['record_date']} - {record['record_type']} - {record['description'][:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {record['record_type']}")
                    st.write(f"**Date:** {record['record_date']}")
                    st.write(f"**Description:** {record['description']}")
                
                with col2:
                    if record['diagnosis']:
                        st.write(f"**Diagnosis:** {record['diagnosis']}")
                    if record['treatment']:
                        st.write(f"**Treatment:** {record['treatment']}")
    else:
        st.info("No medical records found.")

def show_symptom_checker():
    st.subheader("AI Symptom Checker")
    
    # Get available symptoms
    symptoms = get_available_symptoms()
    
    if symptoms:
        selected_symptoms = st.multiselect(
            "Select your symptoms:",
            options=symptoms,
            help="Choose all symptoms you're currently experiencing"
        )
        
        if st.button("Analyze Symptoms", key="analyze_symptoms"):
            if selected_symptoms:
                with st.spinner("Analyzing symptoms with AI..."):
                    prediction = predict_disease_from_symptoms(selected_symptoms)
                    
                    if prediction:
                        st.markdown("### Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Possible Condition", prediction['disease'])
                            st.metric("Confidence Level", f"{prediction['confidence']:.1%}")
                        
                        with col2:
                            st.write(f"**Recommended Specialist:** {prediction['recommended_specialist']}")
                            st.write(f"**Description:** {prediction['description']}")
                        
                        st.subheader("Suggested Diagnostic Tests")
                        for test in prediction['suggested_tests']:
                            st.write(f"• {test}")
                        
                        st.subheader("Precautions & Recommendations")
                        for precaution in prediction['precautions']:
                            st.write(f"• {precaution}")
                        
                        # Emergency warning for serious symptoms
                        serious_symptoms = {'chest pain', 'shortness of breath', 'severe headache', 'uncontrolled bleeding'}
                        if any(symptom.lower() in serious_symptoms for symptom in selected_symptoms):
                            st.error("🚨 **Seek immediate medical attention for these symptoms!**")
                    else:
                        st.error("Failed to get prediction. Please try again.")
            else:
                st.warning("Please select at least one symptom.")
    else:
        st.error("Could not load symptoms list. Please try again later.")

def show_profile():
    st.subheader("User Profile")
    
    user_data = get_user_profile()
    
    if user_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Username:**", user_data['username'])
            st.write("**Email:**", user_data['email'])
            st.write("**Full Name:**", user_data['full_name'])
        
        with col2:
            st.write("**Date of Birth:**", user_data['date_of_birth'])
            st.write("**Gender:**", user_data['gender'])
            st.write("**Phone:**", user_data['phone'])
        
        # Profile update form could be added here
    else:
        st.error("Could not load profile data")

# API helper functions
def login_user(username, password):
    try:
        response = requests.post(f"{BACKEND_URL}/login", json={
            "username": username,
            "password": password
        })
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data['access_token']
            st.session_state.user = get_user_profile()
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {e}")
        return False

def register_user(user_data):
    try:
        response = requests.post(f"{BACKEND_URL}/register", json=user_data)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False

def get_user_profile():
    if not st.session_state.token:
        return None
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{BACKEND_URL}/profile", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Profile error: {e}")
        return None

def get_medical_records():
    if not st.session_state.token:
        return []
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{BACKEND_URL}/medical-records", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Medical records error: {e}")
        return []

def add_medical_record(record_data):
    if not st.session_state.token:
        return False
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.post(f"{BACKEND_URL}/medical-records", json=record_data, headers=headers)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Add record error: {e}")
        return False

def get_available_symptoms():
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
        response = requests.get(f"{BACKEND_URL}/symptoms", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Symptoms error: {e}")
        return []

def predict_disease_from_symptoms(symptoms):
    if not st.session_state.token:
        return None
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.post(
            f"{BACKEND_URL}/predict-disease", 
            json={"symptoms": symptoms, "user_id": 1},  # user_id will be handled by backend
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def calculate_age(birthdate_str):
    try:
        birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except:
        return "Unknown"

if __name__ == "__main__":
    main()