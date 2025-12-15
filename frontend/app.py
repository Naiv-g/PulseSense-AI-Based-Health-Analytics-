import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"


def safe_rerun():
    """Helper to trigger a Streamlit rerun."""
    try:
        st.rerun()
    except Exception:
        # Fallback if rerun fails
        st.session_state.setdefault("_rerun", False)
        st.session_state["_rerun"] = not st.session_state["_rerun"]

def main():
    st.set_page_config(
        page_title="PulseSense - AI Health Analytics",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )
    
    # Enhanced Custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.3);
        margin: 1rem 0;
    }
    
    .symptom-tag {
        display: inline-block;
        background: #e0e7ff;
        color: #4f46e5;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox>div>div {
        border-radius: 10px;
    }
    
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content .block-container {
        color: white;
    }
    
    h1, h2, h3 {
        color: #1f2937;
    }
    
    .success-message {
        background: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fef3c7;
        color: #92400e;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .record-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .record-card:hover {
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏥 PulseSense</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">AI-Powered Health Analytics Platform</h3>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Enhanced Sidebar navigation
    if st.session_state.token:
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='color: white; margin-bottom: 0.5rem;'>🏥 PulseSense</h2>
                <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Health Analytics</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            user_name = st.session_state.user.get('full_name', 'User') if st.session_state.user else 'User'
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <p style='color: white; margin: 0; font-weight: 500;'>👤 {user_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown('<h3 style="color: white; font-size: 1.1rem; margin-bottom: 1rem;">Navigation</h3>', unsafe_allow_html=True)
            
            # Navigation with buttons - check state before rerunning
            nav_options = {
                "🏠 Dashboard": "Dashboard",
                "📋 Medical Records": "Medical Records",
                "🔍 Symptom Checker": "Symptom Checker",
                "💬 Health Chatbot": "Health Chatbot",
                "👤 Profile": "Profile"
            }
            
            for icon_text, page in nav_options.items():
                # Highlight current page
                button_style = ""
                if st.session_state.current_page == page:
                    button_style = "background: rgba(255,255,255,0.2);"
                
                if st.button(icon_text, use_container_width=True, key=f"nav_{page}", type="secondary"):
                    if st.session_state.current_page != page:
                        st.session_state.current_page = page
                        st.rerun()
            
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True, key="logout_btn", type="secondary"):
                st.session_state.token = None
                st.session_state.user = None
                st.session_state.current_page = "Dashboard"
                st.rerun()
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
    elif st.session_state.current_page == "Health Chatbot":
        show_health_chatbot()
    elif st.session_state.current_page == "Profile":
        show_profile()

def show_login_page():
    col1, col2, col3 = st.columns([1, 2.5, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <h2 style='color: #667eea; margin-bottom: 0.5rem;'>Welcome to PulseSense</h2>
            <p style='color: #6b7280;'>Sign in to access your health analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.form("login_form"):
                st.markdown("### Login to Your Account")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if login_user(username, password):
                        st.success("✅ Login successful! Redirecting...")
                        st.session_state.current_page = "Dashboard"
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
        
        with tab2:
            with st.form("register_form"):
                st.markdown("### Create New Account")
                col1, col2 = st.columns(2)
                
                with col1:
                    username = st.text_input("Username*", placeholder="Choose a username")
                    full_name = st.text_input("Full Name*", placeholder="Your full name")
                    email = st.text_input("Email*", placeholder="your.email@example.com")
                
                with col2:
                    date_of_birth = st.date_input("Date of Birth*", min_value=datetime(1900, 1, 1))
                    gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                    phone = st.text_input("Phone Number*", placeholder="+1234567890")
                
                password = st.text_input("Password*", type="password", placeholder="Create a strong password")
                confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Re-enter password")
                
                submit = st.form_submit_button("Register", use_container_width=True)
                
                if submit:
                    if password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif not all([username, full_name, email, password, phone]):
                        st.error("❌ Please fill all required fields")
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
                            st.success("✅ Registration successful! Please login.")
                        # Error message is already shown in register_user() function

def show_dashboard():
    st.markdown("### 📊 Health Dashboard")
    
    user_data = get_user_profile()
    medical_records = get_medical_records()
    
    if user_data:
        # Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">Total Records</div>
                <div style="font-size: 2rem; font-weight: 700;">{len(medical_records)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            age = calculate_age(user_data['date_of_birth'])
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">Age</div>
                <div style="font-size: 2rem; font-weight: 700;">{age}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">Gender</div>
                <div style="font-size: 2rem; font-weight: 700;">{user_data['gender'][0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">Status</div>
                <div style="font-size: 2rem; font-weight: 700;">✓</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Actions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #667eea;">👤 Personal Information</h3>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Name:** {user_data['full_name']}")
            st.write(f"**Email:** {user_data['email']}")
            st.write(f"**Phone:** {user_data['phone']}")
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #667eea;">⚡ Quick Actions</h3>
            </div>
            """, unsafe_allow_html=True)
            if st.button("➕ Add Medical Record", use_container_width=True, key="btn_add_record"):
                if st.session_state.current_page != "Medical Records":
                    st.session_state.current_page = "Medical Records"
                    st.rerun()
            if st.button("🔍 Check Symptoms", use_container_width=True, key="btn_check_symptoms"):
                if st.session_state.current_page != "Symptom Checker":
                    st.session_state.current_page = "Symptom Checker"
                    st.rerun()
        
        st.markdown("---")
        
        # Recent Medical Records
        st.markdown("### 📋 Recent Medical Records")
        if medical_records:
            for record in medical_records[:5]:
                st.markdown(f"""
                <div class="record-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: #667eea;">{record['record_type']}</h4>
                        <span style="color: #6b7280; font-size: 0.9rem;">{record['record_date']}</span>
                    </div>
                    <p style="color: #374151; margin: 0.5rem 0;"><strong>Description:</strong> {record['description']}</p>
                    {f"<p style='color: #374151; margin: 0.5rem 0;'><strong>Diagnosis:</strong> {record['diagnosis']}</p>" if record.get('diagnosis') else ""}
                    {f"<p style='color: #374151; margin: 0.5rem 0;'><strong>Treatment:</strong> {record['treatment']}</p>" if record.get('treatment') else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📝 No medical records found. Add your first record to get started!")

def show_medical_records():
    st.markdown("### 📋 Medical Records Management")
    
    # Track form submission counter to prevent duplicates
    if 'record_form_counter' not in st.session_state:
        st.session_state.record_form_counter = 0
    if 'pending_record_submission' not in st.session_state:
        st.session_state.pending_record_submission = None
    
    # Process pending submission if exists (prevents duplicate on rerun)
    if st.session_state.pending_record_submission:
        record_data = st.session_state.pending_record_submission
        st.session_state.pending_record_submission = None  # Clear immediately
        
        if add_medical_record(record_data):
            st.success("✅ Medical record added successfully!")
        else:
            st.error("❌ Failed to add medical record")
        st.rerun()
    
    # Add new record form
    with st.expander("➕ Add New Medical Record", expanded=True):
        form_key = f"add_medical_record_{st.session_state.record_form_counter}"
        with st.form(form_key, clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                record_type = st.selectbox("Record Type*", [
                    "Diagnosis", "Treatment", "Surgery", "Medication",
                    "Allergy", "Chronic Condition", "Other"
                ], key=f"type_{form_key}")
                record_date = st.date_input("Record Date*", key=f"date_{form_key}")
            
            with col2:
                diagnosis = st.text_input("Diagnosis (if any)", placeholder="Enter diagnosis if available", key=f"diagnosis_{form_key}")
                treatment = st.text_area("Treatment Details (if any)", placeholder="Enter treatment details", key=f"treatment_{form_key}")
            
            description = st.text_area("Description*", placeholder="Describe the medical record in detail...", height=100, key=f"desc_{form_key}")
            
            submitted = st.form_submit_button("💾 Add Medical Record", use_container_width=True)
            
            if submitted:
                if not description:
                    st.error("❌ Description is required")
                else:
                    record_data = {
                        "record_type": record_type,
                        "description": description,
                        "diagnosis": diagnosis or "",
                        "treatment": treatment or "",
                        "record_date": record_date.isoformat()
                    }
                    
                    # Store in pending and increment counter to create new form on rerun
                    st.session_state.pending_record_submission = record_data
                    st.session_state.record_form_counter += 1
                    st.rerun()
    
    st.markdown("---")
    
    # Display existing records
    st.markdown("### 📚 Your Medical Records")
    records = get_medical_records()
    
    if records:
        for idx, record in enumerate(records):
            record_id = record.get('id', idx)
            
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"""
                <div class="record-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: #667eea;">{record['record_type']}</h4>
                        <span style="color: #6b7280; font-size: 0.9rem;">📅 {record['record_date']}</span>
                    </div>
                    <p style="color: #374151; margin: 0.5rem 0;"><strong>Description:</strong> {record['description']}</p>
                    {f"<p style='color: #374151; margin: 0.5rem 0;'><strong>Diagnosis:</strong> {record['diagnosis']}</p>" if record.get('diagnosis') else ""}
                    {f"<p style='color: #374151; margin: 0.5rem 0;'><strong>Treatment:</strong> {record['treatment']}</p>" if record.get('treatment') else ""}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                if st.button("🗑️ Delete", key=f"delete_{record_id}", type="secondary", use_container_width=True):
                    if delete_medical_record(record_id):
                        st.success("✅ Record deleted successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete record")
    else:
        st.info("📝 No medical records found. Add your first record above!")

def show_symptom_checker():
    st.markdown("### 🔍 AI Symptom Checker")
    st.markdown("Select your symptoms below and get an AI-powered analysis of possible conditions.")
    
    symptoms = get_available_symptoms()
    
    if symptoms:
        selected_symptoms = st.multiselect(
            "Select your symptoms:",
            options=symptoms,
            help="Choose all symptoms you're currently experiencing",
            placeholder="Start typing or select from the list..."
        )
        
        if selected_symptoms:
            st.markdown("### Selected Symptoms")
            symptom_tags = " ".join([f'<span class="symptom-tag">{symptom}</span>' for symptom in selected_symptoms])
            st.markdown(f'<div style="margin: 1rem 0;">{symptom_tags}</div>', unsafe_allow_html=True)
        
        if st.button("🤖 Analyze Symptoms with AI", use_container_width=True, type="primary"):
            if selected_symptoms:
                if not st.session_state.token:
                    st.warning("⚠️ Please log in to use the AI Symptom Checker.")
                else:
                    with st.spinner("🔬 Analyzing symptoms with AI... This may take a few seconds."):
                        prediction = predict_disease_from_symptoms(selected_symptoms)
                        
                        if prediction:
                            st.markdown("---")
                            st.markdown("### 📊 Analysis Results")
                            
                            # Main prediction card
                            st.markdown(f"""
                            <div class="prediction-card">
                                <div style="text-align: center; margin-bottom: 1.5rem;">
                                    <h2 style="color: white; margin: 0; font-size: 2rem;">{prediction['disease']}</h2>
                                    <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;">Confidence: {prediction['confidence']:.1%}</p>
                                </div>
                                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                                    <p style="color: white; margin: 0;"><strong>Recommended Specialist:</strong> {prediction['recommended_specialist']}</p>
                                    <p style="color: white; margin: 0.5rem 0 0 0;"><strong>Description:</strong> {prediction['description']}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show input symptoms
                            st.markdown("### 🔍 Your Symptoms")
                            symptom_display = " ".join([f'<span class="symptom-tag">{symptom}</span>' for symptom in selected_symptoms])
                            st.markdown(f'<div style="margin: 1rem 0;">{symptom_display}</div>', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                <div class="info-card">
                                    <h4 style="margin-top: 0; color: #667eea;">🔬 Suggested Diagnostic Tests</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                for test in prediction['suggested_tests']:
                                    st.markdown(f"✅ {test}")
                            
                            with col2:
                                st.markdown("""
                                <div class="info-card">
                                    <h4 style="margin-top: 0; color: #667eea;">💡 Precautions & Recommendations</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                for precaution in prediction['precautions']:
                                    st.markdown(f"📌 {precaution}")
                            
                            # Help needed section
                            st.markdown("### 🆘 Help & Next Steps")
                            st.markdown("""
                            <div class="info-card">
                                <h4 style="margin-top: 0; color: #667eea;">What to do next:</h4>
                                <ul style="color: #374151; line-height: 1.8;">
                                    <li><strong>Consult a healthcare professional:</strong> This analysis is for informational purposes only and should not replace professional medical advice.</li>
                                    <li><strong>Monitor your symptoms:</strong> Keep track of any changes or worsening of symptoms.</li>
                                    <li><strong>Seek immediate care if:</strong> Symptoms worsen, you experience severe pain, difficulty breathing, or other emergency symptoms.</li>
                                    <li><strong>Prepare for your appointment:</strong> Bring a list of your symptoms, medications, and any relevant medical history.</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to get prediction. Please try again.")
            else:
                st.warning("⚠️ Please select at least one symptom to analyze.")
    else:
        st.error("❌ Could not load symptoms list. Please try again later.")

def show_profile():
    st.markdown("### 👤 User Profile")
    
    user_data = get_user_profile()
    
    if user_data:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">👤</div>
                <h3 style="color: white; margin: 0;">{}</h3>
            </div>
            """.format(user_data['full_name']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #667eea;">Personal Information</h3>
            </div>
            """, unsafe_allow_html=True)
            
            profile_data = [
                ("👤 Username", user_data['username']),
                ("📧 Email", user_data['email']),
                ("📝 Full Name", user_data['full_name']),
                ("📅 Date of Birth", user_data['date_of_birth']),
                ("⚧️ Gender", user_data['gender']),
                ("📱 Phone", user_data['phone'])
            ]
            
            for label, value in profile_data:
                st.markdown(f"**{label}:** {value}")
    else:
        st.error("❌ Could not load profile data")

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
        response = requests.post(f"{BACKEND_URL}/register", json=user_data, timeout=10)
        if response.status_code == 200:
            return True
        else:
            # Try to get detailed error message from response
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", f"Registration failed (Status: {response.status_code})")
                st.error(f"❌ {error_detail}")
                # Also log to console for debugging
                print(f"Registration failed: {error_detail}")
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")
            except Exception as parse_error:
                st.error(f"❌ Registration failed. Status code: {response.status_code}")
                st.error(f"Response: {response.text[:200]}")  # Show first 200 chars of response
                print(f"Failed to parse error response: {parse_error}")
                print(f"Raw response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to server. Please ensure the backend is running at http://localhost:8000")
        return False
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
        return False
    except Exception as e:
        st.error(f"❌ Registration error: {str(e)}")
        print(f"Registration exception: {type(e).__name__}: {str(e)}")
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

def delete_medical_record(record_id):
    """Delete a medical record by ID."""
    if not st.session_state.token:
        return False
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.delete(f"{BACKEND_URL}/medical-records/{record_id}", headers=headers)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Delete record error: {e}")
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
            json={"symptoms": symptoms, "user_id": 1},
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

def show_health_chatbot():
    st.markdown("### 💬 Health Chatbot - Ask About Diseases")
    st.markdown("Ask me anything about diseases, symptoms, treatments, and health information!")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        # Load chat history from backend
        try:
            history = get_chatbot_history()
            if history:
                # Reverse to show oldest first
                for conv in reversed(history):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": conv["user_message"],
                        "timestamp": conv.get("created_at", "")
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": conv["bot_response"],
                        "timestamp": conv.get("created_at", "")
                    })
        except Exception as e:
            st.warning(f"Could not load chat history: {e}")
    
    # Initialize pending message tracking
    if 'pending_chat_message' not in st.session_state:
        st.session_state.pending_chat_message = None
    if 'chat_form_counter' not in st.session_state:
        st.session_state.chat_form_counter = 0
    
    # Process pending message if exists (prevents duplicate on rerun)
    if st.session_state.pending_chat_message:
        pending_msg = st.session_state.pending_chat_message
        st.session_state.pending_chat_message = None  # Clear immediately
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": pending_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            response = send_chatbot_query(pending_msg)
            
            if response:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Sorry, I encountered an error. Please try again.",
                    "timestamp": datetime.now().isoformat()
                })
        
        st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; 
                            margin-left: 20%; text-align: right;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">You</div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f3f4f6; color: #374151; padding: 1rem; border-radius: 10px; 
                            margin-bottom: 1rem; margin-right: 20%; border-left: 4px solid #667eea;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: #667eea;">🤖 Health Assistant</div>
                    <div style="white-space: pre-wrap;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    # Example questions
    st.markdown("**💡 Example Questions:**")
    col1, col2, col3 = st.columns(3)
    example_questions = [
        "What are the symptoms of influenza?",
        "Tell me about migraines",
        "How to prevent common cold?"
    ]
    
    for idx, question in enumerate(example_questions):
        with [col1, col2, col3][idx]:
            if st.button(f"💬 {question[:30]}...", use_container_width=True, key=f"example_{idx}"):
                if not st.session_state.pending_chat_message:
                    st.session_state.pending_chat_message = question
                    st.rerun()
    
    # Chat input form
    form_key = f"chat_form_{st.session_state.chat_form_counter}"
    with st.form(form_key, clear_on_submit=True):
        user_input = st.text_area(
            "Type your question here...",
            height=100,
            placeholder="e.g., What are the symptoms of diabetes?",
            key=f"chat_input_{form_key}"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("📤 Send", use_container_width=True)
        
        if submit_button and user_input.strip():
            if not st.session_state.pending_chat_message:
                # Store in pending and increment counter to create new form on rerun
                st.session_state.pending_chat_message = user_input.strip()
                st.session_state.chat_form_counter += 1
                st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

def send_chatbot_query(message):
    """Send a message to the chatbot API."""
    if not st.session_state.token:
        return None
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.post(
            f"{BACKEND_URL}/chatbot/query",
            json={"message": message},
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Chatbot error: {e}")
        return None

def get_chatbot_history():
    """Get chatbot conversation history from backend."""
    if not st.session_state.token:
        return []
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{BACKEND_URL}/chatbot/history", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        return []

if __name__ == "__main__":
    main()
