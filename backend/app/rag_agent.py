from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Any
import json
import re

class HealthcareRAGAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.setup_llm()
        self.setup_knowledge_base()
    
    def setup_llm(self):
        """Initialize a simpler LLM for medical queries"""
        try:
            
            print("Loading medical LLM...")
            
            
            model_name = "microsoft/BioGPT-Large"
            
            try:
                tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
                model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/BioGPT-Large",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            except:
                print("Falling back to smaller model...")
                model_name = "google/flan-t5-base"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("LLM loaded successfully")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.setup_fallback_llm()
    
    def setup_fallback_llm(self):
        """Setup a very basic fallback"""
        try:
            self.llm = HuggingFacePipeline.from_model_id(
                model_id="google/flan-t5-small",
                task="text2text-generation",
                model_kwargs={"torch_dtype": torch.float16}
            )
        except Exception as e:
            print(f"Error loading fallback LLM: {e}")
            # Create a mock LLM for development
            self.llm = None
    
    def setup_knowledge_base(self):
        """Setup medical knowledge base with comprehensive data"""
        medical_knowledge = [
            Document(
                page_content="""
                Condition: Common Cold
                Symptoms: cough, sore throat, runny nose, sneezing, mild fever, headache, body aches
                Description: Viral infection of the upper respiratory tract
                Treatment: rest, fluids, over-the-counter cold medicine, pain relievers
                Duration: 7-10 days
                Specialist: General Physician
                Precautions: Wash hands frequently, avoid close contact with sick people
                """,
                metadata={"condition": "Common Cold", "type": "respiratory"}
            ),
            Document(
                page_content="""
                Condition: Influenza (Flu)
                Symptoms: high fever, body aches, fatigue, headache, chills, cough, sore throat
                Description: Viral infection affecting respiratory system, more severe than common cold
                Treatment: antiviral medication, rest, fluids, pain relievers
                Prevention: annual flu vaccine
                Specialist: General Physician
                Precautions: Get flu shot, avoid crowded places during flu season
                """,
                metadata={"condition": "Influenza", "type": "respiratory"}
            ),
            Document(
                page_content="""
                Condition: Migraine
                Symptoms: throbbing headache, nausea, sensitivity to light, sensitivity to sound, dizziness
                Description: Neurological condition characterized by intense headaches
                Treatment: pain relievers, triptans, preventive medications, rest in dark room
                Triggers: stress, certain foods, hormonal changes, lack of sleep
                Specialist: Neurologist
                Precautions: Identify and avoid triggers, maintain regular sleep schedule
                """,
                metadata={"condition": "Migraine", "type": "neurological"}
            ),
            Document(
                page_content="""
                Condition: Gastroenteritis
                Symptoms: diarrhea, vomiting, abdominal pain, nausea, fever, loss of appetite
                Description: Inflammation of stomach and intestines, often called stomach flu
                Treatment: hydration, bland diet, rest, anti-nausea medication
                Prevention: proper hand hygiene, avoid contaminated food/water
                Specialist: Gastroenterologist
                Precautions: Practice good hygiene, drink clean water, eat well-cooked food
                """,
                metadata={"condition": "Gastroenteritis", "type": "digestive"}
            ),
            Document(
                page_content="""
                Condition: Hypertension
                Symptoms: often asymptomatic, may include headaches, shortness of breath, nosebleeds
                Description: High blood pressure that can lead to serious health issues
                Treatment: lifestyle changes, medication, regular monitoring
                Risk factors: age, family history, obesity, high salt intake, stress
                Specialist: Cardiologist
                Precautions: Regular exercise, low-salt diet, maintain healthy weight
                """,
                metadata={"condition": "Hypertension", "type": "cardiovascular"}
            ),
            Document(
                page_content="""
                Condition: Asthma
                Symptoms: shortness of breath, wheezing, chest tightness, coughing
                Description: Chronic inflammatory disease of the airways
                Treatment: inhalers, corticosteroids, avoiding triggers
                Triggers: allergens, cold air, exercise, smoke
                Specialist: Pulmonologist
                Precautions: Avoid triggers, use inhaler as prescribed, have action plan
                """,
                metadata={"condition": "Asthma", "type": "respiratory"}
            )
        ]
        
        try:
            self.vector_store = FAISS.from_documents(medical_knowledge, self.embeddings)
            
            if self.llm:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
            print("Knowledge base setup completed")
        except Exception as e:
            print(f"Error setting up knowledge base: {e}")
    
    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Predict disease based on symptoms using RAG"""
        if not symptoms:
            return self.get_fallback_response([])
        
        symptom_text = ", ".join(symptoms)
        
        # If LLM is not available, use rule-based approach
        if not self.llm or not self.qa_chain:
            return self.rule_based_prediction(symptoms)
        
        query = f"""
        Based on these symptoms: {symptom_text}
        
        Provide a JSON response with:
        - disease: most likely condition
        - confidence: number between 0 and 1
        - description: brief explanation
        - recommended_specialist: which doctor to see
        - suggested_tests: list of diagnostic tests
        - precautions: list of precautions
        
        Be concise and medical accurate.
        """
        
        try:
            result = self.qa_chain({"query": query})
            response_text = result["result"]
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, parse the text response
                return self.parse_text_response(response_text, symptoms)
                
        except Exception as e:
            print(f"Error in disease prediction: {e}")
            return self.rule_based_prediction(symptoms)
    
    def parse_text_response(self, response: str, symptoms: List[str]) -> Dict[str, Any]:
        """Parse text response into structured format"""
        # Simple text parsing as fallback
        return {
            "disease": self.extract_condition(response) or "Medical Consultation Recommended",
            "confidence": 0.7,
            "description": self.extract_description(response) or "Based on symptoms, professional evaluation is advised",
            "recommended_specialist": self.extract_specialist(response) or "General Physician",
            "suggested_tests": ["Physical Examination", "Basic Blood Tests"],
            "precautions": ["Rest", "Stay Hydrated", "Monitor Symptoms", "Seek Medical Advice"]
        }
    
    def extract_condition(self, text: str) -> str:
        """Extract condition from text response"""
        conditions = ["Common Cold", "Influenza", "Migraine", "Gastroenteritis", "Hypertension", "Asthma"]
        for condition in conditions:
            if condition.lower() in text.lower():
                return condition
        return ""
    
    def extract_description(self, text: str) -> str:
        """Extract description from text response"""
        # Simple extraction - look for sentences after keywords
        lines = text.split('.')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['description', 'condition', 'is a']):
                return line.strip()
        return text[:100] + "..." if len(text) > 100 else text
    
    def extract_specialist(self, text: str) -> str:
        """Extract specialist from text response"""
        specialists = ["General Physician", "Neurologist", "Gastroenterologist", "Cardiologist", "Pulmonologist"]
        for specialist in specialists:
            if specialist.lower() in text.lower():
                return specialist
        return "General Physician"
    
    def rule_based_prediction(self, symptoms: List[str]) -> Dict[str, Any]:
        """Rule-based fallback prediction"""
        symptom_set = set(s.lower() for s in symptoms)
        
        # Define symptom patterns for common conditions
        conditions = {
            "Common Cold": {
                "symptoms": {"cough", "sore throat", "runny nose", "sneezing"},
                "specialist": "General Physician",
                "description": "Viral upper respiratory infection",
                "tests": ["Physical Exam", "Throat Swab"],
                "precautions": ["Rest", "Hydration", "Over-the-counter cold medicine"]
            },
            "Influenza": {
                "symptoms": {"fever", "body aches", "fatigue", "chills", "headache"},
                "specialist": "General Physician", 
                "description": "Viral respiratory infection",
                "tests": ["Flu Test", "Physical Exam"],
                "precautions": ["Rest", "Fluids", "Antiviral medication if early"]
            },
            "Migraine": {
                "symptoms": {"headache", "nausea", "dizziness", "sensitivity to light"},
                "specialist": "Neurologist",
                "description": "Neurological headache disorder",
                "tests": ["Neurological Exam", "MRI if severe"],
                "precautions": ["Rest in dark room", "Avoid triggers", "Hydration"]
            },
            "Gastroenteritis": {
                "symptoms": {"nausea", "vomiting", "diarrhea", "abdominal pain"},
                "specialist": "Gastroenterologist",
                "description": "Stomach and intestinal inflammation",
                "tests": ["Stool Test", "Physical Exam"],
                "precautions": ["Hydration", "Bland diet", "Rest"]
            }
        }
        
        # Find best matching condition
        best_match = None
        max_matches = 0
        
        for condition, data in conditions.items():
            matches = len(symptom_set.intersection(data["symptoms"]))
            if matches > max_matches:
                max_matches = matches
                best_match = condition
        
        if best_match and max_matches >= 2:
            data = conditions[best_match]
            confidence = min(0.3 + (max_matches * 0.15), 0.9)  # Dynamic confidence
            return {
                "disease": best_match,
                "confidence": confidence,
                "description": data["description"],
                "recommended_specialist": data["specialist"],
                "suggested_tests": data["tests"],
                "precautions": data["precautions"]
            }
        else:
            return self.get_fallback_response(symptoms)
    
    def get_fallback_response(self, symptoms: List[str]) -> Dict[str, Any]:
        """Provide generic fallback response"""
        return {
            "disease": "Consult Healthcare Provider",
            "confidence": 0.3,
            "description": "Symptoms require professional medical evaluation for accurate diagnosis",
            "recommended_specialist": "General Physician",
            "suggested_tests": ["Physical examination", "Basic blood tests"],
            "precautions": ["Monitor symptoms", "Seek medical attention if worsening", "Rest and hydrate"]
        }

# Global RAG agent instance
rag_agent = HealthcareRAGAgent()