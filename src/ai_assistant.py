"""
AI Assistant Module for CKD Prediction System

This module provides AI-powered explanations and insights for:
- What CKD is and its implications
- How predictions are made
- Risk level interpretations
- Natural language explanations
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class CKDAssistant:
    """
    AI Assistant for explaining CKD predictions and providing health insights.
    
    Provides:
    - Disease information
    - Prediction explanations
    - Risk interpretations
    - Actionable recommendations
    """
    
    def __init__(self):
        """Initialize the AI assistant with knowledge base."""
        self.ckd_info = self._load_ckd_knowledge()
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.7,
            'high': 1.0
        }
    
    def _load_ckd_knowledge(self) -> Dict:
        """Load CKD knowledge base."""
        return {
            'definition': """
Chronic Kidney Disease (CKD) is a long-term condition characterized by gradual loss of 
kidney function over time. The kidneys filter waste and excess fluids from the blood, 
which are then excreted in urine. When chronic kidney disease reaches an advanced stage, 
dangerous levels of fluid, electrolytes, and wastes can build up in the body.
            """.strip(),
            
            'stages': {
                'Stage 1': 'Kidney damage with normal kidney function (GFR ≥90)',
                'Stage 2': 'Kidney damage with mild loss of function (GFR 60-89)',
                'Stage 3a': 'Mild to moderate loss of function (GFR 45-59)',
                'Stage 3b': 'Moderate to severe loss of function (GFR 30-44)',
                'Stage 4': 'Severe loss of function (GFR 15-29)',
                'Stage 5': 'Kidney failure (GFR <15)'
            },
            
            'risk_factors': [
                'Diabetes (Type 1 or Type 2)',
                'High blood pressure (Hypertension)',
                'Heart disease',
                'Family history of kidney disease',
                'Obesity',
                'Older age',
                'Smoking',
                'Abnormal kidney structure',
                'Frequent use of medications that can damage kidneys'
            ],
            
            'symptoms': [
                'Fatigue and weakness',
                'Difficulty concentrating',
                'Decreased appetite',
                'Trouble sleeping',
                'Muscle cramps at night',
                'Swollen feet and ankles',
                'Dry, itchy skin',
                'Frequent urination, especially at night',
                'Blood in urine',
                'Foamy urine'
            ],
            
            'prevention': [
                'Control blood sugar if diabetic',
                'Maintain healthy blood pressure',
                'Eat a healthy, low-sodium diet',
                'Exercise regularly',
                'Maintain a healthy weight',
                'Avoid smoking',
                'Limit alcohol consumption',
                'Stay hydrated',
                'Avoid overuse of pain medications',
                'Get regular kidney function tests'
            ]
        }
    
    def explain_ckd(self) -> str:
        """
        Provide comprehensive explanation of CKD.
        
        Returns:
            Formatted string explaining CKD
        """
        explanation = """
╔══════════════════════════════════════════════════════════════════════╗
║           UNDERSTANDING CHRONIC KIDNEY DISEASE (CKD)                 ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT IS CKD?
━━━━━━━━━━━━
{definition}

CKD STAGES:
━━━━━━━━━━━
{stages}

COMMON RISK FACTORS:
━━━━━━━━━━━━━━━━━━━━
{risk_factors}

WARNING SYMPTOMS:
━━━━━━━━━━━━━━━━━
{symptoms}

PREVENTION TIPS:
━━━━━━━━━━━━━━━━
{prevention}
        """.format(
            definition=self.ckd_info['definition'],
            stages='\n'.join([f"  • {k}: {v}" for k, v in self.ckd_info['stages'].items()]),
            risk_factors='\n'.join([f"  • {rf}" for rf in self.ckd_info['risk_factors']]),
            symptoms='\n'.join([f"  • {s}" for s in self.ckd_info['symptoms']]),
            prevention='\n'.join([f"  • {p}" for p in self.ckd_info['prevention']])
        )
        
        return explanation
    
    def explain_prediction_methodology(self) -> str:
        """
        Explain how the prediction system works.
        
        Returns:
            Formatted string explaining the methodology
        """
        return """
╔══════════════════════════════════════════════════════════════════════╗
║               HOW OUR PREDICTION SYSTEM WORKS                        ║
╚══════════════════════════════════════════════════════════════════════╝

DATA COLLECTION:
━━━━━━━━━━━━━━━━
Our system analyzes multiple health indicators including:
  • Demographic information (age, gender)
  • Blood pressure measurements (systolic and diastolic)
  • Blood test results (glucose, creatinine, BUN, hemoglobin, albumin)
  • Urine analysis (specific gravity, red blood cells, pus cells)
  • Medical history (hypertension, diabetes, heart disease)
  • Lifestyle factors (smoking, BMI)
  • Family history of kidney disease

MACHINE LEARNING MODELS:
━━━━━━━━━━━━━━━━━━━━━━━━
We employ multiple advanced ML algorithms:

  1. CLASSIFICATION MODELS (CKD: Yes/No)
     • Logistic Regression - Baseline interpretable model
     • Random Forest - Ensemble of decision trees
     • XGBoost - Gradient boosting with high accuracy
     • Neural Network (MLP) - Deep learning approach

  2. REGRESSION MODELS (Kidney Function Score)
     • Linear Regression - Baseline prediction
     • Gradient Boosting - Advanced ensemble method
     • Neural Network (MLP) - Non-linear pattern recognition

PREDICTION PROCESS:
━━━━━━━━━━━━━━━━━━━
  1. Patient data is preprocessed and normalized
  2. Features are encoded and scaled
  3. Multiple models analyze the data
  4. Predictions are aggregated for robustness
  5. Risk scores and explanations are generated

MODEL VALIDATION:
━━━━━━━━━━━━━━━━━
Our models are validated using:
  • Train-test split (80-20)
  • Cross-validation (5-fold)
  • Multiple metrics (Accuracy, F1, ROC-AUC, RMSE, R²)

EXPLAINABILITY:
━━━━━━━━━━━━━━━
We use SHAP (SHapley Additive exPlanations) to explain:
  • Which factors contributed most to the prediction
  • How each factor influenced the risk score
  • Personalized recommendations based on individual factors
        """
    
    def interpret_risk_level(self, probability: float) -> Dict:
        """
        Interpret the risk probability and provide detailed insights.
        
        Args:
            probability: CKD probability (0-1)
            
        Returns:
            Dictionary with risk interpretation
        """
        if probability < self.risk_thresholds['low']:
            level = 'LOW'
            color = 'green'
            description = """
Your risk of developing Chronic Kidney Disease is currently LOW. This is encouraging, 
but maintaining kidney health should still be a priority. Continue with healthy habits 
and regular check-ups.
            """.strip()
            recommendations = [
                "Continue regular health check-ups",
                "Maintain a balanced, low-sodium diet",
                "Stay physically active",
                "Keep blood pressure and blood sugar in healthy ranges",
                "Stay hydrated with adequate water intake"
            ]
        elif probability < self.risk_thresholds['moderate']:
            level = 'MODERATE'
            color = 'yellow'
            description = """
Your risk of developing Chronic Kidney Disease is MODERATE. This suggests some risk 
factors are present that warrant attention. It's important to work with healthcare 
providers to address modifiable risk factors.
            """.strip()
            recommendations = [
                "Schedule a consultation with your healthcare provider",
                "Get comprehensive kidney function tests",
                "If diabetic, optimize blood sugar control",
                "Monitor blood pressure regularly",
                "Review current medications with your doctor",
                "Consider dietary modifications (low sodium, moderate protein)",
                "Increase physical activity if sedentary"
            ]
        else:
            level = 'HIGH'
            color = 'red'
            description = """
Your risk of developing Chronic Kidney Disease is HIGH. This indicates significant 
risk factors are present. Immediate medical attention and lifestyle modifications 
are strongly recommended.
            """.strip()
            recommendations = [
                "URGENT: Consult a nephrologist (kidney specialist)",
                "Get comprehensive kidney function tests immediately",
                "Strictly control blood pressure and blood sugar",
                "Review all medications for kidney impact",
                "Follow a kidney-friendly diet (DASH diet recommended)",
                "Quit smoking if applicable",
                "Avoid NSAIDs and other nephrotoxic substances",
                "Monitor kidney function regularly"
            ]
        
        return {
            'level': level,
            'color': color,
            'probability': probability,
            'percentage': f"{probability * 100:.1f}%",
            'description': description,
            'recommendations': recommendations
        }
    
    def interpret_kidney_function_score(self, score: float) -> Dict:
        """
        Interpret the kidney function score.
        
        Args:
            score: Kidney function score (typically 0-100)
            
        Returns:
            Dictionary with interpretation
        """
        if score >= 90:
            stage = 'Normal/Stage 1'
            status = 'NORMAL'
            color = 'green'
            description = "Kidney function is normal or near-normal."
        elif score >= 60:
            stage = 'Stage 2 (Mild)'
            status = 'MILDLY REDUCED'
            color = 'lightgreen'
            description = "Mildly decreased kidney function. Usually no symptoms."
        elif score >= 45:
            stage = 'Stage 3a (Mild-Moderate)'
            status = 'MILD-MODERATELY REDUCED'
            color = 'yellow'
            description = "Mild to moderate decrease in kidney function."
        elif score >= 30:
            stage = 'Stage 3b (Moderate-Severe)'
            status = 'MODERATELY-SEVERELY REDUCED'
            color = 'orange'
            description = "Moderate to severe decrease in kidney function."
        elif score >= 15:
            stage = 'Stage 4 (Severe)'
            status = 'SEVERELY REDUCED'
            color = 'orangered'
            description = "Severe decrease in kidney function. Preparation for dialysis may be needed."
        else:
            stage = 'Stage 5 (Kidney Failure)'
            status = 'KIDNEY FAILURE'
            color = 'red'
            description = "Kidney failure. Dialysis or transplant typically required."
        
        return {
            'score': score,
            'stage': stage,
            'status': status,
            'color': color,
            'description': description
        }
    
    def generate_patient_report(
        self,
        patient_data: Dict,
        ckd_probability: float,
        kidney_function_score: float,
        top_risk_factors: List[str] = None,
        top_protective_factors: List[str] = None
    ) -> str:
        """
        Generate a comprehensive patient report.
        
        Args:
            patient_data: Dictionary of patient features
            ckd_probability: Predicted CKD probability
            kidney_function_score: Predicted kidney function score
            top_risk_factors: List of top risk factors from explainability
            top_protective_factors: List of protective factors
            
        Returns:
            Formatted patient report string
        """
        risk_interpretation = self.interpret_risk_level(ckd_probability)
        kidney_interpretation = self.interpret_kidney_function_score(kidney_function_score)
        
        # Build patient info section
        patient_info = f"""
PATIENT INFORMATION:
━━━━━━━━━━━━━━━━━━━
  • Age: {patient_data.get('age', 'N/A')}
  • Gender: {patient_data.get('gender', 'N/A')}
  • BMI: {patient_data.get('bmi', 'N/A')}
  • Blood Pressure: {patient_data.get('blood_pressure_systolic', 'N/A')}/{patient_data.get('blood_pressure_diastolic', 'N/A')} mmHg
  • Blood Glucose: {patient_data.get('blood_glucose', 'N/A')} mg/dL
  • Serum Creatinine: {patient_data.get('serum_creatinine', 'N/A')} mg/dL
  • Hemoglobin: {patient_data.get('hemoglobin', 'N/A')} g/dL
        """
        
        # Build conditions section
        conditions = []
        if patient_data.get('hypertension') == 'Yes':
            conditions.append('Hypertension')
        if patient_data.get('diabetes_mellitus') == 'Yes':
            conditions.append('Diabetes Mellitus')
        if patient_data.get('coronary_artery_disease') == 'Yes':
            conditions.append('Coronary Artery Disease')
        if patient_data.get('anemia') == 'Yes':
            conditions.append('Anemia')
        
        conditions_text = ', '.join(conditions) if conditions else 'None reported'
        
        # Build risk factors section
        risk_factors_text = ""
        if top_risk_factors:
            risk_factors_text = "\nKEY RISK FACTORS IDENTIFIED:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for factor in top_risk_factors[:5]:
                risk_factors_text += f"  ⚠ {factor.replace('_', ' ').title()}\n"
        
        protective_factors_text = ""
        if top_protective_factors:
            protective_factors_text = "\nPROTECTIVE FACTORS:\n━━━━━━━━━━━━━━━━━━━\n"
            for factor in top_protective_factors[:5]:
                protective_factors_text += f"  ✓ {factor.replace('_', ' ').title()}\n"
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    CKD RISK ASSESSMENT REPORT                        ║
╚══════════════════════════════════════════════════════════════════════╝

{patient_info}
EXISTING CONDITIONS: {conditions_text}

═══════════════════════════════════════════════════════════════════════

PREDICTION RESULTS:
━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────┐
  │  CKD RISK PROBABILITY: {risk_interpretation['percentage']:>6}                     │
  │  RISK LEVEL: {risk_interpretation['level']:<12}                           │
  └─────────────────────────────────────────────────────┘

{risk_interpretation['description']}

  ┌─────────────────────────────────────────────────────┐
  │  KIDNEY FUNCTION SCORE: {kidney_function_score:>5.1f}                       │
  │  STATUS: {kidney_interpretation['status']:<20}               │
  │  STAGE: {kidney_interpretation['stage']:<25}          │
  └─────────────────────────────────────────────────────┘

{kidney_interpretation['description']}
{risk_factors_text}{protective_factors_text}
RECOMMENDATIONS:
━━━━━━━━━━━━━━━━
"""
        for i, rec in enumerate(risk_interpretation['recommendations'], 1):
            report += f"  {i}. {rec}\n"
        
        report += """
═══════════════════════════════════════════════════════════════════════

DISCLAIMER: This assessment is generated by an AI-based prediction system 
and should NOT replace professional medical advice. Please consult with 
qualified healthcare providers for proper diagnosis and treatment.

═══════════════════════════════════════════════════════════════════════
        """
        
        return report
    
    def get_quick_summary(
        self,
        ckd_probability: float,
        kidney_function_score: float
    ) -> str:
        """
        Get a quick one-line summary of the prediction.
        
        Args:
            ckd_probability: Predicted CKD probability
            kidney_function_score: Predicted kidney function score
            
        Returns:
            Quick summary string
        """
        risk = self.interpret_risk_level(ckd_probability)
        kidney = self.interpret_kidney_function_score(kidney_function_score)
        
        return f"CKD Risk: {risk['level']} ({risk['percentage']}) | Kidney Function: {kidney['status']} (Score: {kidney_function_score:.1f})"
    
    def get_lifestyle_recommendations(self, patient_data: Dict) -> List[str]:
        """
        Generate personalized lifestyle recommendations.
        
        Args:
            patient_data: Dictionary of patient features
            
        Returns:
            List of personalized recommendations
        """
        recommendations = []
        
        # Age-based recommendations
        age = patient_data.get('age', 0)
        if age > 60:
            recommendations.append("Regular kidney function monitoring is especially important at your age")
        
        # BMI-based recommendations
        bmi = patient_data.get('bmi', 0)
        if bmi > 30:
            recommendations.append("Consider a weight management program - obesity increases CKD risk")
        elif bmi > 25:
            recommendations.append("Maintaining a healthy weight can help protect kidney function")
        
        # Blood pressure-based recommendations
        systolic = patient_data.get('blood_pressure_systolic', 0)
        if systolic > 140:
            recommendations.append("Blood pressure control is critical - aim for <130/80 mmHg")
        elif systolic > 120:
            recommendations.append("Monitor blood pressure regularly and maintain it in healthy range")
        
        # Glucose-based recommendations
        glucose = patient_data.get('blood_glucose', 0)
        if glucose > 126:
            recommendations.append("Blood sugar management is essential for kidney protection")
        elif glucose > 100:
            recommendations.append("Pre-diabetic glucose levels - consider dietary modifications")
        
        # Condition-based recommendations
        if str(patient_data.get('hypertension', '')).lower() == 'yes':
            recommendations.append("Follow hypertension management plan strictly")
        
        if str(patient_data.get('diabetes_mellitus', '')).lower() == 'yes':
            recommendations.append("Optimal diabetes control can slow kidney disease progression")
        
        if str(patient_data.get('smoking', '')).lower() == 'yes':
            recommendations.append("Smoking cessation is one of the best things you can do for kidney health")
        
        if str(patient_data.get('family_history_ckd', '')).lower() == 'yes':
            recommendations.append("Family history of CKD increases your risk—get regular kidney function tests (creatinine, eGFR) annually")
        
        # General recommendations
        recommendations.extend([
            "Stay hydrated with 6-8 glasses of water daily",
            "Limit sodium intake to less than 2,300mg per day",
            "Engage in 150 minutes of moderate exercise weekly"
        ])
        
        return recommendations


# Singleton instance for easy access
assistant = CKDAssistant()


def get_ckd_explanation() -> str:
    """Get CKD explanation."""
    return assistant.explain_ckd()


def get_methodology_explanation() -> str:
    """Get prediction methodology explanation."""
    return assistant.explain_prediction_methodology()


def get_risk_interpretation(probability: float) -> Dict:
    """Get risk level interpretation."""
    return assistant.interpret_risk_level(probability)


def get_kidney_score_interpretation(score: float) -> Dict:
    """Get kidney function score interpretation."""
    return assistant.interpret_kidney_function_score(score)


def generate_report(
    patient_data: Dict,
    ckd_probability: float,
    kidney_function_score: float,
    risk_factors: List[str] = None,
    protective_factors: List[str] = None
) -> str:
    """Generate patient report."""
    return assistant.generate_patient_report(
        patient_data, ckd_probability, kidney_function_score,
        risk_factors, protective_factors
    )


if __name__ == "__main__":
    # Demo the assistant
    print(assistant.explain_ckd())
    print(assistant.explain_prediction_methodology())
    
    # Test risk interpretation
    for prob in [0.15, 0.45, 0.85]:
        print(f"\nRisk for probability {prob}:")
        interpretation = assistant.interpret_risk_level(prob)
        print(f"  Level: {interpretation['level']}")
        print(f"  Description: {interpretation['description'][:100]}...")
