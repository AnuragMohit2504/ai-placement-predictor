import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random
from pathlib import Path
from datetime import datetime

# Add backend to path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from backend.tools import predict_placement

# Page config
st.set_page_config(
    page_title="IQ Quiz & Placement Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .quiz-question {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# IQ Test Question Bank
IQ_QUESTIONS = {
    "Logical Reasoning": [
        {
            "question": "If all roses are flowers and some flowers fade quickly, then:",
            "options": ["All roses fade quickly", "Some roses might fade quickly", "No roses fade quickly", "All flowers are roses"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "Complete the sequence: 2, 6, 12, 20, 30, ?",
            "options": ["40", "42", "38", "36"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "If A = 1, B = 2, C = 3, what is the sum of CAT?",
            "options": ["24", "23", "22", "25"],
            "correct": 0,
            "difficulty": "easy"
        },
        {
            "question": "If 5 workers take 5 hours to complete 5 tasks, how long for 100 workers to complete 100 tasks?",
            "options": ["100 hours", "20 hours", "5 hours", "1 hour"],
            "correct": 2,
            "difficulty": "hard"
        },
        {
            "question": "What comes next: J, F, M, A, M, ?",
            "options": ["J", "S", "N", "D"],
            "correct": 0,
            "difficulty": "medium"
        }
    ],
    "Pattern Recognition": [
        {
            "question": "Find the odd one out: 3, 5, 7, 9, 12, 13",
            "options": ["3", "9", "12", "13"],
            "correct": 2,
            "difficulty": "easy"
        },
        {
            "question": "Complete: 1, 4, 9, 16, 25, ?",
            "options": ["30", "35", "36", "49"],
            "correct": 2,
            "difficulty": "easy"
        },
        {
            "question": "What's the pattern: AB, CD, EF, GH, ?",
            "options": ["IJ", "HI", "JK", "IK"],
            "correct": 0,
            "difficulty": "easy"
        },
        {
            "question": "Find the next: 2, 5, 11, 23, 47, ?",
            "options": ["94", "95", "96", "97"],
            "correct": 1,
            "difficulty": "hard"
        },
        {
            "question": "Complete: Z, X, V, T, R, ?",
            "options": ["Q", "P", "O", "N"],
            "correct": 1,
            "difficulty": "medium"
        }
    ],
    "Mathematical Ability": [
        {
            "question": "If x + 5 = 12, what is x?",
            "options": ["5", "6", "7", "8"],
            "correct": 2,
            "difficulty": "easy"
        },
        {
            "question": "What is 15% of 200?",
            "options": ["25", "30", "35", "40"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "If a shirt costs $80 after a 20% discount, what was the original price?",
            "options": ["$96", "$100", "$104", "$110"],
            "correct": 1,
            "difficulty": "hard"
        },
        {
            "question": "Simplify: (8 + 2) × 5 - 10",
            "options": ["30", "40", "50", "60"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "If 3x = 27, what is x²?",
            "options": ["9", "27", "81", "243"],
            "correct": 2,
            "difficulty": "medium"
        }
    ],
    "Verbal Reasoning": [
        {
            "question": "Choose the word most similar to 'HAPPY':",
            "options": ["Sad", "Joyful", "Angry", "Tired"],
            "correct": 1,
            "difficulty": "easy"
        },
        {
            "question": "Complete: Book is to Reading as Fork is to ?",
            "options": ["Eating", "Cooking", "Kitchen", "Food"],
            "correct": 0,
            "difficulty": "easy"
        },
        {
            "question": "Find the antonym of 'ABUNDANT':",
            "options": ["Plentiful", "Scarce", "Many", "Rich"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "Doctor : Patient :: Teacher : ?",
            "options": ["School", "Student", "Book", "Class"],
            "correct": 1,
            "difficulty": "easy"
        },
        {
            "question": "Which word doesn't belong: Apple, Banana, Carrot, Orange",
            "options": ["Apple", "Banana", "Carrot", "Orange"],
            "correct": 2,
            "difficulty": "easy"
        }
    ],
    "Spatial Reasoning": [
        {
            "question": "How many faces does a cube have?",
            "options": ["4", "6", "8", "12"],
            "correct": 1,
            "difficulty": "easy"
        },
        {
            "question": "If you fold a paper in half 3 times and make one cut, how many pieces will you have?",
            "options": ["4", "6", "8", "9"],
            "correct": 3,
            "difficulty": "hard"
        },
        {
            "question": "A clock shows 3:15. What is the angle between hour and minute hands?",
            "options": ["0°", "7.5°", "15°", "22.5°"],
            "correct": 1,
            "difficulty": "hard"
        },
        {
            "question": "How many edges does a triangular pyramid have?",
            "options": ["4", "5", "6", "7"],
            "correct": 2,
            "difficulty": "medium"
        },
        {
            "question": "If a square is rotated 90° clockwise, it will look:",
            "options": ["Different", "The same", "Larger", "Smaller"],
            "correct": 1,
            "difficulty": "easy"
        }
    ]
}

# Initialize session state
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False
if "iq_score" not in st.session_state:
    st.session_state.iq_score = None
if "test_data" not in st.session_state:
    st.session_state.test_data = []

def generate_quiz():
    """Generate 10 random questions (2 from each section)"""
    quiz = []
    for section, questions in IQ_QUESTIONS.items():
        selected = random.sample(questions, 2)  # 2 questions per section
        for q in selected:
            quiz.append({
                "section": section,
                "question": q["question"],
                "options": q["options"],
                "correct": q["correct"],
                "difficulty": q["difficulty"]
            })
    random.shuffle(quiz)
    return quiz

def calculate_iq(answers, questions):
    """Calculate IQ score based on answers"""
    correct_count = 0
    difficulty_bonus = 0
    
    for i, answer in enumerate(answers):
        if answer == questions[i]["correct"]:
            correct_count += 1
            # Bonus points for harder questions
            if questions[i]["difficulty"] == "hard":
                difficulty_bonus += 2
            elif questions[i]["difficulty"] == "medium":
                difficulty_bonus += 1
    
    # Base IQ calculation
    base_score = (correct_count / len(questions)) * 100
    
    # Convert to IQ scale (70-160)
    # 50% correct = 100 IQ (average)
    # 100% correct = 140 IQ (high)
    # 0% correct = 70 IQ (low)
    
    iq = 70 + (base_score / 100) * 70 + difficulty_bonus
    iq = min(max(iq, 70), 160)  # Clamp between 70-160
    
    return round(iq, 1), correct_count

# Main title
st.markdown('<p class="main-header">🧠 IQ Assessment & Placement Prediction</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Test Status")
    
    if st.session_state.quiz_completed:
        st.success("✅ IQ Test Completed")
        st.metric("Your IQ Score", st.session_state.iq_score)
    else:
        st.info("📝 Take the IQ test first")
    
    st.markdown("---")
    st.header("📈 Statistics")
    st.metric("Tests Completed", len(st.session_state.test_data))
    
    if len(st.session_state.test_data) > 0:
        df = pd.DataFrame(st.session_state.test_data)
        avg_iq = df['iq'].mean()
        st.metric("Average IQ", f"{avg_iq:.1f}")
        
        if st.button("💾 Download Test Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "iq_test_data.csv",
                "text/csv"
            )

# Main content
tab1, tab2, tab3 = st.tabs(["🧠 IQ Test", "🎓 Placement Prediction", "📊 Results"])

# ===================================
# TAB 1: IQ Test
# ===================================
with tab1:
    st.header("🧠 IQ Assessment Test")
    st.info("This test consists of 10 questions across 5 cognitive areas. Answer carefully!")
    
    if not st.session_state.quiz_started:
        st.markdown("""
        ### Test Sections:
        - 🧩 Logical Reasoning
        - 🔍 Pattern Recognition  
        - 🔢 Mathematical Ability
        - 📝 Verbal Reasoning
        - 🎲 Spatial Reasoning
        
        **Instructions:**
        - You will get 2 questions from each section (10 total)
        - Questions are randomized each time
        - Choose the best answer for each question
        - Your IQ score will be calculated automatically
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start IQ Test", type="primary", use_container_width=True):
                st.session_state.quiz_started = True
                st.session_state.quiz_questions = generate_quiz()
                st.session_state.current_question = 0
                st.session_state.user_answers = []
                st.session_state.quiz_completed = False
                st.rerun()
    
    elif st.session_state.quiz_started and not st.session_state.quiz_completed:
        # Show progress
        progress = st.session_state.current_question / len(st.session_state.quiz_questions)
        st.progress(progress)
        st.write(f"Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_questions)}")
        
        # Get current question
        q = st.session_state.quiz_questions[st.session_state.current_question]
        
        # Section header
        st.markdown(f'<div class="section-header">{q["section"]}</div>', unsafe_allow_html=True)
        
        # Question
        st.markdown(f'<div class="quiz-question"><h3>{q["question"]}</h3></div>', unsafe_allow_html=True)
        
        # Options
        st.write("")
        answer = st.radio(
            "Select your answer:",
            options=range(len(q["options"])),
            format_func=lambda x: f"{chr(65+x)}. {q['options'][x]}",
            key=f"q_{st.session_state.current_question}"
        )
        
        st.write("")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("Next Question ➡️", type="primary", use_container_width=True):
                st.session_state.user_answers.append(answer)
                
                if st.session_state.current_question < len(st.session_state.quiz_questions) - 1:
                    st.session_state.current_question += 1
                    st.rerun()
                else:
                    # Quiz completed
                    iq_score, correct_count = calculate_iq(
                        st.session_state.user_answers,
                        st.session_state.quiz_questions
                    )
                    st.session_state.iq_score = iq_score
                    st.session_state.quiz_completed = True
                    st.rerun()
    
    elif st.session_state.quiz_completed:
        st.success("🎉 IQ Test Completed!")
        
        col1, col2, col3 = st.columns(3)
        
        iq_score, correct_count = calculate_iq(
            st.session_state.user_answers,
            st.session_state.quiz_questions
        )
        
        with col1:
            st.metric("Your IQ Score", f"{iq_score}")
        with col2:
            st.metric("Correct Answers", f"{correct_count}/10")
        with col3:
            accuracy = (correct_count / 10) * 100
            st.metric("Accuracy", f"{accuracy:.0f}%")
        
        # IQ interpretation
        st.markdown("---")
        st.subheader("📊 Score Interpretation")
        
        if iq_score >= 130:
            interpretation = "🌟 **Exceptional** - Very superior intelligence"
            color = "#10b981"
        elif iq_score >= 120:
            interpretation = "⭐ **Superior** - Above average intelligence"
            color = "#3b82f6"
        elif iq_score >= 110:
            interpretation = "✨ **High Average** - Above average"
            color = "#8b5cf6"
        elif iq_score >= 90:
            interpretation = "👍 **Average** - Normal intelligence"
            color = "#f59e0b"
        elif iq_score >= 80:
            interpretation = "📌 **Low Average** - Below average"
            color = "#ef4444"
        else:
            interpretation = "📍 **Below Average** - Needs improvement"
            color = "#dc2626"
        
        st.markdown(f"<div style='background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; font-size: 1.2em;'>{interpretation}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Test", use_container_width=True):
                st.session_state.quiz_started = False
                st.session_state.quiz_completed = False
                st.session_state.user_answers = []
                st.session_state.current_question = 0
                st.rerun()
        
        with col2:
            if st.button("➡️ Continue to Placement Prediction", type="primary", use_container_width=True):
                st.switch_page
                st.info("Go to the 'Placement Prediction' tab to continue")

# ===================================
# TAB 2: Placement Prediction
# ===================================
with tab2:
    st.header("🎓 Placement Prediction")
    
    if not st.session_state.quiz_completed:
        st.warning("⚠️ Please complete the IQ test first!")
        st.info("Go to the 'IQ Test' tab to take the assessment")
    else:
        st.success(f"✅ IQ Score Recorded: {st.session_state.iq_score}")
        
        st.markdown("---")
        st.subheader("Enter Your Academic Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cgpa = st.number_input(
                "CGPA (0-10 scale)",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
                help="Enter your cumulative GPA"
            )
            
            branch = st.selectbox(
                "Branch/Department",
                ["Computer Science", "IT", "ECE", "EEE", "Mechanical", "Civil", "Other"]
            )
        
        with col2:
            year = st.selectbox(
                "Current Year",
                ["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduated"]
            )
            
            placement_status = st.radio(
                "Actual Placement Status (for training data)",
                ["Not Yet Decided", "Placed", "Not Placed"]
            )
        
        st.markdown("---")
        
        if st.button("🔮 Predict My Placement", type="primary", use_container_width=True):
            with st.spinner("Analyzing your profile..."):
                # Make prediction
                try:
                    pred, prob, influence = predict_placement(cgpa, st.session_state.iq_score)
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        status = "✅ PLACED" if pred == 1 else "❌ NOT PLACED"
                        color = "#10b981" if pred == 1 else "#ef4444"
                        st.markdown(f"<div style='background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;'><h2>{status}</h2></div>", unsafe_allow_html=True)
                    
                    with result_col2:
                        st.metric("Confidence", f"{prob:.1%}")
                    
                    with result_col3:
                        st.metric("Key Factor", influence)
                    
                    # Detailed breakdown
                    st.markdown("---")
                    st.subheader("📈 Profile Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Your Scores:**")
                        st.write(f"• CGPA: {cgpa}/10")
                        st.write(f"• IQ Score: {st.session_state.iq_score}")
                        st.write(f"• Branch: {branch}")
                    
                    with col2:
                        st.markdown("**Recommendations:**")
                        if pred == 0:
                            st.write("• Focus on improving your CGPA" if influence == "CGPA" else "• Work on aptitude skills")
                            st.write("• Build more projects")
                            st.write("• Practice coding regularly")
                        else:
                            st.write("• Keep up the good work!")
                            st.write("• Prepare for interviews")
                            st.write("• Build your portfolio")
                    
                    # Save to test data
                    test_record = {
                        "cgpa": cgpa,
                        "iq": st.session_state.iq_score,
                        "branch": branch,
                        "year": year,
                        "predicted_placement": pred,
                        "confidence": prob,
                        "actual_placement": 1 if placement_status == "Placed" else (0 if placement_status == "Not Placed" else None),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.test_data.append(test_record)
                    
                    st.success(f"✅ Prediction saved! Total records: {len(st.session_state.test_data)}")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
                    st.info("Make sure the model is trained. Check if backend/placement_model.pkl exists.")

# ===================================
# TAB 3: Results & Analytics
# ===================================
with tab3:
    st.header("📊 Test Results & Analytics")
    
    if len(st.session_state.test_data) == 0:
        st.info("No test data available yet. Complete the IQ test and placement prediction first.")
    else:
        df = pd.DataFrame(st.session_state.test_data)
        
        # Summary metrics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tests", len(df))
        with col2:
            avg_cgpa = df['cgpa'].mean()
            st.metric("Avg CGPA", f"{avg_cgpa:.2f}")
        with col3:
            avg_iq = df['iq'].mean()
            st.metric("Avg IQ", f"{avg_iq:.1f}")
        with col4:
            placed_count = df['predicted_placement'].sum()
            st.metric("Predicted Placed", f"{placed_count}/{len(df)}")
        
        st.markdown("---")
        
        # Data table
        st.subheader("📋 All Test Records")
        st.dataframe(df, use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CGPA vs IQ scatter
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            colors = df['predicted_placement'].map({1: '#10b981', 0: '#ef4444'})
            ax1.scatter(df['cgpa'], df['iq'], c=colors, s=100, alpha=0.6, edgecolors='black')
            ax1.set_xlabel('CGPA', fontweight='bold')
            ax1.set_ylabel('IQ Score', fontweight='bold')
            ax1.set_title('CGPA vs IQ Distribution', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#10b981', label='Predicted Placed'),
                Patch(facecolor='#ef4444', label='Predicted Not Placed')
            ]
            ax1.legend(handles=legend_elements)
            st.pyplot(fig1)
        
        with col2:
            # Prediction distribution
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            pred_counts = df['predicted_placement'].value_counts()
            pred_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                           colors=['#ef4444', '#10b981'], 
                           labels=['Not Placed', 'Placed'])
            ax2.set_ylabel('')
            ax2.set_title('Placement Prediction Distribution', fontweight='bold')
            st.pyplot(fig2)
        
        # Download button
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Complete Test Data (CSV)",
            csv,
            "iq_placement_test_data.csv",
            "text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🧠 IQ Assessment & Placement Prediction System</p>
    <p style='font-size: 0.9rem;'>Built with Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)