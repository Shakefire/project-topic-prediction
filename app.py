import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request, session
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

STUDENTS_PATH = os.path.join(DATA_DIR, "students.csv")
PROJECTS_PATH = os.path.join(DATA_DIR, "projects.csv")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_ranker.json")

app = Flask(__name__)
app.secret_key = 'naub_recommendation_system_2024'

print("=" * 60)
print("LOADING RECOMMENDATION SYSTEM")
print("=" * 60)

print("\n[1/5] Loading datasets...")
students = pd.read_csv(STUDENTS_PATH)
projects = pd.read_csv(PROJECTS_PATH)
print(f"  Students: {students.shape}")
print(f"  Projects: {projects.shape}")

print("\n[2/5] Loading encoders...")
with open(ENCODERS_PATH, 'rb') as f:
    encoders = pickle.load(f)

print("\n[3/5] Loading XGBoost model...")
model = xgb.Booster()
model.load_model(MODEL_PATH)

# Default feature names for the new model (20 features)
DEFAULT_FEATURE_NAMES_20 = [
    's_dept', 's_year', 's_gpa', 's_cs_grade', 's_pref_diff', 's_num_past',
    'p_dept', 'p_diff', 'p_cat', 'p_avg_grade', 'p_times_sel', 'p_research_area',
    'skill_match', 'dept_match', 'diff_match',
    'content_score', 'collab_score',
    'research_overlap', 'tech_similarity', 'career_alignment'
]

# Old feature names (17 features) - for backward compatibility
DEFAULT_FEATURE_NAMES_17 = [
    's_dept', 's_year', 's_gpa', 's_cs_grade', 's_pref_diff', 's_num_past',
    'p_dept', 'p_diff', 'p_cat', 'p_avg_grade', 'p_times_sel',
    'skill_match', 'dept_match', 'diff_match',
    'content_score', 'collab_score', 'skill_overlap'
]

# Detect feature count from the model by trying to predict with dummy data
print("\n[3.5/5] Detecting model feature count...")
try:
    dummy_features_17 = np.zeros((1, 17))
    dummy_dmatrix_17 = xgb.DMatrix(dummy_features_17, feature_names=DEFAULT_FEATURE_NAMES_17)
    model.predict(dummy_dmatrix_17)
    expected_features = 17
    print(f"  Model expects 17 features")
except:
    expected_features = 20
    print(f"  Model expects 20 features")

if expected_features == 20:
    feature_names = DEFAULT_FEATURE_NAMES_20
else:
    feature_names = DEFAULT_FEATURE_NAMES_17

print(f"  Using feature names ({len(feature_names)})")

print("\n[4/5] Building feature matrices...")
dept_encoder = encoders['dept_encoder']
cat_encoder = encoders['cat_encoder']
research_area_encoder = encoders.get('research_area_encoder')
scaler = encoders.get('scaler')

diff_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}

students['dept_encoded'] = dept_encoder.transform(students['department'])
students['pref_diff_encoded'] = students['preferred_difficulty'].map(diff_map)

students['gpa_norm'] = (students['gpa'] - students['gpa'].min()) / (students['gpa'].max() - students['gpa'].min() + 1e-8)
students['avg_cs_grade_norm'] = (students['avg_cs_grade'] - students['avg_cs_grade'].min()) / (students['avg_cs_grade'].max() - students['avg_cs_grade'].min() + 1e-8)

projects['dept_encoded'] = dept_encoder.transform(projects['department'])
projects['diff_encoded'] = projects['difficulty'].map(diff_map)
projects['cat_encoded'] = cat_encoder.transform(projects['category'])

if research_area_encoder:
    projects['research_area_encoded'] = research_area_encoder.transform(projects['research_area'])
else:
    projects['research_area_encoded'] = 0

projects['avg_grade_norm'] = (projects['avg_grade_given'] - projects['avg_grade_given'].min()) / (projects['avg_grade_given'].max() - projects['avg_grade_given'].min() + 1e-8)
projects['times_selected_norm'] = (projects['times_selected'] - projects['times_selected'].min()) / (projects['times_selected'].max() - projects['times_selected'].min() + 1e-8)

student_features_df = students[['dept_encoded', 'year_level', 'gpa_norm',
                                'avg_cs_grade_norm', 'pref_diff_encoded', 'num_past_projects']].to_numpy()
project_features_df = projects[['dept_encoded', 'diff_encoded', 'cat_encoded',
                                'avg_grade_norm', 'times_selected_norm', 'research_area_encoded']].to_numpy()

print("\n[4/5] Loading pre-computed matrices...")
model_dir_files = os.listdir(MODEL_DIR)
print(f"  Files in model dir: {model_dir_files}")

try:
    skill_match_matrix = np.load(os.path.join(MODEL_DIR, "skill_match_matrix.npy"))
    dept_match_matrix = np.load(os.path.join(MODEL_DIR, "dept_match_matrix.npy"))
    diff_match_matrix = np.load(os.path.join(MODEL_DIR, "diff_match_matrix.npy"))
    content_scores_all = np.load(os.path.join(MODEL_DIR, "content_scores_all.npy"))
    collab_scores_all = np.load(os.path.join(MODEL_DIR, "collab_scores_all.npy"))
    
    # Try to load new matrices
    research_overlap_all = np.load(os.path.join(MODEL_DIR, "research_overlap_all.npy"))
    tech_similarity_all = np.load(os.path.join(MODEL_DIR, "tech_similarity_all.npy"))
    career_alignment_all = np.load(os.path.join(MODEL_DIR, "career_alignment_all.npy"))
    print("  Loaded all matrices including new features")
except Exception as e:
    print(f"  Warning: Could not load some matrices: {e}")
    # Create dummy matrices if not available
    n_students = len(students)
    n_projects = len(projects)
    research_overlap_all = np.zeros((n_students, n_projects))
    tech_similarity_all = np.zeros((n_students, n_projects))
    career_alignment_all = np.zeros((n_students, n_projects))

# Pre-compute TF-IDF for new student predictions
print("\n[5/5] Setting up TF-IDF vectorizers...")
interest_vectorizer = encoders.get('interest_vectorizer')
project_vectorizer = encoders.get('project_vectorizer')
research_tag_vectorizer = encoders.get('research_tag_vectorizer')

# Pre-compute project text for TF-IDF
if project_vectorizer:
    projects['text_content'] = projects['title'] + ' ' + projects['description'] + ' ' + projects['research_tags']
    project_tfidf = project_vectorizer.transform(projects['text_content'])
else:
    project_tfidf = None

print("\n" + "=" * 60)
print("SYSTEM READY")
print("=" * 60)


def calculate_skill_match(student_skills, project_skills):
    """Calculate Jaccard similarity between student and project skills"""
    student_set = set([s.lower().strip() for s in student_skills])
    project_set = set([s.lower().strip() for s in project_skills])
    if not student_set or not project_set:
        return 0.0
    intersection = student_set.intersection(project_set)
    union = student_set.union(project_set)
    return len(intersection) / len(union) if union else 0.0


def calculate_research_overlap(student_interests, project_research_tags):
    """Calculate overlap between student interests and project research tags"""
    interest_set = set([i.lower().strip() for i in student_interests])
    tag_set = set([t.lower().strip() for t in project_research_tags])
    if not interest_set or not tag_set:
        return 0.0
    intersection = interest_set.intersection(tag_set)
    union = interest_set.union(tag_set)
    return len(intersection) / len(union) if union else 0.0


def calculate_tech_stack_similarity(student_langs, student_tools, project_skills, project_tech):
    """Calculate tech stack compatibility"""
    student_tech = set([s.lower() for s in student_langs + student_tools])
    project_tech_set = set([s.lower() for s in project_skills + project_tech])
    if not student_tech or not project_tech_set:
        return 0.0
    intersection = student_tech.intersection(project_tech_set)
    union = student_tech.union(project_tech_set)
    return len(intersection) / len(union) if union else 0.0


def calculate_career_alignment(student_dept, student_gpa, student_pref_diff, project_dept, project_diff):
    """Calculate career alignment score"""
    score = 0.0
    
    # Department match (0.4 weight)
    if student_dept == project_dept:
        score += 0.4
    
    # Difficulty match (0.3 weight)
    diff_diff = abs(diff_map.get(student_pref_diff, 1) - diff_map.get(project_diff, 1))
    score += 0.3 * (1 - diff_diff / 2)
    
    # GPA-difficulty fit (0.3 weight)
    if project_diff == 'Easy' and student_gpa >= 2.0:
        score += 0.3
    elif project_diff == 'Medium' and 2.5 <= student_gpa <= 4.0:
        score += 0.3
    elif project_diff == 'Hard' and student_gpa >= 3.0:
        score += 0.3
    else:
        score += 0.15
    
    return min(1.0, score)


def recommend_for_new_student(student_profile, top_n=5):
    """Generate recommendations for a new student"""
    
    # Extract student profile
    department = student_profile['department']
    year_level = int(student_profile['year_level'])
    gpa = float(student_profile['gpa'])
    avg_cs_grade = float(student_profile['avg_cs_grade'])
    preferred_difficulty = student_profile['preferred_difficulty']
    programming_languages = student_profile.get('programming_languages', [])
    frameworks_tools = student_profile.get('frameworks_tools', [])
    interests = student_profile.get('interests', [])
    num_past_projects = int(student_profile.get('num_past_projects', 0))
    past_projects = student_profile.get('past_projects', '')
    
    # Combine all skills
    all_skills = programming_languages + frameworks_tools
    
    # Encode student features
    try:
        s_dept = dept_encoder.transform([department])[0]
    except:
        s_dept = 0
    
    s_year = year_level
    s_gpa = (gpa - students['gpa'].min()) / (students['gpa'].max() - students['gpa'].min() + 1e-8)
    s_cs_grade = (avg_cs_grade - students['avg_cs_grade'].min()) / (students['avg_cs_grade'].max() - students['avg_cs_grade'].min() + 1e-8)
    s_pref_diff = diff_map.get(preferred_difficulty, 1)
    s_num_past = num_past_projects
    
    n_projects = len(projects)
    
    # Calculate similarity scores for all projects
    content_scores = np.zeros(n_projects)
    research_overlap = np.zeros(n_projects)
    tech_similarity = np.zeros(n_projects)
    career_alignment = np.zeros(n_projects)
    skill_match = np.zeros(n_projects)
    dept_match = np.zeros(n_projects)
    diff_match = np.zeros(n_projects)
    
    for j in range(n_projects):
        proj = projects.iloc[j]
        
        # Skill match
        proj_skills = proj['skills_list'] if 'skills_list' in proj else proj['required_skills'].split(', ')
        skill_match[j] = calculate_skill_match(all_skills, proj_skills)
        
        # Department match
        dept_match[j] = 1.0 if department == proj['department'] else 0.0
        
        # Difficulty match
        diff_match[j] = 1.0 - abs(s_pref_diff - proj['diff_encoded']) / 2.0
        
        # Research overlap
        proj_tags = proj['research_tags'].split(', ') if isinstance(proj['research_tags'], str) else []
        research_overlap[j] = calculate_research_overlap(interests, proj_tags)
        
        # Tech similarity
        proj_tech = proj['tech_stack'].split(', ') if isinstance(proj['tech_stack'], str) else []
        tech_similarity[j] = calculate_tech_stack_similarity(programming_languages, frameworks_tools, proj_skills, proj_tech)
        
        # Career alignment
        career_alignment[j] = calculate_career_alignment(department, gpa, preferred_difficulty,
                                                         proj['department'], proj['difficulty'])
        
        # Content score (TF-IDF similarity)
        if project_vectorizer and project_tfidf is not None:
            student_text = ' '.join(interests) + ' ' + ' '.join(all_skills)
            student_vec = project_vectorizer.transform([student_text])
            content_scores[j] = (student_vec * project_tfidf[j].T).toarray()[0][0]
    
    # Normalize content scores
    if content_scores.max() > content_scores.min():
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
    
    # Build feature matrix for XGBoost
    student_features = np.array([s_dept, s_year, s_gpa, s_cs_grade, s_pref_diff, s_num_past])
    student_repeated = np.tile(student_features, (n_projects, 1))
    
    # Check if we're using old (17) or new (20) feature format
    n_features = len(feature_names)
    
    if n_features == 20:
        # New format with research_area and new similarity features
        features_array = np.column_stack([
            student_repeated,           # 6 features
            project_features_df,        # 6 features (includes research_area_encoded)
            skill_match.reshape(-1, 1), # 1
            dept_match.reshape(-1, 1),  # 1
            diff_match.reshape(-1, 1),  # 1
            content_scores.reshape(-1, 1),  # 1
            np.zeros((n_projects, 1)),  # collab scores placeholder - 1
            research_overlap.reshape(-1, 1),  # 1
            tech_similarity.reshape(-1, 1),   # 1
            career_alignment.reshape(-1, 1)   # 1
        ])  # Total: 6 + 6 + 8 = 20
    else:
        # Old format (17 features) - for backward compatibility
        # Use only first 5 project features (exclude research_area_encoded)
        project_features_old = project_features_df[:, :5]
        features_array = np.column_stack([
            student_repeated,           # 6 features
            project_features_old,       # 5 features
            skill_match.reshape(-1, 1), # 1
            dept_match.reshape(-1, 1),  # 1
            diff_match.reshape(-1, 1),  # 1
            content_scores.reshape(-1, 1),  # 1
            np.zeros((n_projects, 1)),  # collab scores - 1
            np.zeros((n_projects, 1))   # skill_overlap placeholder - 1
        ])  # Total: 6 + 5 + 6 = 17
    
    # Predict with XGBoost
    dmatrix = xgb.DMatrix(features_array, feature_names=feature_names)
    ranker_scores = model.predict(dmatrix)
    
    # Normalize ranker scores
    if ranker_scores.std() > 0:
        ranker_scores = (ranker_scores - ranker_scores.min()) / (ranker_scores.max() - ranker_scores.min() + 1e-8)
    else:
        ranker_scores = np.zeros_like(ranker_scores)
    
    # Hybrid scoring
    hybrid_scores = (
        0.25 * content_scores +
        0.20 * np.zeros(n_projects) +  # No collaborative for new students
        0.25 * ranker_scores +
        0.15 * research_overlap +
        0.15 * tech_similarity
    )
    
    # Get top N
    top_indices = np.argsort(hybrid_scores)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        project = projects.iloc[idx]
        
        # Calculate match percentage
        match_pct = round(float(hybrid_scores[idx]) * 100, 1)
        if match_pct < 5:
            match_pct = round(50 + float(hybrid_scores[idx]) * 50, 1)
        
        # Generate explanation
        explanation = generate_explanation(
            student_profile, 
            project,
            {
                'content': float(content_scores[idx]),
                'research': float(research_overlap[idx]),
                'tech': float(tech_similarity[idx]),
                'career': float(career_alignment[idx]),
                'skill_match': float(skill_match[idx]),
                'dept_match': float(dept_match[idx]),
                'diff_match': float(diff_match[idx])
            }
        )
        
        results.append({
            'project_id': project['project_id'],
            'title': project['title'],
            'description': project['description'],
            'research_area': project['research_area'],
            'research_tags': project['research_tags'],
            'difficulty': project['difficulty'],
            'category': project['category'],
            'match_percentage': match_pct,
            'explanation': explanation,
            'scores': {
                'content': round(float(content_scores[idx]), 3),
                'research': round(float(research_overlap[idx]), 3),
                'tech': round(float(tech_similarity[idx]), 3),
                'career': round(float(career_alignment[idx]), 3),
                'ranker': round(float(ranker_scores[idx]), 3)
            }
        })
    
    return results


def generate_explanation(student_profile, project, scores):
    """Generate user-friendly explanation for recommendation"""
    explanations = []
    
    # Research interest match
    if scores['research'] > 0.1:
        explanations.append("Matches your research interests")
    elif scores['content'] > 0.3:
        explanations.append("Aligns with your academic focus")
    
    # Skill compatibility
    if scores['tech'] > 0.2:
        explanations.append("Compatible with your technical skills")
    elif scores['skill_match'] > 0.1:
        explanations.append("Uses technologies you know")
    
    # Academic suitability
    if scores['career'] > 0.5:
        explanations.append("Suitable for your academic performance")
    
    # Difficulty match
    if scores['diff_match'] > 0.6:
        explanations.append("Fits your preferred difficulty level")
    
    # Department alignment
    if scores['dept_match'] > 0.5:
        explanations.append("Aligned with your department")
    
    # Ensure at least one explanation
    if not explanations:
        explanations.append("Recommended based on your overall profile")
    
    return explanations


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/', methods=['GET'])
def index():
    """Home page"""
    departments = sorted(students['department'].unique().tolist())
    research_areas = sorted(projects['research_area'].unique().tolist())
    
    # Get all unique skills
    all_languages = set()
    all_tools = set()
    for _, student in students.iterrows():
        if isinstance(student['programming_languages'], str):
            all_languages.update([s.strip() for s in student['programming_languages'].split(',')])
        if isinstance(student['frameworks_tools'], str):
            all_tools.update([s.strip() for s in student['frameworks_tools'].split(',')])
    
    return render_template('index.html',
                         departments=departments,
                         research_areas=research_areas,
                         programming_languages=sorted(all_languages),
                         frameworks_tools=sorted(all_tools))


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations"""
    # Get form data
    student_profile = {
        'department': request.form.get('department', 'Computer Science'),
        'year_level': request.form.get('year_level', '3'),
        'gpa': request.form.get('gpa', '3.0'),
        'avg_cs_grade': request.form.get('avg_cs_grade', '3.0'),
        'preferred_difficulty': request.form.get('preferred_difficulty', 'Medium'),
        'programming_languages': request.form.getlist('programming_languages'),
        'frameworks_tools': request.form.getlist('frameworks_tools'),
        'interests': request.form.getlist('interests'),
        'num_past_projects': request.form.get('num_past_projects', '0'),
        'past_projects': request.form.get('past_projects', '')
    }
    
    # Generate recommendations
    recommendations = recommend_for_new_student(student_profile, top_n=5)
    
    # Store in session for analytics
    session['student_profile'] = student_profile
    session['recommendations'] = recommendations
    
    return render_template('results.html',
                         student_profile=student_profile,
                         recommendations=recommendations)


@app.route('/analytics', methods=['GET'])
def analytics():
    """Analytics dashboard"""
    student_profile = session.get('student_profile', {})
    recommendations = session.get('recommendations', [])
    
    if not recommendations:
        return render_template('analytics.html',
                             student_profile=None,
                             recommendations=None,
                             chart_data=None)
    
    # Prepare chart data
    match_percentages = [r['match_percentage'] for r in recommendations]
    research_areas = [r['research_area'] for r in recommendations]
    difficulties = [r['difficulty'] for r in recommendations]
    
    # Research area distribution
    area_counts = {}
    for area in research_areas:
        area_counts[area] = area_counts.get(area, 0) + 1
    
    # Difficulty distribution
    diff_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    for diff in difficulties:
        diff_counts[diff] = diff_counts.get(diff, 0) + 1
    
    # Score breakdown
    score_components = {
        'content': np.mean([r['scores']['content'] for r in recommendations]),
        'research': np.mean([r['scores']['research'] for r in recommendations]),
        'tech': np.mean([r['scores']['tech'] for r in recommendations]),
        'career': np.mean([r['scores']['career'] for r in recommendations]),
        'ranker': np.mean([r['scores']['ranker'] for r in recommendations])
    }
    
    chart_data = {
        'match_percentages': match_percentages,
        'project_titles': [r['title'][:30] + '...' if len(r['title']) > 30 else r['title'] for r in recommendations],
        'area_labels': list(area_counts.keys()),
        'area_values': list(area_counts.values()),
        'diff_labels': list(diff_counts.keys()),
        'diff_values': list(diff_counts.values()),
        'score_labels': ['Content', 'Research', 'Tech', 'Career', 'Ranker'],
        'score_values': [score_components['content'], score_components['research'],
                        score_components['tech'], score_components['career'],
                        score_components['ranker']]
    }
    
    return render_template('analytics.html',
                         student_profile=student_profile,
                         recommendations=recommendations,
                         chart_data=chart_data)


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for recommendations"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        recommendations = recommend_for_new_student(data, top_n=5)
        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
