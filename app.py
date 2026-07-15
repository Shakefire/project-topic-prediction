import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
import json
from difflib import SequenceMatcher

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

# Feature names (17 features - matching the trained model)
feature_names = [
    's_dept', 's_year', 's_gpa', 's_cs_grade', 's_pref_diff', 's_num_past',
    'p_dept', 'p_diff', 'p_cat', 'p_avg_grade', 'p_times_sel',
    'skill_match', 'dept_match', 'diff_match',
    'content_score', 'collab_score', 'skill_overlap'
]

print("\n[4/5] Building feature matrices...")
dept_encoder = encoders['dept_encoder']
cat_encoder = encoders['cat_encoder']
scaler = encoders.get('scaler')

diff_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}

students['dept_encoded'] = dept_encoder.transform(students['department'])
students['pref_diff_encoded'] = students['preferred_difficulty'].map(diff_map)
students['gpa_norm'] = (students['gpa'] - students['gpa'].min()) / (students['gpa'].max() - students['gpa'].min() + 1e-8)
students['avg_cs_grade_norm'] = (students['avg_cs_grade'] - students['avg_cs_grade'].min()) / (students['avg_cs_grade'].max() - students['avg_cs_grade'].min() + 1e-8)

projects['dept_encoded'] = dept_encoder.transform(projects['department'])
projects['diff_encoded'] = projects['difficulty'].map(diff_map)
projects['cat_encoded'] = cat_encoder.transform(projects['category'])
projects['avg_grade_norm'] = (projects['avg_grade_given'] - projects['avg_grade_given'].min()) / (projects['avg_grade_given'].max() - projects['avg_grade_given'].min() + 1e-8)
projects['times_selected_norm'] = (projects['times_selected'] - projects['times_selected'].min()) / (projects['times_selected'].max() - projects['times_selected'].min() + 1e-8)

student_features_df = students[['dept_encoded', 'year_level', 'gpa_norm',
                                'avg_cs_grade_norm', 'pref_diff_encoded', 'num_past_projects']].to_numpy()
project_features_df = projects[['dept_encoded', 'diff_encoded', 'cat_encoded',
                                'avg_grade_norm', 'times_selected_norm']].to_numpy()

print("\n[5/5] Loading pre-computed matrices...")
try:
    skill_match_matrix = np.load(os.path.join(MODEL_DIR, "skill_match_matrix.npy"))
    dept_match_matrix = np.load(os.path.join(MODEL_DIR, "dept_match_matrix.npy"))
    diff_match_matrix = np.load(os.path.join(MODEL_DIR, "diff_match_matrix.npy"))
    content_scores_all = np.load(os.path.join(MODEL_DIR, "content_scores_all.npy"))
    collab_scores_all = np.load(os.path.join(MODEL_DIR, "collab_scores_all.npy"))
    print("  Loaded all matrices")
except Exception as e:
    print(f"  Warning: {e}")
    n_students = len(students)
    n_projects = len(projects)
    skill_match_matrix = np.zeros((n_students, n_projects))
    dept_match_matrix = np.zeros((n_students, n_projects))
    diff_match_matrix = np.zeros((n_students, n_projects))
    content_scores_all = np.zeros((n_students, n_projects))
    collab_scores_all = np.zeros((n_students, n_projects))

# Pre-compute TF-IDF
interest_vectorizer = encoders.get('interest_vectorizer')
project_vectorizer = encoders.get('project_vectorizer')

if project_vectorizer:
    projects['text_content'] = projects['title'] + ' ' + projects['description'] + ' ' + projects['research_tags']
    project_tfidf = project_vectorizer.transform(projects['text_content'])
else:
    project_tfidf = None

# Technology categories for the wizard
TECH_CATEGORIES = {
    "Programming": [
        {"name": "Python", "icon": "fab fa-python"},
        {"name": "Java", "icon": "fab fa-java"},
        {"name": "JavaScript", "icon": "fab fa-js"},
        {"name": "C++", "icon": "fas fa-code"},
        {"name": "PHP", "icon": "fab fa-php"},
        {"name": "SQL", "icon": "fas fa-database"},
        {"name": "R", "icon": "fas fa-chart-line"},
        {"name": "Swift", "icon": "fab fa-swift"},
        {"name": "Kotlin", "icon": "fas fa-mobile-alt"},
        {"name": "Go", "icon": "fas fa-arrow-right"},
        {"name": "Rust", "icon": "fas fa-cog"},
        {"name": "TypeScript", "icon": "fas fa-code"},
    ],
    "AI & Data": [
        {"name": "TensorFlow", "icon": "fas fa-brain"},
        {"name": "PyTorch", "icon": "fas fa-fire"},
        {"name": "Scikit-learn", "icon": "fas fa-robot"},
        {"name": "Pandas", "icon": "fas fa-table"},
        {"name": "NumPy", "icon": "fas fa-calculator"},
        {"name": "OpenCV", "icon": "fas fa-eye"},
        {"name": "Keras", "icon": "fas fa-network-wired"},
        {"name": "NLTK", "icon": "fas fa-language"},
        {"name": "SpaCy", "icon": "fas fa-comments"},
        {"name": "Matplotlib", "icon": "fas fa-chart-bar"},
    ],
    "Security": [
        {"name": "Linux", "icon": "fab fa-linux"},
        {"name": "Wireshark", "icon": "fas fa-shield-alt"},
        {"name": "Splunk", "icon": "fas fa-search"},
        {"name": "Metasploit", "icon": "fas fa-bug"},
        {"name": "Nmap", "icon": "fas fa-network-wired"},
        {"name": "Burp Suite", "icon": "fas fa-shield-virus"},
        {"name": "Kubernetes", "icon": "fas fa-dharmachakra"},
    ],
    "Web & Mobile": [
        {"name": "React", "icon": "fab fa-react"},
        {"name": "Node.js", "icon": "fab fa-node-js"},
        {"name": "Flask", "icon": "fas fa-flask"},
        {"name": "Django", "icon": "fas fa-python"},
        {"name": "Express", "icon": "fas fa-server"},
        {"name": "Vue.js", "icon": "fab fa-vuejs"},
        {"name": "Angular", "icon": "fab fa-angular"},
        {"name": "Flutter", "icon": "fas fa-mobile-alt"},
        {"name": "React Native", "icon": "fas fa-mobile-alt"},
        {"name": "HTML/CSS", "icon": "fab fa-html5"},
    ],
    "Cloud & DevOps": [
        {"name": "Docker", "icon": "fab fa-docker"},
        {"name": "AWS", "icon": "fab fa-aws"},
        {"name": "Azure", "icon": "fab fa-microsoft"},
        {"name": "GCP", "icon": "fab fa-google"},
        {"name": "Terraform", "icon": "fas fa-cloud"},
        {"name": "Jenkins", "icon": "fas fa-cogs"},
        {"name": "Git", "icon": "fab fa-git-alt"},
        {"name": "GitHub", "icon": "fab fa-github"},
        {"name": "MongoDB", "icon": "fas fa-database"},
        {"name": "PostgreSQL", "icon": "fas fa-database"},
        {"name": "Redis", "icon": "fas fa-database"},
    ],
}

# Research interests with subfields
RESEARCH_INTERESTS = {
    "Artificial Intelligence": {
        "icon": "fas fa-brain",
        "color": "#8B5CF6",
        "subfields": ["Machine Learning", "Deep Learning", "Neural Networks", "Reinforcement Learning", "Generative AI"]
    },
    "Cybersecurity": {
        "icon": "fas fa-shield-alt",
        "color": "#EF4444",
        "subfields": ["Malware Analysis", "Threat Intelligence", "Digital Forensics", "Penetration Testing", "Network Security"]
    },
    "Data Science": {
        "icon": "fas fa-chart-bar",
        "color": "#10B981",
        "subfields": ["Data Analytics", "Business Intelligence", "Data Mining", "Statistical Analysis", "Predictive Modeling"]
    },
    "Web Development": {
        "icon": "fas fa-globe",
        "color": "#3B82F6",
        "subfields": ["Frontend Development", "Backend Development", "Full Stack", "E-commerce", "Content Management"]
    },
    "Mobile Development": {
        "icon": "fas fa-mobile-alt",
        "color": "#F59E0B",
        "subfields": ["Android Development", "iOS Development", "Cross-Platform", "Mobile UI/UX", "App Security"]
    },
    "Cloud Computing": {
        "icon": "fas fa-cloud",
        "color": "#6366F1",
        "subfields": ["Cloud Architecture", "Serverless", "Containerization", "Microservices", "Cloud Security"]
    },
    "Computer Vision": {
        "icon": "fas fa-eye",
        "color": "#EC4899",
        "subfields": ["Image Classification", "Object Detection", "Image Segmentation", "Video Analysis", "Medical Imaging"]
    },
    "Natural Language Processing": {
        "icon": "fas fa-language",
        "color": "#14B8A6",
        "subfields": ["Text Classification", "Sentiment Analysis", "Machine Translation", "Question Answering", "Chatbots"]
    },
    "Blockchain": {
        "icon": "fas fa-link",
        "color": "#F97316",
        "subfields": ["Smart Contracts", "Cryptocurrency", "DeFi", "Supply Chain", "Voting Systems"]
    },
    "Internet of Things": {
        "icon": "fas fa-wifi",
        "color": "#84CC16",
        "subfields": ["Smart Home", "Industrial IoT", "Wearable Devices", "Sensor Networks", "Edge Computing"]
    },
    "Software Engineering": {
        "icon": "fas fa-code",
        "color": "#06B6D4",
        "subfields": ["Software Architecture", "DevOps", "Testing", "Agile", "Design Patterns"]
    },
    "Networking": {
        "icon": "fas fa-network-wired",
        "color": "#A855F7",
        "subfields": ["Network Design", "Network Security", "SDN", "Wireless Networks", "Network Monitoring"]
    },
}

print("\n" + "=" * 60)
print("SYSTEM READY")
print("=" * 60)


def fuzzy_match(query, choices, threshold=0.3):
    """Fuzzy matching for search queries"""
    query = query.lower().strip()
    results = []
    
    for choice in choices:
        choice_lower = choice.lower()
        
        # Exact match
        if query == choice_lower:
            results.append((choice, 1.0))
            continue
        
        # Startswith match
        if choice_lower.startswith(query):
            results.append((choice, 0.9))
            continue
        
        # Contains match
        if query in choice_lower:
            results.append((choice, 0.7))
            continue
        
        # Abbreviation match (e.g., "tf" -> "tensorflow")
        words = choice_lower.replace('-', ' ').replace('.', ' ').split()
        abbrev = ''.join(w[0] for w in words if w)
        if query == abbrev:
            results.append((choice, 0.8))
            continue
        
        # Sequence matching
        ratio = SequenceMatcher(None, query, choice_lower).ratio()
        if ratio >= threshold:
            results.append((choice, ratio))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results]


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
    
    if student_dept == project_dept:
        score += 0.4
    
    diff_diff = abs(diff_map.get(student_pref_diff, 1) - diff_map.get(project_diff, 1))
    score += 0.3 * (1 - diff_diff / 2)
    
    if project_diff == 'Easy' and student_gpa >= 2.0:
        score += 0.3
    elif project_diff == 'Medium' and 2.5 <= student_gpa <= 4.0:
        score += 0.3
    elif project_diff == 'Hard' and student_gpa >= 3.0:
        score += 0.3
    else:
        score += 0.15
    
    return min(1.0, score)


def generate_explanation(student_profile, project, scores):
    """Generate user-friendly explanation for recommendation"""
    explanations = []
    
    if scores.get('research', 0) > 0.1 or scores.get('content', 0) > 0.3:
        explanations.append({
            "icon": "fas fa-flask",
            "text": "Matches your research interests"
        })
    
    if scores.get('tech', 0) > 0.15 or scores.get('skill_match', 0) > 0.1:
        explanations.append({
            "icon": "fas fa-code",
            "text": "Compatible with your technical skills"
        })
    
    if scores.get('career', 0) > 0.4:
        explanations.append({
            "icon": "fas fa-graduation-cap",
            "text": "Suitable for your academic level"
        })
    
    if scores.get('diff_match', 0) > 0.5:
        explanations.append({
            "icon": "fas fa-chart-line",
            "text": "Fits your preferred difficulty level"
        })
    
    if scores.get('dept_match', 0) > 0.5:
        explanations.append({
            "icon": "fas fa-university",
            "text": "Aligned with your department"
        })
    
    if not explanations:
        explanations.append({
            "icon": "fas fa-star",
            "text": "Recommended based on your overall profile"
        })
    
    return explanations


def recommend_for_new_student(student_profile, top_n=5):
    """Generate recommendations for a new student"""
    
    department = student_profile['department']
    year_level = int(student_profile.get('year_level', 3))
    gpa = float(student_profile.get('gpa', 3.0))
    avg_cs_grade = float(student_profile.get('avg_cs_grade', 3.0))
    preferred_difficulty = student_profile.get('preferred_difficulty', 'Medium')
    programming_languages = student_profile.get('programming_languages', [])
    frameworks_tools = student_profile.get('frameworks_tools', [])
    interests = student_profile.get('interests', [])
    num_past_projects = int(student_profile.get('num_past_projects', 0))
    
    all_skills = programming_languages + frameworks_tools
    
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
    
    content_scores = np.zeros(n_projects)
    research_overlap = np.zeros(n_projects)
    tech_similarity = np.zeros(n_projects)
    career_alignment = np.zeros(n_projects)
    skill_match = np.zeros(n_projects)
    dept_match = np.zeros(n_projects)
    diff_match = np.zeros(n_projects)
    
    for j in range(n_projects):
        proj = projects.iloc[j]
        
        proj_skills = proj['required_skills'].split(', ') if isinstance(proj['required_skills'], str) else []
        skill_match[j] = calculate_skill_match(all_skills, proj_skills)
        
        dept_match[j] = 1.0 if department == proj['department'] else 0.0
        
        diff_match[j] = 1.0 - abs(s_pref_diff - proj['diff_encoded']) / 2.0
        
        proj_tags = proj['research_tags'].split(', ') if isinstance(proj['research_tags'], str) else []
        research_overlap[j] = calculate_research_overlap(interests, proj_tags)
        
        proj_tech = proj['tech_stack'].split(', ') if isinstance(proj['tech_stack'], str) else []
        tech_similarity[j] = calculate_tech_stack_similarity(programming_languages, frameworks_tools, proj_skills, proj_tech)
        
        career_alignment[j] = calculate_career_alignment(department, gpa, preferred_difficulty,
                                                         proj['department'], proj['difficulty'])
        
        if project_vectorizer and project_tfidf is not None:
            student_text = ' '.join(interests) + ' ' + ' '.join(all_skills)
            student_vec = project_vectorizer.transform([student_text])
            content_scores[j] = (student_vec * project_tfidf[j].T).toarray()[0][0]
    
    if content_scores.max() > content_scores.min():
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
    
    student_features = np.array([s_dept, s_year, s_gpa, s_cs_grade, s_pref_diff, s_num_past])
    student_repeated = np.tile(student_features, (n_projects, 1))
    
    features_array = np.column_stack([
        student_repeated,
        project_features_df[:, :5],
        skill_match.reshape(-1, 1),
        dept_match.reshape(-1, 1),
        diff_match.reshape(-1, 1),
        content_scores.reshape(-1, 1),
        np.zeros((n_projects, 1)),
        np.zeros((n_projects, 1))
    ])
    
    dmatrix = xgb.DMatrix(features_array, feature_names=feature_names)
    ranker_scores = model.predict(dmatrix)
    
    if ranker_scores.std() > 0:
        ranker_scores = (ranker_scores - ranker_scores.min()) / (ranker_scores.max() - ranker_scores.min() + 1e-8)
    else:
        ranker_scores = np.zeros_like(ranker_scores)
    
    hybrid_scores = (
        0.25 * content_scores +
        0.20 * np.zeros(n_projects) +
        0.25 * ranker_scores +
        0.15 * research_overlap +
        0.15 * tech_similarity
    )
    
    top_indices = np.argsort(hybrid_scores)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        project = projects.iloc[idx]
        
        match_pct = round(float(hybrid_scores[idx]) * 100, 1)
        if match_pct < 30:
            match_pct = round(50 + float(hybrid_scores[idx]) * 50, 1)
        
        scores_dict = {
            'content': float(content_scores[idx]),
            'research': float(research_overlap[idx]),
            'tech': float(tech_similarity[idx]),
            'career': float(career_alignment[idx]),
            'skill_match': float(skill_match[idx]),
            'dept_match': float(dept_match[idx]),
            'diff_match': float(diff_match[idx])
        }
        
        explanation = generate_explanation(student_profile, project, scores_dict)
        
        # Generate a better description based on research area and interests
        desc = project['description']
        if len(desc) > 150:
            desc = desc[:147] + "..."
        
        results.append({
            'project_id': project['project_id'],
            'title': project['title'],
            'description': desc,
            'research_area': project['research_area'],
            'research_tags': project['research_tags'],
            'difficulty': project['difficulty'],
            'category': project['category'],
            'match_percentage': match_pct,
            'explanation': explanation,
            'scores': scores_dict
        })
    
    return results


@app.route('/')
def landing():
    """Landing page"""
    return render_template('landing.html')


@app.route('/start')
def start_wizard():
    """Start the recommendation wizard"""
    session.clear()
    return redirect(url_for('wizard_step', step=1))


@app.route('/wizard/<int:step>', methods=['GET', 'POST'])
def wizard_step(step):
    """Wizard steps"""
    if step < 1 or step > 4:
        return redirect(url_for('wizard_step', step=1))
    
    if request.method == 'POST':
        # Save current step data
        if step == 1:
            session['department'] = request.form.get('department', 'Computer Science')
            session['year_level'] = request.form.get('year_level', '3')
            session['gpa'] = request.form.get('gpa', '3.0')
            session['avg_cs_grade'] = request.form.get('avg_cs_grade', '3.0')
            session['preferred_difficulty'] = request.form.get('preferred_difficulty', 'Medium')
            return redirect(url_for('wizard_step', step=2))
        
        elif step == 2:
            session['programming_languages'] = request.form.getlist('programming_languages')
            session['frameworks_tools'] = request.form.getlist('frameworks_tools')
            session['num_past_projects'] = request.form.get('num_past_projects', '0')
            return redirect(url_for('wizard_step', step=3))
        
        elif step == 3:
            session['interests'] = request.form.getlist('interests')
            return redirect(url_for('wizard_step', step=4))
    
    # Prepare data for current step
    departments = sorted(students['department'].unique().tolist())
    
    context = {
        'step': step,
        'departments': departments,
        'tech_categories': TECH_CATEGORIES,
        'research_interests': RESEARCH_INTERESTS,
        'session': session
    }
    
    if step == 4:
        # Generate recommendations
        student_profile = {
            'department': session.get('department', 'Computer Science'),
            'year_level': session.get('year_level', '3'),
            'gpa': session.get('gpa', '3.0'),
            'avg_cs_grade': session.get('avg_cs_grade', '3.0'),
            'preferred_difficulty': session.get('preferred_difficulty', 'Medium'),
            'programming_languages': session.get('programming_languages', []),
            'frameworks_tools': session.get('frameworks_tools', []),
            'interests': session.get('interests', []),
            'num_past_projects': session.get('num_past_projects', '0')
        }
        
        recommendations = recommend_for_new_student(student_profile, top_n=5)
        session['recommendations'] = recommendations
        context['recommendations'] = recommendations
        context['student_profile'] = student_profile
    
    template_name = f'wizard_{step}.html'
    return render_template(template_name, **context)


@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    student_profile = {
        'department': session.get('department', 'N/A'),
        'year_level': session.get('year_level', 'N/A'),
        'gpa': session.get('gpa', 'N/A'),
        'avg_cs_grade': session.get('avg_cs_grade', 'N/A'),
        'preferred_difficulty': session.get('preferred_difficulty', 'N/A'),
        'programming_languages': session.get('programming_languages', []),
        'frameworks_tools': session.get('frameworks_tools', []),
        'interests': session.get('interests', []),
    }
    
    recommendations = session.get('recommendations', [])
    
    if not recommendations:
        return render_template('analytics.html', student_profile=student_profile, 
                             recommendations=None, chart_data=None)
    
    match_percentages = [r['match_percentage'] for r in recommendations]
    research_areas = [r['research_area'] for r in recommendations]
    difficulties = [r['difficulty'] for r in recommendations]
    
    area_counts = {}
    for area in research_areas:
        area_counts[area] = area_counts.get(area, 0) + 1
    
    diff_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    for diff in difficulties:
        diff_counts[diff] = diff_counts.get(diff, 0) + 1
    
    score_components = {
        'content': np.mean([r['scores']['content'] for r in recommendations]),
        'research': np.mean([r['scores']['research'] for r in recommendations]),
        'tech': np.mean([r['scores']['tech'] for r in recommendations]),
        'career': np.mean([r['scores']['career'] for r in recommendations]),
    }
    
    chart_data = {
        'match_percentages': match_percentages,
        'project_titles': [r['title'][:25] + '...' if len(r['title']) > 25 else r['title'] for r in recommendations],
        'area_labels': list(area_counts.keys()),
        'area_values': list(area_counts.values()),
        'diff_labels': list(diff_counts.keys()),
        'diff_values': list(diff_counts.values()),
        'score_labels': ['Content Match', 'Research Alignment', 'Tech Compatibility', 'Career Fit'],
        'score_values': [score_components['content'], score_components['research'],
                        score_components['tech'], score_components['career']]
    }
    
    return render_template('analytics.html', student_profile=student_profile,
                         recommendations=recommendations, chart_data=chart_data)


@app.route('/api/search/tech')
def search_tech():
    """API for fuzzy technology search"""
    query = request.args.get('q', '')
    category = request.args.get('category', None)
    
    if not query:
        return jsonify([])
    
    all_tech = []
    if category and category in TECH_CATEGORIES:
        all_tech = [t['name'] for t in TECH_CATEGORIES[category]]
    else:
        for cat_tech in TECH_CATEGORIES.values():
            all_tech.extend([t['name'] for t in cat_tech])
    
    results = fuzzy_match(query, all_tech)
    return jsonify(results[:10])


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
