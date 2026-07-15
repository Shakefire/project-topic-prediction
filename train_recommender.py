import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (auc, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

STUDENTS_PATH = os.path.join(DATA_DIR, "students.csv")
PROJECTS_PATH = os.path.join(DATA_DIR, "projects.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")

TECHNICAL_SKILLS = {
    'python': ['python'],
    'java': ['java'],
    'cpp': ['c++', 'cpp'],
    'javascript': ['javascript', 'js', 'node'],
    'sql': ['sql', 'postgresql', 'mysql', 'sqlite', 'nosql', 'mongodb'],
    'machine_learning': ['machine learning', 'ml', 'scikit-learn', 'tensorflow', 'pytorch', 'keras'],
    'deep_learning': ['deep learning', 'neural network', 'cnn', 'rnn', 'lstm', 'transformer', 'bert'],
    'networking': ['networking', 'network', 'tcp', 'udp', 'routing', 'switching'],
    'cybersecurity': ['cybersecurity', 'cyber security', 'security', 'penetration', 'intrusion', 'forensics', 'incident response'],
    'cloud_computing': ['cloud', 'aws', 'azure', 'gcp', 'google cloud', 'cloud computing'],
    'mobile_development': ['mobile', 'android', 'ios', 'flutter', 'react native', 'kotlin', 'swift'],
    'database_management': ['database', 'db', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle'],
    'software_engineering': ['software engineering', 'software development', 'architecture', 'design pattern', 'oop'],
}

DOMAIN_INTERESTS = {
    'artificial_intelligence': ['artificial intelligence', 'ai', 'machine learning', 'deep learning', 'neural', 'computer vision', 'nlp'],
    'data_science': ['data science', 'data analytics', 'analytics', 'data mining', 'visualization', 'big data'],
    'cybersecurity': ['cybersecurity', 'security', 'crypto', 'penetration', 'incident response', 'forensics'],
    'web_development': ['web development', 'frontend', 'backend', 'full stack', 'react', 'angular', 'vue', 'html', 'css', 'javascript'],
    'mobile_development': ['mobile development', 'android', 'ios', 'flutter', 'react native', 'kotlin', 'swift'],
    'nlp': ['nlp', 'natural language processing', 'language model', 'text classification', 'sentiment'],
    'computer_vision': ['computer vision', 'image recognition', 'object detection', 'segmentation', 'opencv'],
    'iot': ['iot', 'internet of things', 'embedded systems', 'raspberry', 'arduino'],
    'blockchain': ['blockchain', 'smart contract', 'ethereum', 'bitcoin', 'crypto'],
    'embedded_systems': ['embedded systems', 'microcontroller', 'firmware', 'iot', 'raspberry', 'arduino'],
    'software_engineering': ['software engineering', 'software development', 'architecture', 'testing', 'devops'],
}

DIFFICULTY_MAP = {'Easy': 0, 'Medium': 1, 'Hard': 2}


def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_tokens(text):
    text = clean_text(text)
    tokens = re.split(r'[;,/]|\band\b|\bto\b|\bon\b', text)
    return [token.strip() for token in tokens if token.strip()]


def keyword_flags(text, keyword_map):
    text = clean_text(text)
    flags = {}
    for name, patterns in keyword_map.items():
        flags[name] = int(any(pattern in text for pattern in patterns))
    return flags


def build_student_records(students):
    students = students.copy()
    students['department_clean'] = students['department'].astype(str)
    students['pref_diff_clean'] = students['preferred_difficulty'].astype(str)
    students['gpa_norm'] = MinMaxScaler().fit_transform(students[['gpa']])
    students['avg_cs_grade_norm'] = MinMaxScaler().fit_transform(students[['avg_cs_grade']])
    students['interest_text'] = students['interests'].fillna('').astype(str)
    students['skills_text'] = (students['programming_languages'].fillna('') + ' ' +
                               students['frameworks_tools'].fillna('')).astype(str)
    students['skill_flags'] = students['skills_text'].apply(lambda text: keyword_flags(text, TECHNICAL_SKILLS))
    students['domain_flags'] = students['interests'].fillna('').astype(str).apply(lambda text: keyword_flags(text, DOMAIN_INTERESTS))
    students['past_projects_text'] = students['past_projects'].fillna('').astype(str)
    return students


def build_project_records(projects):
    projects = projects.copy()
    projects['department_clean'] = projects['department'].astype(str)
    projects['difficulty_clean'] = projects['difficulty'].astype(str)
    projects['category_clean'] = projects['category'].astype(str)
    projects['avg_grade_norm'] = MinMaxScaler().fit_transform(projects[['avg_grade_given']])
    projects['times_selected_norm'] = MinMaxScaler().fit_transform(projects[['times_selected']])
    projects['project_text'] = (
        projects['title'].fillna('') + ' ' +
        projects['description'].fillna('') + ' ' +
        projects['required_skills'].fillna('') + ' ' +
        projects['tech_stack'].fillna('') + ' ' +
        projects['category'].fillna('')
    ).astype(str)
    projects['skill_text'] = (projects['required_skills'].fillna('') + ' ' + projects['tech_stack'].fillna('')).astype(str)
    projects['skill_flags'] = projects['skill_text'].apply(lambda text: keyword_flags(text, TECHNICAL_SKILLS))
    projects['domain_flags'] = projects['project_text'].apply(lambda text: keyword_flags(text, DOMAIN_INTERESTS))
    return projects


def build_similarity_features(student_row, project_row, student_text_vector, project_text_vector):
    student_skills = np.array(list(student_row['skill_flags'].values()), dtype=int)
    project_skills = np.array(list(project_row['skill_flags'].values()), dtype=int)
    skill_overlap = int(np.logical_and(student_skills, project_skills).sum())
    skill_union = int(np.logical_or(student_skills, project_skills).sum())
    skill_overlap_ratio = skill_overlap / max(skill_union, 1)

    student_domains = np.array(list(student_row['domain_flags'].values()), dtype=int)
    project_domains = np.array(list(project_row['domain_flags'].values()), dtype=int)
    domain_overlap = int(np.logical_and(student_domains, project_domains).sum())
    domain_union = int(np.logical_or(student_domains, project_domains).sum())
    domain_overlap_ratio = domain_overlap / max(domain_union, 1)

    dept_match = float(student_row['department_clean'] == project_row['department_clean'])
    diff_match = 1.0 - abs(student_row['preferred_diff_encoded'] - project_row['difficulty_encoded']) / 2.0
    category_match = float(project_row['category_clean'].lower() in student_row['interest_text'].lower())

    student_vec = student_text_vector.toarray()[0]
    project_vec = project_text_vector.toarray()[0]
    interest_similarity = float(
        np.dot(student_vec, project_vec) /
        (np.linalg.norm(student_vec) * np.linalg.norm(project_vec) + 1e-9)
    )

    return {
        'skill_overlap': skill_overlap,
        'skill_overlap_ratio': skill_overlap_ratio,
        'domain_overlap': domain_overlap,
        'domain_overlap_ratio': domain_overlap_ratio,
        'dept_match': dept_match,
        'diff_match': diff_match,
        'category_match': category_match,
        'interest_similarity': interest_similarity,
    }


def build_feature_vector(student_row, project_row, similarity_features):
    features = [
        student_row['department_encoded'],
        student_row['year_level'],
        student_row['gpa_norm'],
        student_row['avg_cs_grade_norm'],
        student_row['preferred_diff_encoded'],
        student_row['num_past_projects'],
        project_row['department_encoded'],
        project_row['difficulty_encoded'],
        project_row['category_encoded'],
        project_row['avg_grade_norm'],
        project_row['times_selected_norm'],
        similarity_features['skill_overlap'],
        similarity_features['skill_overlap_ratio'],
        similarity_features['domain_overlap'],
        similarity_features['domain_overlap_ratio'],
        similarity_features['dept_match'],
        similarity_features['diff_match'],
        similarity_features['category_match'],
        similarity_features['interest_similarity'],
    ]

    features.extend(list(student_row['skill_flags'].values()))
    features.extend(list(project_row['skill_flags'].values()))
    features.extend(list(student_row['domain_flags'].values()))
    features.extend(list(project_row['domain_flags'].values()))
    return np.array(features, dtype=float)


def run_training():
    print('Loading data...')
    students = pd.read_csv(STUDENTS_PATH)
    projects = pd.read_csv(PROJECTS_PATH)
    history = pd.read_csv(HISTORY_PATH)

    students = build_student_records(students)
    projects = build_project_records(projects)

    print('Encoding categorical values...')
    dept_encoder = LabelEncoder().fit(students['department_clean'])
    diff_encoder = LabelEncoder().fit(['Easy', 'Medium', 'Hard'])
    category_encoder = LabelEncoder().fit(projects['category_clean'])

    students['department_encoded'] = dept_encoder.transform(students['department_clean'])
    students['preferred_diff_encoded'] = diff_encoder.transform(students['pref_diff_clean'].fillna('Easy'))
    projects['department_encoded'] = dept_encoder.transform(projects['department_clean'])
    projects['difficulty_encoded'] = diff_encoder.transform(projects['difficulty_clean'].fillna('Easy'))
    projects['category_encoded'] = category_encoder.transform(projects['category_clean'])

    print('Fitting vectorizers...')
    text_vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    combined_text = pd.concat([students['interest_text'], projects['project_text']], ignore_index=True)
    combined_matrix = text_vectorizer.fit_transform(combined_text)
    interest_matrix = combined_matrix[:len(students)]
    project_matrix = combined_matrix[len(students):]

    student_idx = {sid: i for i, sid in enumerate(students['student_id'])}
    project_idx = {pid: i for i, pid in enumerate(projects['project_id'])}

    positive_history = history[
        (history['completion_status'] == 'completed') &
        (history['rating'] >= 4)
    ].copy()
    positive_history = positive_history[positive_history['student_id'].isin(student_idx) & positive_history['project_id'].isin(project_idx)]

    print(f'Positive interactions: {len(positive_history)}')

    random_negatives = []
    rng = np.random.default_rng(42)
    all_pairs = set(zip(history['student_id'], history['project_id']))
    for _, row in positive_history.iterrows():
        sid = row['student_id']
        for _ in range(3):
            candidate = rng.choice(projects['project_id'].values)
            if (sid, candidate) in all_pairs:
                continue
            random_negatives.append({'student_id': sid, 'project_id': candidate, 'label': 0})
    print(f'Random negative pairs: {len(random_negatives)}')

    negative_history = history[~history.index.isin(positive_history.index)].copy()
    negative_history = negative_history[negative_history['student_id'].isin(student_idx) & negative_history['project_id'].isin(project_idx)]
    negative_history = negative_history.assign(label=0)[['student_id', 'project_id', 'label']]
    print(f'Negative historical interactions: {len(negative_history)}')

    training_rows = []

    positives = positive_history.assign(label=1)[['student_id', 'project_id', 'label']]
    training_rows.extend(positives.to_dict(orient='records'))
    training_rows.extend(random_negatives)
    training_rows.extend(negative_history.to_dict(orient='records'))

    dataframe = pd.DataFrame(training_rows).drop_duplicates(['student_id', 'project_id', 'label'])
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'Total training rows: {len(dataframe)}')

    print('Building feature matrix...')
    feature_vectors = []
    labels = []

    for i, row in dataframe.iterrows():
        s = row['student_id']
        p = row['project_id']
        student_row = students.loc[students['student_id'] == s].iloc[0]
        project_row = projects.loc[projects['project_id'] == p].iloc[0]
        student_index = students.index[students['student_id'] == s][0]
        project_index = projects.index[projects['project_id'] == p][0]
        similarity = build_similarity_features(
            student_row,
            project_row,
            interest_matrix[student_index],
            project_matrix[project_index],
        )
        vector = build_feature_vector(student_row, project_row, similarity)
        feature_vectors.append(vector)
        labels.append(row['label'])

    X = np.vstack(feature_vectors)
    y = np.array(labels, dtype=int)
    print(f'Feature matrix shape: {X.shape}')

    feature_names = [
        'student_dept', 'student_year', 'student_gpa', 'student_cs_grade',
        'student_pref_diff', 'student_num_past', 'project_dept', 'project_diff',
        'project_category', 'project_avg_grade', 'project_times_selected',
        'skill_overlap', 'skill_overlap_ratio', 'domain_overlap', 'domain_overlap_ratio',
        'dept_match', 'diff_match', 'category_match', 'interest_similarity',
    ]
    feature_names.extend([f'student_skill_{k}' for k in TECHNICAL_SKILLS])
    feature_names.extend([f'project_skill_{k}' for k in TECHNICAL_SKILLS])
    feature_names.extend([f'student_domain_{k}' for k in DOMAIN_INTERESTS])
    feature_names.extend([f'project_domain_{k}' for k in DOMAIN_INTERESTS])

    print('Splitting train/test sets...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')

    print('Training XGBoost classifier...')
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    print('Evaluation metrics:')
    for name, value in metrics.items():
        print(f'  {name}: {value:.4f}')

    print('Saving artifacts...')
    with open(os.path.join(MODELS_DIR, 'recommender.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump({
            'dept_encoder': dept_encoder,
            'diff_encoder': diff_encoder,
            'category_encoder': category_encoder,
        }, f)

    with open(os.path.join(MODELS_DIR, 'encoder.pkl'), 'wb') as f:
        pickle.dump({
            'technical_skills': TECHNICAL_SKILLS,
            'domain_interests': DOMAIN_INTERESTS,
        }, f)

    with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump({
            'text_vectorizer': text_vectorizer,
        }, f)

    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump({
            'gpa_scaler': MinMaxScaler().fit(students[['gpa']]),
            'avg_cs_grade_scaler': MinMaxScaler().fit(students[['avg_cs_grade']]),
            'project_avg_grade_scaler': MinMaxScaler().fit(projects[['avg_grade_given']]),
            'project_times_selected_scaler': MinMaxScaler().fit(projects[['times_selected']]),
        }, f)

    metadata = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'model_type': 'xgboost_classifier',
        'target': 'completed project with rating >= 4',
        'feature_names': feature_names,
        'technical_skills': list(TECHNICAL_SKILLS.keys()),
        'domain_interests': list(DOMAIN_INTERESTS.keys()),
        'students': int(students.shape[0]),
        'projects': int(projects.shape[0]),
        'positive_samples': int(len(positive_history)),
        'train_rows': int(X_train.shape[0]),
        'test_rows': int(X_test.shape[0]),
    }
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Saved model artifacts in models/')


if __name__ == '__main__':
    run_training()
