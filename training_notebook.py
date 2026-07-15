import os
import sys

os.makedirs(r"C:\kb\student project\model", exist_ok=True)
os.makedirs(r"C:\kb\student project\plots", exist_ok=True)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import xgboost as xgb
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

PLOTS_DIR = r"C:\kb\student project\plots"

def save_plot(filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {filepath}")
    plt.show()

print("=" * 60)
print("HYBRID STUDENT PROJECT RECOMMENDATION SYSTEM")
print("=" * 60)

# ============================================================
# SECTION 1: DATA LOADING
# ============================================================
print("\n[1/10] Loading data...")

students = pd.read_csv(r"C:\kb\student project\dataset\students.csv")
projects = pd.read_csv(r"C:\kb\student project\dataset\projects.csv")
history = pd.read_csv(r"C:\kb\student project\dataset\history.csv")

print(f"  Students: {students.shape}")
print(f"  Projects: {projects.shape}")
print(f"  History: {history.shape}")

# ============================================================
# SECTION 2: EDA VISUALIZATIONS
# ============================================================
print("\n[2/10] Creating EDA visualizations...")

# Plot 1: Dataset Overview
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
dept_counts = students['department'].value_counts()
axes[0].pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Students by Department', fontsize=12, fontweight='bold')

proj_dept = projects['department'].value_counts()
axes[1].barh(proj_dept.index, proj_dept.values, color=sns.color_palette("husl", len(proj_dept)))
axes[1].set_title('Projects by Department', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')

complexity_counts = projects['difficulty'].value_counts()
colors = {'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}
axes[2].bar(complexity_counts.index, complexity_counts.values,
            color=[colors.get(x, '#95a5a6') for x in complexity_counts.index])
axes[2].set_title('Project Difficulty Distribution', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Count')

plt.suptitle('Dataset Overview', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('01_dataset_overview.png')

# Plot 2: Student Demographics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(students['gpa'], bins=20, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(students['gpa'].mean(), color='red', linestyle='--', label=f'Mean: {students["gpa"].mean():.2f}')
axes[0].set_title('Student GPA Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('GPA')
axes[0].set_ylabel('Count')
axes[0].legend()

students.boxplot(column='gpa', by='department', ax=axes[1], rot=45)
axes[1].set_title('GPA by Department', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Department')
axes[1].set_ylabel('GPA')
plt.sca(axes[1])
plt.xticks(rotation=45, ha='right')

year_counts = students['year_level'].value_counts().sort_index()
axes[2].bar(year_counts.index.astype(str), year_counts.values, color=['#9b59b6', '#3498db'])
axes[2].set_title('Students by Year Level', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Year Level')
axes[2].set_ylabel('Count')

plt.suptitle('Student Demographics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('02_student_demographics.png')

# Plot 3: Project Categories
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
cat_counts = projects['category'].value_counts().head(20)
axes[0].barh(cat_counts.index[::-1], cat_counts.values[::-1], color=sns.color_palette("viridis", 20))
axes[0].set_title('Top 20 Project Categories', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Projects')

dept_diff = pd.crosstab(projects['department'], projects['difficulty'])
dept_diff.plot(kind='bar', stacked=True, ax=axes[1], color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[1].set_title('Projects by Department & Difficulty', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Department')
axes[1].set_ylabel('Count')
axes[1].legend(title='Difficulty')
plt.sca(axes[1])
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
save_plot('03_project_categories.png')

# Plot 4: Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
all_titles = ' '.join(projects['title'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(all_titles)
axes[0].imshow(wordcloud, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Project Titles Word Cloud', fontsize=12, fontweight='bold')

all_interests = ' '.join(students['interests'].tolist())
wordcloud2 = WordCloud(width=800, height=400, background_color='white', colormap='plasma', max_words=100).generate(all_interests)
axes[1].imshow(wordcloud2, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Student Interests Word Cloud', fontsize=12, fontweight='bold')

plt.tight_layout()
save_plot('04_wordclouds.png')

# Plot 5: History Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].hist(history['grade'], bins=20, color='#9b59b6', edgecolor='white', alpha=0.8)
axes[0, 0].axvline(history['grade'].mean(), color='red', linestyle='--', label=f'Mean: {history["grade"].mean():.2f}')
axes[0, 0].set_title('Grade Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Grade')
axes[0, 0].legend()

status_counts = history['completion_status'].value_counts()
colors_status = {'completed': '#2ecc71', 'dropped': '#f39c12', 'failed': '#e74c3c'}
axes[0, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
               colors=[colors_status.get(x, '#95a5a6') for x in status_counts.index])
axes[0, 1].set_title('Completion Status', fontsize=12, fontweight='bold')

rating_counts = history['rating'].value_counts().sort_index()
axes[1, 0].bar(rating_counts.index, rating_counts.values, color=sns.color_palette("YlOrRd", 5))
axes[1, 0].set_title('Rating Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Rating')
axes[1, 0].set_ylabel('Count')

sem_counts = history['semester'].value_counts()
axes[1, 1].barh(sem_counts.index, sem_counts.values, color=sns.color_palette("coolwarm", len(sem_counts)))
axes[1, 1].set_title('Interactions by Semester', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Count')

plt.suptitle('Interaction History Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('05_history_analysis.png')

# Plot 6: Skills Distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
all_langs = []
for langs in students['programming_languages'].str.split(', '):
    all_langs.extend(langs)
lang_counts = pd.Series(all_langs).value_counts().head(15)
axes[0].barh(lang_counts.index[::-1], lang_counts.values[::-1], color=sns.color_palette("Set2", 15))
axes[0].set_title('Top 15 Programming Languages', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count')

all_tools = []
for tools in students['frameworks_tools'].str.split(', '):
    all_tools.extend(tools)
tool_counts = pd.Series(all_tools).value_counts().head(15)
axes[1].barh(tool_counts.index[::-1], tool_counts.values[::-1], color=sns.color_palette("Set3", 15))
axes[1].set_title('Top 15 Frameworks & Tools', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')

plt.suptitle('Student Skills Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('06_skills_distribution.png')

# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================
print("\n[3/10] Feature engineering...")

dept_encoder = LabelEncoder()
cat_encoder = LabelEncoder()
scaler = MinMaxScaler()

students['dept_encoded'] = dept_encoder.fit_transform(students['department'])
diff_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
students['pref_diff_encoded'] = students['preferred_difficulty'].map(diff_map)
students['gpa_norm'] = scaler.fit_transform(students[['gpa']])
students['avg_cs_grade_norm'] = scaler.fit_transform(students[['avg_cs_grade']])
students['prog_langs_list'] = students['programming_languages'].str.split(', ')
students['fw_tools_list'] = students['frameworks_tools'].str.split(', ')

projects['dept_encoded'] = dept_encoder.transform(projects['department'])
projects['diff_encoded'] = projects['difficulty'].map(diff_map)
projects['cat_encoded'] = cat_encoder.fit_transform(projects['category'])
projects['avg_grade_norm'] = scaler.fit_transform(projects[['avg_grade_given']])
projects['times_selected_norm'] = scaler.fit_transform(projects[['times_selected']])
projects['skills_list'] = projects['required_skills'].str.split(', ')
projects['tech_list'] = projects['tech_stack'].str.split(', ')

all_prog_langs = sorted(set(l for langs in students['prog_langs_list'] for l in langs))
all_tools = sorted(set(t for tools in students['fw_tools_list'] for t in tools))
all_skills = sorted(set(s for skills in projects['skills_list'] for s in skills))

print(f"  Unique: {len(all_prog_langs)} languages, {len(all_tools)} tools, {len(all_skills)} skills")

prog_lang_matrix = np.zeros((len(students), len(all_prog_langs)))
for i, langs in enumerate(students['prog_langs_list']):
    for lang in langs:
        if lang in all_prog_langs:
            prog_lang_matrix[i, all_prog_langs.index(lang)] = 1

tools_matrix = np.zeros((len(students), len(all_tools)))
for i, tools in enumerate(students['fw_tools_list']):
    for tool in tools:
        if tool in all_tools:
            tools_matrix[i, all_tools.index(tool)] = 1

skills_matrix = np.zeros((len(projects), len(all_skills)))
for i, skills in enumerate(projects['skills_list']):
    for skill in skills:
        if skill in all_skills:
            skills_matrix[i, all_skills.index(skill)] = 1

# TF-IDF
interest_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
interest_tfidf = interest_vectorizer.fit_transform(students['interests'])

projects['text_content'] = projects['title'] + ' ' + projects['description']
project_vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
project_tfidf = project_vectorizer.fit_transform(projects['text_content'])

past_proj_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
students['past_projects_clean'] = students['past_projects'].replace('None', '')
past_proj_tfidf = past_proj_vectorizer.fit_transform(students['past_projects_clean'])

# Plot 7: Feature Correlations
fig, ax = plt.subplots(figsize=(12, 8))
numerical_features = students[['gpa', 'avg_cs_grade', 'year_level', 'num_past_projects']].copy()
numerical_features['dept_encoded'] = students['dept_encoded']
numerical_features['pref_diff_encoded'] = students['pref_diff_encoded']
corr_matrix = numerical_features.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
ax.set_title('Student Feature Correlations', fontsize=14, fontweight='bold')
plt.tight_layout()
save_plot('07_feature_correlations.png')

# ============================================================
# SECTION 4: CONTENT-BASED COMPONENT
# ============================================================
print("\n[4/10] Building content-based component...")

def calculate_skill_match(student_skills, project_skills):
    student_set = set([s.lower().strip() for s in student_skills])
    project_set = set([s.lower().strip() for s in project_skills])
    if not student_set or not project_set:
        return 0.0
    intersection = student_set.intersection(project_set)
    union = student_set.union(project_set)
    return len(intersection) / len(union) if union else 0.0

def calculate_content_scores_vectorized(student_idx, student_row):
    """Vectorized content score calculation using matrix operations"""
    student_skills = student_row['prog_langs_list'] + student_row['fw_tools_list']
    
    # Text similarity (interests + skills vs project text)
    student_text = student_row['interests'] + ' ' + ' '.join(student_skills)
    student_vec = project_vectorizer.transform([student_text])
    text_sim = cosine_similarity(student_vec, project_tfidf).flatten()
    
    # Past project similarity
    past_text = student_row['past_projects_clean'] if student_row['past_projects_clean'] else ''
    past_vec = project_vectorizer.transform([past_text])
    past_sim = cosine_similarity(past_vec, project_tfidf).flatten()
    
    # Skill match (Jaccard) - vectorized
    student_skill_set = set([s.lower().strip() for s in student_skills])
    skill_matches = np.array([len(student_skill_set.intersection(set([s.lower() for s in p_skills]))) / 
                              max(len(student_skill_set.union(set([s.lower() for s in p_skills]))), 1)
                              for p_skills in projects['skills_list']])
    
    # Department match
    dept_match = (projects['department'] == student_row['department']).astype(float).values
    
    # Difficulty match
    diff_match = (1.0 - abs(student_row['pref_diff_encoded'] - projects['diff_encoded']) / 2.0).values
    
    # GPA fit
    gpa = student_row['gpa']
    gpa_fit = np.where(projects['difficulty'] == 'Easy', 
                       np.where(gpa >= 2.0, 1.0, 0.5),
                       np.where(projects['difficulty'] == 'Medium',
                                np.where((gpa >= 2.5) & (gpa <= 4.0), 1.0, 0.5),
                                np.where(gpa >= 3.0, 1.0, 0.3)))
    
    # Weighted combination
    content_scores = (0.25 * skill_matches + 0.25 * text_sim + 0.15 * dept_match +
                      0.15 * diff_match + 0.10 * gpa_fit + 0.10 * past_sim)
    return content_scores

print("  Calculating content-based scores for student 0...")
content_scores = calculate_content_scores_vectorized(0, students.iloc[0])
print(f"  Content scores range: [{content_scores.min():.4f}, {content_scores.max():.4f}]")

# Plot 8: Content-Based Scores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(content_scores, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(content_scores.mean(), color='red', linestyle='--', label=f'Mean: {content_scores.mean():.4f}')
axes[0].set_title('Content-Based Score Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Score')
axes[0].set_ylabel('Count')
axes[0].legend()

top_indices = np.argsort(content_scores)[-10:][::-1]
top_scores = content_scores[top_indices]
top_categories = [projects.iloc[i]['category'] for i in top_indices]
colors = sns.color_palette("viridis", 10)
axes[1].barh(range(10), top_scores, color=colors)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(top_categories, fontsize=9)
axes[1].set_title('Top 10 Project Categories (Content-Based)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Content Score')
axes[1].invert_yaxis()

plt.suptitle(f'Content-Based Analysis for Student 0 ({students.iloc[0]["department"]})', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('08_content_based_scores.png')

# ============================================================
# SECTION 5: COLLABORATIVE FILTERING
# ============================================================
print("\n[5/10] Building collaborative filtering...")

student_id_to_idx = {sid: i for i, sid in enumerate(students['student_id'])}
project_id_to_idx = {pid: i for i, pid in enumerate(projects['project_id'])}
idx_to_project_id = {i: pid for pid, i in project_id_to_idx.items()}

n_students = len(students)
n_projects = len(projects)

rows, cols, vals = [], [], []
for _, row in history.iterrows():
    if row['student_id'] in student_id_to_idx and row['project_id'] in project_id_to_idx:
        rows.append(student_id_to_idx[row['student_id']])
        cols.append(project_id_to_idx[row['project_id']])
        vals.append(row['rating'])

interaction_matrix = csr_matrix((vals, (rows, cols)), shape=(n_students, n_projects))
print(f"  Interaction matrix: {interaction_matrix.shape}, Sparsity: {1 - interaction_matrix.nnz / (n_students * n_projects):.4%}")

# Plot 9: Interaction Matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sample_size = 50
sample_matrix = interaction_matrix[:sample_size, :sample_size].toarray()
axes[0].spy(sample_matrix, markersize=2, color='#3498db')
axes[0].set_title(f'Interaction Matrix (First {sample_size}x{sample_size})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Projects')
axes[0].set_ylabel('Students')

interactions_per_student = np.array(interaction_matrix.sum(axis=1)).flatten()
axes[1].hist(interactions_per_student, bins=20, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].axvline(interactions_per_student.mean(), color='blue', linestyle='--', label=f'Mean: {interactions_per_student.mean():.1f}')
axes[1].set_title('Interactions per Student', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Interactions')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.suptitle('Collaborative Filtering Data', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('09_collaborative_matrix.png')

# SVD
from sklearn.decomposition import TruncatedSVD
n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
student_factors = svd.fit_transform(interaction_matrix)
project_factors = svd.components_.T
print(f"  SVD: student_factors={student_factors.shape}, project_factors={project_factors.shape}")
print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

# Plot 10: SVD Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(range(1, n_components + 1), svd.explained_variance_ratio_, 'bo-')
axes[0].set_title('SVD Explained Variance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Component')
axes[0].set_ylabel('Explained Variance Ratio')

cumvar = np.cumsum(svd.explained_variance_ratio_)
axes[1].plot(range(1, n_components + 1), cumvar, 'go-')
axes[1].set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance')
axes[1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
axes[1].legend()

sample_factors = student_factors[:20, :10]
sns.heatmap(sample_factors, cmap='coolwarm', ax=axes[2], xticklabels=5, yticklabels=5)
axes[2].set_title('Student Latent Factors (Sample)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Factor')
axes[2].set_ylabel('Student')

plt.suptitle('SVD Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('10_svd_analysis.png')

def get_collaborative_scores(student_idx):
    student_vec = student_factors[student_idx]
    scores = np.dot(project_factors, student_vec)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores

collab_scores = get_collaborative_scores(0)
print(f"  Collaborative scores range: [{collab_scores.min():.4f}, {collab_scores.max():.4f}]")

# Plot 11: Content vs Collaborative
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(content_scores, collab_scores, alpha=0.3, s=10, color='#9b59b6')
axes[0].set_xlabel('Content-Based Score')
axes[0].set_ylabel('Collaborative Score')
axes[0].set_title('Content vs Collaborative Scores', fontsize=12, fontweight='bold')
corr = np.corrcoef(content_scores, collab_scores)[0, 1]
axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0].transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[1].hist(content_scores, bins=30, alpha=0.5, label='Content-Based', color='#3498db')
axes[1].hist(collab_scores, bins=30, alpha=0.5, label='Collaborative', color='#e74c3c')
axes[1].set_title('Score Distributions Comparison', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Score')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.suptitle('Content vs Collaborative Filtering', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('11_content_vs_collab.png')

# ============================================================
# SECTION 6: TRAINING DATA GENERATION
# ============================================================
print("\n[6/10] Generating training data...")

positive_pairs = []
for _, row in history.iterrows():
    if row['student_id'] in student_id_to_idx and row['project_id'] in project_id_to_idx:
        s_idx = student_id_to_idx[row['student_id']]
        p_idx = project_id_to_idx[row['project_id']]
        grade = row['grade']
        status = 1 if row['completion_status'] == 'completed' else 0
        # Convert to integer relevance (0-4) for XGBoost ranking
        raw_score = grade * 0.7 + status * 5 * 0.3
        relevance = min(4, max(0, int(round(raw_score))))
        positive_pairs.append((s_idx, p_idx, relevance))

negative_pairs = []
np.random.seed(42)
for s_idx, p_idx, _ in positive_pairs:
    for _ in range(3):
        neg_p_idx = np.random.randint(0, n_projects)
        negative_pairs.append((s_idx, neg_p_idx, 0))

all_pairs = positive_pairs + negative_pairs
print(f"  Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)}, Total: {len(all_pairs)}")

# Plot 12: Training Data
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pair_types = ['Positive', 'Negative']
pair_counts = [len(positive_pairs), len(negative_pairs)]
axes[0].pie(pair_counts, labels=pair_types, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0].set_title('Training Pair Distribution', fontsize=12, fontweight='bold')

relevance_scores = [p[2] for p in all_pairs]
axes[1].hist(relevance_scores, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
axes[1].set_title('Relevance Score Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Relevance Score')
axes[1].set_ylabel('Count')

pairs_per_student = defaultdict(int)
for s_idx, _, _ in all_pairs:
    pairs_per_student[s_idx] += 1
pairs_counts = list(pairs_per_student.values())
axes[2].hist(pairs_counts, bins=20, color='#9b59b6', edgecolor='white', alpha=0.8)
axes[2].set_title('Training Pairs per Student', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Number of Pairs')
axes[2].set_ylabel('Count')

plt.suptitle('Training Data Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('12_training_data.png')

# ============================================================
# SECTION 7: FEATURE EXTRACTION
# ============================================================
print("\n[7/10] Extracting features (optimized)...")

# Pre-compute all student features
student_features_df = students[['dept_encoded', 'year_level', 'gpa_norm', 'avg_cs_grade_norm', 
                                 'pref_diff_encoded', 'num_past_projects']].values

# Pre-compute all project features
project_features_df = projects[['dept_encoded', 'diff_encoded', 'cat_encoded', 
                                 'avg_grade_norm', 'times_selected_norm']].values

# Pre-compute department match matrix
dept_match_matrix = (students['department'].values[:, None] == projects['department'].values[None, :]).astype(float)

# Pre-compute difficulty match matrix
diff_match_matrix = 1.0 - np.abs(students['pref_diff_encoded'].values[:, None] - 
                                  projects['diff_encoded'].values[None, :]) / 2.0

# Pre-compute skill match using sets (optimized)
print("  Pre-computing skill match matrix...")
student_skill_sets = [frozenset([s.lower() for s in students.iloc[i]['prog_langs_list'] + students.iloc[i]['fw_tools_list']])
                      for i in range(n_students)]
project_skill_sets = [frozenset([s.lower() for s in projects.iloc[j]['skills_list']])
                      for j in range(n_projects)]

skill_match_matrix = np.zeros((n_students, n_projects))
for i in range(n_students):
    s_set = student_skill_sets[i]
    if not s_set:
        continue
    for j in range(n_projects):
        p_set = project_skill_sets[j]
        if not p_set:
            continue
        intersection = len(s_set & p_set)
        if intersection > 0:
            union = len(s_set | p_set)
            skill_match_matrix[i, j] = intersection / union

# Pre-compute content scores using TF-IDF similarity
print("  Pre-computing content scores...")
content_scores_all = np.zeros((n_students, n_projects))
for i in range(n_students):
    student = students.iloc[i]
    student_skills = student['prog_langs_list'] + student['fw_tools_list']
    student_text = student['interests'] + ' ' + ' '.join(student_skills)
    student_vec = project_vectorizer.transform([student_text])
    content_scores_all[i] = cosine_similarity(student_vec, project_tfidf).flatten()

# Pre-compute collaborative scores
print("  Pre-computing collaborative scores...")
collab_scores_all = student_factors @ project_factors.T
for i in range(n_students):
    row = collab_scores_all[i]
    collab_scores_all[i] = (row - row.min()) / (row.max() - row.min() + 1e-8)

# Build feature matrix using vectorized operations
print("  Building feature matrix...")
X_list = []
y_list = []

for s_idx, p_idx, relevance in all_pairs:
    features = [
        student_features_df[s_idx, 0],  # s_dept
        student_features_df[s_idx, 1],  # s_year
        student_features_df[s_idx, 2],  # s_gpa
        student_features_df[s_idx, 3],  # s_cs_grade
        student_features_df[s_idx, 4],  # s_pref_diff
        student_features_df[s_idx, 5],  # s_num_past
        project_features_df[p_idx, 0],  # p_dept
        project_features_df[p_idx, 1],  # p_diff
        project_features_df[p_idx, 2],  # p_cat
        project_features_df[p_idx, 3],  # p_avg_grade
        project_features_df[p_idx, 4],  # p_times_sel
        skill_match_matrix[s_idx, p_idx],  # skill_match
        dept_match_matrix[s_idx, p_idx],   # dept_match
        diff_match_matrix[s_idx, p_idx],   # diff_match
        content_scores_all[s_idx, p_idx],  # content_score
        collab_scores_all[s_idx, p_idx],   # collab_score
        0  # skill_overlap (placeholder)
    ]
    X_list.append(features)
    y_list.append(relevance)

X = np.array(X_list)
y = np.array(y_list)

# Calculate groups
groups = []
current_student = None
group_size = 0
for s_idx, _, _ in all_pairs:
    if s_idx != current_student:
        if current_student is not None:
            groups.append(group_size)
        current_student = s_idx
        group_size = 1
    else:
        group_size += 1
groups.append(group_size)

print(f"  Feature matrix: {X.shape}, Groups: {len(groups)}")

# Plot 13: Feature Distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feat_name in enumerate(feature_names[:9]):
    axes[i].hist(X[:, i], bins=30, color=sns.color_palette("husl", 9)[i], edgecolor='white', alpha=0.8)
    axes[i].set_title(f'{feat_name}', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Count')
plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('13_feature_distributions.png')

# Plot 14: Feature Correlations with Target
fig, ax = plt.subplots(figsize=(12, 6))
feature_corrs = []
for i, feat_name in enumerate(feature_names):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    feature_corrs.append((feat_name, corr))
feature_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
feat_names_sorted = [f[0] for f in feature_corrs]
corr_values = [f[1] for f in feature_corrs]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corr_values]
ax.barh(feat_names_sorted, corr_values, color=colors)
ax.set_title('Feature Correlation with Relevance Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
save_plot('14_feature_correlations_target.png')

# ============================================================
# SECTION 8: XGBOOST RANKER TRAINING
# ============================================================
print("\n[8/10] Training XGBoost Ranker...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': 42
}

evals_result = {}
model = xgb.train(params, dtrain, num_boost_round=200,
                  evals=[(dtrain, 'train'), (dtest, 'test')],
                  evals_result=evals_result, early_stopping_rounds=20, verbose_eval=50)

# Plot 15: Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(evals_result['train']['ndcg'], label='Train', color='#3498db')
axes[0].plot(evals_result['test']['ndcg'], label='Test', color='#e74c3c')
axes[0].set_title('NDCG During Training', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Boosting Round')
axes[0].set_ylabel('NDCG')
axes[0].legend()
axes[0].axvline(x=model.best_iteration, color='green', linestyle='--', alpha=0.5)

axes[1].plot(evals_result['train'].get('ndcg', []), label='Train', color='#3498db')
axes[1].plot(evals_result['test'].get('ndcg', []), label='Test', color='#e74c3c')
axes[1].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Boosting Round')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.suptitle('XGBoost Training Progress', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('15_xgboost_training.png')

# Plot 16: Feature Importance
importance = model.get_score(importance_type='gain')
importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
feat_names = [f[0] for f in importance_sorted]
gain_values = [f[1] for f in importance_sorted]
colors = sns.color_palette("viridis", len(feat_names))
axes[0].barh(feat_names[::-1], gain_values[::-1], color=colors)
axes[0].set_title('Feature Importance (Gain)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Gain')

importance_weight = model.get_score(importance_type='weight')
importance_weight_sorted = sorted(importance_weight.items(), key=lambda x: x[1], reverse=True)
feat_names_w = [f[0] for f in importance_weight_sorted]
weight_values = [f[1] for f in importance_weight_sorted]
axes[1].barh(feat_names_w[::-1], weight_values[::-1], color=sns.color_palette("magma", len(feat_names_w)))
axes[1].set_title('Feature Importance (Weight)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Weight')

plt.suptitle('XGBoost Feature Importance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('16_feature_importance.png')

# ============================================================
# SECTION 9: HYBRID MODEL
# ============================================================
print("\n[9/10] Assembling hybrid model...")

class HybridRecommender:
    def __init__(self, xgb_model, content_weight=0.3, collab_weight=0.3, ranker_weight=0.4):
        self.model = xgb_model
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.ranker_weight = ranker_weight

    def recommend(self, student_idx, top_n=5):
        # Get pre-computed scores
        c_scores = content_scores_all[student_idx]
        cf_scores = collab_scores_all[student_idx]
        
        # Build features for ranker using pre-computed matrices
        features_array = np.column_stack([
            np.full(n_projects, student_features_df[student_idx, 0]),  # s_dept
            np.full(n_projects, student_features_df[student_idx, 1]),  # s_year
            np.full(n_projects, student_features_df[student_idx, 2]),  # s_gpa
            np.full(n_projects, student_features_df[student_idx, 3]),  # s_cs_grade
            np.full(n_projects, student_features_df[student_idx, 4]),  # s_pref_diff
            np.full(n_projects, student_features_df[student_idx, 5]),  # s_num_past
            project_features_df[:, 0],  # p_dept
            project_features_df[:, 1],  # p_diff
            project_features_df[:, 2],  # p_cat
            project_features_df[:, 3],  # p_avg_grade
            project_features_df[:, 4],  # p_times_sel
            skill_match_matrix[student_idx],  # skill_match
            dept_match_matrix[student_idx],   # dept_match
            diff_match_matrix[student_idx],   # diff_match
            c_scores,  # content_score
            cf_scores,  # collab_score
            np.zeros(n_projects)  # skill_overlap placeholder
        ])
        
        dmatrix = xgb.DMatrix(features_array, feature_names=feature_names)
        ranker_scores = self.model.predict(dmatrix)
        ranker_scores = (ranker_scores - ranker_scores.min()) / (ranker_scores.max() - ranker_scores.min() + 1e-8)
        
        hybrid_scores = (self.content_weight * c_scores + 
                        self.collab_weight * cf_scores + 
                        self.ranker_weight * ranker_scores)
        
        top_indices = np.argsort(hybrid_scores)[-top_n:][::-1]
        results = [(idx, hybrid_scores[idx], c_scores[idx], 
                   cf_scores[idx], ranker_scores[idx]) for idx in top_indices]
        return results

recommender = HybridRecommender(model)

test_student_idx = 0
print(f"\n  Student 0: {students.iloc[test_student_idx]['department']}, GPA: {students.iloc[test_student_idx]['gpa']}")
print(f"  Interests: {students.iloc[test_student_idx]['interests']}")

recommendations = recommender.recommend(test_student_idx, top_n=5)

print(f"\n  Top 5 Recommendations:")
for i, (proj_idx, score, c, cf, r) in enumerate(recommendations):
    proj = projects.iloc[proj_idx]
    print(f"    {i+1}. {proj['title'][:70]}...")
    print(f"       Category: {proj['category']} | Difficulty: {proj['difficulty']}")
    print(f"       Hybrid: {score:.4f} (C: {c:.4f}, CF: {cf:.4f}, R: {r:.4f})")

# Plot 17: Recommendation Breakdown
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rec_indices = [r[0] for r in recommendations[:5]]
rec_titles = [f"Proj {i+1}" for i in range(5)]
content_vals = [r[2] for r in recommendations[:5]]
collab_vals = [r[3] for r in recommendations[:5]]
ranker_vals = [r[4] for r in recommendations[:5]]

x = np.arange(5)
width = 0.25
axes[0].bar(x - width, content_vals, width, label='Content', color='#3498db')
axes[0].bar(x, collab_vals, width, label='Collaborative', color='#e74c3c')
axes[0].bar(x + width, ranker_vals, width, label='Ranker', color='#2ecc71')
axes[0].set_xlabel('Recommendation')
axes[0].set_ylabel('Score')
axes[0].set_title('Score Breakdown per Recommendation', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(rec_titles)
axes[0].legend()

total_scores = [r[1] for r in recommendations[:5]]
content_contrib = [c * recommender.content_weight for c in content_vals]
collab_contrib = [cf * recommender.collab_weight for cf in collab_vals]
ranker_contrib = [r * recommender.ranker_weight for r in ranker_vals]

axes[1].bar(x, content_contrib, width, label='Content Contribution', color='#3498db')
axes[1].bar(x, collab_contrib, width, bottom=content_contrib, label='Collab Contribution', color='#e74c3c')
axes[1].bar(x, ranker_contrib, width, bottom=[a+b for a, b in zip(content_contrib, collab_contrib)],
            label='Ranker Contribution', color='#2ecc71')
axes[1].set_xlabel('Recommendation')
axes[1].set_ylabel('Weighted Score')
axes[1].set_title('Hybrid Score Composition', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(rec_titles)
axes[1].legend()

plt.suptitle('Recommendation Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('17_recommendation_breakdown.png')

# ============================================================
# SECTION 10: EVALUATION
# ============================================================
print("\n[10/10] Evaluating model...")

def evaluate_model(recommender, test_students, k=5):
    hits = 0
    total = 0
    reciprocal_ranks = []

    for student_idx in test_students[:50]:
        student_id = students.iloc[student_idx]['student_id']
        actual_projects = history[history['student_id'] == student_id]['project_id'].tolist()
        actual_project_ids = set(actual_projects)
        if not actual_project_ids:
            continue
        recs = recommender.recommend(student_idx, top_n=k)
        rec_project_ids = [projects.iloc[r[0]]['project_id'] for r in recs]
        for rank, proj_id in enumerate(rec_project_ids):
            if proj_id in actual_project_ids:
                hits += 1
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        total += 1

    precision_at_k = hits / total if total > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    hit_rate = hits / total if total > 0 else 0
    return {'Precision@K': precision_at_k, 'Hit Rate@K': hit_rate, 'MRR': mrr}

test_student_indices = list(range(min(100, n_students)))
metrics = evaluate_model(recommender, test_student_indices, k=5)

print("\n  Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"    {metric}: {value:.4f}")

# Plot 18: Evaluation Metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Metrics bar chart
metric_names = list(metrics.keys())
metric_values = list(metrics.values())
colors = ['#3498db', '#2ecc71', '#f39c12']
axes[0].bar(metric_names, metric_values, color=colors)
axes[0].set_title('Evaluation Metrics', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1)
for i, v in enumerate(metric_values):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Top recommended categories
top_cats = []
for i in range(min(20, n_students)):
    recs = recommender.recommend(i, top_n=5)
    for r in recs:
        top_cats.append(projects.iloc[r[0]]['category'])

cat_series = pd.Series(top_cats).value_counts().head(10)
axes[1].barh(cat_series.index[::-1], cat_series.values[::-1], color=sns.color_palette("viridis", 10))
axes[1].set_title('Most Recommended Categories', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')

# Difficulty distribution of recommendations
diff_dist = []
for i in range(min(20, n_students)):
    recs = recommender.recommend(i, top_n=5)
    for r in recs:
        diff_dist.append(projects.iloc[r[0]]['difficulty'])

diff_series = pd.Series(diff_dist).value_counts()
colors_diff = {'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}
axes[2].pie(diff_series.values, labels=diff_series.index, autopct='%1.1f%%',
            colors=[colors_diff.get(x, '#95a5a6') for x in diff_series.index])
axes[2].set_title('Recommended Difficulty Distribution', fontsize=12, fontweight='bold')

plt.suptitle('Model Evaluation Summary', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_plot('18_evaluation_summary.png')

# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================
print("\n[Saving] Model artifacts...")
import pickle

model.save_model(r"C:\kb\student project\model\xgboost_ranker.json")

with open(r"C:\kb\student project\model\encoders.pkl", 'wb') as f:
    pickle.dump({
        'dept_encoder': dept_encoder,
        'cat_encoder': cat_encoder,
        'interest_vectorizer': interest_vectorizer,
        'project_vectorizer': project_vectorizer,
        'past_proj_vectorizer': past_proj_vectorizer,
        'scaler': scaler,
        'feature_names': feature_names,
        'all_prog_langs': all_prog_langs,
        'all_tools': all_tools,
        'all_skills': all_skills
    }, f)

np.save(r"C:\kb\student project\model\student_factors.npy", student_factors)
np.save(r"C:\kb\student project\model\project_factors.npy", project_factors)
np.save(r"C:\kb\student project\model\content_scores.npy", content_scores)
np.save(r"C:\kb\student project\model\collab_scores.npy", collab_scores)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel saved to: C:\\kb\\student project\\model\\")
print(f"Plots saved to: C:\\kb\\student project\\plots\\")
print(f"\nTotal visualizations: 18")
print(f"  01_dataset_overview.png")
print(f"  02_student_demographics.png")
print(f"  03_project_categories.png")
print(f"  04_wordclouds.png")
print(f"  05_history_analysis.png")
print(f"  06_skills_distribution.png")
print(f"  07_feature_correlations.png")
print(f"  08_content_based_scores.png")
print(f"  09_collaborative_matrix.png")
print(f"  10_svd_analysis.png")
print(f"  11_content_vs_collab.png")
print(f"  12_training_data.png")
print(f"  13_feature_distributions.png")
print(f"  14_feature_correlations_target.png")
print(f"  15_xgboost_training.png")
print(f"  16_feature_importance.png")
print(f"  17_recommendation_breakdown.png")
print(f"  18_evaluation_summary.png")
