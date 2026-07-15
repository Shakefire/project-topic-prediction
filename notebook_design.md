# Hybrid Student Project Recommendation System - Training Notebook Design

## Notebook Structure

### 1. Data Loading & Exploration
- Load students.csv, projects.csv, history.csv
- Basic statistics and distributions
- Missing value analysis
- Data quality checks

### 2. Feature Engineering Pipeline
#### 2.1 Student Features
- `department_encoded` - Label encoding
- `year_level` - Numeric
- `gpa` - Normalized
- `avg_cs_grade` - Normalized
- `programming_languages_vector` - Multi-hot encoding (17 languages)
- `frameworks_tools_vector` - Multi-hot encoding (30 tools)
- `interests_embedding` - TF-IDF or Sentence-BERT
- `preferred_difficulty_encoded` - Label encoding
- `num_past_projects` - Numeric
- `past_projects_embedding` - TF-IDF on past project titles

#### 2.2 Project Features
- `category_encoded` - Label encoding
- `department_encoded` - Label encoding
- `difficulty_encoded` - Label encoding
- `required_skills_vector` - Multi-hot encoding
- `tech_stack_vector` - Multi-hot encoding
- `avg_grade_given` - Normalized
- `times_selected` - Normalized
- `title_embedding` - TF-IDF or Sentence-BERT
- `description_embedding` - TF-IDF or Sentence-BERT

#### 2.3 Compatibility Features (Student × Project)
- `skill_match_pct` - Jaccard similarity of skills
- `interest_similarity` - Cosine similarity of interest/category vectors
- `dept_match` - Binary (same department?)
- `difficulty_match` - Ordinal alignment
- `gpa_difficulty_fit` - GPA vs difficulty appropriateness
- `past_project_similarity` - Cosine similarity with past projects
- `tech_overlap` - Technology stack overlap percentage

### 3. Content-Based Component
- TF-IDF vectorization of interests, skills, project descriptions
- Cosine similarity calculation
- Generate content-based scores for all student-project pairs

### 4. Collaborative Filtering Component
- User-Item matrix (students × projects)
- SVD decomposition
- Generate collaborative scores

### 5. Training Data Preparation
- Generate positive pairs (from history)
- Generate negative pairs (sampling strategy)
- Create feature matrix X and relevance scores y
- Train/validation/test split

### 6. XGBoost Ranker Training
- Feature importance analysis
- Hyperparameter tuning
- Cross-validation
- Model training

### 7. Hybrid Model Assembly
- Combine content-based + collaborative + ranker scores
- Weight optimization
- Final ranking function

### 8. Evaluation
- Precision@K, Recall@K, NDCG@K
- Hit Rate
- Mean Reciprocal Rank
- Coverage and diversity metrics

### 9. Inference Pipeline
- New student input
- Feature extraction
- Score calculation
- Top-N recommendations output
