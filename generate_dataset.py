import csv
import random
import os

random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
NUM_STUDENTS = 500
OUTPUT_DIR = r"C:\kb\student project\dataset"

DEPARTMENTS = ["Computer Science", "Cybersecurity", "Software Engineering", "Information Systems", "Information Technology"]
DEPT_CODES = {"Computer Science": "COS", "Cybersecurity": "CYB", "Software Engineering": "SFE", "Information Systems": "IFS", "Information Technology": "IFT"}

YEAR_LEVELS = [3, 4]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
COMPLETION_STATUSES = ["completed", "dropped", "failed"]
SEMESTERS = ["Fall 2022", "Spring 2023", "Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"]

# Interest areas mapped to departments
INTERESTS_BY_DEPT = {
    "Computer Science": ["machine learning", "deep learning", "natural language processing", "computer vision", "data mining", "algorithms", "bioinformatics", "neural networks", "reinforcement learning", "generative AI", "recommendation systems", "graph algorithms", "optimization", "quantum computing"],
    "Cybersecurity": ["network security", "malware analysis", "cryptography", "penetration testing", "digital forensics", "incident response", "threat intelligence", "privacy", "access control", "compliance", "IoT security", "cloud security", "AI security", "operational technology security"],
    "Software Engineering": ["web development", "mobile development", "devops", "testing", "software architecture", "API development", "microservices", "cloud computing", "database design", "UI/UX design", "game development", "e-commerce", "enterprise systems", "progressive web apps"],
    "Information Systems": ["business intelligence", "data analytics", "enterprise resource planning", "customer relationship management", "business process management", "data warehouse", "supply chain", "decision support", "data governance", "e-government", "healthcare systems", "retail systems"],
    "Information Technology": ["cloud infrastructure", "networking", "database administration", "system administration", "IoT", "edge computing", "monitoring", "virtualization", "containerization", "green IT", "infrastructure automation", "storage systems"]
}

# Programming languages and tools
PROGRAMMING_LANGUAGES = ["Python", "Java", "C++", "JavaScript", "C#", "PHP", "Ruby", "Go", "Rust", "Kotlin", "Swift", "R", "MATLAB", "TypeScript", "Scala"]
FRAMEWORKS_TOOLS = ["React", "Angular", "Vue.js", "Django", "Flask", "Spring Boot", "Node.js", "Express", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "Docker", "Kubernetes", "AWS", "Azure", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Git", "Linux", "Wireshark", "Burp Suite", "Metasploit", "Nmap", "Splunk", "Elasticsearch", "Hadoop", "Spark"]

# Skills required per project area
SKILLS_BY_AREA = {
    "Classification": ["python", "scikit-learn", "pandas", "numpy", "matplotlib"],
    "Regression": ["python", "scikit-learn", "pandas", "numpy", "statistics"],
    "Clustering": ["python", "scikit-learn", "pandas", "numpy", "matplotlib"],
    "Anomaly Detection": ["python", "scikit-learn", "pandas", "numpy", "statistics"],
    "CNN": ["python", "tensorflow", "keras", "pytorch", "opencv", "numpy"],
    "RNN": ["python", "tensorflow", "keras", "pytorch", "numpy"],
    "Generative": ["python", "tensorflow", "pytorch", "keras", "numpy"],
    "Transformer": ["python", "pytorch", "transformers", "huggingface", "numpy"],
    "NLP-Text": ["python", "nltk", "spacy", "scikit-learn", "pandas"],
    "NLP-NER": ["python", "nltk", "spacy", "transformers", "tensorflow"],
    "NLP-MT": ["python", "transformers", "pytorch", "tensorflow"],
    "NLP-QA": ["python", "transformers", "pytorch", "elasticsearch"],
    "CV-Class": ["python", "opencv", "tensorflow", "pytorch", "keras"],
    "CV-Detection": ["python", "opencv", "tensorflow", "pytorch", "yolo"],
    "CV-Segmentation": ["python", "opencv", "tensorflow", "pytorch", "keras"],
    "Intrusion": ["python", "wireshark", "scikit-learn", "pandas", "linux"],
    "Host-IDS": ["python", "linux", "windows", "sysmon", "splunk"],
    "Malware": ["python", "c", "assembly", "ida pro", "ghidra", "sandbox"],
    "Anomaly": ["python", "scikit-learn", "pandas", "numpy", "statistics"],
    "Fraud": ["python", "scikit-learn", "pandas", "numpy", "sql"],
    "Web-Sec": ["python", "javascript", "burp suite", "owasp", "sql"],
    "IAM": ["python", "java", "ldap", "oauth", "jwt", "sql"],
    "Cloud Sec": ["python", "aws", "azure", "docker", "kubernetes", "terraform"],
    "IoT Sec": ["python", "c", "mqtt", "linux", "embedded systems"],
    "Crypto": ["python", "c", "openssl", "mathematics", "linux"],
    "Compliance": ["python", "sql", "excel", "reporting", "linux"],
    "ERP": ["java", "sql", "python", "javascript", "sap"],
    "CRM": ["java", "sql", "python", "javascript", "salesforce"],
    "BPM": ["java", "sql", "python", "javascript", "bpmn"],
    "DW": ["sql", "python", "etl", "tableau", "power bi"],
    "Finance": ["java", "sql", "python", "javascript", "excel"],
    "HR": ["java", "sql", "python", "javascript", "excel"],
    "Database": ["sql", "python", "mongodb", "postgresql", "mysql"],
    "NoSQL": ["python", "mongodb", "cassandra", "redis", "neo4j"],
    "Recommender": ["python", "scikit-learn", "pandas", "numpy", "surprise"],
    "Content-Based": ["python", "scikit-learn", "pandas", "numpy", "nlp"],
    "Blockchain": ["solidity", "javascript", "web3", "python", "ethereum"],
    "Quantum": ["python", "qiskit", "cirq", "mathematics", "physics"],
    "Data Tools": ["python", "pandas", "numpy", "sql", "excel"],
    "Basic Tools": ["python", "javascript", "html", "css", "sql"],
    "Web Scraping": ["python", "beautifulsoup", "selenium", "scrapy", "javascript"],
    "Analysis": ["python", "pandas", "matplotlib", "seaborn", "tableau"],
    "Gaming": ["c++", "c#", "unity", "unreal engine", "opengl"],
    "Testing": ["python", "java", "selenium", "junit", "pytest"],
    "Architecture": ["java", "python", "docker", "kubernetes", "microservices"],
    "DevOps": ["docker", "kubernetes", "jenkins", "terraform", "ansible", "linux"],
    "Web-App": ["javascript", "html", "css", "react", "node.js", "sql"],
    "API": ["javascript", "python", "node.js", "express", "rest", "graphql"],
    "Mobile": ["java", "kotlin", "swift", "react native", "flutter", "sql"],
    "Data Mining": ["python", "scikit-learn", "pandas", "numpy", "weka"],
    "Time Series": ["python", "pandas", "numpy", "statsmodels", "tensorflow"],
    "Visualization": ["python", "matplotlib", "seaborn", "plotly", "d3.js", "tableau"],
    "Analytics": ["python", "pandas", "sql", "tableau", "power bi"],
    "ML": ["python", "scikit-learn", "tensorflow", "pytorch", "pandas"],
    "Feature": ["python", "scikit-learn", "pandas", "numpy", "featuretools"],
    "Interpretability": ["python", "scikit-learn", "lime", "shap", "pandas"],
    "Ensemble": ["python", "scikit-learn", "xgboost", "lightgbm", "pandas"],
    "Prediction": ["python", "scikit-learn", "pandas", "numpy", "tensorflow"],
    "Pentest": ["python", "metasploit", "nmap", "burp suite", "linux"],
    "Threat": ["python", "splunk", "elasticsearch", "mitre att&ck", "linux"],
    "Incident": ["python", "splunk", "forensics", "linux", "windows"],
    "Hunting": ["python", "splunk", "elasticsearch", "yara", "linux"],
    "Forensics": ["python", "autopsy", "volatility", "wireshark", "linux"],
    "Vulnerability": ["python", "nmap", "burp suite", "metasploit", "linux"],
    "Privacy": ["python", "cryptography", "mathematics", "sql", "linux"],
    "Access": ["python", "java", "ldap", "oauth", "rbac"],
    "Identity": ["python", "java", "biometrics", "ldap", "oauth"],
    "Supply": ["java", "sql", "python", "excel", "erp"],
    "Manufacturing": ["java", "sql", "python", "plc", "scada"],
    "MRP": ["java", "sql", "python", "excel", "erp"],
    "Quality": ["java", "sql", "python", "excel", "statistics"],
    "Container": ["docker", "kubernetes", "helm", "linux", "yaml"],
    "Cloud": ["aws", "azure", "gcp", "terraform", "docker", "kubernetes"],
    "IaC": ["terraform", "ansible", "cloudformation", "python", "yaml"],
    "Platform": ["docker", "kubernetes", "python", "go", "linux"],
    "PWA": ["javascript", "html", "css", "service workers", "react"],
    "SPA": ["javascript", "html", "css", "react", "angular", "vue.js"],
    "Framework": ["javascript", "html", "css", "webpack", "babel"],
    "Backend": ["python", "java", "javascript", "node.js", "sql", "rest"],
    "Accessibility": ["javascript", "html", "css", "wcag", "screen readers"],
    "i18n": ["javascript", "html", "css", "react", "i18next"],
    "Performance": ["javascript", "python", "chrome devtools", "lighthouse", "sql"],
    "Maintenance": ["java", "python", "javascript", "git", "sonarqube"],
    "Security": ["python", "java", "owasp", "burp suite", "sonarqube"],
    "Pattern": ["java", "python", "c++", "design patterns", "solid"],
    "Neural": ["python", "pytorch", "tensorflow", "numpy", "mathematics"],
    "Unsupervised": ["python", "pytorch", "tensorflow", "scikit-learn", "numpy"],
    "Physics": ["python", "pytorch", "tensorflow", "mathematics", "physics"],
    "Neuromorphic": ["python", "pytorch", "c++", "hardware", "neuroscience"],
    "Adversarial": ["python", "pytorch", "tensorflow", "adversarial robustness"],
    "Bio": ["python", "r", "biopython", "statistics", "genomics"],
    "OT": ["python", "c", "scada", "modbus", "industrial protocols"],
    "Monitoring": ["python", "linux", "grafana", "prometheus", "splunk"],
    "Concurrent": ["java", "python", "c++", "go", "rust", "multithreading"],
    "Distributed": ["java", "python", "go", "raft", "paxos", "consensus"],
    "Collaboration": ["javascript", "python", "websockets", "operational transform"],
    "Scalability": ["java", "python", "docker", "kubernetes", "load balancing"],
    "Reactive": ["java", "javascript", "python", "rxjava", "reactive streams"],
    "Vision": ["python", "opencv", "pytorch", "tensorflow", "3d geometry"],
    "Video": ["python", "opencv", "ffmpeg", "pytorch", "tensorflow"],
    "Graphics": ["c++", "opengl", "vulkan", "glsl", "unity"],
    "Human": ["python", "opencv", "pytorch", "tensorflow", "mediapipe"],
    "AI Security": ["python", "pytorch", "tensorflow", "scikit-learn", "linux"],
    "Response": ["python", "splunk", "soar", "linux", "incident handling"],
    "Predictive": ["python", "scikit-learn", "pandas", "numpy", "statistics"],
    "Automation": ["python", "ansible", "terraform", "jenkins", "linux"],
    "Retail": ["java", "sql", "python", "javascript", "pos systems"],
    "E-Commerce": ["java", "javascript", "sql", "react", "node.js", "stripe"],
    "Loyalty": ["java", "sql", "python", "javascript", "analytics"],
    "Omnichannel": ["java", "sql", "python", "javascript", "api integration"],
    "Replication": ["sql", "python", "postgresql", "mysql", "mongodb"],
    "Big Data": ["python", "hadoop", "spark", "kafka", "sql"],
    "Backup": ["python", "linux", "sql", "aws", "azure"],
    "Edge": ["python", "c", "iot", "docker", "kubernetes", "linux"],
    "Environment": ["python", "sql", "excel", "iot", "sensors"],
    "Specialized": ["python", "c", "iot", "sensors", "embedded systems"],
    "Analytics": ["python", "pandas", "sql", "tableau", "power bi"],
    "Governance": ["python", "sql", "excel", "data catalog", "metadata"],
    "Strategy": ["python", "sql", "excel", "tableau", "power bi"],
    "Organization": ["python", "sql", "excel", "hr analytics"],
    "Network": ["python", "linux", "tcp/ip", "sdn", "wireshark"],
    "Storage": ["python", "linux", "aws", "azure", "storage systems"],
    "Virtualization": ["vmware", "docker", "kubernetes", "linux", "python"],
    "Messaging": ["java", "python", "kafka", "rabbitmq", "redis"],
    "Integration": ["java", "python", "rest", "soap", "etl"],
    "Observability": ["python", "grafana", "prometheus", "elasticsearch", "jaeger"],
    "Deployment": ["docker", "kubernetes", "jenkins", "argocd", "helm"],
    "Resilience": ["java", "python", "docker", "kubernetes", "circuit breaker"],
    "Advanced": ["python", "java", "sql", "tableau", "power bi"],
    "Social": ["javascript", "python", "react", "node.js", "mongodb", "redis"],
    "Healthcare": ["java", "python", "sql", "hl7", "fhir"],
    "Finance": ["java", "python", "sql", "blockchain", "fintech"],
    "EdTech": ["java", "javascript", "python", "sql", "lms"],
    "Logistics": ["java", "python", "sql", "gps", "routing algorithms"],
    "Documents": ["python", "java", "javascript", "pdf", "excel"],
    "Media": ["python", "javascript", "ffmpeg", "opencv", "streaming"],
    "XR": ["c#", "unity", "unreal engine", "c++", "3d modeling"],
    "iOS": ["swift", "xcode", "cloudkit", "arkit", "swiftui"],
    "Productivity": ["java", "javascript", "python", "react native", "flutter"],
    "Education": ["java", "javascript", "python", "react", "sql"],
    "Utilities": ["java", "javascript", "python", "c++", "system programming"],
    "Communication": ["javascript", "python", "websockets", "webrtc", "node.js"],
    "Notifications": ["javascript", "python", "firebase", "apns", "fcm"],
    "Business": ["java", "javascript", "python", "sql", "excel"],
    "Automation": ["python", "java", "javascript", "kafka", "redis"],
    "Resilience": ["java", "python", "docker", "kubernetes"],
    "Integration": ["java", "python", "rest", "soap", "etl"],
    "E-Gov": ["java", "javascript", "python", "sql", "cloud"],
    "Admin": ["java", "javascript", "python", "sql", "excel"],
    "Safety": ["java", "python", "sql", "compliance", "reporting"],
    "Justice": ["java", "javascript", "python", "sql", "blockchain"],
    "Planning": ["java", "javascript", "python", "sql", "gis"],
    "Energy": ["python", "iot", "sensors", "sql", "monitoring"],
    "Water": ["python", "iot", "sensors", "sql", "monitoring"],
    "Sustainability": ["python", "sql", "excel", "reporting", "iot"],
    "Circular": ["python", "sql", "excel", "reporting", "supply chain"],
    "UX": ["javascript", "html", "css", "figma", "user research"],
    "Design": ["javascript", "html", "css", "figma", "illustrator"],
    "Accessibility": ["javascript", "html", "css", "wcag", "aria"],
    "VR": ["c#", "unity", "unreal engine", "c++", "3d modeling"],
    "AR": ["c#", "unity", "arkit", "arcore", "3d modeling"],
    "MR": ["c#", "unity", "mixed reality toolkit", "3d modeling"],
    "GIS": ["python", "javascript", "postgis", "qgis", "leaflet"],
    "Location": ["python", "javascript", "gps", "maps api", "sql"],
    "Maps": ["javascript", "python", "leaflet", "mapbox", "google maps"],
    "Remote": ["python", "r", "satellite imagery", "gis", "machine learning"],
    "Navigation": ["python", "javascript", "gps", "routing", "algorithms"],
    "Environmental": ["python", "r", "iot", "sensors", "statistics"],
    "Quantum": ["python", "qiskit", "cirq", "mathematics", "physics"],
    "Advanced": ["python", "c++", "hardware", "specialized computing"],
    "Legal": ["python", "nlp", "sql", "document processing"],
    "RealEstate": ["python", "sql", "gis", "valuation models"],
    "Academic": ["python", "javascript", "sql", "lms", "research tools"],
    "Sports": ["python", "statistics", "machine learning", "sql", "visualization"],
    "Fashion": ["python", "machine learning", "image processing", "sql"],
    "Tourism": ["java", "javascript", "python", "sql", "maps api"],
    "Food": ["java", "javascript", "python", "sql", "pos systems"],
    "Hospitality": ["java", "javascript", "python", "sql", "booking systems"],
    "Events": ["java", "javascript", "python", "sql", "ticketing"],
}

# Map areas to departments
AREA_TO_DEPT = {
    "Classification": "Computer Science", "Regression": "Computer Science", "Clustering": "Computer Science",
    "Anomaly Detection": "Computer Science", "CNN": "Computer Science", "RNN": "Computer Science",
    "Generative": "Computer Science", "Transformer": "Computer Science", "NLP-Text": "Computer Science",
    "NLP-NER": "Computer Science", "NLP-MT": "Computer Science", "NLP-QA": "Computer Science",
    "CV-Class": "Computer Science", "CV-Detection": "Computer Science", "CV-Segmentation": "Computer Science",
    "Data Mining": "Computer Science", "Time Series": "Computer Science", "Visualization": "Computer Science",
    "Analytics": "Computer Science", "Database": "Computer Science", "NoSQL": "Computer Science",
    "Recommender": "Computer Science", "Content-Based": "Computer Science", "Blockchain": "Computer Science",
    "Quantum": "Computer Science", "Data Tools": "Computer Science", "Basic Tools": "Computer Science",
    "Web Scraping": "Computer Science", "Analysis": "Computer Science", "Gaming": "Computer Science",
    "ML": "Computer Science", "Feature": "Computer Science", "Interpretability": "Computer Science",
    "Ensemble": "Computer Science", "Prediction": "Computer Science", "Neural": "Computer Science",
    "Unsupervised": "Computer Science", "Physics": "Computer Science", "Neuromorphic": "Computer Science",
    "Adversarial": "Computer Science", "Bio": "Computer Science", "Algorithms": "Computer Science",
    "Optimization": "Computer Science", "Geometry": "Computer Science", "String": "Computer Science",
    "Vision": "Computer Science", "Video": "Computer Science", "Graphics": "Computer Science",
    "Human": "Computer Science",
    "Intrusion": "Cybersecurity", "Host-IDS": "Cybersecurity", "Malware": "Cybersecurity",
    "Anomaly": "Cybersecurity", "Fraud": "Cybersecurity", "Web-Sec": "Cybersecurity",
    "IAM": "Cybersecurity", "Cloud Sec": "Cybersecurity", "IoT Sec": "Cybersecurity",
    "Crypto": "Cybersecurity", "Compliance": "Cybersecurity", "Pentest": "Cybersecurity",
    "Threat": "Cybersecurity", "Incident": "Cybersecurity", "Hunting": "Cybersecurity",
    "Forensics": "Cybersecurity", "Vulnerability": "Cybersecurity", "Privacy": "Cybersecurity",
    "Access": "Cybersecurity", "Identity": "Cybersecurity", "OT": "Cybersecurity",
    "AI Security": "Cybersecurity", "Response": "Cybersecurity", "Predictive": "Cybersecurity",
    "ERP": "Information Systems", "CRM": "Information Systems", "BPM": "Information Systems",
    "DW": "Information Systems", "Finance": "Information Systems", "HR": "Information Systems",
    "Data Gov": "Information Systems", "Enterprise": "Information Systems",
    "Advanced": "Information Systems", "Analytics": "Information Systems",
    "Strategy": "Information Systems", "Organization": "Information Systems",
    "Supply": "Information Systems", "Manufacturing": "Information Systems",
    "MRP": "Information Systems", "Quality": "Information Systems",
    "Retail": "Information Systems", "E-Commerce": "Information Systems",
    "Loyalty": "Information Systems", "Omnichannel": "Information Systems",
    "Governance": "Information Systems", "Operations": "Information Systems",
    "Market": "Information Systems", "E-Gov": "Information Systems",
    "Admin": "Information Systems", "Safety": "Information Systems",
    "Justice": "Information Systems", "Planning": "Information Systems",
    "Testing": "Software Engineering", "Architecture": "Software Engineering",
    "DevOps": "Software Engineering", "Web-App": "Software Engineering",
    "API": "Software Engineering", "Mobile": "Software Engineering",
    "Productivity": "Software Engineering", "Education": "Software Engineering",
    "Utilities": "Software Engineering", "Communication": "Software Engineering",
    "Notifications": "Software Engineering", "Business": "Software Engineering",
    "Automation": "Software Engineering", "Resilience": "Software Engineering",
    "Integration": "Software Engineering", "Social": "Software Engineering",
    "Healthcare": "Software Engineering", "EdTech": "Software Engineering",
    "Logistics": "Software Engineering", "Documents": "Software Engineering",
    "Media": "Software Engineering", "XR": "Software Engineering",
    "iOS": "Software Engineering", "PWA": "Software Engineering",
    "SPA": "Software Engineering", "Framework": "Software Engineering",
    "Backend": "Software Engineering", "Accessibility": "Software Engineering",
    "i18n": "Software Engineering", "Performance": "Software Engineering",
    "Maintenance": "Software Engineering", "Security": "Software Engineering",
    "Pattern": "Software Engineering", "Concurrent": "Software Engineering",
    "Distributed": "Software Engineering", "Collaboration": "Software Engineering",
    "Scalability": "Software Engineering", "Reactive": "Software Engineering",
    "UX": "Software Engineering", "Design": "Software Engineering",
    "VR": "Software Engineering", "AR": "Software Engineering",
    "MR": "Software Engineering", "Advanced": "Software Engineering",
    "Database": "Information Technology", "Cloud": "Information Technology",
    "IoT": "Information Technology", "SysAdmin": "Information Technology",
    "Monitoring": "Information Technology", "Environment": "Information Technology",
    "Specialized": "Information Technology", "Edge": "Information Technology",
    "Network": "Information Technology", "Storage": "Information Technology",
    "Virtualization": "Information Technology", "Container": "Information Technology",
    "IaC": "Information Technology", "Platform": "Information Technology",
    "Messaging": "Information Technology", "Observability": "Information Technology",
    "Deployment": "Information Technology", "Energy": "Information Technology",
    "Water": "Information Technology", "Sustainability": "Information Technology",
    "Circular": "Information Technology", "Green": "Information Technology",
    "Backup": "Information Technology", "Replication": "Information Technology",
    "Big Data": "Information Technology",
}

# Project catalog (parsed from the original file)
PROJECTS = []

def parse_projects():
    """Parse projects from the original CSV data"""
    projects_raw = []
    
    # Read the original file
    with open(r"C:\kb\student project\Book2.csv", 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 4 and row[0].strip():
                pid = row[0].strip()
                title = row[1].strip()
                complexity_str = row[2].strip()
                area = row[3].strip()
                
                # Parse complexity
                if "Easy" in complexity_str:
                    complexity = "Easy"
                elif "Medium" in complexity_str:
                    complexity = "Medium"
                elif "Hard" in complexity_str:
                    complexity = "Hard"
                else:
                    complexity = "Medium"
                
                # Determine department from ID prefix
                dept_prefix = pid.split("-")[0] if "-" in pid else "COS"
                dept_map = {"COS": "Computer Science", "CYB": "Cybersecurity", "SFE": "Software Engineering", 
                           "IFS": "Information Systems", "IFT": "Information Technology", "ET": "Computer Science",
                           "SD": "Software Engineering", "EXX": "Computer Science"}
                department = dept_map.get(dept_prefix, "Computer Science")
                
                # Get required skills
                skills = SKILLS_BY_AREA.get(area, ["python", "sql"])
                
                # Generate tech stack from skills
                tech_stack = ", ".join(random.sample(skills, min(len(skills), random.randint(3, 5))))
                required_skills = ", ".join(skills)
                
                # Generate description from title
                description = f"A project focused on {title.lower()}"
                
                # Times selected (popularity)
                if complexity == "Easy":
                    times_selected = random.randint(5, 30)
                elif complexity == "Medium":
                    times_selected = random.randint(2, 20)
                else:
                    times_selected = random.randint(1, 10)
                
                # Average grade given
                if complexity == "Easy":
                    avg_grade = round(random.uniform(3.0, 4.0), 2)
                elif complexity == "Medium":
                    avg_grade = round(random.uniform(2.5, 3.8), 2)
                else:
                    avg_grade = round(random.uniform(2.0, 3.5), 2)
                
                projects_raw.append({
                    "project_id": pid,
                    "title": title,
                    "description": description,
                    "category": area,
                    "department": department,
                    "difficulty": complexity,
                    "required_skills": required_skills,
                    "tech_stack": tech_stack,
                    "avg_grade_given": avg_grade,
                    "times_selected": times_selected
                })
    
    return projects_raw

def generate_students(n):
    """Generate n realistic student profiles"""
    students = []
    for i in range(1, n + 1):
        sid = f"STU-{i:04d}"
        dept = random.choice(DEPARTMENTS)
        year = random.choice(YEAR_LEVELS)
        
        # GPA distribution (normal-ish)
        gpa = round(random.gauss(3.0, 0.6), 2)
        gpa = max(1.0, min(5.0, gpa))
        
        # Average CS grade (correlated with GPA)
        avg_cs_grade = round(gpa + random.gauss(0, 0.3), 2)
        avg_cs_grade = max(1.0, min(5.0, avg_cs_grade))
        
        # Programming languages (more for higher GPA)
        num_langs = random.randint(2, min(6, int(gpa * 1.5) + 1))
        prog_langs = ", ".join(random.sample(PROGRAMMING_LANGUAGES, num_langs))
        
        # Frameworks/tools
        num_tools = random.randint(2, min(8, int(gpa * 1.5) + 2))
        frameworks = ", ".join(random.sample(FRAMEWORKS_TOOLS, num_tools))
        
        # Interests (based on department)
        dept_interests = INTERESTS_BY_DEPT[dept]
        num_interests = random.randint(2, min(5, len(dept_interests)))
        interests = ", ".join(random.sample(dept_interests, num_interests))
        
        # Preferred difficulty (based on GPA and year)
        if gpa >= 3.5 and year == 4:
            pref_diff = random.choice(["Medium", "Hard", "Hard"])
        elif gpa >= 3.0:
            pref_diff = random.choice(["Easy", "Medium", "Medium"])
        else:
            pref_diff = random.choice(["Easy", "Easy", "Medium"])
        
        # Past projects (list of project titles the student has done)
        num_past = random.randint(0, 3)
        past_projects_list = []
        past_grades_list = []
        for _ in range(num_past):
            past_grade = round(random.gauss(gpa, 0.3), 2)
            past_grade = max(1.0, min(5.0, past_grade))
            past_grades_list.append(str(past_grade))
        
        num_past_projects = len(past_projects_list)
        past_projects_str = "; ".join(past_projects_list) if past_projects_list else "None"
        past_grades_str = "; ".join(past_grades_list) if past_grades_list else "None"
        
        students.append({
            "student_id": sid,
            "department": dept,
            "year_level": year,
            "gpa": gpa,
            "avg_cs_grade": avg_cs_grade,
            "programming_languages": prog_langs,
            "frameworks_tools": frameworks,
            "interests": interests,
            "preferred_difficulty": pref_diff,
            "past_projects": past_projects_str,
            "num_past_projects": num_past_projects,
            "past_project_grades": past_grades_str
        })
    
    return students

def generate_history(students, projects):
    """Generate student-project interaction history"""
    history = []
    record_id = 1
    
    for student in students:
        # Each student has done 1-3 projects
        num_interactions = random.randint(1, 3)
        
        # Filter projects suitable for student's department
        suitable_projects = [p for p in projects if p["department"] == student["department"]]
        if not suitable_projects:
            suitable_projects = projects
        
        selected_projects = random.sample(suitable_projects, min(num_interactions, len(suitable_projects)))
        
        for project in selected_projects:
            # Grade (correlated with student GPA and project difficulty)
            base_grade = student["gpa"]
            if project["difficulty"] == "Easy":
                grade = round(base_grade + random.gauss(0.2, 0.3), 2)
            elif project["difficulty"] == "Medium":
                grade = round(base_grade + random.gauss(0, 0.4), 2)
            else:
                grade = round(base_grade + random.gauss(-0.3, 0.5), 2)
            grade = max(1.0, min(5.0, grade))
            
            # Completion status (higher GPA = more likely to complete)
            if student["gpa"] >= 3.5:
                status = random.choices(COMPLETION_STATUSES, weights=[0.85, 0.10, 0.05])[0]
            elif student["gpa"] >= 2.5:
                status = random.choices(COMPLETION_STATUSES, weights=[0.70, 0.20, 0.10])[0]
            else:
                status = random.choices(COMPLETION_STATUSES, weights=[0.50, 0.30, 0.20])[0]
            
            semester = random.choice(SEMESTERS)
            
            # Rating (1-5, correlated with grade)
            rating = min(5, max(1, int(round(grade))))
            
            # Update student's past projects
            if student["past_projects"] == "None":
                student["past_projects"] = project["title"]
                student["past_project_grades"] = str(grade)
            else:
                student["past_projects"] += "; " + project["title"]
                student["past_project_grades"] += "; " + str(grade)
            student["num_past_projects"] += 1
            
            # Update project times_selected
            project["times_selected"] += 1
            
            history.append({
                "record_id": record_id,
                "student_id": student["student_id"],
                "project_id": project["project_id"],
                "grade": grade,
                "completion_status": status,
                "semester": semester,
                "rating": rating
            })
            record_id += 1
    
    return history

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Parsing projects from original file...")
    projects = parse_projects()
    print(f"Found {len(projects)} projects")
    
    print("Generating students...")
    students = generate_students(NUM_STUDENTS)
    print(f"Generated {len(students)} students")
    
    print("Generating interaction history...")
    history = generate_history(students, projects)
    print(f"Generated {len(history)} interaction records")
    
    # Write projects CSV
    print("Writing projects.csv...")
    with open(os.path.join(OUTPUT_DIR, "projects.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["project_id", "title", "description", "category", "department", 
                                                "difficulty", "required_skills", "tech_stack", "avg_grade_given", "times_selected"])
        writer.writeheader()
        for p in projects:
            writer.writerow(p)
    
    # Write students CSV
    print("Writing students.csv...")
    with open(os.path.join(OUTPUT_DIR, "students.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "department", "year_level", "gpa", "avg_cs_grade",
                                                "programming_languages", "frameworks_tools", "interests", 
                                                "preferred_difficulty", "past_projects", "num_past_projects", "past_project_grades"])
        writer.writeheader()
        for s in students:
            writer.writerow(s)
    
    # Write history CSV
    print("Writing history.csv...")
    with open(os.path.join(OUTPUT_DIR, "history.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "student_id", "project_id", "grade", 
                                                "completion_status", "semester", "rating"])
        writer.writeheader()
        for h in history:
            writer.writerow(h)
    
    print(f"\nDataset generated successfully in: {OUTPUT_DIR}")
    print(f"  - projects.csv: {len(projects)} records")
    print(f"  - students.csv: {len(students)} records")
    print(f"  - history.csv: {len(history)} records")

if __name__ == "__main__":
    main()
