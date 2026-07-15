import csv
import random
import os
import re

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

# ============================================================
# TASK 1: Research Areas and Tags Mapping
# ============================================================
RESEARCH_AREAS = [
    "Artificial Intelligence", "Data Science", "Cybersecurity", "Computer Vision",
    "Natural Language Processing", "Software Engineering", "Web Development",
    "Mobile Development", "Networking", "Cloud Computing", "Blockchain",
    "Internet of Things", "Embedded Systems", "Database Systems", "Algorithms and Theory"
]

# Category to Research Area mapping
CATEGORY_TO_RESEARCH_AREA = {
    # AI/ML categories
    "Classification": "Artificial Intelligence",
    "Regression": "Artificial Intelligence",
    "Clustering": "Artificial Intelligence",
    "Anomaly Detection": "Artificial Intelligence",
    "ML": "Artificial Intelligence",
    "Feature": "Data Science",
    "Interpretability": "Artificial Intelligence",
    "Ensemble": "Artificial Intelligence",
    "Prediction": "Artificial Intelligence",
    "Neural": "Artificial Intelligence",
    "Unsupervised": "Artificial Intelligence",
    "Adversarial": "Artificial Intelligence",
    
    # Deep Learning
    "CNN": "Computer Vision",
    "RNN": "Natural Language Processing",
    "Generative": "Artificial Intelligence",
    "Transformer": "Natural Language Processing",
    
    # NLP
    "NLP-Text": "Natural Language Processing",
    "NLP-NER": "Natural Language Processing",
    "NLP-MT": "Natural Language Processing",
    "NLP-QA": "Natural Language Processing",
    
    # Computer Vision
    "CV-Class": "Computer Vision",
    "CV-Detection": "Computer Vision",
    "CV-Segmentation": "Computer Vision",
    "Vision": "Computer Vision",
    "Video": "Computer Vision",
    "Graphics": "Computer Vision",
    "Human": "Computer Vision",
    
    # Cybersecurity
    "Intrusion": "Cybersecurity",
    "Host-IDS": "Cybersecurity",
    "Malware": "Cybersecurity",
    "Anomaly": "Cybersecurity",
    "Fraud": "Cybersecurity",
    "Web-Sec": "Cybersecurity",
    "IAM": "Cybersecurity",
    "Cloud Sec": "Cloud Computing",
    "IoT Sec": "Internet of Things",
    "Crypto": "Cybersecurity",
    "Compliance": "Cybersecurity",
    "Pentest": "Cybersecurity",
    "Threat": "Cybersecurity",
    "Incident": "Cybersecurity",
    "Hunting": "Cybersecurity",
    "Forensics": "Cybersecurity",
    "Vulnerability": "Cybersecurity",
    "Privacy": "Cybersecurity",
    "Access": "Cybersecurity",
    "Identity": "Cybersecurity",
    "OT": "Cybersecurity",
    "AI Security": "Cybersecurity",
    "Response": "Cybersecurity",
    "Predictive": "Data Science",
    
    # Data Science
    "Data Mining": "Data Science",
    "Time Series": "Data Science",
    "Visualization": "Data Science",
    "Analytics": "Data Science",
    "Bio": "Data Science",
    
    # Software Engineering
    "Testing": "Software Engineering",
    "Architecture": "Software Engineering",
    "DevOps": "Software Engineering",
    "Web-App": "Web Development",
    "API": "Web Development",
    "Mobile": "Mobile Development",
    "Pattern": "Software Engineering",
    "Concurrent": "Software Engineering",
    "Distributed": "Software Engineering",
    "Collaboration": "Software Engineering",
    "Scalability": "Software Engineering",
    "Reactive": "Software Engineering",
    "Performance": "Software Engineering",
    "Maintenance": "Software Engineering",
    "Security": "Cybersecurity",
    
    # Web/Mobile
    "PWA": "Web Development",
    "SPA": "Web Development",
    "Framework": "Web Development",
    "Backend": "Web Development",
    "Accessibility": "Web Development",
    "i18n": "Web Development",
    "iOS": "Mobile Development",
    "Productivity": "Mobile Development",
    "Education": "Software Engineering",
    "Utilities": "Software Engineering",
    "Communication": "Web Development",
    "Notifications": "Web Development",
    "Social": "Web Development",
    "XR": "Computer Vision",
    "VR": "Computer Vision",
    "AR": "Computer Vision",
    "MR": "Computer Vision",
    
    # Database
    "Database": "Database Systems",
    "NoSQL": "Database Systems",
    "DW": "Database Systems",
    "Data Gov": "Database Systems",
    "Data Tools": "Data Science",
    "Basic Tools": "Software Engineering",
    "Web Scraping": "Data Science",
    "Analysis": "Data Science",
    "Recommender": "Artificial Intelligence",
    "Content-Based": "Artificial Intelligence",
    
    # Blockchain
    "Blockchain": "Blockchain",
    
    # Quantum
    "Quantum": "Algorithms and Theory",
    
    # IoT/Cloud
    "IoT": "Internet of Things",
    "Edge": "Internet of Things",
    "Cloud": "Cloud Computing",
    "Container": "Cloud Computing",
    "IaC": "Cloud Computing",
    "Platform": "Cloud Computing",
    "Network": "Networking",
    "SysAdmin": "Networking",
    "Monitoring": "Networking",
    "Virtualization": "Cloud Computing",
    "Storage": "Database Systems",
    "Messaging": "Software Engineering",
    "Observability": "Software Engineering",
    "Deployment": "Cloud Computing",
    
    # Information Systems
    "ERP": "Software Engineering",
    "CRM": "Software Engineering",
    "BPM": "Software Engineering",
    "Finance": "Software Engineering",
    "HR": "Software Engineering",
    "Enterprise": "Software Engineering",
    "Advanced": "Data Science",
    "Strategy": "Data Science",
    "Organization": "Software Engineering",
    "Supply": "Software Engineering",
    "Manufacturing": "Software Engineering",
    "MRP": "Software Engineering",
    "Quality": "Software Engineering",
    "Retail": "Software Engineering",
    "E-Commerce": "Web Development",
    "Loyalty": "Software Engineering",
    "Omnichannel": "Software Engineering",
    "Governance": "Database Systems",
    "Operations": "Software Engineering",
    "Market": "Data Science",
    "E-Gov": "Software Engineering",
    "Admin": "Software Engineering",
    "Safety": "Cybersecurity",
    "Justice": "Software Engineering",
    "Planning": "Software Engineering",
    
    # Other
    "Gaming": "Software Engineering",
    "Automation": "Software Engineering",
    "Resilience": "Software Engineering",
    "Integration": "Software Engineering",
    "Healthcare": "Software Engineering",
    "EdTech": "Software Engineering",
    "Logistics": "Software Engineering",
    "Documents": "Software Engineering",
    "Media": "Software Engineering",
    "Business": "Software Engineering",
    "Energy": "Internet of Things",
    "Water": "Internet of Things",
    "Sustainability": "Internet of Things",
    "Specialized": "Internet of Things",
    "Environment": "Internet of Things",
    "Circular": "Internet of Things",
    "UX": "Web Development",
    "Design": "Web Development",
    "GIS": "Data Science",
    "Location": "Internet of Things",
    "Maps": "Data Science",
    "Remote": "Data Science",
    "Navigation": "Algorithms and Theory",
    "Environmental": "Internet of Things",
    "Physics": "Algorithms and Theory",
    "Neuromorphic": "Embedded Systems",
    "Bio": "Data Science",
    "Legal": "Natural Language Processing",
    "RealEstate": "Software Engineering",
    "Academic": "Software Engineering",
    "Sports": "Data Science",
    "Fashion": "Data Science",
    "Tourism": "Software Engineering",
    "Food": "Software Engineering",
    "Hospitality": "Software Engineering",
    "Events": "Software Engineering",
    "Algorithms": "Algorithms and Theory",
    "Optimization": "Algorithms and Theory",
    "Geometry": "Algorithms and Theory",
    "String": "Algorithms and Theory",
    "Advanced": "Algorithms and Theory",
}

# Research tags by research area
RESEARCH_TAGS_BY_AREA = {
    "Artificial Intelligence": [
        "machine learning", "deep learning", "neural networks", "classification",
        "regression", "clustering", "prediction", "pattern recognition",
        "reinforcement learning", "generative ai", "ensemble methods",
        "feature engineering", "model interpretability", "transfer learning"
    ],
    "Data Science": [
        "data analysis", "data visualization", "statistical modeling",
        "predictive analytics", "data mining", "big data", "etl",
        "business intelligence", "data warehousing", "exploratory analysis",
        "time series", "forecasting", "data preprocessing"
    ],
    "Cybersecurity": [
        "network security", "intrusion detection", "malware analysis",
        "penetration testing", "digital forensics", "cryptography",
        "vulnerability assessment", "threat intelligence", "incident response",
        "access control", "authentication", "encryption", "firewall",
        "security auditing", "compliance", "privacy"
    ],
    "Computer Vision": [
        "image classification", "object detection", "image segmentation",
        "face recognition", "video analysis", "3d reconstruction",
        "medical imaging", "augmented reality", "visual computing",
        "image processing", "feature extraction", "cnn"
    ],
    "Natural Language Processing": [
        "text classification", "sentiment analysis", "named entity recognition",
        "machine translation", "question answering", "text generation",
        "language modeling", "information extraction", "text mining",
        "speech processing", "dialogue systems", "transformers"
    ],
    "Software Engineering": [
        "software architecture", "design patterns", "testing",
        "code quality", "devops", "agile", "version control",
        "refactoring", "software maintenance", "requirements engineering",
        "software testing", "continuous integration"
    ],
    "Web Development": [
        "frontend", "backend", "full stack", "responsive design",
        "rest api", "graphql", "single page application", "progressive web app",
        "web security", "web performance", "user experience",
        "content management", "e-commerce"
    ],
    "Mobile Development": [
        "android", "ios", "cross-platform", "react native", "flutter",
        "mobile ui", "push notifications", "offline storage",
        "mobile security", "app performance", "native development"
    ],
    "Networking": [
        "network protocols", "routing", "switching", "tcp ip",
        "network management", "software defined networking", "network monitoring",
        "load balancing", "dns", "dhcp", "vpn", "network security"
    ],
    "Cloud Computing": [
        "aws", "azure", "gcp", "serverless", "containers",
        "microservices", "cloud architecture", "cloud security",
        "infrastructure as code", "cloud migration", "devops",
        "container orchestration", "service mesh"
    ],
    "Blockchain": [
        "smart contracts", "cryptocurrency", "decentralized applications",
        "consensus algorithms", "distributed ledger", "ethereum",
        "blockchain security", "token economics", "defi"
    ],
    "Internet of Things": [
        "sensor networks", "embedded systems", "iot protocols",
        "edge computing", "iot security", "smart devices",
        "industrial iot", "home automation", "wearables",
        "mqtt", "iot analytics", "device management"
    ],
    "Embedded Systems": [
        "microcontrollers", "real-time systems", "firmware",
        "hardware interfacing", "rtos", "embedded programming",
        "circuit design", "pcb design", "sensor integration",
        "low power design", "embedded linux"
    ],
    "Database Systems": [
        "sql", "nosql", "database design", "query optimization",
        "data modeling", "database administration", "replication",
        "sharding", "database security", "data migration",
        "stored procedures", "indexing"
    ],
    "Algorithms and Theory": [
        "algorithm design", "complexity analysis", "data structures",
        "graph algorithms", "optimization", "computational geometry",
        "dynamic programming", "greedy algorithms", "divide and conquer",
        "approximation algorithms", "quantum computing"
    ]
}

# ============================================================
# TASK 2: Realistic Interest-to-Skills Mapping
# ============================================================
INTEREST_TO_SKILLS = {
    # AI/ML Interests
    "machine learning": {
        "languages": ["Python", "R"],
        "frameworks": ["Scikit-learn", "TensorFlow", "PyTorch", "Pandas", "NumPy", "Jupyter", "Keras"],
        "tools": ["Git", "Docker", "Linux"]
    },
    "deep learning": {
        "languages": ["Python"],
        "frameworks": ["TensorFlow", "PyTorch", "Keras", "NumPy", "CUDA"],
        "tools": ["Git", "Docker", "Linux", "Jupyter"]
    },
    "natural language processing": {
        "languages": ["Python"],
        "frameworks": ["NLTK", "SpaCy", "Transformers", "PyTorch", "TensorFlow"],
        "tools": ["Git", "Jupyter", "Hugging Face"]
    },
    "computer vision": {
        "languages": ["Python", "C++"],
        "frameworks": ["OpenCV", "PyTorch", "TensorFlow", "Keras"],
        "tools": ["Git", "Docker", "CUDA", "Jupyter"]
    },
    "data mining": {
        "languages": ["Python", "R"],
        "frameworks": ["Pandas", "Scikit-learn", "NumPy", "Weka"],
        "tools": ["Git", "Jupyter", "SQL"]
    },
    "algorithms": {
        "languages": ["Python", "C++", "Java"],
        "frameworks": ["NumPy", "SciPy"],
        "tools": ["Git", "Linux"]
    },
    "bioinformatics": {
        "languages": ["Python", "R"],
        "frameworks": ["Biopython", "Pandas", "NumPy"],
        "tools": ["Git", "Jupyter", "BLAST"]
    },
    "neural networks": {
        "languages": ["Python"],
        "frameworks": ["PyTorch", "TensorFlow", "Keras", "NumPy"],
        "tools": ["Git", "Docker", "CUDA"]
    },
    "reinforcement learning": {
        "languages": ["Python"],
        "frameworks": ["PyTorch", "TensorFlow", "OpenAI Gym", "Stable Baselines"],
        "tools": ["Git", "Docker", "Linux"]
    },
    "generative ai": {
        "languages": ["Python"],
        "frameworks": ["PyTorch", "TensorFlow", "Transformers", "Diffusers"],
        "tools": ["Git", "Docker", "CUDA", "Hugging Face"]
    },
    "recommendation systems": {
        "languages": ["Python"],
        "frameworks": ["Scikit-learn", "Surprise", "TensorFlow", "Pandas"],
        "tools": ["Git", "SQL", "Docker"]
    },
    "graph algorithms": {
        "languages": ["Python", "C++"],
        "frameworks": ["NetworkX", "Neo4j", "Graph-tool"],
        "tools": ["Git", "Linux"]
    },
    "optimization": {
        "languages": ["Python", "C++"],
        "frameworks": ["SciPy", "PuLP", "OR-Tools", "NumPy"],
        "tools": ["Git", "Linux"]
    },
    "quantum computing": {
        "languages": ["Python"],
        "frameworks": ["Qiskit", "Cirq", "PennyLane"],
        "tools": ["Git", "Jupyter"]
    },
    
    # Cybersecurity Interests
    "network security": {
        "languages": ["Python", "C"],
        "frameworks": ["Scapy", "Wireshark"],
        "tools": ["Nmap", "Metasploit", "Linux", "TCPDump", "Snort"]
    },
    "malware analysis": {
        "languages": ["Python", "C", "Assembly"],
        "frameworks": ["YARA", "PEfile"],
        "tools": ["IDA Pro", "Ghidra", "Sandbox", "Linux", "OllyDbg"]
    },
    "cryptography": {
        "languages": ["Python", "C"],
        "frameworks": ["PyCryptodome", "OpenSSL"],
        "tools": ["Git", "Linux", "Mathematics"]
    },
    "penetration testing": {
        "languages": ["Python", "Bash"],
        "frameworks": ["Requests", "Scapy"],
        "tools": ["Metasploit", "Nmap", "Burp Suite", "John the Ripper", "Linux"]
    },
    "digital forensics": {
        "languages": ["Python"],
        "frameworks": ["Volatility", "Autopsy"],
        "tools": ["Wireshark", "FTK", "EnCase", "Linux", "Sleuth Kit"]
    },
    "incident response": {
        "languages": ["Python", "Bash"],
        "frameworks": ["Splunk", "ELK Stack"],
        "tools": ["SOAR", "SIEM", "Linux", "Windows"]
    },
    "threat intelligence": {
        "languages": ["Python"],
        "frameworks": ["MISP", "OpenCTI"],
        "tools": ["Splunk", "Elasticsearch", "Linux", "MITRE ATT&CK"]
    },
    "privacy": {
        "languages": ["Python"],
        "frameworks": ["Cryptography", "PySyft"],
        "tools": ["Linux", "SQL", "Anonymization Tools"]
    },
    "access control": {
        "languages": ["Python", "Java"],
        "frameworks": ["OAuth", "LDAP"],
        "tools": ["Active Directory", "Keycloak", "Linux"]
    },
    "compliance": {
        "languages": ["Python", "SQL"],
        "frameworks": ["Pandas", "Excel"],
        "tools": ["GRC Tools", "Linux", "Reporting"]
    },
    "IoT security": {
        "languages": ["Python", "C"],
        "frameworks": ["MQTT", "Scapy"],
        "tools": ["Wireshark", "Linux", "Embedded Systems"]
    },
    "cloud security": {
        "languages": ["Python"],
        "frameworks": ["Boto3", "Azure SDK"],
        "tools": ["AWS", "Azure", "Docker", "Kubernetes", "Terraform"]
    },
    "AI security": {
        "languages": ["Python"],
        "frameworks": ["PyTorch", "TensorFlow", "CleverHans"],
        "tools": ["Linux", "Git", "Docker"]
    },
    "operational technology security": {
        "languages": ["Python", "C"],
        "frameworks": ["Modbus", "DNP3"],
        "tools": ["SCADA", "Wireshark", "Linux"]
    },
    
    # Software Engineering Interests
    "web development": {
        "languages": ["JavaScript", "HTML", "CSS", "Python"],
        "frameworks": ["React", "Node.js", "Django", "Express", "Vue.js"],
        "tools": ["Git", "Docker", "Webpack", "NPM"]
    },
    "mobile development": {
        "languages": ["Java", "Kotlin", "Swift", "Dart"],
        "frameworks": ["Flutter", "React Native", "Android SDK", "iOS SDK"],
        "tools": ["Git", "Android Studio", "Xcode"]
    },
    "devops": {
        "languages": ["Python", "Bash", "Go"],
        "frameworks": ["Docker", "Kubernetes", "Terraform", "Ansible"],
        "tools": ["Jenkins", "GitLab CI", "Prometheus", "Grafana", "Linux"]
    },
    "testing": {
        "languages": ["Python", "Java", "JavaScript"],
        "frameworks": ["Pytest", "JUnit", "Selenium", "Jest"],
        "tools": ["Git", "Docker", "CI/CD"]
    },
    "software architecture": {
        "languages": ["Java", "Python", "C#"],
        "frameworks": ["Spring Boot", "Django", ".NET"],
        "tools": ["Docker", "Kubernetes", "Git", "UML Tools"]
    },
    "API development": {
        "languages": ["Python", "JavaScript", "Go"],
        "frameworks": ["FastAPI", "Flask", "Express", "GraphQL"],
        "tools": ["Git", "Docker", "Postman", "Swagger"]
    },
    "microservices": {
        "languages": ["Java", "Python", "Go"],
        "frameworks": ["Spring Boot", "FastAPI", "gRPC"],
        "tools": ["Docker", "Kubernetes", "Istio", "RabbitMQ"]
    },
    "cloud computing": {
        "languages": ["Python", "Go"],
        "frameworks": ["Boto3", "Terraform", "Ansible"],
        "tools": ["AWS", "Azure", "GCP", "Docker", "Kubernetes"]
    },
    "database design": {
        "languages": ["SQL", "Python"],
        "frameworks": ["SQLAlchemy", "Django ORM"],
        "tools": ["PostgreSQL", "MySQL", "MongoDB", "Redis"]
    },
    "UI/UX design": {
        "languages": ["HTML", "CSS", "JavaScript"],
        "frameworks": ["React", "Vue.js", "Figma"],
        "tools": ["Figma", "Adobe XD", "Sketch"]
    },
    "game development": {
        "languages": ["C++", "C#"],
        "frameworks": ["Unity", "Unreal Engine"],
        "tools": ["Git", "Blender", "Photoshop"]
    },
    "e-commerce": {
        "languages": ["JavaScript", "Python", "PHP"],
        "frameworks": ["React", "Node.js", "Django", "Shopify"],
        "tools": ["Git", "Docker", "SQL", "Stripe"]
    },
    "enterprise systems": {
        "languages": ["Java", "C#", "Python"],
        "frameworks": ["Spring Boot", ".NET", "SAP"],
        "tools": ["Docker", "Kubernetes", "SQL", "ERP Systems"]
    },
    "progressive web apps": {
        "languages": ["JavaScript", "HTML", "CSS"],
        "frameworks": ["React", "Vue.js", "Workbox"],
        "tools": ["Git", "Webpack", "Lighthouse"]
    },
    
    # Information Systems Interests
    "business intelligence": {
        "languages": ["SQL", "Python"],
        "frameworks": ["Pandas", "Tableau", "Power BI"],
        "tools": ["Excel", "SQL Server", "ETL Tools"]
    },
    "data analytics": {
        "languages": ["Python", "R", "SQL"],
        "frameworks": ["Pandas", "NumPy", "Matplotlib", "Seaborn"],
        "tools": ["Jupyter", "Tableau", "Excel"]
    },
    "enterprise resource planning": {
        "languages": ["Java", "SQL", "Python"],
        "frameworks": ["SAP", "Oracle ERP"],
        "tools": ["SQL", "Excel", "ERP Systems"]
    },
    "customer relationship management": {
        "languages": ["Java", "JavaScript", "Python"],
        "frameworks": ["Salesforce", "Dynamics CRM"],
        "tools": ["SQL", "Excel", "CRM Platforms"]
    },
    "business process management": {
        "languages": ["Java", "Python"],
        "frameworks": ["Camunda", "Activiti"],
        "tools": ["BPMN", "SQL", "Excel"]
    },
    "data warehouse": {
        "languages": ["SQL", "Python"],
        "frameworks": ["Snowflake", "Redshift", "BigQuery"],
        "tools": ["ETL Tools", "Airflow", "dbt"]
    },
    "supply chain": {
        "languages": ["Java", "Python", "SQL"],
        "frameworks": ["Pandas", "Excel"],
        "tools": ["ERP Systems", "SQL", "Visualization Tools"]
    },
    "decision support": {
        "languages": ["Python", "R", "SQL"],
        "frameworks": ["Scikit-learn", "Pandas"],
        "tools": ["Tableau", "Excel", "SQL"]
    },
    "data governance": {
        "languages": ["Python", "SQL"],
        "frameworks": ["Apache Atlas", "Collibra"],
        "tools": ["Metadata Tools", "SQL", "Excel"]
    },
    "e-government": {
        "languages": ["Java", "JavaScript", "Python"],
        "frameworks": ["Spring Boot", "React"],
        "tools": ["SQL", "Docker", "Cloud Platforms"]
    },
    "healthcare systems": {
        "languages": ["Java", "Python", "SQL"],
        "frameworks": ["HL7", "FHIR"],
        "tools": ["SQL", "Docker", "Healthcare Platforms"]
    },
    "retail systems": {
        "languages": ["Java", "JavaScript", "Python"],
        "frameworks": ["React", "Node.js"],
        "tools": ["SQL", "POS Systems", "E-commerce Platforms"]
    },
    
    # Information Technology Interests
    "cloud infrastructure": {
        "languages": ["Python", "Go", "Bash"],
        "frameworks": ["Terraform", "Ansible", "CloudFormation"],
        "tools": ["AWS", "Azure", "GCP", "Docker", "Kubernetes"]
    },
    "networking": {
        "languages": ["Python", "C"],
        "frameworks": ["Scapy", "Netmiko"],
        "tools": ["Wireshark", "Nmap", "GNS3", "Linux"]
    },
    "database administration": {
        "languages": ["SQL", "Python", "Bash"],
        "frameworks": ["PostgreSQL", "MySQL", "MongoDB"],
        "tools": ["pgAdmin", "MySQL Workbench", "Linux"]
    },
    "system administration": {
        "languages": ["Bash", "Python"],
        "frameworks": ["Ansible", "Puppet", "Chef"],
        "tools": ["Linux", "Windows Server", "Docker", "Monitoring Tools"]
    },
    "IoT": {
        "languages": ["Python", "C", "C++"],
        "frameworks": ["MQTT", "Raspberry Pi", "Arduino"],
        "tools": ["Linux", "Sensors", "Embedded Systems"]
    },
    "edge computing": {
        "languages": ["Python", "C++"],
        "frameworks": ["TensorFlow Lite", "ONNX Runtime"],
        "tools": ["Docker", "Kubernetes", "Edge Devices"]
    },
    "monitoring": {
        "languages": ["Python", "Go"],
        "frameworks": ["Prometheus", "Grafana", "ELK Stack"],
        "tools": ["Linux", "Docker", "Nagios"]
    },
    "virtualization": {
        "languages": ["Python", "Bash"],
        "frameworks": ["VMware", "KVM", "Hyper-V"],
        "tools": ["Docker", "Kubernetes", "Linux"]
    },
    "containerization": {
        "languages": ["Python", "Go", "Bash"],
        "frameworks": ["Docker", "Kubernetes", "Podman"],
        "tools": ["Helm", "Istio", "Linux"]
    },
    "green IT": {
        "languages": ["Python"],
        "frameworks": ["Monitoring Tools"],
        "tools": ["Linux", "Cloud Platforms", "Energy Monitoring"]
    },
    "infrastructure automation": {
        "languages": ["Python", "Bash", "Go"],
        "frameworks": ["Terraform", "Ansible", "Pulumi"],
        "tools": ["Jenkins", "GitLab CI", "Docker"]
    },
    "storage systems": {
        "languages": ["Python", "C"],
        "frameworks": ["Ceph", "MinIO"],
        "tools": ["Linux", "AWS S3", "Azure Blob"]
    }
}

# Department to primary interests mapping
DEPT_TO_INTERESTS = {
    "Computer Science": [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "data mining", "algorithms", "neural networks",
        "reinforcement learning", "generative ai", "recommendation systems",
        "graph algorithms", "optimization", "quantum computing", "bioinformatics"
    ],
    "Cybersecurity": [
        "network security", "malware analysis", "cryptography",
        "penetration testing", "digital forensics", "incident response",
        "threat intelligence", "privacy", "access control", "compliance",
        "IoT security", "cloud security", "AI security", "operational technology security"
    ],
    "Software Engineering": [
        "web development", "mobile development", "devops", "testing",
        "software architecture", "API development", "microservices",
        "cloud computing", "database design", "UI/UX design",
        "game development", "e-commerce", "enterprise systems", "progressive web apps"
    ],
    "Information Systems": [
        "business intelligence", "data analytics", "enterprise resource planning",
        "customer relationship management", "business process management",
        "data warehouse", "supply chain", "decision support", "data governance",
        "e-government", "healthcare systems", "retail systems"
    ],
    "Information Technology": [
        "cloud infrastructure", "networking", "database administration",
        "system administration", "IoT", "edge computing", "monitoring",
        "virtualization", "containerization", "green IT", "infrastructure automation",
        "storage systems"
    ]
}

# Skills required per project area (updated with consistent naming)
SKILLS_BY_AREA = {
    "Classification": ["Python", "Scikit-learn", "Pandas", "NumPy", "Matplotlib"],
    "Regression": ["Python", "Scikit-learn", "Pandas", "NumPy", "Statsmodels"],
    "Clustering": ["Python", "Scikit-learn", "Pandas", "NumPy", "Matplotlib"],
    "Anomaly Detection": ["Python", "Scikit-learn", "Pandas", "NumPy", "SciPy"],
    "CNN": ["Python", "TensorFlow", "PyTorch", "OpenCV", "Keras", "NumPy"],
    "RNN": ["Python", "TensorFlow", "PyTorch", "Keras", "NumPy"],
    "Generative": ["Python", "TensorFlow", "PyTorch", "Keras", "NumPy"],
    "Transformer": ["Python", "PyTorch", "Transformers", "Hugging Face", "NumPy"],
    "NLP-Text": ["Python", "NLTK", "SpaCy", "Scikit-learn", "Pandas"],
    "NLP-NER": ["Python", "NLTK", "SpaCy", "Transformers", "TensorFlow"],
    "NLP-MT": ["Python", "Transformers", "PyTorch", "TensorFlow"],
    "NLP-QA": ["Python", "Transformers", "PyTorch", "Elasticsearch"],
    "CV-Class": ["Python", "OpenCV", "TensorFlow", "PyTorch", "Keras"],
    "CV-Detection": ["Python", "OpenCV", "TensorFlow", "PyTorch", "YOLO"],
    "CV-Segmentation": ["Python", "OpenCV", "TensorFlow", "PyTorch", "Keras"],
    "Intrusion": ["Python", "Wireshark", "Scikit-learn", "Pandas", "Linux"],
    "Host-IDS": ["Python", "Linux", "Windows", "Sysmon", "Splunk"],
    "Malware": ["Python", "C", "Assembly", "IDA Pro", "Ghidra"],
    "Anomaly": ["Python", "Scikit-learn", "Pandas", "NumPy", "SciPy"],
    "Fraud": ["Python", "Scikit-learn", "Pandas", "NumPy", "SQL"],
    "Web-Sec": ["Python", "JavaScript", "Burp Suite", "OWASP", "SQL"],
    "IAM": ["Python", "Java", "LDAP", "OAuth", "JWT", "SQL"],
    "Cloud Sec": ["Python", "AWS", "Azure", "Docker", "Kubernetes", "Terraform"],
    "IoT Sec": ["Python", "C", "MQTT", "Linux", "Embedded Systems"],
    "Crypto": ["Python", "C", "OpenSSL", "Mathematics", "Linux"],
    "Compliance": ["Python", "SQL", "Excel", "Linux"],
    "ERP": ["Java", "SQL", "Python", "JavaScript", "SAP"],
    "CRM": ["Java", "SQL", "Python", "JavaScript", "Salesforce"],
    "BPM": ["Java", "SQL", "Python", "JavaScript", "BPMN"],
    "DW": ["SQL", "Python", "ETL", "Tableau", "Power BI"],
    "Finance": ["Java", "SQL", "Python", "JavaScript", "Excel"],
    "HR": ["Java", "SQL", "Python", "JavaScript", "Excel"],
    "Database": ["SQL", "Python", "MongoDB", "PostgreSQL", "MySQL"],
    "NoSQL": ["Python", "MongoDB", "Cassandra", "Redis", "Neo4j"],
    "Recommender": ["Python", "Scikit-learn", "Pandas", "NumPy", "Surprise"],
    "Content-Based": ["Python", "Scikit-learn", "Pandas", "NumPy", "NLTK"],
    "Blockchain": ["Solidity", "JavaScript", "Web3", "Python", "Ethereum"],
    "Quantum": ["Python", "Qiskit", "Cirq", "Mathematics"],
    "Data Tools": ["Python", "Pandas", "NumPy", "SQL", "Excel"],
    "Basic Tools": ["Python", "JavaScript", "HTML", "CSS", "SQL"],
    "Web Scraping": ["Python", "BeautifulSoup", "Selenium", "Scrapy", "JavaScript"],
    "Analysis": ["Python", "Pandas", "Matplotlib", "Seaborn", "Tableau"],
    "Gaming": ["C++", "C#", "Unity", "Unreal Engine", "OpenGL"],
    "Testing": ["Python", "Java", "Selenium", "JUnit", "Pytest"],
    "Architecture": ["Java", "Python", "Docker", "Kubernetes"],
    "DevOps": ["Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible", "Linux"],
    "Web-App": ["JavaScript", "HTML", "CSS", "React", "Node.js", "SQL"],
    "API": ["JavaScript", "Python", "Node.js", "Express", "REST", "GraphQL"],
    "Mobile": ["Java", "Kotlin", "Swift", "React Native", "Flutter", "SQL"],
    "Data Mining": ["Python", "Scikit-learn", "Pandas", "NumPy", "Weka"],
    "Time Series": ["Python", "Pandas", "NumPy", "Statsmodels", "TensorFlow"],
    "Visualization": ["Python", "Matplotlib", "Seaborn", "Plotly", "Tableau"],
    "Analytics": ["Python", "Pandas", "SQL", "Tableau", "Power BI"],
    "ML": ["Python", "Scikit-learn", "TensorFlow", "PyTorch", "Pandas"],
    "Feature": ["Python", "Scikit-learn", "Pandas", "NumPy", "Featuretools"],
    "Interpretability": ["Python", "Scikit-learn", "LIME", "SHAP", "Pandas"],
    "Ensemble": ["Python", "Scikit-learn", "XGBoost", "LightGBM", "Pandas"],
    "Prediction": ["Python", "Scikit-learn", "Pandas", "NumPy", "TensorFlow"],
    "Pentest": ["Python", "Metasploit", "Nmap", "Burp Suite", "Linux"],
    "Threat": ["Python", "Splunk", "Elasticsearch", "MITRE ATT&CK", "Linux"],
    "Incident": ["Python", "Splunk", "Forensics", "Linux", "Windows"],
    "Hunting": ["Python", "Splunk", "Elasticsearch", "YARA", "Linux"],
    "Forensics": ["Python", "Autopsy", "Volatility", "Wireshark", "Linux"],
    "Vulnerability": ["Python", "Nmap", "Burp Suite", "Metasploit", "Linux"],
    "Privacy": ["Python", "Cryptography", "Mathematics", "SQL", "Linux"],
    "Access": ["Python", "Java", "LDAP", "OAuth", "RBAC"],
    "Identity": ["Python", "Java", "Biometrics", "LDAP", "OAuth"],
    "Supply": ["Java", "SQL", "Python", "Excel", "ERP"],
    "Manufacturing": ["Java", "SQL", "Python", "PLC", "SCADA"],
    "MRP": ["Java", "SQL", "Python", "Excel", "ERP"],
    "Quality": ["Java", "SQL", "Python", "Excel", "Statistics"],
    "Container": ["Docker", "Kubernetes", "Helm", "Linux", "YAML"],
    "Cloud": ["AWS", "Azure", "GCP", "Terraform", "Docker", "Kubernetes"],
    "IaC": ["Terraform", "Ansible", "CloudFormation", "Python", "YAML"],
    "Platform": ["Docker", "Kubernetes", "Python", "Go", "Linux"],
    "PWA": ["JavaScript", "HTML", "CSS", "Service Workers", "React"],
    "SPA": ["JavaScript", "HTML", "CSS", "React", "Angular", "Vue.js"],
    "Framework": ["JavaScript", "HTML", "CSS", "Webpack", "Babel"],
    "Backend": ["Python", "Java", "JavaScript", "Node.js", "SQL", "REST"],
    "Accessibility": ["JavaScript", "HTML", "CSS", "WCAG"],
    "i18n": ["JavaScript", "HTML", "CSS", "React", "i18next"],
    "Performance": ["JavaScript", "Python", "Chrome DevTools", "Lighthouse", "SQL"],
    "Maintenance": ["Java", "Python", "JavaScript", "Git", "SonarQube"],
    "Security": ["Python", "Java", "OWASP", "Burp Suite", "SonarQube"],
    "Pattern": ["Java", "Python", "C++", "Design Patterns", "SOLID"],
    "Neural": ["Python", "PyTorch", "TensorFlow", "NumPy", "Mathematics"],
    "Unsupervised": ["Python", "PyTorch", "TensorFlow", "Scikit-learn", "NumPy"],
    "Physics": ["Python", "PyTorch", "TensorFlow", "Mathematics", "Physics"],
    "Neuromorphic": ["Python", "PyTorch", "C++", "Hardware"],
    "Adversarial": ["Python", "PyTorch", "TensorFlow"],
    "Bio": ["Python", "R", "Biopython", "Statistics", "Genomics"],
    "OT": ["Python", "C", "SCADA", "Modbus", "Industrial Protocols"],
    "Monitoring": ["Python", "Linux", "Grafana", "Prometheus", "Splunk"],
    "Concurrent": ["Java", "Python", "C++", "Go", "Rust"],
    "Distributed": ["Java", "Python", "Go", "Raft", "Paxos"],
    "Collaboration": ["JavaScript", "Python", "WebSockets"],
    "Scalability": ["Java", "Python", "Docker", "Kubernetes"],
    "Reactive": ["Java", "JavaScript", "Python", "RxJava"],
    "Vision": ["Python", "OpenCV", "PyTorch", "TensorFlow"],
    "Video": ["Python", "OpenCV", "FFmpeg", "PyTorch", "TensorFlow"],
    "Graphics": ["C++", "OpenGL", "Vulkan", "GLSL", "Unity"],
    "Human": ["Python", "OpenCV", "PyTorch", "TensorFlow", "MediaPipe"],
    "AI Security": ["Python", "PyTorch", "TensorFlow", "Scikit-learn", "Linux"],
    "Response": ["Python", "Splunk", "SOAR", "Linux"],
    "Predictive": ["Python", "Scikit-learn", "Pandas", "NumPy", "Statistics"],
    "Automation": ["Python", "Ansible", "Terraform", "Jenkins", "Linux"],
    "Retail": ["Java", "SQL", "Python", "JavaScript"],
    "E-Commerce": ["Java", "JavaScript", "SQL", "React", "Node.js"],
    "Loyalty": ["Java", "SQL", "Python", "JavaScript"],
    "Omnichannel": ["Java", "SQL", "Python", "JavaScript"],
    "Replication": ["SQL", "Python", "PostgreSQL", "MySQL", "MongoDB"],
    "Big Data": ["Python", "Hadoop", "Spark", "Kafka", "SQL"],
    "Backup": ["Python", "Linux", "SQL", "AWS", "Azure"],
    "Edge": ["Python", "C", "IoT", "Docker", "Kubernetes", "Linux"],
    "Environment": ["Python", "SQL", "Excel", "IoT", "Sensors"],
    "Specialized": ["Python", "C", "IoT", "Sensors", "Embedded Systems"],
    "Governance": ["Python", "SQL", "Excel", "Data Catalog"],
    "Strategy": ["Python", "SQL", "Excel", "Tableau", "Power BI"],
    "Organization": ["Python", "SQL", "Excel"],
    "Network": ["Python", "Linux", "TCP/IP", "SDN", "Wireshark"],
    "Storage": ["Python", "Linux", "AWS", "Azure"],
    "Virtualization": ["VMware", "Docker", "Kubernetes", "Linux", "Python"],
    "Messaging": ["Java", "Python", "Kafka", "RabbitMQ", "Redis"],
    "Integration": ["Java", "Python", "REST", "SOAP", "ETL"],
    "Observability": ["Python", "Grafana", "Prometheus", "Elasticsearch"],
    "Deployment": ["Docker", "Kubernetes", "Jenkins", "ArgoCD", "Helm"],
    "Resilience": ["Java", "Python", "Docker", "Kubernetes"],
    "Advanced": ["Python", "Java", "SQL", "Tableau", "Power BI"],
    "Social": ["JavaScript", "Python", "React", "Node.js", "MongoDB"],
    "Healthcare": ["Java", "Python", "SQL", "HL7", "FHIR"],
    "Finance": ["Java", "Python", "SQL"],
    "EdTech": ["Java", "JavaScript", "Python", "SQL"],
    "Logistics": ["Java", "Python", "SQL"],
    "Documents": ["Python", "Java", "JavaScript"],
    "Media": ["Python", "JavaScript", "FFmpeg", "OpenCV"],
    "XR": ["C#", "Unity", "Unreal Engine", "C++"],
    "iOS": ["Swift", "Xcode", "CloudKit", "ARKit", "SwiftUI"],
    "Productivity": ["Java", "JavaScript", "Python", "React Native", "Flutter"],
    "Education": ["Java", "JavaScript", "Python", "React", "SQL"],
    "Utilities": ["Java", "JavaScript", "Python", "C++"],
    "Communication": ["JavaScript", "Python", "WebSockets", "WebRTC", "Node.js"],
    "Notifications": ["JavaScript", "Python", "Firebase"],
    "Business": ["Java", "JavaScript", "Python", "SQL", "Excel"],
    "E-Gov": ["Java", "JavaScript", "Python", "SQL"],
    "Admin": ["Java", "JavaScript", "Python", "SQL", "Excel"],
    "Safety": ["Java", "Python", "SQL"],
    "Justice": ["Java", "JavaScript", "Python", "SQL"],
    "Planning": ["Java", "JavaScript", "Python", "SQL"],
    "Energy": ["Python", "IoT", "Sensors", "SQL"],
    "Water": ["Python", "IoT", "Sensors", "SQL"],
    "Sustainability": ["Python", "SQL", "Excel", "IoT"],
    "Circular": ["Python", "SQL", "Excel"],
    "UX": ["JavaScript", "HTML", "CSS", "Figma"],
    "Design": ["JavaScript", "HTML", "CSS", "Figma", "Illustrator"],
    "GIS": ["Python", "JavaScript", "PostGIS", "QGIS"],
    "Location": ["Python", "JavaScript", "GPS", "Maps API", "SQL"],
    "Maps": ["JavaScript", "Python", "Leaflet", "Mapbox"],
    "Remote": ["Python", "R", "GIS", "Machine Learning"],
    "Navigation": ["Python", "JavaScript", "GPS", "Algorithms"],
    "Environmental": ["Python", "R", "IoT", "Sensors", "Statistics"],
    "Legal": ["Python", "NLP", "SQL"],
    "RealEstate": ["Python", "SQL", "GIS"],
    "Academic": ["Python", "JavaScript", "SQL"],
    "Sports": ["Python", "Statistics", "Machine Learning", "SQL"],
    "Fashion": ["Python", "Machine Learning", "Image Processing", "SQL"],
    "Tourism": ["Java", "JavaScript", "Python", "SQL"],
    "Food": ["Java", "JavaScript", "Python", "SQL"],
    "Hospitality": ["Java", "JavaScript", "Python", "SQL"],
    "Events": ["Java", "JavaScript", "Python", "SQL"],
    "Gaming": ["C++", "C#", "Unity", "Unreal Engine", "OpenGL"],
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
    "MR": "Software Engineering",
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


def get_research_area(category):
    """Get research area from category"""
    return CATEGORY_TO_RESEARCH_AREA.get(category, "Computer Science")


def get_research_tags(research_area, category, title, skills):
    """Generate research tags based on research area, category, title, and skills"""
    tags = set()
    
    # Add tags from research area
    if research_area in RESEARCH_TAGS_BY_AREA:
        # Select 2-4 tags from the research area
        num_tags = random.randint(2, 4)
        area_tags = random.sample(RESEARCH_TAGS_BY_AREA[research_area], 
                                  min(num_tags, len(RESEARCH_TAGS_BY_AREA[research_area])))
        tags.update(area_tags)
    
    # Add category as tag
    category_tag = category.lower().replace("-", " ").replace("_", " ")
    tags.add(category_tag)
    
    # Extract keywords from title
    title_keywords = extract_keywords_from_title(title)
    tags.update(title_keywords[:2])  # Add up to 2 title keywords
    
    # Add relevant skill-based tags
    skill_tags = get_skill_based_tags(skills)
    tags.update(skill_tags[:2])  # Add up to 2 skill tags
    
    return ", ".join(sorted(tags)[:6])  # Return max 6 tags


def extract_keywords_from_title(title):
    """Extract meaningful keywords from project title"""
    # Common stop words to exclude
    stop_words = {"a", "an", "the", "and", "or", "for", "in", "on", "at", "to", "of", 
                  "with", "by", "from", "using", "based", "system", "framework", "approach",
                  "implementation", "design", "development", "analysis", "application"}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', title.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Return unique keywords
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords[:3]


def get_skill_based_tags(skills):
    """Generate tags based on required skills"""
    skill_to_tags = {
        "python": "python programming",
        "tensorflow": "deep learning",
        "pytorch": "deep learning",
        "keras": "neural networks",
        "scikit-learn": "machine learning",
        "opencv": "computer vision",
        "nltk": "natural language processing",
        "spacy": "natural language processing",
        "transformers": "transformers",
        "wireshark": "network analysis",
        "metasploit": "penetration testing",
        "nmap": "network scanning",
        "burp suite": "web security",
        "docker": "containerization",
        "kubernetes": "orchestration",
        "aws": "cloud computing",
        "azure": "cloud computing",
        "sql": "databases",
        "mongodb": "nosql",
        "react": "frontend",
        "node.js": "backend",
        "flutter": "cross-platform",
        "unity": "game development",
        "solidity": "blockchain",
        "qiskit": "quantum computing"
    }
    
    tags = []
    for skill in skills[:3]:  # Check first 3 skills
        skill_lower = skill.lower()
        if skill_lower in skill_to_tags:
            tags.append(skill_to_tags[skill_lower])
    
    return tags


def parse_projects():
    """Parse projects from the original CSV data with research areas and tags"""
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
                skills = SKILLS_BY_AREA.get(area, ["Python", "SQL"])
                
                # Generate tech stack from skills
                tech_stack = ", ".join(random.sample(skills, min(len(skills), random.randint(3, 5))))
                required_skills = ", ".join(skills)
                
                # Generate description from title
                description = f"A project focused on {title.lower()}"
                
                # Get research area and tags
                research_area = get_research_area(area)
                research_tags = get_research_tags(research_area, area, title, skills)
                
                # Times selected (popularity) - reduced influence
                if complexity == "Easy":
                    times_selected = random.randint(2, 15)
                elif complexity == "Medium":
                    times_selected = random.randint(1, 10)
                else:
                    times_selected = random.randint(0, 5)
                
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
                    "research_area": research_area,
                    "research_tags": research_tags,
                    "avg_grade_given": avg_grade,
                    "times_selected": times_selected
                })
    
    return projects_raw


def generate_students(n, projects):
    """Generate n realistic student profiles with interest-aligned skills"""
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
        
        # Select interests based on department
        dept_interests = DEPT_TO_INTERESTS.get(dept, DEPT_TO_INTERESTS["Computer Science"])
        
        # Primary interest (60-80% of skills come from here)
        primary_interest = random.choice(dept_interests)
        
        # Secondary interest (20-40% of skills come from here)
        secondary_interests = [i for i in dept_interests if i != primary_interest]
        secondary_interest = random.choice(secondary_interests) if secondary_interests else None
        
        # Build skills based on interests
        all_languages = set()
        all_frameworks = set()
        all_tools = set()
        
        # Primary interest skills (60-80%)
        if primary_interest in INTEREST_TO_SKILLS:
            primary_skills = INTEREST_TO_SKILLS[primary_interest]
            all_languages.update(primary_skills.get("languages", []))
            all_frameworks.update(primary_skills.get("frameworks", []))
            all_tools.update(primary_skills.get("tools", []))
        
        # Secondary interest skills (20-40%)
        if secondary_interest and secondary_interest in INTEREST_TO_SKILLS:
            secondary_skills = INTEREST_TO_SKILLS[secondary_interest]
            # Add a subset of secondary skills
            num_langs = random.randint(1, min(2, len(secondary_skills.get("languages", []))))
            num_frameworks = random.randint(1, min(3, len(secondary_skills.get("frameworks", []))))
            num_tools = random.randint(1, min(2, len(secondary_skills.get("tools", []))))
            
            all_languages.update(random.sample(secondary_skills.get("languages", []), 
                                               min(num_langs, len(secondary_skills.get("languages", [])))))
            all_frameworks.update(random.sample(secondary_skills.get("frameworks", []), 
                                                min(num_frameworks, len(secondary_skills.get("frameworks", [])))))
            all_tools.update(random.sample(secondary_skills.get("tools", []), 
                                           min(num_tools, len(secondary_skills.get("tools", [])))))
        
        # Add some common skills based on department
        common_skills = {
            "Computer Science": {"languages": ["Python"], "frameworks": ["Git"], "tools": ["Linux", "Jupyter"]},
            "Cybersecurity": {"languages": ["Python", "Linux"], "frameworks": ["Wireshark"], "tools": ["Nmap", "Metasploit"]},
            "Software Engineering": {"languages": ["JavaScript", "Python"], "frameworks": ["Git", "Docker"], "tools": ["VS Code"]},
            "Information Systems": {"languages": ["SQL", "Python"], "frameworks": ["Excel"], "tools": ["Tableau"]},
            "Information Technology": {"languages": ["Python", "Bash"], "frameworks": ["Docker"], "tools": ["Linux"]}
        }
        
        if dept in common_skills:
            all_languages.update(common_skills[dept]["languages"])
            all_frameworks.update(common_skills[dept]["frameworks"])
            all_tools.update(common_skills[dept]["tools"])
        
        # Convert to lists and limit size
        prog_langs = list(all_languages)[:random.randint(3, 6)]
        frameworks = list(all_frameworks)[:random.randint(3, 7)]
        tools = list(all_tools)[:random.randint(2, 5)]
        
        # Combine frameworks and tools
        frameworks_tools = frameworks + tools
        frameworks_tools = list(set(frameworks_tools))[:random.randint(4, 8)]
        
        # Build interests string
        interests_list = [primary_interest]
        if secondary_interest:
            interests_list.append(secondary_interest)
        # Add one more interest for variety
        other_interests = [i for i in dept_interests if i not in interests_list]
        if other_interests and random.random() > 0.5:
            interests_list.append(random.choice(other_interests))
        
        interests = ", ".join(interests_list)
        
        # Preferred difficulty (based on GPA and year)
        if gpa >= 3.5 and year == 4:
            pref_diff = random.choice(["Medium", "Hard", "Hard"])
        elif gpa >= 3.0:
            pref_diff = random.choice(["Easy", "Medium", "Medium"])
        else:
            pref_diff = random.choice(["Easy", "Easy", "Medium"])
        
        # Past projects (aligned with interests)
        num_past = random.randint(0, 3)
        past_projects_list = []
        past_grades_list = []
        
        # Get projects aligned with student's interests
        interest_aligned_projects = [p for p in projects 
                                     if any(interest.lower() in p["research_tags"].lower() 
                                            for interest in interests_list)]
        
        for _ in range(num_past):
            if interest_aligned_projects:
                past_project = random.choice(interest_aligned_projects)
                past_projects_list.append(past_project["title"])
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
            "programming_languages": ", ".join(prog_langs),
            "frameworks_tools": ", ".join(frameworks_tools),
            "interests": interests,
            "preferred_difficulty": pref_diff,
            "past_projects": past_projects_str,
            "num_past_projects": num_past_projects,
            "past_project_grades": past_grades_str
        })
    
    return students


def generate_history(students, projects):
    """Generate realistic student-project interaction history"""
    history = []
    record_id = 1
    
    for student in students:
        # Each student has done 1-3 projects
        num_interactions = random.randint(1, 3)
        
        # Get student interests
        student_interests = [i.strip().lower() for i in student["interests"].split(",")]
        
        # Filter projects that align with student interests
        aligned_projects = []
        for p in projects:
            # Check if project research tags match student interests
            project_tags = p["research_tags"].lower()
            interest_match = any(interest in project_tags for interest in student_interests)
            
            # Check department match
            dept_match = p["department"] == student["department"]
            
            # Prioritize aligned projects
            if interest_match or dept_match:
                aligned_projects.append(p)
        
        # If not enough aligned projects, add some from other areas
        if len(aligned_projects) < num_interactions:
            other_projects = [p for p in projects if p not in aligned_projects]
            aligned_projects.extend(random.sample(other_projects, 
                                                  min(num_interactions - len(aligned_projects), 
                                                      len(other_projects))))
        
        # Select projects
        selected_projects = random.sample(aligned_projects, min(num_interactions, len(aligned_projects)))
        
        for project in selected_projects:
            # Grade (correlated with student GPA and project difficulty)
            base_grade = student["gpa"]
            
            # Interest alignment bonus
            project_tags = project["research_tags"].lower()
            interest_bonus = 0.2 if any(interest in project_tags for interest in student_interests) else 0
            
            if project["difficulty"] == "Easy":
                grade = round(base_grade + interest_bonus + random.gauss(0.2, 0.3), 2)
            elif project["difficulty"] == "Medium":
                grade = round(base_grade + interest_bonus + random.gauss(0, 0.4), 2)
            else:
                grade = round(base_grade + interest_bonus + random.gauss(-0.3, 0.5), 2)
            grade = max(1.0, min(5.0, grade))
            
            # Completion status (higher GPA and interest alignment = more likely to complete)
            completion_bonus = 0.1 if interest_bonus > 0 else 0
            if student["gpa"] >= 3.5:
                status = random.choices(COMPLETION_STATUSES, weights=[0.85 + completion_bonus, 0.10, 0.05 - completion_bonus])[0]
            elif student["gpa"] >= 2.5:
                status = random.choices(COMPLETION_STATUSES, weights=[0.70 + completion_bonus, 0.20, 0.10 - completion_bonus])[0]
            else:
                status = random.choices(COMPLETION_STATUSES, weights=[0.50 + completion_bonus, 0.30, 0.20 - completion_bonus])[0]
            
            semester = random.choice(SEMESTERS)
            
            # Rating (1-5, correlated with grade and interest alignment)
            rating = min(5, max(1, int(round(grade + (0.5 if interest_bonus > 0 else 0)))))
            
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
    
    print("=" * 60)
    print("DATASET GENERATION WITH RESEARCH AREAS AND REALISTIC SKILLS")
    print("=" * 60)
    
    print("\n[1/4] Parsing projects from original file...")
    projects = parse_projects()
    print(f"  Found {len(projects)} projects")
    
    # Show research area distribution
    research_areas = {}
    for p in projects:
        ra = p["research_area"]
        research_areas[ra] = research_areas.get(ra, 0) + 1
    print("\n  Research Area Distribution:")
    for ra, count in sorted(research_areas.items(), key=lambda x: -x[1]):
        print(f"    {ra}: {count}")
    
    print("\n[2/4] Generating students with realistic skills...")
    students = generate_students(NUM_STUDENTS, projects)
    print(f"  Generated {len(students)} students")
    
    print("\n[3/4] Generating interaction history...")
    history = generate_history(students, projects)
    print(f"  Generated {len(history)} interaction records")
    
    # Write projects CSV
    print("\n[4/4] Writing CSV files...")
    print("  Writing projects.csv...")
    with open(os.path.join(OUTPUT_DIR, "projects.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["project_id", "title", "description", "category", "department", 
                                                "difficulty", "required_skills", "tech_stack", 
                                                "research_area", "research_tags",
                                                "avg_grade_given", "times_selected"])
        writer.writeheader()
        for p in projects:
            writer.writerow(p)
    
    # Write students CSV
    print("  Writing students.csv...")
    with open(os.path.join(OUTPUT_DIR, "students.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "department", "year_level", "gpa", "avg_cs_grade",
                                                "programming_languages", "frameworks_tools", "interests", 
                                                "preferred_difficulty", "past_projects", "num_past_projects", 
                                                "past_project_grades"])
        writer.writeheader()
        for s in students:
            writer.writerow(s)
    
    # Write history CSV
    print("  Writing history.csv...")
    with open(os.path.join(OUTPUT_DIR, "history.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "student_id", "project_id", "grade", 
                                                "completion_status", "semester", "rating"])
        writer.writeheader()
        for h in history:
            writer.writerow(h)
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - projects.csv: {len(projects)} records")
    print(f"  - students.csv: {len(students)} records")
    print(f"  - history.csv: {len(history)} records")
    
    # Show sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA VERIFICATION")
    print("=" * 60)
    
    print("\nSample Project:")
    sample_proj = projects[0]
    for key, value in sample_proj.items():
        print(f"  {key}: {value}")
    
    print("\nSample Student:")
    sample_stu = students[0]
    for key, value in sample_stu.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
