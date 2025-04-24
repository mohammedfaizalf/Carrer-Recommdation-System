import streamlit as st
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(page_title="AI Career Guidance System", layout="wide")
st.title("AI-Based Career Guidance System")

class ScrapingConfig:
    """Configuration class for scraping parameters"""
    RETRY_COUNT = 3
    TIMEOUT = 30
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]

class WebScraper:
    """Simple webscraper with retry logic"""
    
    @staticmethod
    def get_random_headers() -> Dict[str, str]:
        """Generate random headers for request"""
        return {
            'User-Agent': random.choice(ScrapingConfig.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
    
    @staticmethod
    def fetch_page(url: str) -> Optional[str]:
        """Fetch webpage with retry logic"""
        for attempt in range(ScrapingConfig.RETRY_COUNT):
            try:
                headers = WebScraper.get_random_headers()
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=ScrapingConfig.TIMEOUT
                )
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < ScrapingConfig.RETRY_COUNT - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return None

class CareerRoadmaps:
    def __init__(self):
        self.roadmaps = {
            "Data Scientist": {
                "Foundational": [
                    "Python Programming",
                    "Statistics & Probability",
                    "Mathematics for Machine Learning",
                    "SQL Basics"
                ],
                "Intermediate": [
                    "Data Preprocessing & Analysis",
                    "Machine Learning Algorithms",
                    "Data Visualization",
                    "Feature Engineering"
                ],
                "Advanced": [
                    "Deep Learning",
                    "Natural Language Processing",
                    "Big Data Technologies",
                    "MLOps"
                ],
                "Expert": [
                    "Advanced ML Architecture Design",
                    "Research & Paper Publication",
                    "Team Leadership",
                    "Business Strategy"
                ]
            },
            "Full Stack Developer": {
                "Foundational": [
                    "HTML/CSS",
                    "JavaScript Basics",
                    "Version Control (Git)",
                    "Basic Database Concepts"
                ],
                "Intermediate": [
                    "React/Angular",
                    "Node.js",
                    "RESTful APIs",
                    "SQL & NoSQL Databases"
                ],
                "Advanced": [
                    "System Design",
                    "Cloud Services (AWS/Azure)",
                    "Security Best Practices",
                    "Performance Optimization"
                ],
                "Expert": [
                    "Architecture Patterns",
                    "Scalability & DevOps",
                    "Team Leadership",
                    "Technical Strategy"
                ]
            },
            "DevOps Engineer": {
                "Foundational": [
                    "Linux Administration",
                    "Scripting (Python/Shell)",
                    "Networking Basics",
                    "Version Control"
                ],
                "Intermediate": [
                    "Docker & Containers",
                    "CI/CD Pipelines",
                    "Infrastructure as Code",
                    "Cloud Platforms"
                ],
                "Advanced": [
                    "Kubernetes",
                    "Monitoring & Logging",
                    "Security Practices",
                    "Automation"
                ],
                "Expert": [
                    "Multi-Cloud Architecture",
                    "SRE Practices",
                    "Team Leadership",
                    "DevOps Strategy"
                ]
            },
            "AI Engineer": {
                "Foundational": [
                    "Python Programming",
                    "Linear Algebra & Calculus",
                    "Probability & Statistics",
                    "Data Structures"
                ],
                "Intermediate": [
                    "Machine Learning Basics",
                    "Deep Learning Fundamentals",
                    "Neural Networks",
                    "Computer Vision Basics"
                ],
                "Advanced": [
                    "Advanced Deep Learning",
                    "Natural Language Processing",
                    "Reinforcement Learning",
                    "MLOps & Deployment"
                ],
                "Expert": [
                    "AI Research",
                    "Custom Architecture Design",
                    "Team Leadership",
                    "AI Strategy & Ethics"
                ]
            },
            "Frontend Developer": {
                "Foundational": [
                    "HTML",
                    "CSS",
                    "JavaScript Basics",
                    "Version Control"
                ],
                "Intermediate": [
                    "React/Vue.js/Angular",
                    "Responsive Design",
                    "TypeScript",
                    "API Integration"
                ],
                "Advanced": [
                    "Performance Optimization",
                    "Advanced Framework Techniques",
                    "Web Security",
                    "Testing Frameworks"
                ],
                "Expert": [
                    "Frontend Architecture",
                    "Design Systems",
                    "Team Leadership",
                    "Technical Strategy"
                ]
            },
            "Backend Developer": {
                "Foundational": [
                    "Java/Python Basics",
                    "Data Structures & Algorithms",
                    "Basic SQL",
                    "Version Control"
                ],
                "Intermediate": [
                    "Node.js/Spring/Django",
                    "API Development",
                    "Database Management",
                    "Server-Side Logic"
                ],
                "Advanced": [
                    "System Design",
                    "Scalability",
                    "Performance Optimization",
                    "Cloud Integration"
                ],
                "Expert": [
                    "Microservices Architecture",
                    "DevOps Integration",
                    "Team Leadership",
                    "Backend Strategy"
                ]
            },
            "Mobile Developer": {
                "Foundational": [
                    "Java/Kotlin Basics",
                    "Swift Basics",
                    "Mobile UI/UX Design",
                    "Version Control"
                ],
                "Intermediate": [
                    "React Native/Flutter",
                    "Android/iOS Development",
                    "Mobile APIs",
                    "Database Integration"
                ],
                "Advanced": [
                    "Performance Optimization",
                    "Advanced Animations",
                    "App Security",
                    "Publishing Apps"
                ],
                "Expert": [
                    "Cross-Platform Strategies",
                    "Team Leadership",
                    "Mobile Architecture",
                    "Technical Strategy"
                ]
            },
            "Cloud Engineer": {
                "Foundational": [
                    "Linux Basics",
                    "Networking Basics",
                    "Scripting (Python/Bash)",
                    "Version Control"
                ],
                "Intermediate": [
                    "Cloud Basics (AWS/Azure/GCP)",
                    "Virtualization",
                    "Infrastructure as Code",
                    "Containerization"
                ],
                "Advanced": [
                    "Kubernetes",
                    "Cloud Security",
                    "Multi-Cloud Management",
                    "Monitoring & Optimization"
                ],
                "Expert": [
                    "Cloud Architecture Design",
                    "Disaster Recovery",
                    "Team Leadership",
                    "Cloud Strategy"
                ]
            },
            "Cybersecurity Specialist": {
                "Foundational": [
                    "Linux Basics",
                    "Networking & Security Fundamentals",
                    "Scripting (Python/Bash)",
                    "Version Control"
                ],
                "Intermediate": [
                    "Ethical Hacking",
                    "Penetration Testing",
                    "Firewalls & Intrusion Detection",
                    "Cryptography Basics"
                ],
                "Advanced": [
                    "Advanced Threat Detection",
                    "Incident Response",
                    "Security Automation",
                    "Compliance & Auditing"
                ],
                "Expert": [
                    "Cybersecurity Strategy",
                    "Advanced Cryptography",
                    "Team Leadership",
                    "Security Research"
                ]
            },
            "Machine Learning Engineer": {
                "Foundational": [
                    "Python Programming",
                    "Mathematics for ML",
                    "Statistics & Probability",
                    "Data Structures"
                ],
                "Intermediate": [
                    "Machine Learning Basics",
                    "Deep Learning Fundamentals",
                    "Scikit-learn",
                    "Data Engineering"
                ],
                "Advanced": [
                    "TensorFlow/PyTorch",
                    "NLP",
                    "Computer Vision",
                    "MLOps"
                ],
                "Expert": [
                    "Advanced Model Deployment",
                    "Custom AI Architectures",
                    "Team Leadership",
                    "AI Strategy"
                ]
            },
            "Blockchain Developer": {
                "Foundational": [
                    "Blockchain Basics",
                    "Smart Contracts",
                    "Version Control",
                    "Programming Basics (Solidity)"
                ],
                "Intermediate": [
                    "Ethereum Development",
                    "Web3 Integration",
                    "Cryptographic Techniques",
                    "Blockchain Testing"
                ],
                "Advanced": [
                    "Scalability Solutions",
                    "Advanced Smart Contracts",
                    "Decentralized Apps",
                    "Security in Blockchain"
                ],
                "Expert": [
                    "Custom Blockchain Solutions",
                    "Research & Development",
                    "Team Leadership",
                    "Blockchain Strategy"
                ]
            },
            "Data Engineer": {
                "Foundational": [
                    "SQL Basics",
                    "Python Programming",
                    "Data Warehousing",
                    "Version Control"
                ],
                "Intermediate": [
                    "ETL Pipelines",
                    "Big Data Processing",
                    "Hadoop/Spark",
                    "NoSQL Databases"
                ],
                "Advanced": [
                    "Kafka Integration",
                    "Data Lake Design",
                    "Performance Tuning",
                    "Real-Time Data Processing"
                ],
                "Expert": [
                    "Data Architecture Design",
                    "Data Governance",
                    "Team Leadership",
                    "Strategic Data Solutions"
                ]
            },
            "Game Developer": {
                "Foundational": [
                    "C++/C# Basics",
                    "Game Design Fundamentals",
                    "Version Control",
                    "Basic Physics"
                ],
                "Intermediate": [
                    "Unity/Unreal Engine",
                    "Scripting for Games",
                    "3D Modelling",
                    "Gameplay Mechanics"
                ],
                "Advanced": [
                    "Optimization Techniques",
                    "AI in Games",
                    "Advanced Physics Engines",
                    "Multiplayer Networking"
                ],
                "Expert": [
                    "Game Engine Development",
                    "Team Leadership",
                    "Custom Tools & Plugins",
                    "Game Development Strategy"
                ]
            },
            "QA Engineer": {
        "Foundational": [
            "Manual Testing",
            "Software Testing Fundamentals",
            "Test Planning & Execution"
        ],
        "Intermediate": [
            "Automation Testing",
            "Selenium",
            "API Testing",
            "CI/CD" 
        ],
        "Advanced": [
            "Test Automation Frameworks",
            "Performance Testing",
            "Security Testing",
            "Test Management Tools (Jira, TestRail)"
        ],
        "Expert": [
            "Test Architecture",
            "AI/ML in Testing",
            "Test Strategy & Leadership",
            "Agile Methodologies" 
        ]
    },
    "Systems Architect": {
        "Foundational": [
            "System Design Principles",
            "Software Architecture Patterns",
            "Data Structures & Algorithms" 
        ],
        "Intermediate": [
            "Microservices Architecture",
            "Cloud Computing Fundamentals (AWS, Azure, GCP)",
            "Networking Concepts"
        ],
        "Advanced": [
            "Distributed Systems",
            "Scalability & Performance",
            "Security Architecture",
            "DevOps Practices"
        ],
        "Expert": [
            "Enterprise Architecture",
            "Emerging Technologies (AI/ML, Blockchain)",
            "Leadership & Communication",
            "Business Acumen"
        ]
    },
    "AI Researcher": {
        "Foundational": [
            "Python Programming",
            "Mathematics (Linear Algebra, Calculus)",
            "Machine Learning Basics"
        ],
        "Intermediate": [
            "Deep Learning (Neural Networks, CNN, RNN)",
            "Natural Language Processing",
            "Computer Vision"
        ],
        "Advanced": [
            "Reinforcement Learning",
            "AI Ethics & Safety",
            "Research Methodology",
            "Publication & Presentations"
        ],
        "Expert": [
            "Cutting-edge AI Research",
            "Team Leadership & Mentorship",
            "AI Product Development",
            "Industry Collaboration"
        ]
    },
    "Data Analyst": {
        "Foundational": [
            "Data Analysis Fundamentals",
            "SQL Basics",
            "Data Visualization (Excel, Tableau)" 
        ],
        "Intermediate": [
            "Python for Data Analysis (Pandas, NumPy)",
            "Data Wrangling & Cleaning",
            "Statistical Analysis"
        ],
        "Advanced": [
            "Big Data Technologies (Hadoop, Spark)",
            "Data Mining & Machine Learning",
            "Business Intelligence" 
        ],
        "Expert": [
            "Data Strategy & Governance",
            "Advanced Predictive Modeling",
            "Data Storytelling & Communication",
            "Leadership & Team Management"
        ]
    },
    "Software Engineer": {
        "Foundational": [
            "Programming Fundamentals (Java, C++, Python)",
            "Data Structures & Algorithms",
            "Object-Oriented Programming"
        ],
        "Intermediate": [
            "Software Design Patterns",
            "Version Control (Git)",
            "Agile Methodologies" 
        ],
        "Advanced": [
            "Cloud Computing (AWS, Azure, GCP)",
            "Microservices Architecture",
            "DevOps Practices (CI/CD)"
        ],
        "Expert": [
            "System Design & Architecture",
            "Technical Leadership",
            "Mentorship & Coaching",
            "Industry Best Practices"
        ]
    },
    "Business Intelligence Analyst": {
        "Foundational": [
            "Data Analysis Fundamentals",
            "SQL Basics",
            "Business Concepts & Metrics" 
        ],
        "Intermediate": [
            "Data Visualization Tools (Tableau, Power BI)",
            "Data Warehousing & ETL",
            "Business Intelligence Reporting"
        ],
        "Advanced": [
            "Predictive Analytics & Machine Learning",
            "Data Governance & Security",
            "Business Strategy & Decision Making" 
        ],
        "Expert": [
            "Data-driven Decision Support",
            "Leadership & Communication",
            "Industry Expertise",
            "Data Strategy & Roadmap Development"
        ]
    },
    "Cloud Solutions Architect": {
        "Foundational": [
            "Cloud Computing Fundamentals (AWS, Azure, GCP)",
            "Networking Concepts",
            "Security Fundamentals" 
        ],
        "Intermediate": [
            "Cloud Services (Compute, Storage, Networking)",
            "Infrastructure as Code (IaC) - Terraform",
            "Containerization (Docker, Kubernetes)"
        ],
        "Advanced": [
            "Cloud Security Architecture",
            "Cloud Migration & Modernization",
            "High Availability & Disaster Recovery" 
        ],
        "Expert": [
            "Enterprise Cloud Strategy",
            "Cloud Financial Management",
            "Innovation & Emerging Technologies",
            "Leadership & Communication"
        ]
    },
    "Network Engineer": {
        "Foundational": [
            "Networking Fundamentals (TCP/IP, OSI Model)",
            "Linux/Unix Administration",
            "Network Devices (Routers, Switches)" 
        ],
        "Intermediate": [
            "Routing & Switching Protocols",
            "Network Security (Firewalls, IDS/IPS)",
            "Cloud Networking"
        ],
        "Advanced": [
            "Network Automation & Orchestration",
            "Network Performance Monitoring & Troubleshooting",
            "SD-WAN & Network Virtualization" 
        ],
        "Expert": [
            "Network Architecture & Design",
            "Security Engineering",
            "Cloud Networking Strategies",
            "Leadership & Team Management"
        ]
    },
    "Database Administrator": {
        "Foundational": [
            "Relational Databases (MySQL, PostgreSQL)",
            "SQL Fundamentals",
            "Database Design & Modeling" 
        ],
        "Intermediate": [
            "NoSQL Databases (MongoDB, Cassandra)",
            "Database Performance Tuning",
            "Database Security"
        ],
        "Advanced": [
            "Database Administration & Maintenance",
            "High Availability & Disaster Recovery",
            "Cloud Databases (AWS RDS, Azure SQL)" 
        ],
        "Expert": [
            "Database Architecture & Strategy",
            "Data Governance & Compliance",
            "Leadership & Team Management",
            "Emerging Database Technologies"
        ]
    },
    "Digital Marketing Specialist": {
        "Foundational": [
            "Digital Marketing Fundamentals",
            "SEO Basics",
            "Social Media Marketing" 
        ],
        "Intermediate": [
            "Content Marketing",
            "Google Analytics",
            "Paid Advertising (Google Ads, Social Media Ads)"
        ],
        "Advanced": [
            "Data-driven Marketing",
            "Email Marketing",
            "Marketing Automation" 
        ],
            "Expert": [
            "Digital Marketing Strategy",
            "Team Leadership & Management",
            "Marketing Analytics & ROI",
            "Emerging Digital Marketing Trends"
        ]
    }
}
    
    def get_roadmap(self, career_path: str) -> Dict[str, List[str]]:
        """Get the roadmap for a specific career path"""
        return self.roadmaps.get(career_path, {})
    
    def get_next_steps(self, career_path: str, current_skills: List[str]) -> List[str]:
        """Suggest next steps based on current skills"""
        roadmap = self.get_roadmap(career_path)
        if not roadmap:
            return []
        
        # Flatten all skills from the roadmap
        all_skills = []
        for level_skills in roadmap.values():
            all_skills.extend(level_skills)
        
        # Find missing skills
        missing_skills = [skill for skill in all_skills if skill not in current_skills]
        
        # Return top 5 most important missing skills
        return missing_skills[:5]
    
def fetch_linkedin_jobs(location: str, domain: str) -> pd.DataFrame:
    """Fetch job data from LinkedIn with descriptions"""
    jobs_data = []
    
    try:
        search_query = f"{domain} jobs {location}".replace(" ", "%20")
        url = f"https://www.linkedin.com/jobs/search?keywords={search_query}&location={location}"
        
        html_content = WebScraper.fetch_page(url)
        if not html_content:
            return pd.DataFrame()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        job_cards = soup.find_all('div', class_='base-card')
        
        for card in job_cards[:10]:
            try:
                title_elem = card.find('h3', class_='base-search-card__title')
                company_elem = card.find('h4', class_='base-search-card__subtitle')
                location_elem = card.find('span', class_='job-search-card__location')
                description_elem = card.find('div', class_='base-search-card__metadata')
                
                # Extract job link to get full description
                job_link = card.find('a', class_='base-card__full-link')
                job_description = ""
                
                if job_link:
                    job_url = job_link.get('href')
                    job_page = WebScraper.fetch_page(job_url)
                    if job_page:
                        job_soup = BeautifulSoup(job_page, 'html.parser')
                        desc_div = job_soup.find('div', class_='show-more-less-html__markup')
                        if desc_div:
                            job_description = desc_div.get_text(strip=True)
                
                jobs_data.append({
                    "Job Title": title_elem.text.strip() if title_elem else "N/A",
                    "Company": company_elem.text.strip() if company_elem else "N/A",
                    "Location": location_elem.text.strip() if location_elem else location,
                    "Description": job_description if job_description else "No description available",
                    "Source": "LinkedIn"
                })
            except Exception as e:
                logger.error(f"Error parsing LinkedIn job card: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error fetching LinkedIn jobs: {str(e)}")
    
    return pd.DataFrame(jobs_data)
                    
# Cache for storing scraped data
class DataCache:
    """Simple cache implementation with TTL"""
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())

# Initialize cache
data_cache = DataCache()

# Main Streamlit interface
st.subheader("Real-Time Job Postings by Location and Domain")
location = st.text_input("Enter Location:", "Bangalore")
domain = st.text_input("Enter Domain:", "Python Developer")

source = ("LinkedIn")

if st.button("Fetch Job Postings"):
    cache_key = f"jobs_{location}_{domain}_{source}"
    cached_data = data_cache.get(cache_key)
    
    if cached_data is not None:
        st.write(cached_data)
    else:
        with st.spinner("Fetching job data..."):
            try:
                jobs = []
                
                if source in ["LinkedIn"]:
                    linkedin_jobs = fetch_linkedin_jobs(location, domain)
                    if not linkedin_jobs.empty:
                        jobs.append(linkedin_jobs)
                
                if jobs:
                    all_jobs = pd.concat(jobs, ignore_index=True)
                    data_cache.set(cache_key, all_jobs)
                    
                    # Display results
                    st.write(f"Found {len(all_jobs)} jobs:")
                    st.dataframe(all_jobs)
                    
                    # Create a bar chart of jobs by company
                    if len(all_jobs) > 0:
                        company_counts = all_jobs['Company'].value_counts()
                        st.bar_chart(company_counts)
                else:
                    st.warning("No jobs found. Try different search criteria.")
                    
            except Exception as e:
                st.error(f"Error fetching job data: {str(e)}")
                logger.error(f"Job fetch error: {str(e)}")


# Add system status monitoring
st.sidebar.header("System Status")
if st.sidebar.button("Check System Status"):
    status = {
        "Cache Size": len(data_cache.cache),
        "Last Update": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    st.sidebar.json(status)

# Add cache control
if st.sidebar.button("Clear Cache"):
    data_cache.cache.clear()
    st.sidebar.success("Cache cleared successfully!")

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.set_device(0)  # Use first GPU
    st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("GPU not detected, using CPU")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load AI models with GPU support
@st.cache_resource
def load_models():
    try:
        spacy.require_gpu()
    except Exception as e:
        st.warning(f"GPU not available for spaCy: {e}")
    
    nlp = spacy.load("en_core_web_sm")

    skill_matcher = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    skill_matcher.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    job_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

    return nlp, skill_matcher, sentiment_analyzer, job_classifier

nlp, skill_matcher, sentiment_analyzer, job_classifier = load_models()

class AIJobAnalyzer:
    """GPU-accelerated AI-powered job analysis tools"""
    
    def __init__(self):
        self.technical_skills = {
            "Programming Languages": [
                "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift",
                "Kotlin", "Go", "Rust", "TypeScript", "HTML", "CSS", "SQL", "Lua",
                "Solidity", "R", "MATLAB", "Scala", "Perl", "Shell", "C"
            ],
            "Frameworks & Libraries": [
                "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
                "Spring", "PyTorch", "TensorFlow", "Keras", "Scikit-learn",
                "Pandas", "NumPy", "Express.js", "jQuery", "Bootstrap",
                "Laravel", "Ruby on Rails", "ASP.NET", "Tesseract"
            ],
            "Databases": [
                "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis",
                "Cassandra", "MariaDB", "DynamoDB", "Firebase"
            ],
            "Tools & Platforms": [
                "Git", "Docker", "Kubernetes", "Jenkins", "AWS", "Azure",
                "Google Cloud", "Linux", "Unix", "Windows", "Maven",
                "Gradle", "Jira", "Confluence", "VS Code", "Visual Studio",
                "IntelliJ", "Eclipse", "Jupyter", "Postman"
            ],
            "Concepts & Technologies": [
                "Machine Learning", "Deep Learning", "AI", "Blockchain",
                "DevOps", "CI/CD", "REST API", "GraphQL", "Microservices",
                "Cloud Computing", "Big Data", "Data Science", "Agile",
                "Scrum", "Testing", "Web Development", "Mobile Development",
                "System Design", "Computer Vision", "NLP"
            ]
        }
        self.batch_size = 32

        self.job_levels = {
            "entry": ["entry level", "junior", "fresher", "0-2 years", "graduate"],
            "mid": ["mid level", "intermediate", "2-5 years", "experienced"],
            "senior": ["senior", "lead", "architect", "5+ years", "manager"]
        }

    def determine_job_level(self, description: str) -> str:
        """Determine job level from description"""
        description = description.lower()
        
        # Check experience years
        experience_pattern = r'(\d+)[\+\s]*(?:years?|yrs?)'
        matches = re.findall(experience_pattern, description)
        if matches:
            years = int(matches[0])
            if years <= 2:
                return "entry level"
            elif years <= 5:
                return "mid level"
            else:
                return "senior level"
        
        # Check level keywords
        for level, keywords in self.job_levels.items():
            if any(keyword in description for keyword in keywords):
                return level
        
        return "unspecified"
        
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using improved pattern matching"""
        # Convert text to lowercase for better matching
        text_lower = text.lower()
        detected_skills = set()
        
        # Flatten the skills dictionary for easier searching
        all_skills = []
        for category in self.technical_skills.values():
            all_skills.extend(category)
        
        # Create variations of skills for better matching
        skill_variations = {}
        for skill in all_skills:
            pattern = re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE)
            if pattern.search(text_lower):
                detected_skills.add(skill)
            skill_lower = skill.lower()
            variations = [
                skill_lower,
                skill_lower.replace(' ', ''),
                skill_lower.replace('.', ''),
                skill_lower.replace('-', ''),
                skill_lower.replace('js', 'javascript'),
                skill_lower.replace('py', 'python')
            ]
            skill_variations[skill] = variations
        
        # Find skills in text
        for skill, variations in skill_variations.items():
            if any(variation in text_lower for variation in variations):
                detected_skills.add(skill)
        
        # Special case handling for combined skills
        if "html" in text_lower and "css" in text_lower:
            detected_skills.add("HTML/CSS")
        
        # Filter out generic terms that might be falsely detected
        exclude_terms = {"a", "in", "on", "at", "the", "and", "or", "for", "to", "of"}
        detected_skills = {skill for skill in detected_skills if skill.lower() not in exclude_terms}
        
        # Sort skills alphabetically for consistent output
        return sorted(list(detected_skills))
    
    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks that fit within model's max length"""
        if not text.strip():
            return []
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def analyze_job_description(self, description: str) -> Dict:
        """Perform GPU-accelerated AI analysis of job description"""
        if not description or description == "No description available":
            return {
                "skills_required": [],
                "job_level": "unknown",
                "sentiment": {"label": "neutral", "score": 0.0},
                "key_requirements": []
            }
        
        # Extract skills
        skills = self.extract_skills(description)
        
        # Determine job level
        job_level = self.determine_job_level(description)
        
        # Analyze sentiment
        try:
            sentiment_result = sentiment_analyzer(description[:512])[0]
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            sentiment_result = {"label": "neutral", "score": 0.0}
        
        # Extract key requirements
        requirements = []
        doc = nlp(description[:10000])  # Limit text length for spaCy
        for sent in doc.sents:
            if any(keyword in sent.text.lower() 
                  for keyword in ["required", "must have", "minimum", "qualification"]):
                requirements.append(sent.text.strip())
        
        return {
            "skills_required": skills,
            "job_level": job_level,
            "sentiment": sentiment_result,
            "key_requirements": requirements[:5]  # Limit to top 5 requirements
        }
    
    def generate_job_match_score(self, job_skills: List[str], 
                               candidate_skills: List[str]) -> float:
        """Calculate GPU-accelerated job match score"""
        if not job_skills or not candidate_skills:
            return 0.0
        
        with torch.cuda.amp.autocast():
            # Convert skills to embeddings on GPU
            job_embeddings = skill_matcher.encode(job_skills, 
                                                convert_to_tensor=True,
                                                max_length=512,
                                                truncation=True).to(device)
            candidate_embeddings = skill_matcher.encode(candidate_skills, 
                                                      convert_to_tensor=True,
                                                      max_length=512,
                                                      truncation=True).to(device)
            if job_embeddings.size(0) == 0 or candidate_embeddings.size(0) == 0:
                logger.error("Embeddings are empty, match score cannot be calculated.")
                return 0.0

            
            # Calculate similarity matrix on GPU
            similarity_matrix = torch.nn.functional.cosine_similarity(
                job_embeddings.unsqueeze(1),
                candidate_embeddings.unsqueeze(0)
            )
            
            # Calculate match score
            match_score = similarity_matrix.max(dim=1)[0].mean().item() * 100
        
        return round(match_score, 2)
    
    def suggest_skill_improvements(self, job_skills: List[str], candidate_skills: List[str], threshold: float = 0.5) -> List[str]:
        """Suggest skills to learn using GPU-accelerated similarity matching."""
        # Check for empty inputs
        if not job_skills:
            logger.warning("Job skills list is empty.")
            return []
        if not candidate_skills:
            logger.warning("Candidate skills list is empty.")
            return job_skills  # Suggest all job skills as missing if none are present

        try:
            # Convert skills to embeddings
            job_embeddings = skill_matcher.encode(job_skills, 
                                                  convert_to_tensor=True,
                                                  max_length=512,
                                                  truncation=True).to(device)
            candidate_embeddings = skill_matcher.encode(candidate_skills, 
                                                        convert_to_tensor=True,
                                                        max_length=512,
                                                        truncation=True).to(device)
        
            if job_embeddings.size(0) == 0 or candidate_embeddings.size(0) == 0:
                logger.error("Embeddings are empty.")
                return job_skills  # Default to all job skills as missing
        
            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(
                job_embeddings.unsqueeze(1),
                candidate_embeddings.unsqueeze(0)
            )
        
            max_similarities, _ = similarities.max(dim=1)
        
            # Find missing skills
            missing_skills = [
                job_skill for job_skill, max_sim in zip(job_skills, max_similarities)
                if max_sim.item() < threshold
                ]
            return missing_skills
    
        except Exception as e:
            logger.error(f"Error in suggest_skill_improvements: {str(e)}")
            return []

# Initialize AI analyzer
analyzer = AIJobAnalyzer()
roadmap_generator = CareerRoadmaps()

# Job search section
st.header("Smart Job Search")
location = st.text_input("Location:", "Bangalore")
domain = st.text_input("Domain:", "Python Developer")

# Resume analysis section
st.header("AI Resume Analysis")
resume_text = st.text_area("Paste your resume text for AI analysis:", height=200)

if resume_text:
    with st.spinner("Analyzing your resume with AI..."):
        resume_skills = analyzer.extract_skills(resume_text)
        st.subheader("Skills Detected")
        st.write(resume_skills)

# Job search and analysis
if st.button("Search and Analyze Jobs"):
    with st.spinner("Searching and analyzing jobs with AI..."):
        # Fetch jobs (using previous scraping code)
        jobs_df = fetch_linkedin_jobs(location, domain)  # Reuse previous scraping function
        
        if not jobs_df.empty:
            # Analyze each job with AI
            analyses = []
            for _, job in jobs_df.iterrows():
                description = job.get('Description', '')
                analysis = analyzer.analyze_job_description(description)
                
                # Calculate match score if resume provided
                if resume_text:
                    job_skills = analysis['skills_required']
                    match_score = analyzer.generate_job_match_score(
                        job_skills, resume_skills
                    )
                    analysis['match_score'] = match_score
                    
                    # Get skill improvement suggestions
                    missing_skills = analyzer.suggest_skill_improvements(
                        job_skills, resume_skills
                    )
                    analysis['skill_gaps'] = missing_skills
                
                analyses.append(analysis)
            
            # Display results
            st.subheader("AI Analysis Results")
            
            for i, (_, job) in enumerate(jobs_df.iterrows()):
                analysis = analyses[i]
                
                with st.expander(f"{job['Job Title']} at {job['Company']}"):
                    st.write("**Job Level:**", analysis['job_level'])
                    st.write("**Required Skills:**", ", ".join(analysis['skills_required']))
                    st.write("**Sentiment:**", analysis['sentiment']['label'], 
                            f"(Score: {analysis['sentiment']['score']:.2f})")
                    
                    if 'match_score' in analysis:
                        st.write("**Match Score:**", f"{analysis['match_score']}%")
                        if analysis['skill_gaps']:
                            st.write("**Suggested Skills to Learn:**", 
                                   ", ".join(analysis['skill_gaps']))
                    
                    st.write("**Key Requirements:**")
                    for req in analysis['key_requirements']:
                        st.write(f"- {req}")
            
            # Create visualizations
            if 'match_score' in analyses[0]:
                match_scores = [a['match_score'] for a in analyses]
                companies = jobs_df['Company'].tolist()
                
                match_df = pd.DataFrame({
                    'Company': companies,
                    'Match Score': match_scores
                })
                
                st.subheader("Job Match Scores")
                st.bar_chart(match_df.set_index('Company'))
            
            # Skill frequency analysis
            all_skills = []
            for analysis in analyses:
                all_skills.extend(analysis['skills_required'])
            
            skill_freq = pd.DataFrame(
                pd.Series(all_skills).value_counts()
            ).reset_index()
            skill_freq.columns = ['Skill', 'Frequency']
            
            st.subheader("Most In-Demand Skills")
            st.bar_chart(skill_freq.set_index('Skill'))
            
        else:
            st.error("No jobs found. Try different search criteria.")

# Career recommendations and roadmaps
if resume_text:
    st.header("AI Career Recommendations & Roadmaps")
    with st.spinner("Generating personalized career recommendations and roadmaps..."):
        resume_analysis = analyzer.analyze_job_description(resume_text)
        detected_skills = resume_analysis['skills_required']
        
        career_paths = {
            "Data Scientist": ["Python", "Machine Learning", "Data Science", "SQL", "R", "Pandas", "NumPy"],
            "Full Stack Developer": ["JavaScript", "React", "Node.js", "HTML", "CSS", "SQL", "Express.js", "MongoDB"],
            "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Azure", "CI/CD", "Linux", "Jenkins", "Terraform"],
            "AI Engineer": ["Python", "Deep Learning", "Machine Learning", "TensorFlow", "PyTorch", "NLP"],
            "Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "Vue.js", "Angular", "TypeScript"],
            "Backend Developer": ["Java", "Node.js", "Spring", "SQL", "NoSQL", "Python", "Ruby on Rails", "Django"],
            "Mobile Developer": ["Swift", "Kotlin", "Java", "React Native", "Android", "iOS"],
            "Cloud Engineer": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform", "Linux"],
            "Cybersecurity Specialist": ["Linux", "Security", "Firewalls", "Ethical Hacking", "Penetration Testing", "NLP"],
            "Machine Learning Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "NLP"],
            "Blockchain Developer": ["Solidity", "Blockchain", "Ethereum", "Smart Contracts", "Web3", "Cryptography"],
            "Data Engineer": ["Python", "SQL", "ETL", "Big Data", "Hadoop", "Spark", "MongoDB", "Kafka"],
            "Game Developer": ["C++", "C#", "Unity", "Unreal Engine", "Game Design", "MATLAB", "Python"],
            "QA Engineer": ["Automation Testing", "Selenium", "Jenkins", "Python", "Manual Testing", "CI/CD"],
            "Systems Architect": ["System Design", "Microservices", "Cloud Computing", "DevOps", "Agile", "CI/CD"],
            "AI Researcher": ["Python", "Deep Learning", "AI", "TensorFlow", "PyTorch", "Matlab", "Computer Vision"],
            "Data Analyst": ["Python", "SQL", "Data Analysis", "Excel", "Pandas", "Data Visualization"],
            "Software Engineer": ["Java", "C", "C++", "Python", "Git", "Jenkins", "Agile", "Linux"],
            "Business Intelligence Analyst": ["SQL", "Data Analysis", "Excel", "Tableau", "Power BI", "Python"],
            "Cloud Solutions Architect": ["AWS", "Azure", "Google Cloud", "Terraform", "Kubernetes", "Microservices"],
            "Network Engineer": ["Linux", "Networking", "Security", "Cloud Computing", "Cisco", "AWS", "DevOps"],
            "Database Administrator": ["MySQL", "PostgreSQL", "MongoDB", "SQL", "Oracle", "Redis", "MariaDB"],
            "Digital Marketing Specialist": ["SEO", "Content Marketing", "Google Analytics", "Python", "Data Science", "Agile"]
        }
        
        # Calculate match scores and display recommendations with roadmaps
        career_matches = {}
        for career, skills in career_paths.items():
            match_score = analyzer.generate_job_match_score(skills, detected_skills)
            career_matches[career] = match_score
        
        for career, score in sorted(career_matches.items(), key=lambda x: x[1], reverse=True):
            with st.expander(f"{career} - {score:.1f}% match"):
                # Display career roadmap
                roadmap = roadmap_generator.get_roadmap(career)
                for level, skills in roadmap.items():
                    st.subheader(level)
                    for skill in skills:
                        if skill in detected_skills:
                            st.markdown(f"✅ {skill}")
                        else:
                            st.markdown(f"⭕ {skill}")
                
                # Show next recommended steps
                st.subheader("Recommended Next Steps")
                next_steps = roadmap_generator.get_next_steps(career, detected_skills)
                for step in next_steps:
                    st.write(f"- Focus on learning: {step}")

st.sidebar.title("About AI Features")
st.sidebar.markdown("""
This system uses several AI components:
- BERT-based skill matching
- Sentiment analysis of job descriptions
- NLP-based skill extraction
- AI job category prediction
- Semantic similarity for job matching
- Career path recommendation engine
""")

# Display some tips
st.sidebar.header("Search Tips")
st.sidebar.markdown("""
- Try different job titles (e.g., "Software Engineer" vs "Developer")
- Use common location names (e.g., "Bangalore" instead of "Bengaluru")
- Include specific skills in the domain (e.g., "Python Developer")
""")
