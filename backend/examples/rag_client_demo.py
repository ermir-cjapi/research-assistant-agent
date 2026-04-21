"""
Example: Client Demo for RAG-Enhanced Research Assistant

This demonstrates how to interact with the RAG-enhanced server API.
Shows document upload, knowledge base management, and questioning.

Usage:
1. Start the server: python examples/rag_enhanced_server.py
2. Run this client: python examples/rag_client_demo.py
"""

import requests
import json
import time
from pathlib import Path
import tempfile

# Server configuration
BASE_URL = "http://127.0.0.1:8000"

class RAGClient:
    """Client for interacting with RAG-Enhanced Research Assistant API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        
    def check_server(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except:
            return False
    
    def upload_document(self, file_path: str) -> dict:
        """Upload a document to the knowledge base."""
        with open(file_path, 'rb') as f:
            files = {"file": (Path(file_path).name, f, "text/plain")}
            response = requests.post(f"{self.base_url}/upload", files=files)
            response.raise_for_status()
            return response.json()
    
    def upload_text_content(self, content: str, filename: str) -> dict:
        """Upload text content as a document."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            result = self.upload_document(temp_path)
            return result
        finally:
            Path(temp_path).unlink()
    
    def get_knowledge_base_info(self) -> dict:
        """Get information about the knowledge base."""
        response = requests.get(f"{self.base_url}/knowledge-base")
        response.raise_for_status()
        return response.json()
    
    def ask_question(self, question: str, thread_id: str = "demo") -> dict:
        """Ask a question to the research assistant."""
        data = {
            "question": question,
            "thread_id": thread_id
        }
        response = requests.post(f"{self.base_url}/ask", json=data)
        response.raise_for_status()
        return response.json()
    
    def search_knowledge_base(self, query: str, limit: int = 4) -> dict:
        """Search the knowledge base directly."""
        params = {"query": query, "limit": limit}
        response = requests.get(f"{self.base_url}/search", params=params)
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, doc_id: str) -> dict:
        """Delete a document from the knowledge base."""
        response = requests.delete(f"{self.base_url}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()


def create_sample_documents():
    """Create sample documents for testing."""
    
    documents = {
        "machine_learning_basics.txt": """
        # Machine Learning Fundamentals
        
        Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.
        
        ## Types of Machine Learning
        
        1. **Supervised Learning**: Uses labeled data to train models
           - Classification: Predicting categories (spam/not spam)
           - Regression: Predicting continuous values (house prices)
           
        2. **Unsupervised Learning**: Finds patterns in unlabeled data
           - Clustering: Grouping similar data points
           - Dimensionality Reduction: Simplifying complex data
           
        3. **Reinforcement Learning**: Learns through interaction and feedback
           - Used in game playing, robotics, autonomous vehicles
        
        ## Popular Algorithms
        - Linear Regression: Simple relationship modeling
        - Decision Trees: Rule-based decision making
        - Neural Networks: Complex pattern recognition
        - Support Vector Machines: Classification with optimal boundaries
        - Random Forest: Ensemble of decision trees
        
        ## Applications
        Machine learning powers modern applications like recommendation systems, image recognition, natural language processing, and predictive analytics.
        """,
        
        "data_science_workflow.txt": """
        # Data Science Project Workflow
        
        A typical data science project follows a structured approach to extract insights from data.
        
        ## 1. Problem Definition
        - Define business objectives
        - Identify success metrics
        - Understand stakeholder needs
        
        ## 2. Data Collection
        - Gather relevant data sources
        - Consider data quality and completeness
        - Plan for data storage and access
        
        ## 3. Data Exploration and Cleaning
        - Exploratory Data Analysis (EDA)
        - Handle missing values
        - Remove or correct inconsistencies
        - Feature engineering
        
        ## 4. Modeling
        - Select appropriate algorithms
        - Train and validate models
        - Hyperparameter tuning
        - Model evaluation metrics
        
        ## 5. Deployment and Monitoring
        - Production deployment
        - Performance monitoring
        - Model retraining schedules
        - A/B testing
        
        ## Tools and Technologies
        - Python: pandas, scikit-learn, numpy
        - R: statistical computing
        - SQL: database querying
        - Visualization: matplotlib, seaborn, Tableau
        - Cloud: AWS, GCP, Azure
        """,
        
        "ai_ethics_guide.txt": """
        # AI Ethics and Responsible Development
        
        As artificial intelligence becomes more prevalent, ethical considerations become increasingly important.
        
        ## Core Principles
        
        1. **Fairness**: AI systems should not discriminate
           - Avoid bias in training data
           - Test for disparate impact across groups
           - Ensure equitable outcomes
        
        2. **Transparency**: Users should understand AI decisions
           - Explainable AI techniques
           - Clear documentation
           - Open communication about limitations
        
        3. **Privacy**: Protect user data and personal information
           - Data minimization principles
           - Secure data handling
           - User consent and control
        
        4. **Accountability**: Clear responsibility for AI decisions
           - Human oversight and intervention
           - Audit trails and logging
           - Clear governance structures
        
        ## Common Ethical Challenges
        
        - **Algorithmic Bias**: Models reflecting historical inequalities
        - **Privacy Violations**: Unauthorized use of personal data
        - **Job Displacement**: Automation replacing human workers
        - **Misuse Potential**: Deepfakes, surveillance, manipulation
        
        ## Best Practices
        
        - Diverse development teams
        - Regular bias testing and mitigation
        - Stakeholder engagement throughout development
        - Continuous monitoring post-deployment
        - Clear policies and guidelines
        
        ## Regulatory Landscape
        Emerging regulations like EU AI Act, GDPR, and various national AI strategies are shaping responsible AI development globally.
        """
    }
    
    return documents


def run_comprehensive_demo():
    """Run a comprehensive demonstration of RAG capabilities."""
    
    client = RAGClient()
    
    print("🤖 RAG-Enhanced Research Assistant Client Demo")
    print("=" * 60)
    
    # Check server
    if not client.check_server():
        print("❌ Server not running. Please start: python examples/rag_enhanced_server.py")
        return
    
    print("✅ Server is running")
    
    # Check initial knowledge base
    print("\n📊 Initial Knowledge Base Status:")
    kb_info = client.get_knowledge_base_info()
    print(f"   Documents: {kb_info['total_documents']}")
    print(f"   Chunks: {kb_info['total_chunks']}")
    
    # Upload sample documents
    print("\n📤 Uploading sample documents...")
    sample_docs = create_sample_documents()
    uploaded_docs = []
    
    for filename, content in sample_docs.items():
        try:
            result = client.upload_text_content(content, filename)
            uploaded_docs.append(result)
            print(f"   ✅ {filename}: {result['chunks_created']} chunks")
        except Exception as e:
            print(f"   ❌ {filename}: {e}")
    
    # Check updated knowledge base
    print("\n📊 Updated Knowledge Base Status:")
    kb_info = client.get_knowledge_base_info()
    print(f"   Documents: {kb_info['total_documents']}")
    print(f"   Chunks: {kb_info['total_chunks']}")
    
    print("\n   Document Details:")
    for doc_id, info in kb_info['documents'].items():
        print(f"     - {info['filename']}: {info['chunks']} chunks")
    
    # Demo questions
    print("\n" + "=" * 60)
    print("🔍 QUESTION ANSWERING DEMOS")
    print("=" * 60)
    
    demo_questions = [
        {
            "question": "What are the main types of machine learning?",
            "description": "Knowledge Base Query - ML Types"
        },
        {
            "question": "What are the steps in a typical data science workflow?",
            "description": "Knowledge Base Query - Data Science"
        },
        {
            "question": "What are the core principles of AI ethics?",
            "description": "Knowledge Base Query - AI Ethics"
        },
        {
            "question": "Compare supervised learning with what Wikipedia says about it",
            "description": "Multi-source Comparison"
        },
        {
            "question": "How many types of machine learning are mentioned in my documents? Calculate 2024 + that number.",
            "description": "Knowledge Base + Calculator"
        },
        {
            "question": "Search my knowledge base for information about bias, then tell me what Wikipedia says about algorithmic bias",
            "description": "Sequential Multi-source Query"
        }
    ]
    
    thread_id = "comprehensive_demo"
    
    for i, demo in enumerate(demo_questions, 1):
        print(f"\n--- Question {i}: {demo['description']} ---")
        print(f"Q: {demo['question']}")
        
        try:
            result = client.ask_question(demo['question'], thread_id)
            print(f"\nA: {result['answer']}")
            
            if result['sources_used']:
                print(f"\n📚 Sources Used: {', '.join(result['sources_used'])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()  # Add spacing
        time.sleep(1)  # Prevent rate limiting
    
    # Direct search demo
    print("\n" + "=" * 60)
    print("🔍 DIRECT KNOWLEDGE BASE SEARCH DEMO")
    print("=" * 60)
    
    search_queries = [
        "neural networks",
        "data cleaning",
        "algorithmic bias",
        "fairness in AI"
    ]
    
    for query in search_queries:
        print(f"\n--- Searching: '{query}' ---")
        try:
            results = client.search_knowledge_base(query, limit=2)
            print(f"Found {len(results['results'])} relevant chunks:")
            
            for j, result in enumerate(results['results'], 1):
                filename = result['metadata'].get('original_filename', 'Unknown')
                content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"\n   {j}. From: {filename}")
                print(f"      {content_preview}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Memory/conversation demo
    print("\n" + "=" * 60)
    print("🧠 CONVERSATION MEMORY DEMO")
    print("=" * 60)
    
    memory_thread = "memory_demo"
    
    conversation = [
        "What is supervised learning?",
        "What are some examples of that?",
        "How does it differ from unsupervised learning?",
        "Can you summarize what we've discussed?"
    ]
    
    for i, question in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Q: {question}")
        
        try:
            result = client.ask_question(question, memory_thread)
            print(f"A: {result['answer']}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETED")
    print("=" * 60)
    
    print("\n🎯 Key Features Demonstrated:")
    print("   ✅ Document upload and processing")
    print("   ✅ Vector search and retrieval") 
    print("   ✅ Multi-source information synthesis")
    print("   ✅ Tool integration (calculator, Wikipedia)")
    print("   ✅ Conversation memory across turns")
    print("   ✅ Source attribution")
    print("   ✅ Direct knowledge base search")
    
    print("\n💡 Next Steps:")
    print("   - Try uploading your own documents")
    print("   - Experiment with complex multi-step questions")
    print("   - Integrate with your own applications via the API")
    print("   - Customize the tools and prompts for your domain")


if __name__ == "__main__":
    run_comprehensive_demo()