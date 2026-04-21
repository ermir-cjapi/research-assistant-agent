"""
Test script to verify RAG dependencies and setup

Run this to check if all RAG-related packages are installed correctly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    
    tests = []
    
    # Core LangChain
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        tests.append(("✅ langchain_openai", "OpenAI integration working"))
    except ImportError as e:
        tests.append(("❌ langchain_openai", f"Import error: {e}"))
    
    # Document loaders
    try:
        from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
        tests.append(("✅ document_loaders", "PDF, text, and web loaders working"))
    except ImportError as e:
        tests.append(("❌ document_loaders", f"Import error: {e}"))
    
    # Vector stores
    try:
        from langchain_community.vectorstores import FAISS
        tests.append(("✅ faiss", "FAISS vector store working"))
    except ImportError as e:
        tests.append(("❌ faiss", f"Import error: {e}"))
    
    try:
        from langchain_community.vectorstores import Chroma
        tests.append(("✅ chroma", "Chroma vector store working"))
    except ImportError as e:
        tests.append(("❌ chroma", f"Import error: {e}"))
    
    # Text processing
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        tests.append(("✅ text_splitter", "Text splitting working"))
    except ImportError as e:
        tests.append(("❌ text_splitter", f"Import error: {e}"))
    
    # Supporting libraries
    try:
        import pypdf
        tests.append(("✅ pypdf", "PDF processing working"))
    except ImportError as e:
        tests.append(("❌ pypdf", f"Import error: {e}"))
    
    try:
        import bs4
        tests.append(("✅ beautifulsoup4", "HTML parsing working"))
    except ImportError as e:
        tests.append(("❌ beautifulsoup4", f"Import error: {e}"))
    
    # LangGraph
    try:
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import ToolNode
        tests.append(("✅ langgraph", "Graph workflows working"))
    except ImportError as e:
        tests.append(("❌ langgraph", f"Import error: {e}"))
    
    return tests

def test_basic_functionality():
    """Test basic RAG functionality without API calls."""
    
    tests = []
    
    # Test text splitter
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This is a test document. " * 20
        chunks = splitter.split_text(text)
        
        if len(chunks) > 1:
            tests.append(("✅ text_splitting", f"Successfully split text into {len(chunks)} chunks"))
        else:
            tests.append(("⚠️ text_splitting", "Text splitting works but no chunks created"))
            
    except Exception as e:
        tests.append(("❌ text_splitting", f"Error: {e}"))
    
    # Test document creation
    try:
        from langchain_core.documents import Document
        
        doc = Document(
            page_content="Test content",
            metadata={"source": "test", "date": "2024"}
        )
        
        tests.append(("✅ document_creation", "Document objects working"))
        
    except Exception as e:
        tests.append(("❌ document_creation", f"Error: {e}"))
    
    # Test tool creation
    try:
        from langchain_core.tools import tool
        
        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Processed: {query}"
        
        result = test_tool.invoke({"query": "test"})
        tests.append(("✅ tool_creation", "Tool creation and invocation working"))
        
    except Exception as e:
        tests.append(("❌ tool_creation", f"Error: {e}"))
    
    return tests

def main():
    print("🧪 RAG Setup Test")
    print("=" * 50)
    
    # Test imports
    print("\n📦 Testing Package Imports:")
    import_tests = test_imports()
    
    for status, message in import_tests:
        print(f"   {status} {message}")
    
    # Count results
    passed = sum(1 for status, _ in import_tests if status.startswith("✅"))
    failed = sum(1 for status, _ in import_tests if status.startswith("❌"))
    
    print(f"\n📊 Import Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n💡 To fix import errors, run:")
        print("   pip install -r requirements.txt")
        print("\n   Or install specific packages:")
        if any("faiss" in msg for _, msg in import_tests if "❌" in _):
            print("   pip install faiss-cpu")
        if any("pypdf" in msg for _, msg in import_tests if "❌" in _):
            print("   pip install pypdf")
        if any("chroma" in msg for _, msg in import_tests if "❌" in _):
            print("   pip install chromadb")
        if any("beautifulsoup4" in msg for _, msg in import_tests if "❌" in _):
            print("   pip install beautifulsoup4 lxml")
    
    # Test functionality
    print("\n⚙️ Testing Basic Functionality:")
    func_tests = test_basic_functionality()
    
    for status, message in func_tests:
        print(f"   {status} {message}")
    
    # Count results
    passed_func = sum(1 for status, _ in func_tests if status.startswith("✅"))
    failed_func = sum(1 for status, _ in func_tests if status.startswith("❌"))
    
    print(f"\n📊 Functionality Results: {passed_func} passed, {failed_func} failed")
    
    # Overall result
    total_passed = passed + passed_func
    total_failed = failed + failed_func
    
    print(f"\n🎯 Overall: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("\n🎉 All tests passed! RAG setup is ready.")
        print("\nNext steps:")
        print("   1. Set up your .env file with OPENAI_API_KEY")
        print("   2. Run: python examples/rag_with_langchain.py")
        print("   3. Or start server: python examples/rag_enhanced_server.py")
    else:
        print(f"\n⚠️ {total_failed} issues found. Please fix before using RAG features.")

if __name__ == "__main__":
    main()