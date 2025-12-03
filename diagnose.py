"""Diagnostic script - run this to see what's being extracted from your PDF."""
import sys
import os
from dotenv import load_dotenv

load_dotenv()


# Test 1: Check what PyPDFLoader extracts
def test_pypdf(pdf_path: str):
    print("=" * 60)
    print("TEST 1: PyPDFLoader extraction")
    print("=" * 60)
    
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"Pages extracted: {len(docs)}")
    print(f"Total characters: {total_chars}")
    
    for i, doc in enumerate(docs[:3]):  # First 3 pages
        content = doc.page_content.strip()
        print(f"\n--- Page {i+1} ({len(content)} chars) ---")
        print(content[:500] if content else "[EMPTY]")
    
    return total_chars


# Test 2: Check what pymupdf/fitz extracts (better for slides)
def test_pymupdf(pdf_path: str):
    print("\n" + "=" * 60)
    print("TEST 2: PyMuPDF (fitz) extraction")
    print("=" * 60)
    
    try:
        import fitz  # pymupdf
    except ImportError:
        print("pymupdf not installed. Run: pip install pymupdf")
        return 0
    
    doc = fitz.open(pdf_path)
    total_chars = 0
    
    print(f"Pages: {len(doc)}")
    
    for i, page in enumerate(doc):
        text = page.get_text()
        total_chars += len(text)
        
        if i < 3:  # First 3 pages
            print(f"\n--- Page {i+1} ({len(text)} chars) ---")
            print(text[:500] if text.strip() else "[EMPTY]")
    
    print(f"\nTotal characters: {total_chars}")
    doc.close()
    return total_chars


# Test 3: Check ChromaDB contents
def test_chromadb():
    print("\n" + "=" * 60)
    print("TEST 3: ChromaDB contents")
    print("=" * 60)
    
    import chromadb
    from chromadb.config import Settings
    import tempfile
    import glob
    
    # Find existing chroma directories
    chroma_dirs = glob.glob("/tmp/chroma_*")
    
    if not chroma_dirs:
        print("No ChromaDB directories found in /tmp/")
        return
    
    for chroma_dir in chroma_dirs:
        print(f"\nChecking: {chroma_dir}")
        try:
            client = chromadb.PersistentClient(
                path=chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_collection("documents")
            count = collection.count()
            print(f"  Documents in collection: {count}")
            
            if count > 0:
                # Get sample
                results = collection.get(limit=3, include=["documents", "metadatas"])
                for i, doc in enumerate(results["documents"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    print(f"\n  Chunk {i+1}: {meta}")
                    print(f"  Content: {doc[:200]}...")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose.py <path_to_pdf>")
        print("\nRunning ChromaDB check only...")
        test_chromadb()
    else:
        pdf_path = sys.argv[1]
        
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
        
        pypdf_chars = test_pypdf(pdf_path)
        pymupdf_chars = test_pymupdf(pdf_path)
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS")
        print("=" * 60)
        
        if pypdf_chars < 100:
            print("❌ PyPDFLoader extracted almost nothing!")
            print("   This PDF likely has text as images/graphics.")
            if pymupdf_chars > pypdf_chars:
                print(f"   ✅ PyMuPDF extracted {pymupdf_chars} chars - USE THIS INSTEAD")
            else:
                print("   You may need OCR (pytesseract) for this PDF.")
        else:
            print(f"✅ PyPDFLoader extracted {pypdf_chars} chars")
        
        test_chromadb()
