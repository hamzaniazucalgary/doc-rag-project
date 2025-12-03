"""ChromaDB wrapper for vector storage."""
import os
import shutil
import chromadb
from chromadb.config import Settings
from typing import Optional

from config import TOP_K, SIMILARITY_THRESHOLD


class VectorStore:
    """Wrapper for ChromaDB operations."""
    
    def __init__(self, persist_dir: str):
        """
        Initialize ChromaDB with persistent storage.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict]
    ) -> None:
        """Add chunks to the collection."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def query(
        self,
        query_embedding: list[float],
        n_results: int = TOP_K,
        where: Optional[dict] = None
    ) -> dict:
        """
        Query similar chunks with fallback logic.
        
        Returns dict with keys: ids, documents, metadatas, distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Flatten results (query returns nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def filter_by_threshold(self, results: dict) -> tuple[dict, bool]:
        """
        Filter results by similarity threshold.
        
        Returns: (filtered_results, is_low_confidence)
        """
        filtered = {k: [] for k in results.keys()}
        
        for i, dist in enumerate(results["distances"]):
            if dist < SIMILARITY_THRESHOLD:
                for key in results.keys():
                    filtered[key].append(results[key][i])
        
        # Fallback: if nothing passes threshold, return top-1
        if not filtered["ids"] and results["ids"]:
            for key in results.keys():
                filtered[key] = [results[key][0]]
            return filtered, True  # Low confidence flag
        
        return filtered, False
    
    def doc_exists(self, doc_id: str) -> bool:
        """Check if a document is already indexed."""
        results = self.collection.get(
            where={"doc_id": doc_id},
            limit=1
        )
        return len(results["ids"]) > 0
    
    def delete_doc(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        return len(results["ids"])
    
    def get_all_doc_ids(self) -> list[str]:
        """Get unique doc_ids in the collection."""
        results = self.collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in results["metadatas"]:
            if meta and "doc_id" in meta:
                doc_ids.add(meta["doc_id"])
        return list(doc_ids)
    
    def clear(self) -> None:
        """Clear all data and reset."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    @staticmethod
    def cleanup_persist_dir(persist_dir: str) -> None:
        """Remove persistence directory entirely."""
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
