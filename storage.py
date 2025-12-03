import os
import shutil
import chromadb
from chromadb.config import Settings
from typing import Optional
from config import TOP_K, SIMILARITY_THRESHOLD


class VectorStore:
    def __init__(self, persist_dir: str):
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
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def filter_by_threshold(self, results: dict) -> tuple[dict, bool]:
        filtered = {k: [] for k in results.keys()}
        
        for i, dist in enumerate(results["distances"]):
            if dist < SIMILARITY_THRESHOLD:
                for key in results.keys():
                    filtered[key].append(results[key][i])
        
        if not filtered["ids"] and results["ids"]:
            for key in results.keys():
                filtered[key] = [results[key][0]]
            return filtered, True
        
        return filtered, False
    
    def doc_exists(self, doc_id: str) -> bool:
        results = self.collection.get(
            where={"doc_id": doc_id},
            limit=1
        )
        return len(results["ids"]) > 0
    
    def delete_doc(self, doc_id: str) -> int:
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        return len(results["ids"])
    
    def get_all_doc_ids(self) -> list[str]:
        results = self.collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in results["metadatas"]:
            if meta and "doc_id" in meta:
                doc_ids.add(meta["doc_id"])
        return list(doc_ids)
    
    def clear(self) -> None:
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    @staticmethod
    def cleanup_persist_dir(persist_dir: str) -> None:
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
