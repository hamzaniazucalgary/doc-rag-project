"""
Evaluation framework for RAG pipeline.
Implements LLM-as-judge metrics for faithfulness, relevancy, and retrieval accuracy.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import LLM_MODEL, FAITHFULNESS_PROMPT, RELEVANCY_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """Single evaluation test case."""
    question: str
    expected_answer: str
    doc_id: Optional[str] = None
    expected_page: Optional[int] = None
    tags: list[str] = field(default_factory=list)  # e.g., ["factual", "multi-hop"]
    id: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        # Support both new (input/expected_output) and old (question/expected_answer) schemas
        question = data.get("input") or data.get("question")
        expected_answer = data.get("expected_output") or data.get("expected_answer")
        
        if not question or not expected_answer:
            raise ValueError("Test case must contain 'input'/'question' and 'expected_output'/'expected_answer'")
            
        return cls(
            question=question,
            expected_answer=expected_answer,
            doc_id=data.get("doc_id"),
            expected_page=data.get("expected_page"),
            tags=data.get("tags", []),
            id=data.get("id"),
            description=data.get("description")
        )


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    question: str
    generated_answer: str
    expected_answer: str
    retrieved_pages: list[int]
    expected_page: Optional[int]
    faithfulness_score: float
    relevancy_score: float
    retrieval_hit: bool
    latency_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    """Complete evaluation report."""
    results: list[EvalResult]
    avg_faithfulness: float
    avg_relevancy: float
    retrieval_accuracy: float
    avg_latency_ms: float
    n_cases: int
    n_errors: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "summary": {
                "avg_faithfulness": round(self.avg_faithfulness, 3),
                "avg_relevancy": round(self.avg_relevancy, 3),
                "retrieval_accuracy": round(self.retrieval_accuracy, 3),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "n_cases": self.n_cases,
                "n_errors": self.n_errors,
                "timestamp": self.timestamp
            },
            "results": [r.to_dict() for r in self.results]
        }
    
    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self):
        """Print formatted summary to console."""
        print("\n" + "=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)
        print(f"Test Cases:         {self.n_cases}")
        print(f"Errors:             {self.n_errors}")
        print("-" * 50)
        print(f"Avg Faithfulness:   {self.avg_faithfulness:.3f}")
        print(f"Avg Relevancy:      {self.avg_relevancy:.3f}")
        print(f"Retrieval Accuracy: {self.retrieval_accuracy:.1%}")
        print(f"Avg Latency:        {self.avg_latency_ms:.0f}ms")
        print("=" * 50 + "\n")


class RAGEvaluator:
    """
    Evaluates RAG pipeline using LLM-as-judge approach.
    
    Metrics:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Relevancy: Does the answer address the question?
    - Retrieval Accuracy: Did we retrieve the correct source page?
    """
    
    def __init__(
        self,
        test_cases: list[EvalCase],
        judge_model: str = LLM_MODEL,
        verbose: bool = False
    ):
        self.test_cases = test_cases
        self.llm = ChatOpenAI(model=judge_model, temperature=0)
        self.verbose = verbose
    
    @classmethod
    def from_json(cls, path: str, **kwargs) -> "RAGEvaluator":
        """Load test cases from JSON file."""
        with open(path) as f:
            data = json.load(f)
        cases = [EvalCase.from_dict(c) for c in data]
        return cls(test_cases=cases, **kwargs)
    
    def evaluate_single(
        self,
        case: EvalCase,
        pipeline_func: Callable[[str], dict]
    ) -> EvalResult:
        """
        Evaluate a single test case.
        
        Args:
            case: Test case to evaluate
            pipeline_func: Function that takes question string and returns dict with:
                - "answer": generated answer string
                - "context": retrieved context string
                - "chunks": list of chunk dicts with "page" key
        
        Returns:
            EvalResult with all metrics
        """
        import time
        
        start_time = time.time()
        error = None
        
        try:
            result = pipeline_func(case.question)
            latency_ms = (time.time() - start_time) * 1000
            
            answer = result.get("answer", "")
            context = result.get("context", "")
            chunks = result.get("chunks", [])
            
            # Extract retrieved pages
            retrieved_pages = [c.get("page", 0) for c in chunks if isinstance(c, dict)]
            
            # LLM-as-judge scoring
            faithfulness = self._score_faithfulness(context, answer)
            relevancy = self._score_relevancy(case.question, answer)
            
            # Retrieval accuracy
            retrieval_hit = (
                case.expected_page in retrieved_pages 
                if case.expected_page is not None 
                else True
            )
            
            if self.verbose:
                print(f"Q: {case.question[:50]}...")
                print(f"  Faithfulness: {faithfulness:.2f}, Relevancy: {relevancy:.2f}, Hit: {retrieval_hit}")
            
        except Exception as e:
            logger.exception(f"Error evaluating case: {case.question[:50]}")
            latency_ms = (time.time() - start_time) * 1000
            error = str(e)
            answer = ""
            retrieved_pages = []
            faithfulness = 0.0
            relevancy = 0.0
            retrieval_hit = False
        
        return EvalResult(
            question=case.question,
            generated_answer=answer,
            expected_answer=case.expected_answer,
            retrieved_pages=retrieved_pages,
            expected_page=case.expected_page,
            faithfulness_score=faithfulness,
            relevancy_score=relevancy,
            retrieval_hit=retrieval_hit,
            latency_ms=latency_ms,
            error=error
        )
    
    def run_all(
        self,
        pipeline_func: Callable[[str], dict],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> EvalReport:
        """
        Run evaluation on all test cases.
        
        Args:
            pipeline_func: RAG pipeline function
            progress_callback: Optional callback(current, total) for progress updates
        
        Returns:
            EvalReport with aggregated metrics
        """
        results = []
        
        for i, case in enumerate(self.test_cases):
            if progress_callback:
                progress_callback(i + 1, len(self.test_cases))
            
            result = self.evaluate_single(case, pipeline_func)
            results.append(result)
        
        # Calculate aggregates (excluding errors for averages)
        valid_results = [r for r in results if r.error is None]
        
        if valid_results:
            avg_faithfulness = sum(r.faithfulness_score for r in valid_results) / len(valid_results)
            avg_relevancy = sum(r.relevancy_score for r in valid_results) / len(valid_results)
            retrieval_accuracy = sum(r.retrieval_hit for r in valid_results) / len(valid_results)
            avg_latency = sum(r.latency_ms for r in valid_results) / len(valid_results)
        else:
            avg_faithfulness = avg_relevancy = retrieval_accuracy = avg_latency = 0.0
        
        return EvalReport(
            results=results,
            avg_faithfulness=avg_faithfulness,
            avg_relevancy=avg_relevancy,
            retrieval_accuracy=retrieval_accuracy,
            avg_latency_ms=avg_latency,
            n_cases=len(self.test_cases),
            n_errors=len(results) - len(valid_results)
        )
    
    def _score_faithfulness(self, context: str, answer: str) -> float:
        """Score if answer is grounded in context."""
        if not answer or not context:
            return 0.0
        
        prompt = FAITHFULNESS_PROMPT.format(context=context[:4000], answer=answer)
        return self._get_score(prompt)
    
    def _score_relevancy(self, question: str, answer: str) -> float:
        """Score if answer addresses the question."""
        if not answer:
            return 0.0
        
        prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
        return self._get_score(prompt)
    
    def _get_score(self, prompt: str) -> float:
        """Get numeric score from LLM."""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()
            # Extract number from response
            score = float(score_text.split()[0])
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse score: {e}")
            return 0.0
        except Exception as e:
            logger.exception("Error getting score from LLM")
            return 0.0


def create_sample_test_cases() -> list[dict]:
    """Generate sample test case structure for user reference."""
    return [
        {
            "question": "What is the main topic of the document?",
            "expected_answer": "The document discusses...",
            "doc_id": "abc123",  # Optional: MD5 hash of the PDF
            "expected_page": 1,   # Optional: page where answer should be found
            "tags": ["overview", "simple"]
        },
        {
            "question": "What are the key findings mentioned in section 2?",
            "expected_answer": "The key findings include...",
            "expected_page": 5,
            "tags": ["specific", "section-reference"]
        },
        {
            "question": "How does concept A relate to concept B?",
            "expected_answer": "Concept A and B are related because...",
            "tags": ["multi-hop", "reasoning"]
        }
    ]


if __name__ == "__main__":
    # Generate sample test cases file
    sample_cases = create_sample_test_cases()
    with open("test_cases_sample.json", "w") as f:
        json.dump(sample_cases, f, indent=2)
    print("Created test_cases_sample.json - edit this with your actual test cases")
