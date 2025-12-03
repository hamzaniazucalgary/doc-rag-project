import re
import logging
from typing import Generator, Optional, Callable
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    AGENT_MODEL,
    AGENT_MAX_ITERATIONS,
    AGENT_SYSTEM_PROMPT,
    EMBEDDING_MODEL,
    TEMPERATURE
)

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    step_num: int
    thought: str
    action: str
    action_input: str
    observation: str
    
    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation
        }


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_retrievals: int = 0
    sources: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "total_retrievals": self.total_retrievals,
            "sources": self.sources,
            "error": self.error
        }


class RAGAgent:
    def __init__(
        self,
        retriever,
        max_iterations: int = AGENT_MAX_ITERATIONS,
        model: str = AGENT_MODEL,
        verbose: bool = False
    ):
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=TEMPERATURE
        )
        self.embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    def run(self, question: str) -> AgentResult:
        """Run the agent to answer a question."""
        steps = []
        all_sources = []
        scratchpad = ""
        
        system_prompt = AGENT_SYSTEM_PROMPT.format(max_iterations=self.max_iterations)
        
        for iteration in range(self.max_iterations + 1):
            user_prompt = f"Question: {question}\n\n{scratchpad}"
            
            if self.verbose:
                logger.info(f"Iteration {iteration + 1}, scratchpad length: {len(scratchpad)}")
            
            try:
                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                response_text = response.content
            except Exception as e:
                logger.exception("LLM call failed")
                return AgentResult(
                    answer="I encountered an error while processing your question.",
                    steps=steps,
                    total_retrievals=len([s for s in steps if s.action == "search"]),
                    sources=all_sources,
                    error=str(e)
                )
            
            parsed = self._parse_response(response_text)
            
            if parsed["type"] == "answer":
                step = AgentStep(
                    step_num=iteration + 1,
                    thought=parsed["thought"],
                    action="answer",
                    action_input="",
                    observation=parsed["answer"]
                )
                steps.append(step)
                
                return AgentResult(
                    answer=parsed["answer"],
                    steps=steps,
                    total_retrievals=len([s for s in steps if s.action == "search"]),
                    sources=self._dedupe_sources(all_sources)
                )
            
            elif parsed["type"] == "search":
                search_query = parsed["query"]
                
                if self.verbose:
                    logger.info(f"Searching: {search_query}")
                
                query_embedding = self.embedder.embed_query(search_query)
                results, _ = self.retriever.retrieve(
                    query=search_query,
                    query_embedding=query_embedding,
                    n_results=3
                )
                
                observation = self._format_search_results(results)
                
                for r in results:
                    all_sources.append(r.to_dict())
                
                step = AgentStep(
                    step_num=iteration + 1,
                    thought=parsed["thought"],
                    action="search",
                    action_input=search_query,
                    observation=observation
                )
                steps.append(step)
                
                scratchpad += f"\nThought: {parsed['thought']}\n"
                scratchpad += f"Action: search(\"{search_query}\")\n"
                scratchpad += f"Observation: {observation}\n"
            
            else:
                logger.warning(f"Malformed agent response: {response_text[:200]}")
                scratchpad += "\n[System: Please respond with either a search action or a final answer in the correct format.]\n"
        
        # Max iterations reached
        logger.warning("Agent reached max iterations without answering")
        
        if all_sources:
            final_answer = self._force_answer(question, all_sources)
        else:
            final_answer = "I was unable to find sufficient information to answer this question after multiple searches."
        
        return AgentResult(
            answer=final_answer,
            steps=steps,
            total_retrievals=len([s for s in steps if s.action == "search"]),
            sources=self._dedupe_sources(all_sources)
        )
    
    def run_streaming(
        self,
        question: str,
        step_callback: Optional[Callable[[AgentStep], None]] = None
    ) -> Generator[str, None, AgentResult]:
        """Run the agent with streaming output."""
        steps = []
        all_sources = []
        scratchpad = ""
        
        system_prompt = AGENT_SYSTEM_PROMPT.format(max_iterations=self.max_iterations)
        
        for iteration in range(self.max_iterations + 1):
            user_prompt = f"Question: {question}\n\n{scratchpad}"
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            response_text = response.content
            
            parsed = self._parse_response(response_text)
            
            if parsed["type"] == "answer":
                step = AgentStep(
                    step_num=iteration + 1,
                    thought=parsed["thought"],
                    action="answer",
                    action_input="",
                    observation=""
                )
                steps.append(step)
                
                if step_callback:
                    step_callback(step)
                
                for char in parsed["answer"]:
                    yield char
                
                return AgentResult(
                    answer=parsed["answer"],
                    steps=steps,
                    total_retrievals=len([s for s in steps if s.action == "search"]),
                    sources=self._dedupe_sources(all_sources)
                )
            
            elif parsed["type"] == "search":
                search_query = parsed["query"]
                
                query_embedding = self.embedder.embed_query(search_query)
                results, _ = self.retriever.retrieve(
                    query=search_query,
                    query_embedding=query_embedding,
                    n_results=3
                )
                
                observation = self._format_search_results(results)
                
                for r in results:
                    all_sources.append(r.to_dict())
                
                step = AgentStep(
                    step_num=iteration + 1,
                    thought=parsed["thought"],
                    action="search",
                    action_input=search_query,
                    observation=observation
                )
                steps.append(step)
                
                if step_callback:
                    step_callback(step)
                
                scratchpad += f"\nThought: {parsed['thought']}\n"
                scratchpad += f"Action: search(\"{search_query}\")\n"
                scratchpad += f"Observation: {observation}\n"
            
            else:
                scratchpad += "\n[System: Please respond in correct format.]\n"
        
        # Max iterations
        final_answer = "Unable to find sufficient information."
        for char in final_answer:
            yield char
        
        return AgentResult(
            answer=final_answer,
            steps=steps,
            total_retrievals=len([s for s in steps if s.action == "search"]),
            sources=self._dedupe_sources(all_sources)
        )
    
    def _parse_response(self, response: str) -> dict:
        """Parse the agent's response to extract action or answer."""
        result = {
            "type": "unknown",
            "thought": "",
            "query": "",
            "answer": ""
        }
        
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Answer:|$)', response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Check for answer
        answer_match = re.search(r'Answer:\s*(.+)', response, re.DOTALL)
        if answer_match:
            result["type"] = "answer"
            result["answer"] = answer_match.group(1).strip()
            return result
        
        # Check for search action
        search_match = re.search(r'Action:\s*search\(["\'](.+?)["\']\)', response)
        if search_match:
            result["type"] = "search"
            result["query"] = search_match.group(1).strip()
            return result
        
        # Try alternative search format
        search_match_alt = re.search(r'search\(["\'](.+?)["\']\)', response)
        if search_match_alt:
            result["type"] = "search"
            result["query"] = search_match_alt.group(1).strip()
            return result
        
        return result
    
    def _format_search_results(self, results: list) -> str:
        """Format search results for the scratchpad."""
        if not results:
            return "No relevant results found."
        
        formatted = []
        for i, r in enumerate(results, 1):
            content = r.content[:400] + "..." if len(r.content) > 400 else r.content
            formatted.append(f"[{i}] ({r.doc_name}, Page {r.page}): {content}")
        
        return "\n\n".join(formatted)
    
    def _force_answer(self, question: str, sources: list[dict]) -> str:
        """Force an answer when max iterations reached."""
        context = "\n\n".join([
            f"({s['doc_name']}, Page {s['page']}): {s['content'][:300]}"
            for s in sources[:5]
        ])
        
        prompt = f"""Based on the following information, provide the best possible answer to the question.
If the information is insufficient, acknowledge what is known and what is missing.

Question: {question}

Available Information:
{context}

Answer:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _dedupe_sources(self, sources: list[dict]) -> list[dict]:
        """Remove duplicate sources."""
        seen = set()
        deduped = []
        
        for s in sources:
            key = (s.get("doc_id", ""), s.get("page", 0), s.get("chunk_index", 0))
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        
        return deduped


class QueryDecomposer:
    """Decompose complex queries into sub-questions."""
    
    DECOMPOSE_PROMPT = """Analyze this question and determine if it needs to be broken into sub-questions.

Question: {question}

If the question is complex (multi-part, requires combining information, or compares multiple things),
break it into 2-4 simpler sub-questions that can be answered independently.

If the question is already simple and focused, return it unchanged.

Format your response as:
- If simple: SIMPLE: [original question]
- If complex:
  SUB1: [first sub-question]
  SUB2: [second sub-question]
  ...

Response:"""
    
    def __init__(self, model: str = AGENT_MODEL):
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    def decompose(self, question: str) -> list[str]:
        """Decompose a question into sub-questions if complex."""
        prompt = self.DECOMPOSE_PROMPT.format(question=question)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        
        if "SIMPLE:" in text:
            return [question]
        
        sub_questions = []
        for line in text.split("\n"):
            if line.strip().startswith("SUB"):
                match = re.search(r'SUB\d+:\s*(.+)', line)
                if match:
                    sub_questions.append(match.group(1).strip())
        
        return sub_questions if sub_questions else [question]
