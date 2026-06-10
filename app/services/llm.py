"""LLM Service - Claude via AWS Bedrock"""
import json
import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Chat message"""
    role: str  # "user", "assistant", "system"
    content: str

@dataclass
class QAResponse:
    """Question answering response"""
    answer: str
    citations: List[Dict[str, Any]]
    sources: List[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LLMService:
    """Claude LLM via AWS Bedrock"""
    
    def __init__(self, model_id: str = None):
        from app.config import CLAUDE_MODEL, AWS_REGION
        from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        
        self.model_id = model_id or CLAUDE_MODEL
        self.aws_region = AWS_REGION
        
        # Initialize boto3 client
        self._client = None
        self._access_key = AWS_ACCESS_KEY_ID
        self._secret_key = AWS_SECRET_ACCESS_KEY
        
        # System prompt for RAG
        self.system_prompt = """You are an expert AI assistant that answers questions based on the provided document context.

Your task:
1. Read the context provided from documents
2. Answer the user's question accurately and comprehensively
3. Cite your sources using the format [Source N] where N is the source number
4. If the answer cannot be determined from the context, clearly state that

Rules:
- Only use information from the provided context
- Be precise and specific in your answers
- Include relevant details and examples when available
- Format citations as [Source 1], [Source 2], etc.
- If there are multiple sources that support the answer, cite all relevant ones"""
    
    @property
    def client(self):
        """Lazy load Bedrock client"""
        if self._client is None:
            import boto3
            
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.aws_region,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key
            )
            logger.info(f"Bedrock client initialized for {self.aws_region}")
        
        return self._client
    
    def invoke(self, prompt: str, max_tokens: int = 2048, 
               temperature: float = 0.5) -> str:
        """Invoke Claude with a prompt"""
        # Format for Claude 3.5 on Bedrock
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": self.system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        # Parse response
        response_body = json.loads(response["body"].read())
        
        return response_body["content"][0]["text"]
    
    def chat(self, messages: List[ChatMessage], 
             max_tokens: int = 2048,
             temperature: float = 0.5) -> str:
        """Chat with conversation history"""
        # Convert messages to Claude format
        claude_messages = []
        
        for msg in messages:
            claude_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": self.system_prompt,
            "messages": claude_messages
        })
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        response_body = json.loads(response["body"].read())
        
        return response_body["content"][0]["text"]
    
    def answer_question(self, question: str, context: str, 
                        citations: List[Dict]) -> QAResponse:
        """Answer a question with context and citations"""
        # Build prompt with context
        prompt = f"""Context from documents:
{context}

---

User Question: {question}

Please provide a comprehensive answer based on the context above. Cite your sources using [Source N] format."""
        
        # Get answer from LLM
        answer = self.invoke(prompt)
        
        # Extract source references from answer
        sourcerefs = re.findall(r'\[Source\s*(\d+)\]', answer)
        
        # Build unique sources
        unique_sources = []
        seen = set()
        for ref in sourcerefs:
            idx = int(ref) - 1
            if 0 <= idx < len(citations) and idx not in seen:
                seen.add(idx)
                unique_sources.append(citations[idx].get("source", "Unknown"))
        
        return QAResponse(
            answer=answer,
            citations=citations,
            sources=unique_sources,
            prompt_tokens=0,  # Bedrock doesn't expose token counts
            completion_tokens=0,
            total_tokens=0
        )


# Global instance
_llm_service: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    """Get or create global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service