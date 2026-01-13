from __future__ import annotations

import logging
import re
from typing import Any

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

from config import settings


logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    Returns: 'hindi', 'english', or 'hinglish'
    """
    # Hindi Unicode range: \u0900-\u097F
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    
    # Count Hindi characters
    hindi_chars = len(hindi_pattern.findall(text))
    total_chars = len(re.findall(r'[a-zA-Z\u0900-\u097F]', text))
    
    if total_chars == 0:
        return 'english'  # Default
    
    hindi_ratio = hindi_chars / total_chars if total_chars > 0 else 0
    
    if hindi_ratio > 0.3:
        # Check if it's mixed (Hinglish) or pure Hindi
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        if english_chars > len(text) * 0.1:  # Has significant English
            return 'hinglish'
        return 'hindi'
    else:
        return 'english'


# System prompt for multilingual PDF QA with table extraction support
SYSTEM_PROMPT = """You are a strict document QA system for multilingual PDFs (Hindi/English/Hinglish).

CRITICAL RULE — LANGUAGE MATCHING:
The question language is: {question_language}
You MUST answer in the EXACT SAME LANGUAGE as the question.
- If question is in Hindi → Answer in Hindi (Devanagari script)
- If question is in English → Answer in English
- If question is in Hinglish → Answer in Hinglish (mix of Hindi and English)

DO NOT switch languages. If you answer in a different language, you have FAILED.

RULE 2 — CONTEXT:
Use ONLY the text inside <CONTEXT>.
Do NOT use outside knowledge.
Do NOT guess or make up information.

If answer is not clearly in context, reply EXACTLY:
The document does not contain sufficient information to answer this question.

RULE 3 — TABLES:
If question is about a table:
- Copy exact values from the context.
- Do not change numbers or words.
- Use markdown table format if helpful.

RULE 4 — ACCURACY:
- Be precise and factual.
- Quote exact text from context when possible.
- Do not paraphrase if exact text is available.

<CONTEXT>
{context}
</CONTEXT>

QUESTION (Language: {question_language}):
{question}

ANSWER (MUST be in {question_language}):
"""



PROMPT_TEMPLATE = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question", "question_language"],
)


def get_llm() -> Ollama:
    """
    Get LLM via Ollama. Default is llama3.1:8b-instruct for better accuracy.
    
    Make sure the model is pulled locally:
      ollama pull llama3.1:8b-instruct-q4_K_M
    Or use other models like:
      ollama pull mistral:7b-instruct
      ollama pull qwen2.5:7b-instruct
    """
    return Ollama(
        model=settings.ollama_model,
        temperature=0.1,  # Lower temperature for more accurate, deterministic answers
        num_ctx=8192,  # Increased context window for better understanding
    )


def build_qa_chain(vectorstore) -> RetrievalQA:
    """
    Build a RetrievalQA chain over the provided Chroma vectorstore with custom prompt.
    Uses a custom chain that includes language detection.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    llm = get_llm()

    # Create a custom chain that handles language detection
    # This wrapper ensures language detection is included in the prompt
    class LanguageAwareQAChain:
        def __init__(self, retriever, llm, prompt_template):
            self.retriever = retriever
            self.llm = llm
            self.prompt_template = prompt_template
        
        def invoke(self, inputs: dict) -> dict:
            question = inputs.get("query", "")
            question_lang = detect_language(question)
            logger.info(f"Detected question language: {question_lang}")
            
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(question)
            if not docs:
                return {
                    "result": "No relevant information found in the document.",
                    "source_documents": []
                }
            
            context = "\n\n".join([f"[Page {doc.metadata.get('page_number', '?')}]\n{doc.page_content}" 
                                  for doc in docs])
            
            # Format prompt with language information
            prompt = self.prompt_template.format(
                context=context,
                question=question,
                question_language=question_lang
            )
            
            # Get answer from LLM
            try:
                answer = self.llm.invoke(prompt)
                # Clean up answer (remove any prompt artifacts)
                answer = answer.strip()
                # Remove the "ANSWER (MUST be in...):" prefix if present
                if "ANSWER" in answer.upper() and ":" in answer:
                    lines = answer.split("\n", 1)
                    if len(lines) > 1:
                        answer = lines[1].strip()
            except Exception as e:
                logger.error(f"Error getting answer from LLM: {e}")
                answer = "Sorry, I encountered an error while processing your question."
            
            return {
                "result": answer,
                "source_documents": docs
            }
    
    return LanguageAwareQAChain(retriever, llm, PROMPT_TEMPLATE)


def ask_question(chain, question: str) -> dict[str, Any]:
    """
    Run a question through the QA chain.
    Detects language and ensures answer matches question language.
    """
    detected_lang = detect_language(question)
    logger.info("Running QA for question (detected language: %s): %s", detected_lang, question)
    
    result = chain.invoke({"query": question})
    answer = result.get("result", "")
    
    # Verify answer language matches (basic check)
    answer_lang = detect_language(answer)
    if detected_lang != answer_lang and detected_lang in ['hindi', 'hinglish']:
        logger.warning(
            "Language mismatch detected. Question: %s, Answer: %s. "
            "The model may not have followed language instructions.",
            detected_lang, answer_lang
        )
    
    return {
        "answer": answer,
        "sources": result.get("source_documents", []),
    }

