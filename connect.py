import os
from typing import Optional, List, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient


# TOKEN SETUP


HF_TOKEN = os.environ.get("HF_TOKEN")






HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"



# CUSTOM LLM 


class HFChatLLM(LLM):
    repo_id: str
    token: Optional[str] = None
    temperature: float = 0.5
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "hf_chat_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        client = InferenceClient(model=self.repo_id, token=self.token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


def load_llm(repo_id):
    return HFChatLLM(
        repo_id=repo_id,
        token=HF_TOKEN,
        temperature=0.5,
        max_tokens=512,
    )



# PROMPT


CUSTOM_PROMPT_TEMPLATE = """
Use the context to answer the question.
If you don't know, say "I don't know".
Do not make up answers.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


# LOAD FAISS


DB_FAISS_PATH = "vectorStore/db_faiss"

if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"FAISS DB not found at: {DB_FAISS_PATH}")

print("Loading FAISS DB...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS DB loaded successfully")



# QA CHAIN


qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt()}
)






query = input("Enter your question: ")

print("Running query...\n")

response = qa_chain.invoke({"query": query})

print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"])