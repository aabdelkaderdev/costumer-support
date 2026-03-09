from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

class KnowledgeBase:
    def __init__(self, vector_store, llm=None):
        self.vector_store = vector_store
        self.llm = llm or ChatGroq(model="qwen/qwen3-32b", temperature=0)
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question):
        """Query the knowledge base"""
        response = self.qa_chain.invoke({"query": question})
        return response.get("result", response)

class KnowledgeBaseRouter:
    def __init__(self, knowledge_bases, llm=None):
        self.knowledge_bases = knowledge_bases
        self.llm = llm or ChatGroq(model="qwen/qwen3-32b", temperature=0)

    def route_query(self, query):
        """Route query to appropriate knowledge base"""
        if not self.knowledge_bases:
            return "No knowledge base available."
            
        product_names = list(self.knowledge_bases.keys())
        prompt = f"""You are a smart routing assistant for customer support.
Determine which product knowledge base this user query belongs to.
Available knowledge bases: {', '.join(product_names)}.
If the query covers multiple products or doesn't fit elegantly into one, reply EXACTLY with "cross-product".
Respond ONLY with the name of the chosen knowledge base or "cross-product".

User Query: {query}
Result:"""

        routing_decision = self.llm.invoke(prompt).content.strip()
        
        for name in product_names:
            if name.lower() in routing_decision.lower():
                return self.knowledge_bases[name].query(query)
                
        # Fallback mechanism for cross-product questions
        combined_context = ""
        for name, kb in self.knowledge_bases.items():
            docs = kb.retriever.invoke(query)
            context_str = "\n".join([doc.page_content for doc in docs])
            combined_context += f"\n--- {name} Context ---\n{context_str}\n"
            
        final_prompt = f"""You are an expert customer support agent handling a question that spans multiple domains.
Using the provided context from various product knowledge bases, synthesize a helpful, comprehensive answer.

{combined_context}

User Query: {query}
Helpful Answer:"""
        
        final_response = self.llm.invoke(final_prompt)
        return final_response.content
