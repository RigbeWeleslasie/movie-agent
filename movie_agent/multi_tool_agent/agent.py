from google.adk.agents import Agent
from google.adk.tools import FunctionTool  # or other appropriate Tool wrapper
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

# Setup embeddings and Chroma vector store path (use your local persistent directory)
PERSIST_DIR = "/chroma_db"  # Update to your synced ChromaDB folder path

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_function,
    collection_name="movie_chunks"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define prompt for movie recommendation explanations
prompt_template = """
You are a movie recommendation assistant.

Analyze the following movie metadata:
{context}

Based on the user's request: "{input}", provide a clear, concise recommendation.

1. Genre Alignment: Explain how the recommended movies align with requested genres.
2. Star Relevance: Highlight stars from recommended movies relevant to the user query.
3. Plot Similarity: Describe plot similarities relevant to the user interest.

Provide your answer clearly, no repetition of context or query. Use bullets or numbered lists.
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

# Setup chain to combine docs and generate response
def make_qa_chain(llm, prompt):
    return create_stuff_documents_chain(llm=llm, prompt=prompt)

class MovieRecommendationTool(Tool):
    def _run(self, query: str) -> str:
        # Retrieve relevant movie chunks
        docs = retriever.get_relevant_documents(query)
        context = " ".join([d.page_content for d in docs])
        # Create chain with the agent's LLM
        chain = make_qa_chain(self.llm, PROMPT)
        # Run chain with user input and retrieved context
        return chain.run(context=context, input=query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async method not implemented.")

# Initialize the root agent with your Gemini model here
root_agent = Agent(
    name="movie_recommender_agent",
    model="gemini-2.0-flash",  # Or your preferred Gemini variant
    description="Agent for recommending movies based on detailed metadata.",
    instruction="You are a helpful assistant that recommends movies using rich metadata.",
    tools=[MovieRecommendationTool()],
)

