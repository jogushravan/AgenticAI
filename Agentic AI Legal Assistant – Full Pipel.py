#Agentic AI Legal Assistant – Full Pipeline
✅ STEP 1: Read PDF and Chunk It
✅ STEP 2: Embed Chunks & Upload to Azure Cognitive Search
✅ STEP 3: Insert Legal Clauses into Neo4j as Nodes
✅ STEP 4: Build Hybrid Retriever (Azure + Neo4j)
✅ STEP 5: Create LangGraph LegalAgent Node Using HybridRetriever
✅ 🧪📊 STEP 6: Evaluation Function (TSR + Hallucination + Relevance)
✅ 🧪 STEP 7: Run a Query and Score It

[PDFs (contracts, HR policies)]
    ↓
[Text Splitter]
    ↓
[Embeddings Generator] ─────┬─────→ [Azure Cognitive Search]
                            └─────→ [Neo4j GraphDB (nodes + edges)]
                                   ↓
[GraphRAG Retriever] ←───── [LangChain HybridRetriever]
                                   ↓
[LegalAgent Node in LangGraph]

# pip install langchain neo4j sentence-transformers openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.retrievers.neo4j_vector import Neo4jVectorRetriever
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util

✅ STEP 1: Read PDF and Chunk It
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("legal_policy.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

✅ STEP 2: Embed Chunks & Upload to Azure Cognitive Search
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

# 🔑 Configure Azure Cognitive Search
service_name = "your-azure-search-service"
index_name = "legal-index"
api_key = "your-azure-api-key"

embedding = OpenAIEmbeddings()

vectorstore = AzureSearch(
    azure_search_name=service_name,
    azure_search_key=api_key,
    index_name=index_name,
    embedding_function=embedding.embed_query,
)

# Embed and Store in Azure Cognitive Search
vectorstore.add_documents(chunks)

✅ STEP 3: Insert Legal Clauses into Neo4j as Nodes
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "your_password"
driver = GraphDatabase.driver(uri, auth=(user, password))

def insert_clause(tx, title, text):
    query = """
    MERGE (c:Clause {title: $title})
    SET c.text = $text
    """
    tx.run(query, title=title, text=text)

with driver.session() as session:
    for doc in chunks:
        title = doc.metadata.get("source", "Unknown")
        text = doc.page_content
        session.write_transaction(insert_clause, title, text)

✅ STEP 4: Build Hybrid Retriever (Azure + Neo4j)
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.neo4j_vector import Neo4jVectorRetriever

# Neo4j retriever setup (assumes embeddings were added via Cypher)
neo4j_retriever = Neo4jVectorRetriever(
    url=uri,
    username=user,
    password=password,
    embedding=embedding,
    index_name="legal_vector",
    node_label="Clause",
    text_node_property="text"
)

# Final hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[vectorstore.as_retriever(), neo4j_retriever],
    weights=[0.6, 0.4]  # Adjust based on performance
)

✅ STEP 5: Create LangGraph LegalAgent Node Using HybridRetriever
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph
#RetrievalQA Chain using Hybrid Retriever
legal_qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAIChat(),
    retriever=hybrid_retriever,
    return_source_documents=True
)

def legal_agent_node(state):
    query = state["query"]
    result = legal_qa_chain.run(query)
    return {"response": result}

graph = StateGraph()
graph.add_node("LegalAgent", legal_agent_node)

✅ 🧪📊 STEP 6: Evaluation Function (TSR + Hallucination + Relevance)
from sentence_transformers import SentenceTransformer, util

def evaluate_response(response, retrieved_docs, expected_keywords):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    relevance = util.cos_sim(
        model.encode(response),
        model.encode(retrieved_docs[0].page_content)
    ).item()

    hallucinated = not any(doc.page_content.lower() in response.lower() for doc in retrieved_docs)
    success = any(k.lower() in response.lower() for k in expected_keywords)

    return {
        "task_success": success,
        "hallucination": hallucinated,
        "relevance_score": round(relevance, 3)
    }
    
✅ 🧪 STEP 7: Run a Query and Score It
query = "What is the termination notice period in New York?"
expected_keywords = ["termination", "notice period"]

result = qa_chain(query)
response = result["result"]
retrieved_docs = result["source_documents"]

metrics = evaluate_response(response, retrieved_docs, expected_keywords)

print("Response:", response)
print("Evaluation Metrics:", metrics)
