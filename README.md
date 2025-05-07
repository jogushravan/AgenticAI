### Levels Of Text Splitting
Level 1: Character Splitting - Simple static character chunks of data<br>
Level 2: Recursive Character Text Splitting - Recursive chunking based on a list of separators<br>
Level 3: Document Specific Splitting - Various chunking methods for different document types (PDF, Python, Markdown)<br>
Level 4: Semantic Splitting - Embedding walk based chunking<br>
Level 5: Agentic Splitting - Experimental method of splitting text with an agent-like system. agentic_chunker.py (LLM based Text splitter)<br>
**  from sklearn.metrics.pairwise import cosine_similarity<br>**
from langchain.text_splitter import MarkdownTextSplitter<br>**
from **unstructured.partition.pdf import partition_pdf**<br>**
from langchain.text_splitter import PythonCodeTextSplitter<br>**
CLIP model summary of Image<br>**
**Input->Rewrite User Query(DSPy)->Retrieve(VectDB)->LLM(Domain Specific)**

### Agentic AI Legal Assistant – Full Pipel.py
✅ STEP 1: Read PDF and Chunk It<br>
✅ STEP 2: Embed Chunks & Upload to Azure Cognitive Search<br>
✅ STEP 3: Insert Legal Clauses into Neo4j as Nodes<br>
✅ STEP 4: Build Hybrid Retriever (Azure + Neo4j)<br>
✅ STEP 5: Create LangGraph LegalAgent Node Using HybridRetriever<br>
✅ 🧪📊 STEP 6: Evaluation Function (TSR + Hallucination + Relevance)<br>
✅ 🧪 STEP 7: Run a Query and Score It<br>

#### langchain-multimodal.ipynb
Extracts Text, Images, and tables and summarizes each of the data, then adds those summarized data to the Vector database <br>
Images will be added separately InMemory with a UUID identifier
![image](https://github.com/user-attachments/assets/4cfb4acf-04d1-4c84-8bbc-058ff941e1d5)

____________________________________________________________________________________________________________________________
#### Multimodal_Rag_ImageText_Azure_Neo4j.py

![image](https://github.com/user-attachments/assets/65cbd3bd-47c9-436e-bd1f-c410db96b421)
![image](https://github.com/user-attachments/assets/3793da68-687a-4f65-86b1-9c0e0f04ba60)
![image](https://github.com/user-attachments/assets/08a55041-30ec-4198-9694-1ba1a4180c71)

#### langchain.retrievers.multi_vector [MultiVectorRetriever] ![image](https://github.com/user-attachments/assets/efcf29ff-74d4-4d68-93c2-27fa10e358a9)
![image](https://github.com/user-attachments/assets/b56fb8ef-e011-4f4c-9f55-19602209c6f7)

