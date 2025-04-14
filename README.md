#### langchain-multimodal.ipynb
Extracts Text, Images, and tables and summarizes each of the data, then adds those summarized data to the Vector database 
Images will be added separately InMemory with a UUID identifier
![image](https://github.com/user-attachments/assets/4cfb4acf-04d1-4c84-8bbc-058ff941e1d5)

____________________________________________________________________________________________________________________________
#### Multimodal_Rag_ImageText_Azure_Neo4j.py

![image](https://github.com/user-attachments/assets/65cbd3bd-47c9-436e-bd1f-c410db96b421)
![image](https://github.com/user-attachments/assets/b07354bc-b598-4fbb-992c-d9fef5db8cd1)
             ┌─────────────────────────────┐
             │      User Uploads Image     │
             └─────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │  UnifiedFeatureExtractor (Python)  │
         └────────────────────────────────────┘
            │             │               │
            ▼             ▼               ▼
      [BLIP]        [CLIP] Image     [EfficientNet-B7]
     Caption      Embedding (512d)   Embedding (2560d)
    ("Red box")   ↕                   └───────────────────────┐
        │        [CLIP] Text Embedding (512d)                 │
        │                    │                                │
        └────────────┬───────┴────────────┬───────────────────┘
                     ▼                    ▼
         ┌──────────────────┐   ┌───────────────────────────┐
         │  Azure Cognitive │   │        Neo4j Graph        │
         │   Search (Vector)│   │  (Image → Tags → KPIs)    │
         └──────────────────┘   └───────────────────────────┘
                     ▲                    ▲
                     │                    │
             Vector similarity       Graph query (Cypher)
              search (KNN)         (e.g. relationships/tags)
                     ▲                    ▲
                     └──────┬─────────────┘
                            ▼
                 ┌────────────────────┐
                 │   Top-k Results     │
                 │ (Images + captions)│
                 └────────────────────┘
