# Required Libraries
import torch
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from langchain.embeddings import OpenAIEmbeddings
from azure.search.documents.indexes.models import VectorSearchAlgorithmKind
from azure.search.documents import SearchClient
from neo4j import GraphDatabase
import numpy as np

# Unified Feature Extractor: EfficientNet + BLIP + CLIP
class UnifiedFeatureExtractor:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        self.efficientnet.classifier = torch.nn.Identity()

        self.clip_model.eval()
        self.blip_model.eval()
        self.efficientnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize(600),
            transforms.CenterCrop(528),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def extract_blip_caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            out = self.blip_model.generate(**inputs)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def extract_clip_embeddings(self, image_path: str, text: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return {
            "clip_image_embedding": outputs.image_embeds[0].numpy().tolist(),
            "clip_text_embedding": outputs.text_embeds[0].numpy().tolist()
        }

    def extract_efficientnet_embedding(self, image_path: str) -> list:
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.efficientnet(img_tensor)
        return features.squeeze().numpy().tolist()

    def extract_all(self, image_path: str, caption: str = None) -> dict:
        if not caption:
            caption = self.extract_blip_caption(image_path)
        clip_embeddings = self.extract_clip_embeddings(image_path, caption)
        effnet_embedding = self.extract_efficientnet_embedding(image_path)
        return {
            "caption": caption,
            "clip_image_embedding": clip_embeddings["clip_image_embedding"],
            "clip_text_embedding": clip_embeddings["clip_text_embedding"],
            "efficientnet_embedding": effnet_embedding
        }

# Azure and Neo4j Setup
search_client = SearchClient(endpoint="YOUR_ENDPOINT", index_name="YOUR_INDEX", credential="YOUR_CREDENTIAL")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def upload_to_azure_cognitive_search(image_url, caption, clip_image, clip_text, effnet, metadata):
    document = {
        "id": metadata["id"],
        "image_url": image_url,
        "caption": caption,
        "clip_image_embedding": clip_image,
        "clip_text_embedding": clip_text,
        "efficientnet_embedding": effnet,
        "tags": metadata.get("tags", [])
    }
    search_client.upload_documents(documents=[document])

def save_to_neo4j(image_id, caption, tags):
    with driver.session() as session:
        session.run("""
        MERGE (i:Image {id: $image_id})
        SET i.caption = $caption
        WITH i
        UNWIND $tags AS tag
        MERGE (t:Tag {name: tag})
        MERGE (i)-[:HAS_TAG]->(t)
        """, image_id=image_id, caption=caption, tags=tags)

def process_and_store(image_path, image_url, image_id, caption, tags):
    extractor = UnifiedFeatureExtractor()
    result = extractor.extract_all(image_path, caption)
    upload_to_azure_cognitive_search(image_url, result["caption"], result["clip_image_embedding"], result["clip_text_embedding"], result["efficientnet_embedding"], {"id": image_id, "tags": tags})
    save_to_neo4j(image_id, result["caption"], tags)

# Retrieval

def get_kpi_from_neo4j(image_id):
    with driver.session() as session:
        result = session.run("""
            MATCH (i:Image {id: $id})-[:HAS_TAG]->(t)
            RETURN i.caption AS caption, collect(t.name) AS tags
        """, id=image_id)
        for record in result:
            print(f"Caption: {record['caption']}, Tags: {record['tags']}")

def retrieve_similar_images(image_path, mode="clip_image", caption_query=""):
    extractor = UnifiedFeatureExtractor()
    result = extractor.extract_all(image_path, caption_query)

    if mode == "clip_image":
        query_vector = result["clip_image_embedding"]
        field = "clip_image_embedding"
    elif mode == "clip_text":
        query_vector = result["clip_text_embedding"]
        field = "clip_text_embedding"
    elif mode == "efficientnet":
        query_vector = result["efficientnet_embedding"]
        field = "efficientnet_embedding"
    else:
        raise ValueError("Unsupported mode")

    results = search_client.search(
        search_text="*",
        vector=query_vector,
        vector_fields=field,
        top_k=5,
        vector_search_algorithm=VectorSearchAlgorithmKind.COSINE
    )

    for result in results:
        print(f"Image: {result['image_url']}, Caption: {result['caption']}, Tags: {result['tags']}")
        get_kpi_from_neo4j(result['id'])

# Example usage:
# process_and_store("product1.jpg", "https://cdn.com/product1.jpg", "prod-101", None, ["logo", "variantA"])
# retrieve_similar_images("new_product.jpg", mode="clip_image")
