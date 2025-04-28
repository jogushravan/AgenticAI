# main.py
import asyncio
from reader import extract_text_and_images
from storage import initialize_light_rag, insert_documents
from pipeline import run_pipeline
from metrics import metrics

async def main():
    global rag
    rag = await initialize_light_rag()

    pdf_data = extract_text_and_images("path_to_your_pdf_folder/document.pdf")
    documents = [item["text"] for item in pdf_data if item.get("text")]

    await insert_documents(rag, documents)

    user_query = "Summarize GDPR impact on NDAs."
    metrics["queries"] += 1
    final_answer = await run_pipeline(user_query)

    print("\n✅ Final Answer:\n", final_answer)
    print("\n📊 Metrics Summary:\n", metrics)

    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
