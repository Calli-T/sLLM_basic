'''
for llama-index extension
'''

'''
# 파인콘 기본 설정
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)
pc.create_index(
    "quickstart", dimension=1536, metric="euclidean", spec=ServerlessSpec("aws", "us-east-1")
)
pinecone_index = pc.Index("quickstart")

# 라마인덱스에 파인콘 인덱스 연결
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
'''
