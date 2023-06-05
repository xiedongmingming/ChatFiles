from llama_index import (
    ComposableGraph,
    GPTListIndex,
    LLMPredictor,
    # GPTSimpleVectorIndex, # 已经废弃了？？用下面的替换
    GPTVectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
    LangchainEmbedding,
    StorageContext,
    load_index_from_storage,
    load_graph_from_storage,
)

from utils import torch_gc

from models.chatglm_llm import ChatGLM

llm = ChatGLM()

llm.load_model(
    model_name_or_path='THUDM/chatglm-6b-int4',
    llm_device='cuda',
)

llm_predictor = LLMPredictor(llm=llm)

from langchain.embeddings import HuggingFaceEmbeddings

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name='GanymedeNil/text2vec-large-chinese',
    model_kwargs={'device': 'cuda'}
))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

doc1 = SimpleDirectoryReader('./data1').load_data()
doc2 = SimpleDirectoryReader('./data2').load_data()
doc3 = SimpleDirectoryReader('./data3').load_data()

torch_gc()

from llama_index import GPTTreeIndex

storage_context = StorageContext.from_defaults()

index1 = GPTTreeIndex.from_documents(doc1, storage_context=storage_context, service_context=service_context)
index2 = GPTTreeIndex.from_documents(doc2, storage_context=storage_context, service_context=service_context)
index3 = GPTTreeIndex.from_documents(doc3, storage_context=storage_context, service_context=service_context)

# index1_summary = "<summary1>"
# index2_summary = "<summary2>"
# index3_summary = "<summary3>"

torch_gc()

summary = index1.query(
    "What is a summary of this document?", retriever_mode="all_leaf"
)
index1_summary = str(summary)

summary = index2.query(
    "What is a summary of this document?", retriever_mode="all_leaf"
)
index2_summary = str(summary)

summary = index3.query(
    "What is a summary of this document?", retriever_mode="all_leaf"
)
index3_summary = str(summary)

torch_gc()

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index1, index2, index3],
    index_summaries=[index1_summary, index2_summary, index3_summary],
    storage_context=storage_context,
)

# set custom retrievers. An example is provided below
custom_query_engines = {
    index.index_id: index.as_query_engine(
        child_branch_factor=2
    )
    for index in [index1, index2, index3]
}
query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines
)

torch_gc()

response = query_engine.query("MG7多少钱？")

# index1.set_index_id("<index_id_1>")
# index2.set_index_id("<index_id_2>")
# index3.set_index_id("<index_id_3>")

# set the ID
graph.root_index.set_index_id("my_id")

# persist to storage
graph.root_index.storage_context.persist(persist_dir="./storage")

# load
from llama_index import StorageContext, load_graph_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")

graph = load_graph_from_storage(storage_context, root_id="my_id")

response = graph.as_query_engine().query("MG7多少钱？")
