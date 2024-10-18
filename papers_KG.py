from dotenv import load_dotenv
import os

# Common data processing
import json
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI


# Warning control
import warnings
warnings.filterwarnings("ignore")

load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Neo4j graph setup
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

# Create constraint for unique paper IDs
kg.query("""
CREATE CONSTRAINT unique_paper IF NOT EXISTS 
    FOR (p:Paper) REQUIRE p.id IS UNIQUE
""")

# Define the query to merge paper nodes
merge_paper_node_query = """
MERGE (paper:Paper {id: $paperParam.id})
    ON CREATE SET 
        paper.title = $paperParam.title,
        paper.keywords = $paperParam.keywords,
        paper.abstract = $paperParam.abstract,
        paper.embedding = $paperParam.embedding
RETURN paper
"""

# Load the JSONML data
with open('/home/souraja/DiffKG/KG/train.jsonl', 'r') as file:
    papers = json.load(file)

# Iterate through each paper and create nodes
node_count = 0
for paper in papers:
    # Generate OpenAI embeddings for the abstract or title
    # (You may want to choose the most appropriate text for your embeddings)
    paper_text = paper['abstract']  # or paper['title'] or a combination
    embedding_response = kg.query(embedding_query, params={"text": paper_text})
    paper['embedding'] = embedding_response['data']['embedding']  # adjust based on your response structure

    print(f"Creating `:Paper` node for paper ID {paper['id']}")
    kg.query(merge_paper_node_query, 
            params={
                'paperParam': paper
            })
    node_count += 1

print(f"Created {node_count} nodes")

# Create vector index for embeddings
kg.query("""
CREATE VECTOR INDEX `paper_embeddings` IF NOT EXISTS
FOR (p:Paper) ON (p.embedding) 
OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'    
}}
""")

# Update nodes with embeddings
kg.query("""
MATCH (paper:Paper) WHERE paper.embedding IS NULL
WITH paper, genai.vector.encode(
  paper.abstract, 
  "OpenAI", 
  {
    token: $openAiApiKey, 
    endpoint: $openAiEndpoint
  }) AS vector
CALL db.create.setNodeVectorProperty(paper, "embedding", vector)
""", 
params={"openAiApiKey": OPENAI_API_KEY, "openAiApiEndpoint": OPENAI_ENDPOINT})
