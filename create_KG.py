from dotenv import load_dotenv
import os
import json
import numpy as np  # For aggregating embeddings
import warnings
from langchain_community.graphs import Neo4jGraph  
from sentence_transformers import SentenceTransformer

# Warning control
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv('.env', override=True)

# Fetch environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

# Initialize Neo4j connection
kg = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD, 
    database=NEO4J_DATABASE
)

# Create unique constraints on Journal and Paper IDs
kg.query("""
CREATE CONSTRAINT unique_journal IF NOT EXISTS 
    FOR (j:Journal) REQUIRE j.name IS UNIQUE
""")

kg.query("""
CREATE CONSTRAINT unique_paper IF NOT EXISTS 
    FOR (p:Paper) REQUIRE p.id IS UNIQUE
""")

# Query to merge journal nodes in the database
merge_journal_node_query = """
MERGE (journal:Journal {name: $journalName})
RETURN journal
"""

# Query to merge paper nodes in the database and link them to their journal
merge_paper_node_query = """
MATCH (journal:Journal {name: $journalName})
MERGE (paper:Paper {id: $paperParam.id})
    ON CREATE SET 
        paper.title = $paperParam.title,
        paper.keywords = $paperParam.keywords,
        paper.abstract = $paperParam.abstract,
        paper.embedding = $paperParam.embedding
MERGE (paper)-[:PUBLISHED_IN]->(journal)
RETURN paper
"""

# Load papers from the JSONL file
with open('/Users/thanish/Desktop/VS_Code/papers.jsonl', 'r') as file:
    papers = []
    for line in file:
        papers.append(json.loads(line))

# Extract unique journal names, including a default for missing journals
journals = {paper.get('journal', 'Unknown') for paper in papers}

# Create journal nodes in the Neo4j database
for journal_name in journals:
    kg.query(merge_journal_node_query, params={'journalName': journal_name})
    print(f"Created :Journal node for journal: {journal_name}")

# Initialize the sentence-transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Dictionary to store embeddings for each journal
journal_embeddings = {journal_name: [] for journal_name in journals}

# Process each paper and add them as child nodes to the corresponding journal node
node_count = 0
for paper in papers:
    # Ensure 'journal' key exists
    journal_name = paper.get('journal', 'Unknown')
    
    # Preprocess the abstract or title for better embedding generation
    paper_text = paper.get('abstract', '')
    if not paper_text.strip():
        continue  # Skip if there's no abstract
    
    # Generate embeddings for the abstract
    embedding = model.encode(paper_text)
    paper['embedding'] = embedding.tolist()  # Convert to list for JSON compatibility
    
    # Add paper embedding to the corresponding journal's list
    journal_embeddings[journal_name].append(embedding)
    
    # Create or update the paper node and link it to the corresponding journal
    kg.query(merge_paper_node_query, params={
        'journalName': journal_name,
        'paperParam': paper
    })
    node_count += 1
    print(f"Created :Paper node for paper ID {paper['id']} and linked to journal {journal_name}")

print(f"Created {node_count} paper nodes and linked them to their respective journals")

# Now calculate the average embedding for each journal and update the journal nodes
for journal_name, embeddings in journal_embeddings.items():
    if embeddings:
        embeddings_matrix = np.vstack(embeddings)  # Stack embeddings into a matrix
        avg_embedding = np.mean(embeddings_matrix, axis=0).tolist()
        # Update the journal node with the average embedding
        kg.query("""
        MATCH (journal:Journal {name: $journalName})
        SET journal.embedding = $embedding
        RETURN journal
        """, params={'journalName': journal_name, 'embedding': avg_embedding})
        print(f"Updated :Journal node for {journal_name} with average embedding")

# Create vector index for the journal embeddings

kg.query("""
CREATE VECTOR INDEX journal_embeddings IF NOT EXISTS
FOR (j:Journal) ON (j.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: "cosine"
    }
}
""")

# Print final message
print(f"Created vector index for journal embeddings")