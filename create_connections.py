from dotenv import load_dotenv
import os
import numpy as np
from langchain_community.graphs import Neo4jGraph
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv('.env', override=True)

# Neo4j connection setup
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

def create_indices():
    """Create indices for the similarity relationship weights"""
    kg.query("""
    CREATE INDEX paper_similarity_weight IF NOT EXISTS
    FOR ()-[r:SIMILAR_TO]-()
    ON (r.weight)
    """)
    
    kg.query("""
    CREATE INDEX journal_similarity_weight IF NOT EXISTS
    FOR ()-[r:SIMILAR_CONTENT]-()
    ON (r.weight)
    """)

def process_paper_similarities():
    """Create paper-to-paper similarity network with threshold 0.5"""
    print("\nProcessing paper similarity network...")
    
    # Get all papers with their embeddings
    papers = kg.query("""
    MATCH (p:Paper)
    RETURN p.id AS id, p.embedding AS embedding
    """)
    
    if not papers:
        print("No papers found in the database!")
        return
    
    print(f"Processing {len(papers)} papers...")
    
    # Extract IDs and embeddings
    paper_ids = [p['id'] for p in papers]
    embeddings = np.array([p['embedding'] for p in papers])
    
    # Calculate similarity matrix
    print("Calculating paper similarity matrix...")
    similarities = cosine_similarity(embeddings)
    
    # Create relationships for papers with similarity > 0.5
    print("Creating SIMILAR_TO relationships for papers...")
    total_relationships = 0
    
    # Calculate total comparisons for progress bar
    total_comparisons = (len(papers) * (len(papers) - 1)) // 2
    
    with tqdm(total=total_comparisons, desc="Creating paper relationships") as pbar:
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):  # Avoid self-loops
                similarity = similarities[i][j]
                if similarity > 0.5:  # Only create relationship if similarity > 0.5
                    kg.query("""
                    MATCH (p1:Paper {id: $id1})
                    MATCH (p2:Paper {id: $id2})
                    MERGE (p1)-[r:SIMILAR_TO]-(p2)
                    SET r.weight = $weight
                    """, params={
                        'id1': paper_ids[i],
                        'id2': paper_ids[j],
                        'weight': float(similarity)
                    })
                    total_relationships += 1
                pbar.update(1)
    
    print(f"\nCreated {total_relationships} paper similarity relationships")

def process_journal_similarities():
    """Create journal-to-journal similarity network (all connections)"""
    print("\nProcessing journal similarity network...")
    
    # Get all journals with their embeddings
    journals = kg.query("""
    MATCH (j:Journal)
    RETURN j.name AS name, j.embedding AS embedding
    """)
    
    if not journals:
        print("No journals found in the database!")
        return
    
    print(f"Processing {len(journals)} journals...")
    
    # Extract names and embeddings
    journal_names = [j['name'] for j in journals]
    embeddings = np.array([j['embedding'] for j in journals])
    
    # Calculate similarity matrix
    print("Calculating journal similarity matrix...")
    similarities = cosine_similarity(embeddings)
    
    # Create relationships between all journals
    print("Creating SIMILAR_CONTENT relationships for journals...")
    total_relationships = 0
    
    # Calculate total comparisons for progress bar
    total_comparisons = (len(journals) * (len(journals) - 1)) // 2
    
    with tqdm(total=total_comparisons, desc="Creating journal relationships") as pbar:
        for i in range(len(journals)):
            for j in range(i + 1, len(journals)):  # Avoid self-loops
                similarity = similarities[i][j]
                # Create relationship for all journal pairs
                kg.query("""
                MATCH (j1:Journal {name: $name1})
                MATCH (j2:Journal {name: $name2})
                MERGE (j1)-[r:SIMILAR_CONTENT]-(j2)
                SET r.weight = $weight
                """, params={
                    'name1': journal_names[i],
                    'name2': journal_names[j],
                    'weight': float(similarity)
                })
                total_relationships += 1
                pbar.update(1)
    
    print(f"\nCreated {total_relationships} journal similarity relationships")

def print_network_statistics():
    """Print statistics about both similarity networks"""
    # Paper network statistics
    paper_stats = kg.query("""
    MATCH (p:Paper)
    OPTIONAL MATCH (p)-[r:SIMILAR_TO]-()
    RETURN 
        count(DISTINCT p) as nodeCount,
        count(DISTINCT r)/2 as relationshipCount,
        avg(r.weight) as avgSimilarity,
        min(r.weight) as minSimilarity,
        max(r.weight) as maxSimilarity
    """)[0]
    
    # Journal network statistics
    journal_stats = kg.query("""
    MATCH (j:Journal)
    OPTIONAL MATCH (j)-[r:SIMILAR_CONTENT]-()
    RETURN 
        count(DISTINCT j) as nodeCount,
        count(DISTINCT r)/2 as relationshipCount,
        avg(r.weight) as avgSimilarity,
        min(r.weight) as minSimilarity,
        max(r.weight) as maxSimilarity
    """)[0]
    
    print("\nPaper Network Statistics:")
    print(f"Total Papers: {paper_stats['nodeCount']}")
    print(f"Total Relationships (similarity > 0.5): {paper_stats['relationshipCount']}")
    print(f"Average Similarity: {paper_stats['avgSimilarity']:.3f}")
    print(f"Similarity Range: {paper_stats['minSimilarity']:.3f} - {paper_stats['maxSimilarity']:.3f}")
    
    print("\nJournal Network Statistics:")
    print(f"Total Journals: {journal_stats['nodeCount']}")
    print(f"Total Relationships: {journal_stats['relationshipCount']}")
    print(f"Average Similarity: {journal_stats['avgSimilarity']:.3f}")
    print(f"Similarity Range: {journal_stats['minSimilarity']:.3f} - {journal_stats['maxSimilarity']:.3f}")

def main():
    # Create indices
    print("Creating indices...")
    create_indices()
    
    # Process paper similarities (with threshold)
    process_paper_similarities()
    
    # Process journal similarities (all connections)
    process_journal_similarities()
    
    # Print statistics
    print("\nCalculating network statistics...")
    print_network_statistics()

if __name__ == "__main__":
    main()