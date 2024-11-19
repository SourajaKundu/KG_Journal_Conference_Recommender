from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv('.env', override=True)

# Neo4j connection setup
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

def extract_graph_to_matrix(kg) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Extract the graph into an adjacency matrix with correct handling of journal relationships
    """
    # First get all journal nodes
    print("Fetching journal nodes...")
    journals = kg.query("""
    MATCH (j:Journal)
    RETURN j.id AS id, j.name AS name
    ORDER BY j.id
    """)
    
    # Then get all paper nodes
    print("Fetching paper nodes...")
    papers = kg.query("""
    MATCH (p:Paper)
    RETURN p.id AS id, p.title AS title
    ORDER BY p.id
    """)
    
    # Create ordered list of all node IDs (journals first, then papers)
    node_ids = [j['id'] for j in journals] + [p['id'] for p in papers]
    
    # Create ID to index mapping for faster lookup
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Initialize empty adjacency matrix
    n_nodes = len(node_ids)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    # 1. Add Journal-Journal similarities
    print("Adding journal-journal similarities...")
    journal_similarities = kg.query("""
    MATCH (j1:Journal)-[r:SIMILAR_CONTENT]->(j2:Journal)
    RETURN j1.id AS journal1_id, j2.id AS journal2_id, r.weight AS weight
    """)
    
    journal_rel_count = 0
    for rel in tqdm(journal_similarities):
        idx1 = id_to_idx[rel['journal1_id']]
        idx2 = id_to_idx[rel['journal2_id']]
        weight = rel['weight']
        # Make it symmetric
        adj_matrix[idx1, idx2] = weight
        adj_matrix[idx2, idx1] = weight
        journal_rel_count += 1
    
    print(f"Added {journal_rel_count} journal-journal relationships")
    
    # 2. Add Paper-Journal relationships
    print("Adding paper-journal relationships...")
    paper_journal_rels = kg.query("""
    MATCH (p:Paper)-[:PUBLISHED_IN]->(j:Journal)
    RETURN p.id AS paper_id, j.id AS journal_id
    """)
    
    for rel in tqdm(paper_journal_rels):
        paper_idx = id_to_idx[rel['paper_id']]
        journal_idx = id_to_idx[rel['journal_id']]
        # Make it bidirectional
        adj_matrix[paper_idx, journal_idx] = 1
        adj_matrix[journal_idx, paper_idx] = 1
    
    # 3. Add Paper-Paper similarities
    print("Adding paper-paper similarities...")
    paper_similarities = kg.query("""
    MATCH (p1:Paper)-[r:SIMILAR_TO]-(p2:Paper)
    WHERE p1.id < p2.id  // To avoid duplicate relationships
    RETURN p1.id AS paper1_id, p2.id AS paper2_id, r.weight AS weight
    """)
    
    for rel in tqdm(paper_similarities):
        idx1 = id_to_idx[rel['paper1_id']]
        idx2 = id_to_idx[rel['paper2_id']]
        # Make it symmetric
        adj_matrix[idx1, idx2] = rel['weight']
        adj_matrix[idx2, idx1] = rel['weight']
    
    # Create info dictionary
    info = {
        'n_journals': len(journals),
        'n_papers': len(papers),
        'total_nodes': n_nodes,
        'journal_indices': (0, len(journals)),
        'paper_indices': (len(journals), n_nodes)
    }
    
    return adj_matrix, node_ids, info

def analyze_matrix_structure(adj_matrix: np.ndarray, info: Dict):
    """
    Analyze and print detailed statistics about the matrix structure
    """
    n_journals = info['n_journals']
    
    # Analyze journal-journal relationships
    journal_submatrix = adj_matrix[:n_journals, :n_journals]
    journal_relationships = np.count_nonzero(np.triu(journal_submatrix, k=1))
    
    # Only calculate statistics if there are relationships
    if journal_relationships > 0:
        nonzero_values = journal_submatrix[np.triu_indices(n_journals, k=1)][
            journal_submatrix[np.triu_indices(n_journals, k=1)] != 0
        ]
        journal_avg_sim = np.mean(nonzero_values)
        journal_sim_range = (np.min(nonzero_values), np.max(nonzero_values))
    else:
        journal_avg_sim = 0
        journal_sim_range = (0, 0)
    
    # Analyze paper-paper relationships
    paper_submatrix = adj_matrix[n_journals:, n_journals:]
    paper_relationships = np.count_nonzero(np.triu(paper_submatrix, k=1))
    paper_avg_sim = np.mean(paper_submatrix[np.nonzero(np.triu(paper_submatrix, k=1))])
    paper_sim_range = (
        np.min(paper_submatrix[np.nonzero(np.triu(paper_submatrix, k=1))]),
        np.max(paper_submatrix[np.nonzero(np.triu(paper_submatrix, k=1))])
    )
    
    # Analyze paper-journal relationships
    paper_journal_submatrix = adj_matrix[n_journals:, :n_journals]
    paper_journal_relationships = np.count_nonzero(paper_journal_submatrix)
    
    print("\nDetailed Matrix Analysis:")
    print("\nJournal-Journal Network:")
    print(f"Total Relationships: {journal_relationships}")
    if journal_relationships > 0:
        print(f"Average Similarity: {journal_avg_sim:.3f}")
        print(f"Similarity Range: {journal_sim_range[0]:.3f} to {journal_sim_range[1]:.3f}")
    
    print("\nPaper-Paper Network:")
    print(f"Total Relationships: {paper_relationships}")
    print(f"Average Similarity: {paper_avg_sim:.3f}")
    print(f"Similarity Range: {paper_sim_range[0]:.3f} to {paper_sim_range[1]:.3f}")
    
    print("\nPaper-Journal Relationships:")
    print(f"Total Relationships: {paper_journal_relationships}")

def save_matrix_to_file(adj_matrix: np.ndarray, node_ids: List[str], info: Dict, 
                       matrix_file: str = 'adjacency_matrix.npz', 
                       mapping_file: str = 'node_mapping.csv'):
    """
    Save the adjacency matrix and node mapping to files
    """
    # Save the sparse matrix
    print(f"\nSaving adjacency matrix to {matrix_file}...")
    np.savez_compressed(matrix_file, matrix=adj_matrix)
    
    # Save the node mapping
    print(f"Saving node mapping to {mapping_file}...")
    mapping_df = pd.DataFrame({
        'index': range(len(node_ids)),
        'node_id': node_ids,
        'type': ['Journal' if i < info['n_journals'] else 'Paper' 
                for i in range(len(node_ids))]
    })
    mapping_df.to_csv(mapping_file, index=False)
    
    # Print matrix statistics
    print("\nMatrix Statistics:")
    print(f"Shape: {adj_matrix.shape}")
    print(f"Non-zero Elements: {np.count_nonzero(adj_matrix)}")
    print(f"Matrix Density: {np.count_nonzero(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1]):.4f}")
    
    # Analyze detailed matrix structure
    analyze_matrix_structure(adj_matrix, info)

def main():
    # Create the adjacency matrix
    adj_matrix, node_ids, info = extract_graph_to_matrix(kg)
    
    # Save the results
    save_matrix_to_file(adj_matrix, node_ids, info)

if __name__ == "__main__":
    main()