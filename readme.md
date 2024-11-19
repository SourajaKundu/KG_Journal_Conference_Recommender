# Academic Journal Recommender System

## Overview
This project implements an intelligent journal recommendation system using Graph Neural Networks (GNN) and Natural Language Processing (NLP) techniques. The system helps researchers identify suitable academic journals for their papers by analyzing the content similarity and analyzing the paper's novelty score.

## Features
- **Journal Recommendations**: Uses GCN (Graph Convolutional Network) to generate journal recommendations based on paper content
- **Novelty Analysis**: Calculates paper novelty scores by comparing with existing publications
- **Interactive Web Interface**: Streamlit-based user interface for easy paper submission and analysis
- **Knowledge Graph**: Neo4j-based graph database storing paper and journal relationships
- **Embedding Generation**: Uses BERT-based models for generating paper and journal embeddings

## Architecture
The system consists of several key components:

### 1. Knowledge Graph (create_KG.py)
- Builds a Neo4j graph database of papers and journals
- Creates relationships between papers and journals
- Generates and stores embeddings for papers and journals

### 2. Graph Neural Network (training-gcn.ipynb)
- Implements a GCN model for learning journal and paper representations
- Features:
  - 384-dimensional embeddings
  - Two GCN layers with batch normalization
  - Dropout for regularization
  - L2 normalization of embeddings

### 3. Connection Management (create_connections.py)
- Manages similarity relationships between papers and journals
- Creates and maintains graph indices
- Implements threshold-based paper similarity connections

### 4. Matrix Generation (extract_matrix.py)
- Converts graph relationships to adjacency matrices
- Handles different types of relationships (paper-paper, journal-journal)
- Generates node mappings for model training

### 5. Web Application (app.py)
- Streamlit-based user interface
- Features:
  - Paper submission interface
  - Journal recommendations
  - Novelty analysis
  - Exportable results

## Technical Details
- **Model Performance**:
  - Top-1 Accuracy: 84.66%
  - Top-3 Accuracy: 92.87%
  - Top-5 Accuracy: 95.01%
  - Top-10 Accuracy: 97.72%

## Installation

### Prerequisites
- Python 3.10 or higher
- Neo4j Database
- CUDA-compatible GPU (for training)

### Setup
1. Clone the repository:
```bash
git clone [repository-url]
cd academic-journal-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

4. Initialize the knowledge graph:
```bash
python create_KG.py
```

5. Generate connection matrices:
```bash
python extract_matrix.py
```

6. Train the model:
```bash
python training-gcn.py
```

7. Launch the web application:
```bash
streamlit run app.py
```

## Usage
1. Access the web interface at `http://localhost:8501`
2. Enter your paper's title and abstract
3. Click "Get Journal Recommendations"
4. Review recommendations and novelty analysis
5. Export results as needed

## Project Structure
```
├── Building_KG/
│   ├── .env                    # Environment variables
│   ├── adjacency_matrix.npz    # Generated adjacency matrix
│   ├── create_connections.py   # Graph relationship management
│   ├── create_KG.py           # Knowledge graph creation
│   ├── extract_matrix.py      # Matrix generation script
│   ├── journal_info.csv       # Journal information data
│   ├── journal_nodes.csv      # Journal nodes data
│   ├── matrix.ipynb           # Matrix manipulation notebook
│   ├── node_mapping.csv       # Node mapping data
│
├── Dataset/
│   ├── papers.jsonl           # Raw papers data
│   ├── papers[1].json        # Additional papers data
│   ├── test.jsonl            # Test dataset
│   ├── train.jsonl           # Training dataset
│   └── updated_dataframe.csv # Processed dataset
│
├── Journal_recomm_app/
│   ├── .gitattributes
│   ├── app.py                # Streamlit application
│   ├── journal_embeddings.npz # Generated embeddings
│   ├── r.ipynb               # Research notebook
│   ├── requirements.txt      # Project dependencies
│   └── updated_dataframe.csv # Application dataset
│
└── Training GCN and EDA/
    ├── embedding_metadata.csv    # Embedding metadata
    ├── journal_embeddings.csv    # Journal embeddings
    ├── journal_gcn_model.pt      # Trained GCN model
    ├── rough.ipynb              # Development notebook
    └── training-gcn (3).ipynb   # Main training notebook
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Built using PyTorch Geometric
- Transformer models from Hugging Face
- Neo4j for graph database
- Streamlit for web interface

## Contact
For questions and feedback, please open an issue in the repository.
