import sys
import argparse
import faiss
import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text


def extract_text_as_chunks(file_path: str) -> list[str]:
    # if not os.path.isfile(file_path):
    #     raise FileNotFoundError(f"File '{file_path}' not found")
    # if not file_path.lower().endswith('.pdf'):
    #     raise ValueError(f"File '{file_path}' is not a PDF")

    text = extract_text_from_pdf(file_path)
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    
    # remove empty string from end when needed 
    if len(chunks[-1]) ==0 :
       del chunks[-1]
       
    return chunks



def create_embeddings(chunks: list[str]) -> list:

    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for all chunks
    embeddings = model.encode(chunks)
    
    return embeddings

# def store_embeddings_into_db(embeddings: list): #, db_path: str):
#     """
#     Store embeddings into a FAISS database at the specified path
    
#     Args:
#         embeddings: List of embeddings to store
#         db_path: Path where the FAISS database should be created/stored
#     """
    
#     # Convert embeddings to numpy array with correct data type
#     embeddings_array = np.array(embeddings).astype('float32')
    
#     # Get dimension of embeddings
#     dimension = embeddings_array.shape[1]
    
#     # Initialize FAISS index - using L2 distance
#     index = faiss.IndexFlatL2(dimension)
    
#     # Add vectors to the index
#     index.add(embeddings_array)
    
#     # # Create directory if it doesn't exist
#     # os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
#     # # Write index to disk
#     # faiss.write_index(index, db_path)

#     return index



def store_embeddings_into_db(embeddings: list, chunks: list[str], db_path: str):
    """
    Store embeddings and their corresponding text chunks into a FAISS database at the specified path
    
    Args:
        embeddings: List of embeddings to store
        chunks: List of text chunks corresponding to the embeddings
        db_path: Path where the FAISS database should be created/stored
    """
    
    # Convert embeddings to numpy array with correct data type
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Get dimension of embeddings
    dimension = embeddings_array.shape[1]
    
    # Initialize FAISS index - using L2 distance
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to the index
    index.add(embeddings_array)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Store the text chunks alongside the index
    metadata = {
        'chunks': chunks
    }
    
    # Write index and metadata to disk
    faiss.write_index(index, db_path)
    with open(db_path + '.metadata', 'wb') as f:
        pickle.dump(metadata, f)

    return index


def prepare_text_for_rag(file_path: str, db_path: str):
    print("Extracting text from PDF...")
    chunks = extract_text_as_chunks(file_path)
    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)
    print("Storing embeddings into FAISS database...")
    index = store_embeddings_into_db(embeddings, chunks=chunks, db_path=db_path)
    print("Index created successfully")
    return index

if __name__=="__main__":
    # Initialize parser
    # parser = argparse.ArgumentParser()
    # parser.add_argument('file_path', type=str, help='Complete path of the file', default="Source Material/TCS - annual-report-2021-2022.pdf")
    # args = parser.parse_args()
    # perform_rag_on_file(args.file_path)

    index = prepare_text_for_rag("Source Material/TCS - annual-report-2021-2022.pdf", "VectorDb/faiss_index.db")   
    print(index)



