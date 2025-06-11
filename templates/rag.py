import os
import re
import glob
import json
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
from tqdm import tqdm

# For parsing Python files and notebooks
import ast
import nbformat

# For PDF processing
import fitz  # PyMuPDF

# For embeddings
from sentence_transformers import SentenceTransformer

# For vector store
import faiss

# For LLM integration (assuming you have a pre-trained model)
from transformers import AutoTokenizer, AutoModelForCausalLM


class CodeChunk:
    """Represents a code chunk with its metadata."""
    
    def __init__(self, content: str, source: str, chunk_type: str, 
                 function_name: Optional[str] = None,
                 class_name: Optional[str] = None,
                 docstring: Optional[str] = None):
        self.content = content
        self.source = source
        self.chunk_type = chunk_type  # 'function', 'class', 'module', 'doc'
        self.function_name = function_name
        self.class_name = class_name
        self.docstring = docstring
    
    def to_dict(self):
        return {
            "content": self.content,
            "source": self.source,
            "chunk_type": self.chunk_type,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "docstring": self.docstring
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            content=data["content"],
            source=data["source"],
            chunk_type=data["chunk_type"],
            function_name=data["function_name"],
            class_name=data["class_name"],
            docstring=data["docstring"]
        )
    
    def __str__(self):
        return f"{self.chunk_type}: {self.function_name or self.class_name or 'snippet'} from {self.source}"


class DocumentProcessor:
    """Processes various document types into code chunks."""
    
    @staticmethod
    def process_python_file(file_path: str) -> List[CodeChunk]:
        """Process a Python file into code chunks."""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the Python file
            tree = ast.parse(content)
            
            # Process module-level docstring
            if ast.get_docstring(tree):
                chunks.append(CodeChunk(
                    content=ast.get_docstring(tree),
                    source=file_path,
                    chunk_type='doc',
                    docstring=ast.get_docstring(tree)
                ))
            
            # Process functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_source = content[node.lineno-1:node.end_lineno]
                    docstring = ast.get_docstring(node)
                    chunks.append(CodeChunk(
                        content=func_source,
                        source=file_path,
                        chunk_type='function',
                        function_name=node.name,
                        docstring=docstring
                    ))
                elif isinstance(node, ast.ClassDef):
                    class_source = content[node.lineno-1:node.end_lineno]
                    docstring = ast.get_docstring(node)
                    chunks.append(CodeChunk(
                        content=class_source,
                        source=file_path,
                        chunk_type='class',
                        class_name=node.name,
                        docstring=docstring
                    ))
                    
                    # Process methods within classes
                    for child_node in ast.iter_child_nodes(node):
                        if isinstance(child_node, ast.FunctionDef):
                            method_source = content[child_node.lineno-1:child_node.end_lineno]
                            method_docstring = ast.get_docstring(child_node)
                            chunks.append(CodeChunk(
                                content=method_source,
                                source=file_path,
                                chunk_type='function',
                                function_name=f"{node.name}.{child_node.name}",
                                class_name=node.name,
                                docstring=method_docstring
                            ))
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
        
        return chunks
    
    @staticmethod
    def process_jupyter_notebook(file_path: str) -> List[CodeChunk]:
        """Process a Jupyter notebook into code chunks."""
        chunks = []
        try:
            nb = nbformat.read(file_path, as_version=4)
            
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    chunks.append(CodeChunk(
                        content=cell.source,
                        source=f"{file_path}#cell{i+1}",
                        chunk_type='module'
                    ))
                elif cell.cell_type == 'markdown':
                    # Only include markdown if it contains code examples (denoted by backticks)
                    code_blocks = re.findall(r'```python(.*?)```', cell.source, re.DOTALL)
                    for code in code_blocks:
                        chunks.append(CodeChunk(
                            content=code.strip(),
                            source=f"{file_path}#cell{i+1}",
                            chunk_type='doc'
                        ))
        except Exception as e:
            print(f"Error processing notebook {file_path}: {str(e)}")
        
        return chunks
    
    @staticmethod
    def process_pdf(file_path: str, chunk_size: int = 1000) -> List[CodeChunk]:
        """Process a PDF document into text chunks."""
        chunks = []
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            # Extract Python code blocks from PDF
            code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
            for code in code_blocks:
                chunks.append(CodeChunk(
                    content=code.strip(),
                    source=f"{file_path}",
                    chunk_type='doc'
                ))
            
            # Chunk the remaining text
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            for i, chunk in enumerate(text_chunks):
                chunks.append(CodeChunk(
                    content=chunk,
                    source=f"{file_path}#chunk{i+1}",
                    chunk_type='doc'
                ))
                
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
        
        return chunks


class RAGPipeline:
    """Complete RAG pipeline for code completion."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.embedding_model = SentenceTransformer(model_name).to(device)
        self.vector_db = None
        self.chunks = []
        self.embeddings = None
        
        # For storing mapping between FAISS indices and chunks
        self.index_to_chunk = {}
        
    def process_documents(self, 
                          python_files: List[str] = None,
                          jupyter_files: List[str] = None,
                          pdf_files: List[str] = None):
        """Process all documents and create chunks."""
        processor = DocumentProcessor()
        
        if python_files:
            for file in tqdm(python_files, desc="Processing Python files"):
                self.chunks.extend(processor.process_python_file(file))
        
        if jupyter_files:
            for file in tqdm(jupyter_files, desc="Processing Jupyter notebooks"):
                self.chunks.extend(processor.process_jupyter_notebook(file))
        
        if pdf_files:
            for file in tqdm(pdf_files, desc="Processing PDF files"):
                self.chunks.extend(processor.process_pdf(file))
        
        print(f"Processed {len(self.chunks)} chunks in total")
        
    def build_vector_store(self):
        """Create embeddings and build the FAISS index."""
        # Create a mapping from index to chunk
        self.index_to_chunk = {i: chunk for i, chunk in enumerate(self.chunks)}
        
        # Prepare texts for embedding
        texts = [
            f"{chunk.chunk_type}: {chunk.function_name or chunk.class_name or ''} "
            f"{chunk.docstring or ''} {chunk.content}"
            for chunk in self.chunks
        ]
        
        # Create embeddings
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index
        print("Building FAISS index...")
        vector_dimension = self.embeddings.shape[1]
        self.vector_db = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity
        self.vector_db.add(self.embeddings)
        
        print(f"Vector store built with {self.vector_db.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[CodeChunk]:
        """Search for relevant code chunks given a query."""
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search the vector store
        scores, indices = self.vector_db.search(query_embedding, k)
        
        # Return the matched chunks
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.chunks):  # Ensure valid index
                results.append(self.index_to_chunk[idx])
        
        return results
    
    def save(self, output_dir: str):
        """Save the pipeline state."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(os.path.join(output_dir, "chunks.json"), "w") as f:
            json.dump(chunks_data, f)
        
        # Save the embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), self.embeddings)
        
        # Save the FAISS index
        faiss.write_index(self.vector_db, os.path.join(output_dir, "vector_db.index"))
        
        print(f"RAG pipeline saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str):
        """Load a previously saved pipeline."""
        pipeline = cls()
        
        # Load the chunks
        with open(os.path.join(input_dir, "chunks.json"), "r") as f:
            chunks_data = json.load(f)
            pipeline.chunks = [CodeChunk.from_dict(chunk) for chunk in chunks_data]
        
        # Rebuild the index-to-chunk mapping
        pipeline.index_to_chunk = {i: chunk for i, chunk in enumerate(pipeline.chunks)}
        
        # Load the embeddings
        pipeline.embeddings = np.load(os.path.join(input_dir, "embeddings.npy"))
        
        # Load the FAISS index
        pipeline.vector_db = faiss.read_index(os.path.join(input_dir, "vector_db.index"))
        
        print(f"RAG pipeline loaded from {input_dir}")
        return pipeline


class CodeCompletionSystem:
    """Complete code completion system using RAG."""
    
    def __init__(self, rag_pipeline: RAGPipeline, model_name: str = "gpt2-medium",
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.rag_pipeline = rag_pipeline
        self.device = device
        
        # Load the code generation model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    def complete_code(self, code_prefix: str, max_length: int = 50, 
                     k_chunks: int = 3, temperature: float = 0.7):
        """Generate code completion with RAG."""
        # Search for relevant chunks
        relevant_chunks = self.rag_pipeline.search(code_prefix, k=k_chunks)
        
        # Create a prompt with the retrieved chunks
        prompt = self._create_prompt(code_prefix, relevant_chunks)
        
        # Generate completion
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Calculate where to truncate from
        prefix_tokens = len(inputs.input_ids[0])
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=prefix_tokens + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Only return the newly generated tokens
        generated_text = self.tokenizer.decode(outputs[0][prefix_tokens:], skip_special_tokens=True)
        
        return generated_text, relevant_chunks
    
    def _create_prompt(self, code_prefix: str, chunks: List[CodeChunk]) -> str:
        """Create a prompt with context from retrieved chunks."""
        prompt = "# Reference code examples:\n"
        
        for i, chunk in enumerate(chunks):
            if chunk.chunk_type == 'function':
                prompt += f"\n# Example {i+1}: Function {chunk.function_name} from {chunk.source}\n"
            elif chunk.chunk_type == 'class':
                prompt += f"\n# Example {i+1}: Class {chunk.class_name} from {chunk.source}\n"
            else:
                prompt += f"\n# Example {i+1}: Code from {chunk.source}\n"
            
            prompt += f"{chunk.content}\n"
        
        prompt += "\n# Now complete this code:\n"
        prompt += code_prefix
        
        return prompt


# Example usage
def main():
    # Set up paths
    python_files = glob.glob("Dataset", recursive=True)
    jupyter_files = glob.glob("Dataset", recursive=True)
    pdf_files = ["pdf\numpy-1.18.pdf"]
    
    # Create the RAG pipeline
    rag = RAGPipeline()
    
    # Process documents
    rag.process_documents(
        python_files=python_files,
        jupyter_files=jupyter_files,
        pdf_files=pdf_files
    )
    
    # Build vector store
    rag.build_vector_store()
    
    # Save the pipeline for future use
    rag.save("./rag_pipeline_data")
    
    # Create the code completion system
    completion_system = CodeCompletionSystem(rag)
    
    # Example query
    code_prefix = "def calculate_matrix_norm(matrix):\n    # Calculate Frobenius norm of a matrix\n    "
    completion, chunks = completion_system.complete_code(code_prefix)
    
    print(f"Code completion: {completion}")
    print("\nRetrieved chunks:")
    for chunk in chunks:
        print(f"- {chunk}")


if __name__ == "__main__":
    main()