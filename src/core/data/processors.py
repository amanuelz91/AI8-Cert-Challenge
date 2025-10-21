"""
Document processing and text chunking utilities.

Handles text splitting, semantic chunking, and document preprocessing.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from src.utils.logging import get_logger
from src.utils.exceptions import DataProcessingError
from src.utils.decorators import timing_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class DocumentProcessor:
    """Base class for document processing operations."""
    
    def __init__(self):
        """Initialize document processor."""
        self.config = get_config()
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed documents
            
        Raises:
            DataProcessingError: If processing fails
        """
        raise NotImplementedError("Subclasses must implement process_documents")


class TextSplitter(DocumentProcessor):
    """Text splitter for document chunking."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            separators: Text separators for splitting
        """
        super().__init__()
        self.chunk_size = chunk_size or self.config.data.chunk_size
        self.chunk_overlap = chunk_overlap or self.config.data.chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    @timing_decorator
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
            
        Raises:
            DataProcessingError: If splitting fails
        """
        try:
            logger.info(f"✂️ Splitting {len(documents)} documents into chunks "
                       f"(size={self.chunk_size}, overlap={self.chunk_overlap})")
            
            split_documents = self.splitter.split_documents(documents)
            
            logger.info(f"✅ Created {len(split_documents)} document chunks")
            return split_documents
        except Exception as e:
            error_msg = f"Failed to split documents: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e


class SemanticSplitter(DocumentProcessor):
    """Semantic chunker for better document boundaries."""
    
    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile"):
        """
        Initialize semantic splitter.
        
        Args:
            embeddings: Embedding model for semantic analysis
            breakpoint_threshold_type: Type of threshold for breakpoints
        """
        super().__init__()
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type
        )
    
    @timing_decorator
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using semantic boundaries.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of semantically-chunked documents
            
        Raises:
            DataProcessingError: If splitting fails
        """
        try:
            logger.info(f"🧠 Semantic splitting {len(documents)} documents")
            
            split_documents = self.splitter.split_documents(documents)
            
            logger.info(f"✅ Created {len(split_documents)} semantic chunks")
            return split_documents
        except Exception as e:
            error_msg = f"Failed to semantically split documents: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e


class DocumentPreprocessor(DocumentProcessor):
    """Preprocessor for document cleaning and formatting."""
    
    def __init__(self, min_length: int = 100, max_length: int = 10000):
        """
        Initialize document preprocessor.
        
        Args:
            min_length: Minimum document length
            max_length: Maximum document length
        """
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
    
    def _clean_text(self, text: str) -> str:
        """
        Clean document text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove excessive redactions (more than 5 XXXX)
        if text.count("XXXX") > 5:
            return ""
        
        # Remove empty or placeholder text
        if text.strip() in ["", "None", "N/A", "N/A.", "N/A,"]:
            return ""
        
        return text.strip()
    
    def _format_csv_document(self, doc: Document) -> Document:
        """
        Format CSV document with structured content.
        
        Args:
            doc: Document to format
            
        Returns:
            Formatted document
        """
        metadata = doc.metadata
        
        # Extract key information
        issue = metadata.get("Issue", "Unknown")
        product = metadata.get("Product", "Unknown")
        narrative = metadata.get("Consumer complaint narrative", "")
        
        # Create structured content
        formatted_content = f"Customer Issue: {issue}\n"
        formatted_content += f"Product: {product}\n"
        formatted_content += f"Complaint Details: {narrative}"
        
        doc.page_content = formatted_content
        return doc
    
    @timing_decorator
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents for quality and formatting.
        
        Args:
            documents: List of documents to preprocess
            
        Returns:
            List of preprocessed documents
            
        Raises:
            DataProcessingError: If preprocessing fails
        """
        try:
            logger.info(f"🔧 Preprocessing {len(documents)} documents")
            
            processed_docs = []
            filter_stats = {
                "too_short": 0,
                "too_long": 0,
                "empty_content": 0,
                "processed": 0
            }
            
            for doc in documents:
                # Clean text content
                cleaned_text = self._clean_text(doc.page_content)
                
                # Apply length filters
                if len(cleaned_text) < self.min_length:
                    filter_stats["too_short"] += 1
                    continue
                
                if len(cleaned_text) > self.max_length:
                    filter_stats["too_long"] += 1
                    continue
                
                if not cleaned_text:
                    filter_stats["empty_content"] += 1
                    continue
                
                # Format document based on type
                if doc.metadata.get("source", "").endswith(".csv"):
                    doc = self._format_csv_document(doc)
                else:
                    doc.page_content = cleaned_text
                
                processed_docs.append(doc)
                filter_stats["processed"] += 1
            
            # Log filter results
            logger.info(f"📊 Preprocessing results:")
            logger.info(f"   ✅ Processed: {filter_stats['processed']}")
            logger.info(f"   ❌ Too short: {filter_stats['too_short']}")
            logger.info(f"   ❌ Too long: {filter_stats['too_long']}")
            logger.info(f"   ❌ Empty content: {filter_stats['empty_content']}")
            
            if len(documents) > 0:
                retention_rate = (filter_stats["processed"] / len(documents)) * 100
                logger.info(f"📈 Retention rate: {retention_rate:.1f}%")
            else:
                logger.warning("⚠️ No documents to process")
            
            return processed_docs
        except Exception as e:
            error_msg = f"Failed to preprocess documents: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Convenience function to split documents.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    splitter = TextSplitter(chunk_size, chunk_overlap)
    return splitter.process_documents(documents)


def preprocess_documents(
    documents: List[Document],
    min_length: int = 100,
    max_length: int = 10000
) -> List[Document]:
    """
    Convenience function to preprocess documents.
    
    Args:
        documents: List of documents to preprocess
        min_length: Minimum document length
        max_length: Maximum document length
        
    Returns:
        List of preprocessed documents
    """
    preprocessor = DocumentPreprocessor(min_length, max_length)
    return preprocessor.process_documents(documents)
