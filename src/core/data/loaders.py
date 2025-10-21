"""
Document loaders for PDF files.

Supports PDF documents with proper error handling.
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from src.utils.logging import get_logger
from src.utils.exceptions import DataProcessingError
from src.config.settings import get_config

logger = get_logger(__name__)


class DocumentLoader:
    """Base class for document loading operations."""
    
    def __init__(self, data_folder: Optional[str] = None):
        """
        Initialize document loader.
        
        Args:
            data_folder: Path to data folder, uses config default if None
        """
        self.config = get_config()
        self.data_folder = data_folder or self.config.data.data_folder
        self._ensure_data_folder()
    
    def _ensure_data_folder(self) -> None:
        """Ensure data folder exists."""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            logger.info(f"📁 Created data folder: {self.data_folder}")
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from the data folder.
        
        Returns:
            List of loaded documents
            
        Raises:
            DataProcessingError: If loading fails
        """
        raise NotImplementedError("Subclasses must implement load_documents")


class PDFDocumentLoader(DocumentLoader):
    """Loader for PDF documents."""
    
    def __init__(self, data_folder: Optional[str] = None, glob_pattern: str = "*.pdf"):
        """
        Initialize PDF loader.
        
        Args:
            data_folder: Path to data folder
            glob_pattern: File pattern to match
        """
        super().__init__(data_folder)
        self.glob_pattern = glob_pattern
    
    def load_documents(self) -> List[Document]:
        """
        Load PDF documents from data folder.
        
        Returns:
            List of PDF documents
            
        Raises:
            DataProcessingError: If loading fails
        """
        try:
            logger.info(f"📄 Loading PDF documents from: {self.data_folder}")
            loader = DirectoryLoader(
                self.data_folder,
                glob=self.glob_pattern,
                loader_cls=PyMuPDFLoader
            )
            documents = loader.load()
            logger.info(f"✅ Loaded {len(documents)} PDF documents")
            return documents
        except Exception as e:
            error_msg = f"Failed to load PDF documents: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e


def load_pdf_documents(data_folder: Optional[str] = None) -> List[Document]:
    """
    Convenience function to load PDF documents.
    
    Args:
        data_folder: Path to data folder
        
    Returns:
        List of PDF documents
    """
    loader = PDFDocumentLoader(data_folder)
    return loader.load_documents()
