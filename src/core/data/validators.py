"""
Data validation utilities for document quality assurance.

Provides validation functions for document content, metadata, and structure.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.utils.logging import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


class DocumentValidator:
    """Validator for document quality and structure."""
    
    def __init__(
        self,
        min_content_length: int = 50,
        max_content_length: int = 50000,
        required_metadata_keys: Optional[List[str]] = None
    ):
        """
        Initialize document validator.
        
        Args:
            min_content_length: Minimum content length
            max_content_length: Maximum content length
            required_metadata_keys: Required metadata keys
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.required_metadata_keys = required_metadata_keys or []
    
    def validate_document(self, document: Document) -> Dict[str, Any]:
        """
        Validate a single document.
        
        Args:
            document: Document to validate
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        # Validate content
        if not document.page_content:
            validation_result["errors"].append("Empty content")
            validation_result["is_valid"] = False
        elif len(document.page_content.strip()) < self.min_content_length:
            validation_result["errors"].append(f"Content too short (< {self.min_content_length} chars)")
            validation_result["is_valid"] = False
        elif len(document.page_content) > self.max_content_length:
            validation_result["warnings"].append(f"Content very long (> {self.max_content_length} chars)")
        
        # Validate metadata
        if not document.metadata:
            validation_result["warnings"].append("No metadata")
        else:
            # Check required metadata keys
            missing_keys = []
            for key in self.required_metadata_keys:
                if key not in document.metadata:
                    missing_keys.append(key)
            
            if missing_keys:
                validation_result["warnings"].append(f"Missing metadata keys: {missing_keys}")
        
        # Content quality checks
        content = document.page_content
        
        # Check for excessive redactions
        if content.count("XXXX") > 10:
            validation_result["warnings"].append("Excessive redactions detected")
        
        # Check for placeholder content
        placeholder_indicators = ["N/A", "None", "TBD", "To be determined"]
        if any(indicator in content for indicator in placeholder_indicators):
            validation_result["warnings"].append("Placeholder content detected")
        
        # Check for encoding issues
        if "â€™" in content or "â€œ" in content or "â€" in content:
            validation_result["warnings"].append("Potential encoding issues detected")
        
        validation_result["metadata"] = {
            "content_length": len(content),
            "metadata_keys": list(document.metadata.keys()) if document.metadata else [],
            "has_redactions": "XXXX" in content,
            "placeholder_content": any(indicator in content for indicator in placeholder_indicators)
        }
        
        return validation_result
    
    def validate_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Validate a list of documents.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            Aggregated validation results
            
        Raises:
            ValidationError: If validation fails
        """
        logger.info(f"🔍 Validating {len(documents)} documents")
        
        validation_results = []
        valid_count = 0
        error_count = 0
        warning_count = 0
        
        for i, document in enumerate(documents):
            try:
                result = self.validate_document(document)
                validation_results.append(result)
                
                if result["is_valid"]:
                    valid_count += 1
                else:
                    error_count += 1
                
                warning_count += len(result["warnings"])
                
            except Exception as e:
                logger.error(f"❌ Validation failed for document {i}: {str(e)}")
                error_count += 1
                validation_results.append({
                    "is_valid": False,
                    "errors": [f"Validation exception: {str(e)}"],
                    "warnings": [],
                    "metadata": {}
                })
        
        # Aggregate results
        aggregated_result = {
            "total_documents": len(documents),
            "valid_documents": valid_count,
            "invalid_documents": error_count,
            "total_warnings": warning_count,
            "validation_rate": (valid_count / len(documents)) * 100 if documents else 0,
            "individual_results": validation_results
        }
        
        logger.info(f"📊 Validation complete:")
        logger.info(f"   ✅ Valid: {valid_count}")
        logger.info(f"   ❌ Invalid: {error_count}")
        logger.info(f"   ⚠️ Warnings: {warning_count}")
        logger.info(f"   📈 Validation rate: {aggregated_result['validation_rate']:.1f}%")
        
        return aggregated_result


def validate_document_content(document: Document) -> bool:
    """
    Quick validation for document content.
    
    Args:
        document: Document to validate
        
    Returns:
        True if document is valid
        
    Raises:
        ValidationError: If document is invalid
    """
    if not document.page_content or len(document.page_content.strip()) < 10:
        raise ValidationError("Document has insufficient content")
    
    if len(document.page_content) > 100000:
        raise ValidationError("Document content is too large")
    
    return True


def validate_document_metadata(document: Document, required_keys: List[str]) -> bool:
    """
    Validate document metadata.
    
    Args:
        document: Document to validate
        required_keys: Required metadata keys
        
    Returns:
        True if metadata is valid
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if not document.metadata:
        raise ValidationError("Document has no metadata")
    
    missing_keys = [key for key in required_keys if key not in document.metadata]
    if missing_keys:
        raise ValidationError(f"Missing required metadata keys: {missing_keys}")
    
    return True


def get_document_statistics(documents: List[Document]) -> Dict[str, Any]:
    """
    Get statistics about a document collection.
    
    Args:
        documents: List of documents to analyze
        
    Returns:
        Dictionary with document statistics
    """
    if not documents:
        return {
            "total_documents": 0,
            "total_content_length": 0,
            "average_content_length": 0,
            "metadata_keys": [],
            "document_types": {}
        }
    
    total_length = sum(len(doc.page_content) for doc in documents)
    content_lengths = [len(doc.page_content) for doc in documents]
    
    # Collect metadata keys
    all_metadata_keys = set()
    document_types = {}
    
    for doc in documents:
        if doc.metadata:
            all_metadata_keys.update(doc.metadata.keys())
            
            # Track document types
            doc_type = doc.metadata.get("source", "unknown")
            if doc_type.endswith(".pdf"):
                doc_type = "PDF"
            elif doc_type.endswith(".csv"):
                doc_type = "CSV"
            else:
                doc_type = "Other"
            
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
    
    return {
        "total_documents": len(documents),
        "total_content_length": total_length,
        "average_content_length": total_length / len(documents),
        "min_content_length": min(content_lengths),
        "max_content_length": max(content_lengths),
        "metadata_keys": sorted(list(all_metadata_keys)),
        "document_types": document_types
    }
