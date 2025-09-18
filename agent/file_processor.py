"""File processing service for document analysis and content extraction."""

import os
import uuid
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import PyPDF2
from docx import Document

logger = logging.getLogger(__name__)


class FileProcessor:
    """Service for processing uploaded files and extracting content."""

    ALLOWED_EXTENSIONS = {
        '.txt', '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss',
        '.md', '.json', '.yaml', '.yml', '.xml', '.sql', '.sh', '.bat',
        '.pdf', '.docx', '.doc', '.c', '.cpp', '.h', '.hpp', '.java',
        '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.r'
    }

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR = Path("data/uploads")

    def __init__(self):
        """Initialize file processor."""
        self.upload_dir = self.UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Validate uploaded file for security and compatibility.

        Args:
            filename: Original filename
            content: File content bytes

        Returns:
            Validation result with status and details
        """
        result = {
            "valid": False,
            "errors": [],
            "file_size": len(content),
            "content_type": None,
            "extension": None
        }

        # Check file size
        if len(content) > self.MAX_FILE_SIZE:
            result["errors"].append(f"File too large. Max size: {self.MAX_FILE_SIZE // (1024*1024)}MB")
            return result

        # Check file extension
        file_path = Path(filename.lower())
        extension = file_path.suffix
        result["extension"] = extension

        if extension not in self.ALLOWED_EXTENSIONS:
            result["errors"].append(f"File type '{extension}' not allowed")
            return result

        # Determine content type
        content_type, _ = mimetypes.guess_type(filename)
        result["content_type"] = content_type or "application/octet-stream"

        # Basic content validation
        try:
            if extension in {'.txt', '.py', '.js', '.md', '.json', '.yaml', '.yml'}:
                # Ensure it's valid text
                content.decode('utf-8')
            elif extension == '.pdf':
                # Basic PDF validation
                if not content.startswith(b'%PDF-'):
                    result["errors"].append("Invalid PDF file format")
                    return result
            elif extension in {'.docx', '.doc'}:
                # Basic Office document validation
                if extension == '.docx' and not content.startswith(b'PK'):
                    result["errors"].append("Invalid DOCX file format")
                    return result
        except UnicodeDecodeError:
            result["errors"].append("File contains invalid text encoding")
            return result

        # Check for potentially malicious patterns
        content_str = content[:1024].decode('utf-8', errors='ignore').lower()
        suspicious_patterns = ['<script', 'javascript:', 'vbscript:', 'data:']
        for pattern in suspicious_patterns:
            if pattern in content_str:
                result["errors"].append("File contains potentially malicious content")
                return result

        result["valid"] = True
        return result

    def save_file(self, filename: str, content: bytes) -> str:
        """Save uploaded file to storage.

        Args:
            filename: Original filename
            content: File content bytes

        Returns:
            Unique file ID
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Preserve original extension
        file_path = Path(filename)
        extension = file_path.suffix

        # Save file with unique name
        save_path = self.upload_dir / f"{file_id}{extension}"
        save_path.write_bytes(content)

        logger.info(f"Saved file {filename} as {save_path}")
        return file_id

    def extract_content(self, file_id: str, filename: str) -> Optional[str]:
        """Extract text content from uploaded file.

        Args:
            file_id: Unique file identifier
            filename: Original filename for extension detection

        Returns:
            Extracted text content or None if extraction failed
        """
        file_path = Path(filename)
        extension = file_path.suffix.lower()

        # Find saved file
        saved_files = list(self.upload_dir.glob(f"{file_id}.*"))
        if not saved_files:
            logger.error(f"File not found: {file_id}")
            return None

        saved_path = saved_files[0]

        try:
            if extension in {'.txt', '.py', '.js', '.ts', '.jsx', '.tsx', '.html',
                           '.css', '.scss', '.md', '.json', '.yaml', '.yml', '.xml',
                           '.sql', '.sh', '.bat', '.c', '.cpp', '.h', '.hpp',
                           '.java', '.go', '.rs', '.php', '.rb', '.swift', '.kt',
                           '.scala', '.r'}:
                # Plain text files
                return saved_path.read_text(encoding='utf-8')

            elif extension == '.pdf':
                return self._extract_pdf_content(saved_path)

            elif extension == '.docx':
                return self._extract_docx_content(saved_path)

            else:
                logger.warning(f"Content extraction not supported for {extension}")
                return None

        except Exception as e:
            logger.error(f"Error extracting content from {file_id}: {e}")
            return None

    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        content = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        content.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {e}")

        return "\n\n".join(content)

    def _extract_docx_content(self, file_path: Path) -> str:
        """Extract text content from DOCX file.

        Args:
            file_path: Path to DOCX file

        Returns:
            Extracted text content
        """
        content = []

        doc = Document(file_path)

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)

        return "\n\n".join(content)

    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored file.

        Args:
            file_id: Unique file identifier

        Returns:
            File information dictionary or None if not found
        """
        saved_files = list(self.upload_dir.glob(f"{file_id}.*"))
        if not saved_files:
            return None

        saved_path = saved_files[0]
        stat = saved_path.stat()

        return {
            "file_id": file_id,
            "file_path": str(saved_path),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "extension": saved_path.suffix
        }

    def delete_file(self, file_id: str) -> bool:
        """Delete a stored file.

        Args:
            file_id: Unique file identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        saved_files = list(self.upload_dir.glob(f"{file_id}.*"))

        for file_path in saved_files:
            try:
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        return False

    def cleanup_old_files(self, max_age_days: int = 7) -> int:
        """Clean up old uploaded files.

        Args:
            max_age_days: Maximum age in days before deletion

        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {file_path}: {e}")

        return deleted_count


# Global file processor instance
file_processor = FileProcessor()