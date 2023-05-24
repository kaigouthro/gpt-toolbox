from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import time

from db import Document, DocumentMetadata

@dataclass
class DocumentExtractorResult:
    document: Document
    metadata: DocumentMetadata

DocumentExtractorResults = List[DocumentExtractorResult]

class AbstractDocumentExtractor(ABC):
    @abstractmethod
    def extract(self, 
                source: str, 
                additional_metadata: DocumentMetadata = None) -> DocumentExtractorResults:
        pass

def to_documents(source: str, additional_metadata: DocumentMetadata = None) -> DocumentExtractorResults:
    universal_metadata = {
        'extracted_at': int(time.time()),
    }

    if additional_metadata is None:
        additional_metadata = {}

    return [DocumentExtractorResult(
        document=source,
        metadata={
            **universal_metadata,
            **additional_metadata,
        }
    )]

class SimpleDocumentExtractor(AbstractDocumentExtractor):
    def extract(self,
                source: str,
                additional_metadata: DocumentMetadata = None) -> DocumentExtractorResults:
        return to_documents(source, additional_metadata)

class DocumentExtractor(AbstractDocumentExtractor):
    def __init__(self, extractors: List[AbstractDocumentExtractor] = None):
        if extractors is None:
            extractors = [SimpleDocumentExtractor()]

        self.extractors = extractors

    def extract(self, 
                source: str, 
                additional_metadata: DocumentMetadata = None) -> DocumentExtractorResults:
        extracted = []

        for extractor in self.extractors:
            extracted.extend(extractor.extract(source, additional_metadata))

        return extracted
