from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

import re

class ContractIngester:
    """
    Handles the ingestion, structural splitting, and cleaning of insurance 
    contract documents to prepare them for vector indexation.
    """

    def __init__(self):
        # Define the markdown header levels to preserve contract hierarchy
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )

        # Configures recursive splitting to respect structural headers while 
        # managing chunk size for semantic retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                ". ",
                " "
            ]
        )

    def clean_text(self, text: str) -> str:
        """
        Cleans raw markdown text by removing OCR noise, decorative artifacts, 
        and structural tables that degrade embedding quality.
        """
        if not text:
            return ""

        # Remove large markdown table artifacts
        text = re.sub(r"\|[-| ]+\|", " ", text)

        # Remove decorative lines
        text = re.sub(r"-{4,}", " ", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove OCR page number artifacts
        text = re.sub(r"Page\s+\d+\s*/\s*\d+", " ", text)

        return text.strip()

    def is_valid_chunk(self, text: str) -> bool:
        """
        Filters out low-quality chunks (too short, noise-heavy, or non-informative)
        to improve retrieval precision.
        """
        if not text:
            return False

        # Filter out chunks that are too small to be meaningful
        if len(text) < 120:
            return False

        # Filter out chunks with excessive table content
        if text.count("|") > 20:
            return False

        # Filter out low-information-density chunks (e.g., numeric lists/noise)
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.55:
            return False

        return True

    def process_text(
        self,
        md_content: str,
        source_name: str
    ):
        """
        Orchestrates the full processing pipeline: clean -> split by headers -> 
        re-split recursively -> filter invalid chunks.
        """
        md_content = self.clean_text(md_content)

        # First pass: Segment by Markdown headers (Article/Section level)
        sections = self.header_splitter.split_text(md_content)

        # Second pass: Recursive character splitting within sections
        chunks = self.text_splitter.split_documents(sections)

        final_chunks = []

        for chunk in chunks:
            content = self.clean_text(chunk.page_content)

            # Apply quality validation filter
            if not self.is_valid_chunk(content):
                continue

            chunk.page_content = content
            chunk.metadata["source"] = source_name

            # Log chunk size for future reranking or diagnostic tasks
            chunk.metadata["chunk_size"] = len(content)

            final_chunks.append(chunk)

        print(
            f"✅ Final clean chunks: {len(final_chunks)}"
        )

        return final_chunks