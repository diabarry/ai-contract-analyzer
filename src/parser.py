import pymupdf4llm
import gc

class ContractParser:
    """
    A utility class to handle PDF parsing and conversion into Markdown format.
    """

    def convert_to_markdown(self, pdf_path: str) -> str:
        """
        Converts a PDF file at the given path into a Markdown string.
        """
        try:
            # Transform PDF content to Markdown while ignoring images and code blocks
            md_text = pymupdf4llm.to_markdown(
                pdf_path,
                write_images=False,
                ignore_code=True
            )

            # Trigger garbage collection to free up memory after document processing
            gc.collect()

            return md_text

        except Exception as e:
            # Log parsing errors and return an empty string as a fallback
            print(f"Parser error: {e}")
            return ""