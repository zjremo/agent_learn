import pymupdf

class PdfReader:
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file and prints the first `num_chars` characters.

        Args:
        pdf_path (str): Path to the PDF file.

        Returns:
        str: Extracted text from the PDF.
        """
        mypdf = pymupdf.open(pdf_path)
        all_text = ""

        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            text = page.get_text("text")
            all_text += text
        return all_text

if __name__ == "__main__":
    file_path = "data/test.pdf"
    pr = PdfReader()
    all_text = pr.extract_text_from_pdf(file_path)
    print(all_text)

