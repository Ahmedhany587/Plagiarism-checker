import os
from src.core.pdf_handler import PDFHandler
from src.core.embedding_generator import EmbeddingGenerator
from src.core.semantic_similarity import PDFSimilarityCalculator
from src.core.sequence_similarity import SequenceSimilarityCalculator
from src.core.exact_match import ExactMatchDetector

def main():
    # Path to directory containing PDF
    pdf_dir = r"C:\Users\EJAD TECH\Desktop\New folder" 

    # Step 1: Load and chunk PDFs
    pdf_handler = PDFHandler(pdf_dir)
    print(f"Found {pdf_handler.get_pdf_count()} PDF files.")
    page_counts = pdf_handler.get_page_counts()
    for pdf, count in page_counts.items():
        print(f"{pdf}: {count} pages")

    chunks = pdf_handler.extract_page_chunks(chunk_size=5000)

    # Step 2: Generate sentence embeddings for all chunks
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(chunks)

    # Step 3: Compute semantic similarity between all PDF pairs
    similarity_calculator = PDFSimilarityCalculator(embeddings)
    semantic_similarity_scores = similarity_calculator.compute_all_pdf_similarities()

    # Display semantic similarity results
    print("\nSemantic Similarity Between PDFs (max, min, mean):")
    for (pdfA, pdfB), (max_sim, min_sim, mean_sim) in semantic_similarity_scores.items():
        print(f"{pdfA} <-> {pdfB}: Max: {max_sim:.2%}, Min: {min_sim:.2%}, Mean: {mean_sim:.2%}")

    # Step 4: Compute sequence similarity between all PDF pairs
    seq_similarity_calculator = SequenceSimilarityCalculator(chunks)
    sequence_similarity_scores = seq_similarity_calculator.compute_all_pdf_similarities()

    # Display sequence similarity results
    print("\nSequence Similarity Between PDFs (max, min, mean):")
    for (pdfA, pdfB), (max_sim, min_sim, mean_sim) in sequence_similarity_scores.items():
        print(f"{pdfA} <-> {pdfB}: Max: {max_sim:.2f}, Min: {min_sim:.2f}, Mean: {mean_sim:.2f}")

    # Step 5: Detect exact matches between PDFs (chunk-based)
    exact_match_detector = ExactMatchDetector()
    exact_matches = exact_match_detector.find_exact_matches(chunks)

    # Display exact match results
    print("\nExact Chunk Matches Between PDFs:")
    if exact_matches:
        for pdfA, chunkA, pdfB, chunkB, _ in exact_matches:
            print(f"Same chunk detected: {pdfA} (chunk {chunkA+1}) <-> {pdfB} (chunk {chunkB+1})")
    else:
        print("No exact chunk matches found.")

if __name__ == "__main__":
    main()
