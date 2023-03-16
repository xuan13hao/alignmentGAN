import numpy as np
from Bio import SeqIO

def hamming_distance(seq1, seq2):
    """
    Computes the Hamming distance between two sequences of equal length.
    """
    return sum(s1 != s2 for s1, s2 in zip(seq1, seq2))

def compute_similarity(query_seq, fasta_file):
    """
    Computes the similarity between a query sequence and all sequences in a FASTA file.
    Returns the sequence with the highest similarity.
    """
    best_similarity = np.inf
    best_sequence = None
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        similarity = hamming_distance(str(query_seq), str(record.seq))
        if similarity < best_similarity:
            best_similarity = similarity
            best_sequence = record.seq
    
    return str(best_sequence)
def hamming_distance(a, b):
    """
    Returns the Hamming distance between two sequences a and b.
    """
    if len(a) != len(b):
        raise ValueError("Sequences must be of equal length")

    return sum(1 for x, y in zip(a, b) if x != y)

def sequence_similarity(a, b):
    """
    Returns a float between 0 and 1 representing the similarity between
    two sequences using Hamming distance.
    """
    distance = hamming_distance(a, b)
    max_distance = max(len(a), len(b))
    return 1 - (distance / max_distance)

target = "AAGCAAATAAATCCTTTCAAACAAATACCACACAAGAAGTCCCCGGGAAGATTCTCTACGAAGTACTCCAAGCAAATCAGGTGTAATTCTGCCAATGTTTC"
#CCAAGATTCGAAGTTCCATGGTCCAGGTTGCTAGTATTACCCAAGCTGGATTAACCCATGGGATAAACTTTGCAGTGTCAAAAGTTCAGAAGAGTCCCCCA
#AAAAGAAAAGAGTGACCAGATGCAGGAGGGGAGGAAAAGCTCAAGGTGAAAGAAGGCTCCCCAGGGCCTGGGAGGTGCCACTAGCACTGCCCAGAAAAACA
fasta_file_path = "real_reads.fa"
fasta_file = "sequences.fasta"
best_sequence = compute_similarity(target, fasta_file_path)
print("Sequence with highest similarity:", best_sequence)
similarity = sequence_similarity(target, best_sequence)
print(f"Similarity between '{best_sequence}' and '{target}': {similarity}")
