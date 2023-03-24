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

# Example usage
target = "CTCATGTCATTTCATTTATTCATTGTTTTTTTTTATTTTTTTATATCTATTTTTTTCATTCATTGTTTTTTTTTTTATATTCATTATCATTATTCATTCAT"
subject = "GAATTCAATTTTAATATAAAAATTTGAAACATCCTGTTTCATTGTAAGACATTGATTAATTCATGTTTTCAACTGGCAAACAGAGAAAAAGGAGGGAAGAG"
similarity = sequence_similarity(target, subject)
print(f"Similarity between '{subject}' and '{target}': {similarity}")