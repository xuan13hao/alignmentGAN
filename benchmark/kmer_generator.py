
# Input parameters
input_file = "test.fasta"
k_min = 50
k_max = 55
output_file = "kmers.fasta"

# Parse reference sequences from input FASTA file
with open(input_file, "r") as f:
    ref_seqs = {}
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        ref_name = lines[i].strip()[1:]
        ref_seq = lines[i+1].strip()
        ref_seqs[ref_name] = ref_seq

# Generate kmers of different lengths for each reference sequence
kmers = {}
for ref_name, ref_seq in ref_seqs.items():
    kmers[ref_name] = []
    for k in range(k_min, k_max+1):
        kmers[ref_name] += [(ref_seq[i:i+k], i+1, i+k) for i in range(len(ref_seq)-k+1)]

# Write kmers to FASTA file
with open(output_file, "w") as f:
    for ref_name, kmer_list in kmers.items():
        for i, kmer in enumerate(kmer_list):
            kmer_seq, start, end = kmer
            f.write(">{}_pos{}:{}-{}\n".format(ref_name, i+1, start, end))
            f.write("{}\n".format(kmer_seq))