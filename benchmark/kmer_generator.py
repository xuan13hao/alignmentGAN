
import fasta
# Input parameters
input_file = "human_transcriptome.fasta"
kmer = [100]
output_file = "kmers.fasta"

# Parse reference sequences from input FASTA file
ref_seqs = {}
for record in fasta.parse(input_file):
    ref_seqs[str(record.id)] = str(record.seq)

# Generate kmers of different lengths for each reference sequence
kmers = {}
for ref_name, ref_seq in ref_seqs.items():
    kmers[ref_name] = []
    for k in kmer:
        kmers[ref_name] += [(ref_seq[i:i+k], i+1, i+k) for i in range(len(ref_seq)-k+1)]

# Write kmers to FASTA file
with open(output_file, "w") as f:
    for ref_name, kmer_list in kmers.items():
        for i, kmer in enumerate(kmer_list):
            kmer_seq, start, end = kmer
            f.write(">{}_pos{}:{}-{}\n".format(ref_name, i+1, start, end))
            f.write("{}\n".format(kmer_seq))
