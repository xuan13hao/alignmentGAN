
import fasta
# Input parameters

input_file = "chrY.fa"
kmer_length = 101
output_file = "chrY_kmers.fa"

# Write kmers to FASTA file
with open(output_file, "w") as f:
    for record in fasta.parse(input_file):
        ref_name = record.id
        ref_seq = str(record.seq)
        for i in range(len(ref_seq) - kmer_length + 1):
            kmer_seq = ref_seq[i:i+kmer_length]
            if "N" not in kmer_seq:
                start = i+1
                end = i+kmer_length
                f.write(">{}_pos{}:{}-{}\n".format(ref_name, i+1, start, end))
                f.write("{}\n".format(kmer_seq))
