import random
from Bio import SeqIO
from Bio.Seq import Seq
import sys

fasta = sys.argv[1]
read_length = sys.argv[2]
out_file = sys.argv[3]
# Set the input fasta file name
fasta_file = fasta

# Set the output fasta file name for the generated reads
output_file = out_file

# Set the read length
read_len = read_length

# Read in the sequences from the input fasta file
seqs = list(SeqIO.parse(fasta_file, "fasta"))

# Shuffle the sequences randomly
random.shuffle(seqs)

# Open the output fasta file for writing
with open(output_file, "w") as f_out:
    for seq in seqs:
        # Calculate the number of reads that can be generated from the sequence
        num_reads = len(seq) // read_len
        # Generate the reads
        for i in range(num_reads):
            # Generate the read sequence and read name
            if i % 2 == 0:
                # Forward read
                read_seq = seq.seq[i*read_len:(i+1)*read_len]
                read_name = f"{seq.id}:{i*read_len+1}-{(i+1)*read_len}_F"
            else:
                # Reverse read
                read_seq = seq.seq.reverse_complement()[i*read_len:(i+1)*read_len]
                read_name = f"{seq.id}:{i*read_len+1}-{(i+1)*read_len}_R"
            # Write the read sequence and read name to the output fasta file
            read = SeqRecord(Seq(read_seq), id=read_name, description="")
            SeqIO.write(read, f_out, "fasta")

