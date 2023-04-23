import pysam
from Bio import SeqIO

# Open the SAM file for reading
samfile = pysam.AlignmentFile("real_illumina_mRNAseq_paired.RazerS3.sam", "r")

# Create a new list to store the extracted reads
reads = []

# Iterate over the reads in the SAM file
for read in samfile:
    # Check if the read is mapped to chr1
    if read.reference_name == "chrY":
        # Add the read sequence and name to the list
        reads.append((read.query_sequence, read.query_name))

# Close the input SAM file
samfile.close()

# Write the extracted reads to a new FASTA file
with open("chrY_reads.fasta", "w") as outfile:
    for read_seq, read_name in reads:
        outfile.write(">{}\n".format(read_name))
        outfile.write(read_seq + "\n")
