import pysam
import sys

sam_file = sys.argv[1]
out_file = sys.argv[2]
# Open SAM file for reading
samfile = pysam.AlignmentFile(sam_file, "r")
# Open fasta file for writing
fastafile = open(out_file, "w")
# Iterate through SAM file
for read in samfile.fetch():
    # Write read to fasta file
    if read.reference_name == "chr1": 
        fastafile.write(">" + read.qname + "\n")
        fastafile.write(read.seq + "\n")

# Close files
samfile.close()
fastafile.close()