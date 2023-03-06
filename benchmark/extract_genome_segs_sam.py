import pysam
import sys
from Bio import SeqIO

sam_file = sys.argv[1]
geno_file = sys.argv[2]
out_file = sys.argv[3]


# Open the SAM file for reading
sam_file = pysam.AlignmentFile(sam_file, 'r')
# Open the genome FASTA file for reading
genome_file = SeqIO.to_dict(SeqIO.parse(geno_file, 'fasta'))
# Open the output FASTA file for writing
output_file = open(out_file, 'w')

# Loop through the aligned reads in the SAM file
for read in sam_file.fetch():

    # Check if the read is mapped to the genome
    if read.is_unmapped:
        continue

    # Get the name and positions of the aligned segment
    chrom = read.reference_name
    start = read.reference_start
    end = read.reference_end

    # Extract the aligned segment from the genome FASTA file: eg.. chr1
    # if chrom == "chr1":
    if end - start <= 101:
        segment = genome_file[chrom][start:end].upper()

    # Write the segment to the output FASTA file
        output_file.write(">{}:{}-{}\n{}\n".format(chrom, start+1, end, segment.seq))

# Close the files
sam_file.close()
output_file.close()