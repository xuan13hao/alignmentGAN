import pysam

# Input SAM file name
sam_file = "real_illumina_mRNAseq_paired.RazerS3.sam"

# Output BAM file name
bam_file = "real_illumina_mRNAseq_paired.RazerS3.bam"

# Open SAM file
sam = pysam.AlignmentFile(sam_file, "r")

# Open BAM file for writing
bam = pysam.AlignmentFile(bam_file, "wb", template=sam)

# Loop through SAM records and write them to the BAM file
for record in sam:
    bam.write(record)

# Close both files
sam.close()
bam.close()
