import pysam

bam_file = "real_illumina_mRNAseq_paired.RazerS3.bam"
out_file = "mutations.txt"

# Open the BAM file using pysam.AlignmentFile()
bam = pysam.AlignmentFile(bam_file, "rb")

# Open an output file for writing mutations
with open(out_file, "w") as out:

    # Loop through the alignment records
    for record in bam.fetch():
        # Skip unmapped reads
        if record.is_unmapped:
            continue
        # Get the reference sequence and aligned read sequence
        ref_seq = bam.get_reference_sequence(record.reference_id, record.reference_start, record.reference_end)
        read_seq = record.query_sequence
        # Find SNPs, insertions, and deletions in the aligned read sequence
        for op, pos, alt in record.get_aligned_pairs(with_seq=True):
            if op == 1 and alt != ref_seq[pos]:
                # SNP
                out.write(f"{record.reference_name}\t{pos+1}\t{ref_seq[pos]}\t{alt}\tSNP\n")
            elif op == 0 and alt is None:
                # Deletion
                out.write(f"{record.reference_name}\t{pos+1}\t{ref_seq[pos]}\t-\tDEL\n")
            elif op == 0 and read_seq[pos-1] != ref_seq[pos-1] and read_seq[pos] == ref_seq[pos]:
                # Insertion
                out.write(f"{record.reference_name}\t{pos}\t{ref_seq[pos-1]}\t{read_seq[pos-1:pos+1]}\tINS\n")

# Close the BAM file
bam.close()
