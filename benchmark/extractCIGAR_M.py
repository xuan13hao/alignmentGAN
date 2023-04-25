import pysam

# # Open SAM file
samfile = pysam.AlignmentFile("real_illumina_mRNAseq_paired.RazerS3.sam", "r")
outputfile = open("cigar_match_output.txt", "w")
# Iterate over each read in the SAM file
for read in samfile:
    # Get the CIGAR string for the read
    cigar = read
    # cigar = read.cigarstring
    # if "M" in cigar and "I" not in cigar and "D" not in cigar:
    #     ops = []
    #     current_op = ""
    #     for char in cigar:
    #         if char.isdigit():
    #             current_op += char
    #         else:
    #             if current_op != "":
    #                 ops.append((int(current_op), char))
    #                 current_op = ""

    #     if current_op != "":
    #         ops.append((int(current_op), char))
    #     expanded_ops = []
    #     for op in ops:
    #         num, char = op
    #         expanded_ops.extend([char] * num)

        # cigar_str = ''.join(expanded_ops)

        # print(expanded_ops)
        # Write the CIGAR string to the output file
    outputfile.write(str(cigar) + "\n")

# Close output file and SAM file
outputfile.close()
samfile.close()

