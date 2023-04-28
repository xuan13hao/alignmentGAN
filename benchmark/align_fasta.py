from Bio import SeqIO
from difflib import SequenceMatcher
from Bio import SeqIO, pairwise2
# Load the sequences from two FASTA files
sequences1 = list(SeqIO.parse("gen.fa", "fasta"))
sequences2 = list(SeqIO.parse("real_1000.fa", "fasta"))

# Define a function to format the alignment with '|' for matches
def format_alignment(align1, align2, score, begin, end):
    matches = ""
    for i in range(begin, end):
        if align1[i] == align2[i]:
            matches += "|"
        else:
            matches += " "
    return align1[begin:end] + "\n" + matches + "\n" + align2[begin:end] + "\n"

# Iterate over the sequences from both files and find similar sequences
for seq1 in sequences1:
    for seq2 in sequences2:
        similarity = SequenceMatcher(None, str(seq1.seq), str(seq2.seq)).ratio()
        if similarity > 0.7:
            # Display the top alignment with '|' for matches
            print(seq1.id + " - " + seq2.id + " (" + str(similarity) + "):\n")
            print(format_alignment(str(seq1.seq), str(seq2.seq), None, 0, len(seq1.seq)))

'''

blastn -query gen.fa -subject mt.fa -out output.txt -outfmt 7
            ACTTGGTTTGTGTTCTTCTTCATATTCTAAAACCATTCCATTTCCAAGCACTTTCAGTCCAATAGTTGTAGGAAATAGCGCTGTTTTTGTTGTGTGCGCAG
         |     |  ||  | ||||    |   | |     | |   | ||       |   |      ||   | |  |    |   |        |
CAACCCACAGCTACTTGGTTTGTGTTCTTCTTCATATTCTAAAACCATTCCATTTCCAAGCACTTTCAGTCCAATAGGTGTAGGAAATAGCGCTGTTTTTG


ACTTGGTTTGTGTTCTTCTTCATATTCTAAAACCATTCCATTTCCAAGCACTTTCAGTCCAATAGTTGTAGGAAATAGCGCTGTTTTTGTTGTGTGCGCAG
      | |||  |  |    ||  |  ||          |       |  |         |    |     | |  ||  | ||| |  |||||||   |
  TTGGTTTGTGTTCTTCTTCATATTCTAAAACCATTCCATTTCCAAGCACTTTCAGTCCAATAGGTGTAGGAAATAGCGCTGTTTTTGTTGTGTGTGCAGGG

  CACAGCTAAGTAGCTCTATTATAATACTTATCCAGTGACTAAAACCAACTTAAACCAGTAAGTGGAGAAATAACATGTTCAAGAACTGTAATGCTGGTTTT
  ||           || |  ||  |    |     |     ||         |         |  || | |  |  |     |    |           |
ACCACAGCTAAGTAGCTCTATTATAATACTTATCCAGTGACTAAAACCAACTTAAACCAGTAAGTGGAGAAATAACATGTTCAAGAACTGTAATGCCGGGT
  
GTTGTTAAAAAAAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGCGGTATCAAAAGATCAAAAACAGTTTTGGA
      |             ||      |    ||||  | |||   |     |  |      |  | ||     |  |  || |    ||          
           AAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGAGGAATCAAAAGATCAAAAGCAGTTTGGGAAGGCCAGAACC


GTTGTTAAAAAAAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGCGGTATCAAAAGATCAAAAACAGTTTTGGA
      ||| | |       |  |  |     ||    |   |         || |      ||     |     |  |  | |   |     |    |  
     TAAAAAAAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGAGGAATCAAAAGATCAAAAGCAGTTTGGGAAGGCC


GTTGTTAAAAAAAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGCGGTATCAAAAGATCAAAAACAGTTTTGGA
      |             ||      |    ||||  | |||   |     |  |      |  | ||     |  |  || |    ||          
           AAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGAGGAATCAAAAGATCAAAAGCAGTTTGGGAAGGCCAGAACC

GTTGTTAAAAAAAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGCGGTATCAAAAGATCAAAAACAGTTTTGGA
      ||| | |       |  |  |     ||    |   |         || |      ||     |     |  |  | |   |     |    |  
     TAAAAAAAATACAGGCTCCCCCACAACTGGGGTGCCTGGGGGGAACTTGGTCTGCTTCAGCCCAAGAGGAATCAAAAGATCAAAAGCAGTTTGGGAAGGCC

ACCCCCAAGCTGGAAACGGCTCTCATCGAGAGTGGGGCAGCCCAGACCTCACCAGAAGCTACCGACACTACAGTGAGGTTCTCCCGAACAGGTCGGTGGTG
  |  |     |||      || |  |    | || |   ||  |           ||    |  ||  |      |   || | |    | ||      |
GACGACTCCTAGGATGATATTCACTGCAGTGGAGGCGGCACCTCGTGGGGCAGCCCAGACCTCACCAGAAGCTACCGACACTACAGTGAGGTTCTCCCAGG


CCAGGTGCTCCACTGACAGTCCCTCCTCTCCGGAGCATTTTGATACCAGAAGGGAAAGCTTCATTCTCCTTGTTGTTGGTTGTTTTTTCCTTTGCTCTTTC
|   |        ||| |                   | |   |       |       ||| ||  |  | |      |   | | |   ||   |    
                           CTCCGGAGCATTTTGATACCAGAAGGGAAAGCTTCATTCTCCTTGTTGTTGGTTGTTTTTTCCTTTGCTCTTTCCCCCTTCCATCTCTGACTTAAGCAAAA

TTTTGGGTGGGAATGCAAAAATTCTCTGCTAAGACTTTTTCAGGTGAACATAACAGACTTGGCCAAGCTAGCATCTTAGCGGAAGCTGATCTCCAATGCTC
  |          |     ||  || |   | |      | |            ||       |      | | || | |   |       ||    |   
AATGCAAAAATTCTCTGCTAAGACTTTTTCAGGTGAACATAACAGACTTGGCCAAGCTAGCATCTTAGCGGAAGCTGATCTCCAATGCTCTTCAGTAGGGT

'''