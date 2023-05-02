# -*- coding: utf-8 -*-
"""
Kmer 
Created on Thu March 10 11:14:08 2023
"""

import re
import torch

# Example CIGAR string
cigar_string = "10M2I4M1D3M"

# Initialize empty lists for insertions, deletions, and matches
insertions = []
deletions = []
matches = []

# Parse the CIGAR string to extract the operation codes and lengths
cigar_ops = re.findall(r"\d+|[A-Z]", cigar_string)

# Loop over the operations to compute the positions of insertions, deletions, and matches
position = 0
for i in range(0, len(cigar_ops), 2):
    length = int(cigar_ops[i])
    op = cigar_ops[i + 1]

    if op == "M":
        for j in range(position, position + length):
            matches.append(j)
        position += length
    elif op == "I":
        for j in range(position, position + length):
            insertions.append(j)
        position += length
    elif op == "D":
        for j in range(position, position + length):
            deletions.append(j)
        position += length

# Pad the shorter lists with zeros
max_length = max(len(matches), len(insertions), len(deletions))
matches += [0] * (max_length - len(matches))
insertions += [0] * (max_length - len(insertions))
deletions += [0] * (max_length - len(deletions))
whole_cigar = [matches,insertions,deletions]
# Combine the three lists into a single tensor
tensor = torch.tensor(whole_cigar)

# Print the resulting tensor
# print("Tensor:", tensor)

def parse_cigar(cigar_string,max_length):
    # Initialize empty lists for insertions, deletions, and matches
    insertions = []
    deletions = []
    matches = []

    # Parse the CIGAR string to extract the operation codes and lengths
    cigar_ops = re.findall(r"\d+|[A-Z]", cigar_string)

    # Loop over the operations to compute the positions of insertions, deletions, and matches
    position = 1
    for i in range(0, len(cigar_ops), 2):
        length = int(cigar_ops[i])
        op = cigar_ops[i + 1]

        if op == "M":
            for j in range(position, position + length):
                matches.append(j)
            position += length
        elif op == "I":
            for j in range(position, position + length):
                insertions.append(j)
            position += length
        elif op == "D":
            for j in range(position, position + length):
                deletions.append(j)
            position += length

    # Pad the shorter lists with zeros
    matches += [0] * (max_length - len(matches))
    insertions += [0] * (max_length - len(insertions))
    deletions += [0] * (max_length - len(deletions))
    whole_cigar = [matches,insertions,deletions]
    return whole_cigar


def cigar_lists(filename,max_length):
    cigar_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Do something with the line
            wc = parse_cigar(line,max_length)
            cigar_list.append(wc)
    tensor = torch.tensor(cigar_list)
    return tensor
filename = "test.txt"
l = cigar_lists(filename,109)
print(l)




#%%
if __name__ == '__main__':

    edits_list = read_file('edits_test.txt',outputFilename='edits.pkl')
    edits_list = read_file('edits_Match.txt',outputFilename='edits_ref.pkl')
    print(edits_list)


