import re
import torch


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

def parseCigar(cigar_string,max_length,insert_length,delete_length):
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
    insertions += [0] * (insert_length - len(insertions))
    deletions += [0] * (delete_length - len(deletions))
    whole_cigar = matches+insertions+deletions
    return whole_cigar



def cigar_lists(filename,max_length,insert_length,delete_length):
    cigar_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Do something with the line
            # wc = parse_cigar(line,max_length)
            wc = parseCigar(line,max_length,insert_length,delete_length)
            cigar_list.append(wc)
    tensor = torch.tensor(cigar_list)
    return tensor

def cigar_to_symbols(cigar):
    symbols = ""
    cigar_tokens = re.findall(r'(\d+)([MIDNSHP=X])', cigar)
    for token in cigar_tokens:
        length, op = int(token[0]), token[1]
        if op == 'M':
            symbols += 'M' * length
        elif op == 'I':
            symbols += 'I' * length
        elif op == 'D':
            symbols += 'D' * length
    return symbols

def generate_edits():
    kmer_dict = {}
    kmer_dict["M"] = 1
    kmer_dict["D"] = 2
    kmer_dict["I"] = 3
    return kmer_dict


def cigar_symbols_lists(filename,max_length = 109):
    cigar_list = []
    cigar_tensors = []
    cigar_dic = generate_edits()
    whole_tensor = []
    with open(filename, 'r') as file:
        for line in file:
            wc = cigar_to_symbols(line)
            cigar_tensors.append(wc)
    for i in cigar_tensors:
        if len(i) < max_length:
            i += "M" * (max_length - len(i))
            cigar_list.append(i)
    for i in cigar_list:
        l = []
        for j in i:
            l.append(cigar_dic[j])
        whole_tensor.append(l)
    # print("length = ",len(whole_tensor))
    tensor = torch.tensor(whole_tensor).int()
    return tensor


# filename = "small_test.txt"
# l = cigar_symbols_lists(filename)
# print(l)