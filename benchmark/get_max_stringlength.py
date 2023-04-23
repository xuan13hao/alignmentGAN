# Open the text file in read mode
with open('cigar_output.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()
    
    # Get the maximum length of all lines
    max_length = max(len(line) for line in lines)
    
print(f"The maximum string length in the file is: {max_length}")
