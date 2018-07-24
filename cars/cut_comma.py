import re

in_file = open('category.csv', 'r')
out_file = open('processed_category.csv', 'w')
pattern = re.compile("^\s+|\s*,\s*|\s+$")

out_file.write("car name \n")

for line in in_file:
    array = [x for x in pattern.split(line) if x]
    for item in array:
        out_file.write(item + " ")
    out_file.write('\n')

out_file.close()
in_file.close()