map_file = open('upos_map', 'r')
source_file = open('CoNLL2009-ST-English-train.txt', 'r')
output_file = open('CoNLL2009-ST-English-train.map', 'w')

map_dict = {}
for line in map_file.readlines():
    pt_pos, u_pos = line.strip().split()
    map_dict[pt_pos] = u_pos

for line in source_file.readlines():
    parts = line.strip().split()
    if len(parts)> 0:
        if map_dict.has_key(parts[4]):
            parts[4] = map_dict[parts[4]]
        else:
            print(parts[4])
        new_line = '\t'.join(parts)
        output_file.write(new_line)
        output_file.write('\n')
    else:
        output_file.write('\n')