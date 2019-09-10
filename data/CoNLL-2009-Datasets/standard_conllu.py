file_in = open("fr-up-test.conllu", 'r')
file_out = open("fr-test", 'w')

start = True

for line in file_in.readlines():
    part = line.strip().split()
    if len(part)<=1:
        file_out.write('\n')
        start = True
        continue
    if start:
        start = False
        continue
    if '-' in part[0]:
        continue
    new_parts = [part[0], part[1],  '_', part[2], part[3],  part[4], '_',  '_', '_', part[6], '_' ]
    new_parts = new_parts + part[7:]
    line = '\t'.join(new_parts)
    file_out.write(line)
    file_out.write('\n')