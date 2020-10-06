def generate_text8_for_sp(input_file, output_file):
    with open(input_file) as fr:
        raw_str = fr.readline().split()
    n = 170000
    res = []
    for i in range(1, n):
        res.append(" ".join(raw_str[(i-1)*(len(raw_str)//n):i*(len(raw_str)//n)]))
    with open(output_file, 'w') as fw:
        for line in res:
            fw.write(line+'\n')
    print('Generate successfully')
generate_text8_for_sp('../data/text8', '../data/text8_for_sp')

