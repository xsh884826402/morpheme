import argparse
import os
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-i','--input_vector_path')
    args = parse.parse_args()
    if args.input_vector_path:
        output_dir = os.path.join(os.path.dirname(args.input_vector_path), 'visual')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_word = 'word_'+os.path.basename(args.input_vector_path)
        output_vector = 'vector' + os.path.basename(args.input_vector_path)
        with open(args.input_vector_path, 'r') as f, open(output_dir+output_word,'w') as fword, \
                open(output_dir+output_vector, 'w') as fvector :
            for line in f:
                line = line.strip().split()
                word = line[0]
                vectors = line[1:]
                fword.write(word+'\n')
                fvector.write("\t".join(vectors)+'\n')
        print('generating Success,Store new word : {} new vector : {}'.
              format(output_dir+output_word, output_dir+output_vector))