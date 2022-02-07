from src.prepare import create_dataset_for_langs
import src
from typing import List
from src.train import main as main_train
import argparse
import pandas as pd
import os
from tqdm import  tqdm



def merge_files(path:str, output_file_name:str):
    with open(output_file_name, 'w') as out_file:
        out_file.writelines(["sentences,lang"])
        files_to_merge = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith(".csv")]
        print(f"Merging files from {path}")
        for i in tqdm(range(len(files_to_merge))):
            filename=files_to_merge[i]
            try:
                if not os.path.isfile(os.path.join(path,filename)):
                    continue
                if filename==os.path.basename(output_file_name) :
                    continue
                line_num=0
                print(f"{filename} >> {output_file_name}")
                with open(os.path.join(path,filename), 'r') as file:
                    while line := file.readline():
                        line_num=line_num+1
                        if line_num==1: 
                            continue
                        out_file.writelines([line])
            except Exception as e:
                print(f"Error when procesig file {filename}")
                raise e
    print("Reshuffling")
    df = pd.read_csv(output_file_name)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_file_name,index=False)


def prepare_ds(langs:List[str], dataset_max_size:int, split:tuple):
    
    create_dataset_for_langs(langs, split[0], split[1],lambda l: l.replace('"',''),limit_lines=dataset_max_size)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Prepare or Train')
    subparsers = parser.add_subparsers(help="prepare | train", dest='command')
    prepare_ds_argparser = subparsers.add_parser('prepare', help='prepares dataset for languages')
    prepare_ds_argparser.add_argument('--languages', help="; seperated list of languages", required=False, default=None)
    prepare_ds_argparser.add_argument('--lang_sentece_limit', help="lower limit to number of sentences to filter out low res languages", required=False, type=int, default=1000000)
    prepare_ds_argparser.add_argument('--dataset_max_size', help="limit the total size of dataset", type=int, required=False, default=1000000)
    prepare_ds_argparser.add_argument('--split', help="split... eq: 80/20", required=False, default="80/20")
    
    

    train_parser = subparsers.add_parser('train', help='')
    train_subparsers = train_parser.add_subparsers(dest="runType")
    train_subparsers.add_parser("local")
    train_subparsers.add_parser("azure")

    args = parser.parse_args()
    print(args.command)
    print(args)
    
    
    # args.command="prepare" # TODO: Remove this
    # args.languages ="si"
    # args.split ="80/20"
    # args.dataset_max_size=1000000
    #main_train("./open-subtitles-dataset", "./outputs/model")
    if args.command=="prepare":
        #args=prepare_ds_argparser.parse_args()
        if args.languages is None:
            assert args.lang_sentece_limit is not None, "one of: languages, lang_sentece_limit  parameters must be provided"
            df = pd.read_csv("languages.csv")
            
            langs = list(df[df["sentences"]>=args.lang_sentece_limit]["language"])
        else:
            langs = [lang.strip() for lang in args.languages.split(";")]
        split_tuple = tuple(int(s) for s in args.split.split("/"))
        assert split_tuple[0]+split_tuple[1]==100, f"sum of split parts must be 100 ... got {split_tuple[0]} / {split_tuple[1]}"
        prepare_ds(langs, args.dataset_max_size, split_tuple)
    elif args.command=="train":
        if args.runType=="local":
            main_train("./open-subtitles-dataset", "./outputs/model")
        elif args.runType=="azure":
            from run_on_azure import run_remote
            run_remote()



        


