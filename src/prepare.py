
import gzip
from tqdm import tqdm
import urllib.request
import os
import typing
import pandas as pd


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_extract_lang(lang_code:str, limit_lines:int = None, preprocess:typing.Callable[[str],str]=None, force=False)->str:
    """Downloads file and extracts it in a cache. Returns path to the final file

    Args:
        lang_code (str): one of languages provieded in languages.csv
        limit_lines (int, optional): Max limit of lines to extract. Can improve performacne if we don't need to extract and process everything Defaults to None.
        preprocess (typing.Callable[[str],str], optional): function to process line as string.  Defaults to None.
        force (bool, optional): If true, donwloads and extracts file even if already exists. Defaults to False.

    Returns:
        str: [description]
    """
    url = f"https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.{lang_code}.gz"

     # Download SEED database
    cache_dir= ".cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    download_target_file=os.path.join(cache_dir,f"OpenSubtitles.raw.{lang_code}.gz")
    if os.path.exists(download_target_file) and not force:
        print(f"{download_target_file} file exists... skipping download")
    else:
        download_url(url, download_target_file)

    sufix = f"_{limit_lines}" if limit_lines else ""
    extracted_target_file= os.path.join(cache_dir,f"OpenSubtitles.raw.{lang_code}{sufix}.txt")
    if  os.path.exists(extracted_target_file) and not force:
        print(f"Extracted {extracted_target_file} file exists... skipping extraction")
    else:
        handle = gzip.open(download_target_file)
        print(f"Extracting to {extracted_target_file}...")
        line_num=0
        with open(extracted_target_file, 'wb') as out:
            for line in handle:
                if preprocess:
                    line = preprocess(line.decode("utf-8") ).encode("utf-8")
                if len(line)<3: #skip extremely short records
                    continue
                out.write(line)
                if limit_lines and line_num>limit_lines:
                    break
                line_num+=1
    return extracted_target_file


def create_dataset_for_langs(lang_list:typing.List[str], train_split_ratio:int,test_split_ratio:int,  preprocess:typing.Callable[[str],str]=None,  limit_lines:int = None, output_dir= "open-subtitles-dataset"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_path = os.path.join(output_dir,"train")
    
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    test_path = os.path.join(output_dir,"test")
    if not os.path.exists(test_path):
        os.makedirs(test_path)


    for i,lang in enumerate(lang_list):
        print(f"Processing {lang} {i}/{len(lang_list)}")
        preprocessed_file = download_and_extract_lang(lang,limit_lines=limit_lines, preprocess=preprocess)

        train_file_path=os.path.join(train_path,f"{lang}.csv")
        test_file_path=os.path.join(test_path,f"{lang}.csv")
        if os.path.exists(train_file_path) and os.path.exists(test_file_path):
            print(f"Dataset for {lang} already exists... skipping")
            continue

        lines=[]
        with open(preprocessed_file, 'r') as file:
            df = pd.DataFrame({"sentences":file.readlines(), "lang":lang})
            df = df[df["sentences"].notnull()]
            df["sentences"] =df["sentences"].map(lambda x: x.strip())
            df = df.sample(frac=1).reset_index(drop=True)

            train_split_size=int(len(df) * train_split_ratio/100)
            test_split_size=int(len(df) * test_split_ratio/100)
            
            assert train_split_size+test_split_size<=len(df), "expected train and test size can't be greater than total dataset size"
            train_df = df[:train_split_size]
            test_df = df[train_split_size:train_split_size+test_split_size]
            
            train_df.to_csv(train_file_path,index=False)
            test_df.to_csv(test_file_path,index=False)



