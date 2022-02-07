1.  Prepare virtual enviroment
```sh
 pipenv install
```


2. get from open-subtitiles-dataset (if you have it) and copy it into

```
./open-subtitles-dataset

    /train

    /text
```

or recreate it
```
 python main.py prepare
```
(downloads lot's of data from internet)


3. run training
   
```
python main.py train local
```