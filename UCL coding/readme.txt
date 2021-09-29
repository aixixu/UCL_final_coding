The data of this project is stored in HEP linux systems and is not saved locally because it is huge
Therefore, if you want to run this code, place the code in HEP linux systems.
Also refer to the practices in https://github.com/ucl-dis-spotify-group-project/podcast-dataset and install Spotify Podcast Dataset software, 
because it includes scripts to read metadata.
Secondly, install the required environment. 
Since the development environment has been configured in HEP linux systems, you only need to switch to the cdtdisspotify environment. 
If you need to start locally, you need to install the following packages:
  - python=3.8.6
  - pip=21.0.1
  - numpy=1.19.5
  - pandas=1.2.3
  - omegaconf=2.0.6 
  - sox=14.4.2
  - pytables=3.6.1
  - matplotlib=3.3.4
  - jupyterlab=3.0.12
  - sentencepiece=0.1.95
  - transformers=4.4.2
  - elasticsearch-dsl=7.3.0
  - pip:
    - tensorflow==2.4.1
    - tf-slim==1.1.0 
    - torch==1.8.1
    - opensmile==2.0.0
    - librosa==0.8.0
    - tqdm==4.56.0

Then download all metadata from HEP linux systems according to these addresses:
opensmile_uri="/unix/cdtdisspotify/index/opensmile"
yamnet_scores_uri="/unix/cdtdisspotify/index/yamnet/scores"
metadata_path="/unix/cdtdisspotify/data/spotify-podcasts-2020/metadata.tsv"
labeled_path ="/unix/cdtdisspotify/index/labeled.csv"

Then create a folder, put my code in the folder, and start running the code!!!!
