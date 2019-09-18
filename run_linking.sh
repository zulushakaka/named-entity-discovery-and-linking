#!/bin/bash

ner_dir=$1
cd xianyang_linking/
source activate xy_linking
python linking.py --run --dir $ner_dir
source deactivate
cd ..
