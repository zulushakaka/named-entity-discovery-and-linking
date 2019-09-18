echo 'Input to EDL is '$1
echo 'Output will be at '$2
mkdir -p $2
source activate ner_bert
cd code_ner_bert/
CUDA_VISIBLE_DEVICES=2 python main.py $1 $2
conda deactivate

: <<'END'
echo 'Input to EDL is '$1
echo 'Output will be at '$2
mkdir -p $2
#source ENV_XY/bin/activate
source activate ner_bert
# cd python_code/
#cd code_ner_bert/python/
cd code_ner_bert/
#python2 main.py $1 $2
#CUDA_VISIBLE_DEVICES=2 python main.py $1 $2

i=0;
max=1
for f in "$1"/* 
do 
  d=dir_$(printf %01d $((i%$max+1))); 
  d="$2/"$d
  mkdir -p $d;
  cp "$f" $d; 
  let i++; 
done
for i in `seq 1 $max`
do
  if [ -d "$2/dir_$i" ]
  then
  mod=$(($i % 3))
  CUDA_VISIBLE_DEVICES=2 python main.py "$2/dir_$i" $2 &
  fi
done
wait
for i in `seq 1 $max`
do
rm -rf "$2/dir_$i"
done
END
conda deactivate
