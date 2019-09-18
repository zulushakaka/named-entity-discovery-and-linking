# use absolute path for input and output directories
# $1: input_dir $2: output_dir
echo 'Input to EDL is '$1
echo 'Output will be at '$2

source ENV_XY/bin/activate

# cd python_code/
cd code2/python/
python2 main.py $1 $2

deactivate
