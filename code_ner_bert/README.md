# Setup:

```bash
conda env create -f ./conda_env_ner_bert.yml
```

# Usage:

```bash
source activate ner_bert
python main.py input_ltf_dir output_dir
rm -f /tmp/corenlp.shutdown
source deactivate
```
