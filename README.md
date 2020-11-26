# Claim-Verification-FakeNews

Experiment 1: es principalmente una copia del respositorio del baseline https://github.com/sheffieldnlp/fever-naacl-2018 , el MLP. 
Mis cambios están en scripts/rte/mlp/train_mlp, donde he añadido el tag —features para poder cambiar de TFIDF a BERT. TFIDF era la representación original y BERT la que yo he añadido. Los scripts de mi añadido están en my_scripts/BERT* y new_features/extractor, que es un script de GEAR modificado. Así que para entrenar con los features de BERT el comando es:

`
PYTHONPATH=experiment-1/MLP_new/ python experiment-1/MLP_new/scripts/rte/mlp/train_mlp.py data/fever/fever.db experiment-1/MLP_new/sampled_data/train.ns.pages.p1.jsonl experiment-1/MLP_new/sampled_data/dev.ns.pages.p1.jsonl --model BERT_concat_model --sentence true --features BERT
`

Pero os faltará tanto la base de datos de FEVER como el sampled_data. 

Experiment 2: es el repositorio de GEAR https://github.com/thunlp/GEAR . La transformación del dataset está en data/MultiFC/tranform/, que es básicamente agrupar los 165 labels de MultiFC en las tres categorías de FEVER y guardar los datasets en el mismo formato para poder usar GEAR con MultiFC. He cambiado también los scripts de evaluación del modelo, y se encuentra en GEAR-MultiFC/my_scripts. 

Transformar el dataset:

```
python data/MultiFC/transform/change_format.py data/MultiFC/dev.tsv --output data/MultiFC/changed_dev.tsv
python data/MultiFC/transform/multifc_to_gear.py data/MultiFC/changed_dev.tsv --output dev_data.tsv
python data/MultiFC/transform/change_format.py data/MultiFC/train.tsv --output data/MultiFC/changed_train.tsv
python data/MultiFC/transform/multifc_to_gear.py data/MultiFC/changed_train.tsv --output train_data.tsv
```

Extraer los features:

```
chmod +x experiment-2/GEAR-MultiFC/feature_extractor/*.sh
experiment-2/GEAR-MultiFC/feature_extractor/dev_extractor.sh
experiment-2/GEAR-MultiFC/feature_extractor/train_extractor.sh
```

Train:

```
CUDA_VISIBLE_DEVICES=0 python experiment-2/GEAR-MultiFC/gear/train.py 
```

Test:

```
python experiment-2/GEAR-MultiFC/gear/test.py 
python experiment-2/GEAR-MultiFC/gear/evaluation.py 
```

Os faltaran los datos de MultiFC, que se pueden descargar aquí. https://competitions.codalab.org/competitions/21163 
