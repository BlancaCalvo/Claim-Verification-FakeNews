# Claim-Verification-FakeNews

He hecho el repositorio que hemos dicho. Contiene el “experimento-1” (i.e. mi intento de transformar paso a paso el baseline a GEAR) y el “experimento-2” (que básicamente es la modificación del corpus MultiFC para poder entrenar el modelo con GEAR). He subido solo lo que creo que se necesita. Os los cuento un poco: 

Experiment 1: es principalmente una copia del respositorio del baseline https://github.com/sheffieldnlp/fever-naacl-2018 , el MLP. 
Mis cambios están en scripts/rte/mlp/train_mlp, donde he añadido el tag —features para poder cambiar de TFIDF a BERT. TFIDF era la representación original y BERT la que yo he añadido. Los scripts de mi añadido están en my_scripts/BERT* y new_features/extractor, que es un script de GEAR modificado. Así que para entrenar con los features de BERT el comando es:

`
PYTHONPATH=experiment-1/MLP_new/ python experiment-1/MLP_new/scripts/rte/mlp/train_mlp.py data/fever/fever.db experiment-1/MLP_new/sampled_data/train.ns.pages.p1.jsonl experiment-1/MLP_new/sampled_data/dev.ns.pages.p1.jsonl --model BERT_concat_model --sentence true --features BERT
`

Pero os faltará tanto la base de datos de FEVER como el sampled_data. 

Experiment 2: es el repositorio de GEAR https://github.com/thunlp/GEAR . Lo que he hecho hasta ahora está en data/MultiFC/tranform/multifc_to_gear, que es básicamente agrupar los 165 labels de MultiFC en las tres categorías de FEVER y guardar los datasets en el mismo formato para poder usar GEAR con MultiFC. En principio ahora estoy sacando los features con GEAR, y si todo va bien, probaré de entrenar. Os voy contando. 
