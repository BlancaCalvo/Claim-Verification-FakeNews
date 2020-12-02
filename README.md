# Claim-Verification-FakeNews

Experiment 1: es principalmente una copia del respositorio del baseline https://github.com/sheffieldnlp/fever-naacl-2018 , el MLP. Tiene algunos cambios para que las evidences procedan de UKP-Athenes y para que los features se generen con BERT, pero nunca lo he llegado a terminar.

Experiment 2: es el repositorio de GEAR https://github.com/thunlp/GEAR . La transformación del dataset está en data/MultiFC/tranform/, que es básicamente agrupar los 165 labels de MultiFC en las tres categorías de FEVER y guardar los datasets en el mismo formato para poder usar GEAR con MultiFC. He cambiado también los scripts de evaluación del modelo, y se encuentra en GEAR-MultiFC/my_scripts. También he modificado las evidencias para que salgan de las "reasons to label" que dieron los factcheckers. Eso se encuentra en GEAR-MultiFC/evidence_stuff/.

Ver README en cada una de estas carpetas.
