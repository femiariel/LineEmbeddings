# Line Embeddings

## Overview

Le but de ce projet est la mise en place d'un serveur d'embeddings qui prend en entrée un fichier excel et renvoie une représentation vectorielle de chaque ligne . Ce serveur est déployé sur une VM GCP . Ainsi nous pouvons requeter l'API et obtenir les résultats au format json et les utiliser dans le cadre de clusterinf basé sur les embeddings par exemple.
Le projet est ouvert à n'importe qui et peut etre utilisé par n'importe qui.
Pour pouvoir l'utiliser, cloner d'abord le projet. Ensuuite executer la commande :
```
docker build -t LineEmbeddings .
```
ensuite executer la commande :
```
docker run -d --name ariel  -p 8000:8000 embedding-server
```
Ensuite accéder à l'interface web de l'api 
```
http://localhost:8000/docs
```
et tester le endpoint POST /embedding


