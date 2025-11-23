

# Ejecución de Tests Funcionales del Modelo churn clientes TELCO

### Paso 1: Fork del Repositorio Original

En el navegador, inicie sesión en Github. Luego, vaya al enlace del proyecto original (https://github.com/sergiobueno137/Modelo-Churn-clientes-TELCO) y dé click al botón "Fork". Esto copiará todo el proyecto en su usuario de Github.


### Paso 2: Configurar git

Abra una Terminal en JupyterLab e ingrese los siguientes comandos

```
git config --global user.name "<USER>"
```

```
git config --global user.email <CORREO>
```

```
git config --list
```

### Paso 3: Clonar el Proyecto desde su propio Github

```
git clone https://github.com/<USER>/Modelo-Churn-clientes-TELCO_2.git
```


### Paso 4: Instalar los pre-requisitos

```
cd Modelo-Churn-clientes-TELCO_2/

```


### Paso 5: Ejecutar las pruebas en el entorno

```
cd src

python make_dataset_v2.py

python train.py

python evaluate.py

python predict.py

cd ..
```


### Paso 6: Guardar los cambios en el Repo

```
git add .

git commit -m "Pruebas Finalizadas"

git push

```

Ingrese su usuario y Personal Access Token de Github. Puede revisar que los cambios se hayan guardado en el repositorio. Luego, puede finalizar JupyterLab ("File" => "Shut Down").
