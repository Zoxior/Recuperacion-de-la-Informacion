# Buscador SciFact: TF-IDF y BM25

Servicio Flask para recuperar documentos del corpus SciFact con modelos lexicos clasicos e indice invertido. No utiliza embeddings.

## Estructura

```text
.
|-- app.py
|-- Procfile
|-- requirements.txt
|-- examen1bim.ipynb
|-- train.tsv
|-- test.tsv
`-- data/scifact/
    |-- corpus/corpus.parquet
    |-- queries/queries.parquet
    `-- qrels/test.tsv
```

## Ejecucion local

Requiere Python 3.10 o superior. Desde la raiz del repositorio:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

La primera ejecucion descarga las stopwords inglesas de NLTK y construye el indice en memoria. El servicio escucha en `http://127.0.0.1:5000`; el puerto se toma de `PORT` cuando esta definida. Para ejecutar el notebook, abre `examen1bim.ipynb` desde esta carpeta. Define `SEARCH_SERVICE_URL` con la URL publicada para habilitar la celda de demostracion HTTP.

## API

`GET /health` responde con el estado y la cantidad de documentos indexados:

```json
{"status":"ok","message":"Servicio de busqueda activo","documents_indexed":5183}
```

`POST /search` requiere `Content-Type: application/json`:

```json
{"query":"What causes cognitive impairment?","model":"bm25","top_k":10}
```

`model` acepta `bm25` o `tfidf`; `top_k` es un entero de 1 a 100 (por defecto 10). La respuesta contiene `model`, `query` y una lista `results` con `rank`, `document_id`, `score`, `title` y `snippet`.

### Postman

1. Para health: crea una solicitud **GET** a `https://TU-SERVICIO.onrender.com/health` y pulsa **Send**.
2. Para buscar: crea una solicitud **POST** a `https://TU-SERVICIO.onrender.com/search`.
3. En **Headers**, agrega `Content-Type` = `application/json`.
4. En **Body**, selecciona **raw** y el tipo **JSON**; pega el ejemplo anterior.
5. Prueba tambien `"model":"tfidf"` y distintos valores de `top_k`.

Casos de error esperados (HTTP 400):

- Consulta vacia: `{"query":"   ","model":"bm25","top_k":10}`
- Modelo desconocido: `{"query":"cancer","model":"dense","top_k":10}`
- `top_k` invalido: `{"query":"cancer","model":"bm25","top_k":0}` o `"top_k":"diez"`
- Cuerpo sin JSON o JSON malformado: mensaje de error claro indicando que se requiere un objeto JSON.

## Despliegue gratuito en Render

Render y otros proveedores pueden cambiar su oferta gratuita; si el plan web gratuito no aparece, elige un proveedor equivalente que permita desplegar desde GitHub. En Render:

1. Sube esta carpeta como raiz del repositorio GitHub (incluye el directorio `data/scifact`). No subas `.venv`, caches ni `node_modules`.
2. En Render, selecciona **New + > Web Service** y conecta el repositorio.
3. Usa **Runtime: Python**, **Root Directory** vacio si estos archivos estan en la raiz del repo, **Build Command** `pip install -r requirements.txt` y **Start Command** `gunicorn app:app --workers 1 --timeout 120`.
4. Selecciona el plan gratuito disponible y crea el servicio. Render define `PORT`; Gunicorn escucha en ese puerto. `Procfile` contiene el mismo comando como referencia.
5. Espera a que el deploy termine y copia la URL `https://...onrender.com` asignada. El primer arranque instala stopwords y construye el indice, asi que puede tardar; el servicio gratuito se suspende tras 15 minutos sin trafico y el reinicio puede tardar alrededor de un minuto.
6. Usa esa URL en las dos solicitudes Postman y en `SEARCH_SERVICE_URL` del notebook.

No se debe fijar `PORT` en Render. Para local, la aplicacion usa 5000 si la variable no existe.

## Evaluacion academica

El notebook carga corpus, consultas y qrels, construye el indice invertido y calcula Precision@10, Recall@10, MAP@10 y nDCG@10 para train/test cuando hay qrels disponibles. Documenta preprocesamiento igual al servicio y compara las metricas reales observadas.
