import fastapi
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import nltk
import uvicorn


app=fastapi.FastAPI()
movies=pd.read_csv("movies_datasert2.csv")

@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes):
    meses_ingles = {
        'enero': 'January',
        'febrero': 'February',
        'marzo': 'March',
        'abril': 'April',
        'mayo': 'May',
        'junio': 'June',
        'julio': 'July',
        'agosto': 'August',
        'septiembre': 'September',
        'octubre': 'October',
        'noviembre': 'November',
        'diciembre': 'December'
    }
    
    mes_ingles = meses_ingles.get(mes.lower())
    if mes_ingles is None:
        raise ValueError('Mes inválido')
    cantidad= len(movies[movies['Mes'] == mes_ingles])
    return {'mes':mes, 'cantidad':cantidad}



@app.get('/peliculas_dias/{dia}')
def peliculas_dia(dia):
    dias_ingles = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miércoles': 'Wednesday',
    'miercoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sábado': 'Saturday',
    'sabado': 'Saturday',
    'domingo': 'Sunday'
    }
    dia_ingles = dias_ingles.get(dia.lower())
    if dia_ingles is None:
        raise ValueError('Día inválido')
    cantidad= len(movies[movies['Dia'] == dia_ingles])
    return {'dia':dia, 'cantidad':cantidad}



@app.get('/franquicia/{franquicia}')
def franquicia(franquicia):
    cantidad=sum(franquicia in franquicias for franquicias in movies['belongs_to_collection']if franquicias is not None)
    ganancia_total = movies[movies['belongs_to_collection']==franquicia]['revenue'].sum()
    ganancia_promedio = ganancia_total/cantidad
    return {'franquicia':franquicia, 'cantidad':cantidad, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}



@app.get('/peliculas_pais/{pais}')
#if paises is not None para saltar los None
def peliculas_pais(pais):
    cantidad_peliculas = sum(pais in paises for paises in movies['production_countries']if paises is not None)
    return {'pais': pais, 'cantidad': cantidad_peliculas}



@app.get('/productoras/{productora}')
def productoras(productora: str):
    
    df_productora = movies[movies['production_companies'].apply(lambda x: productora in str(x))]
    cantidad = len(df_productora)
    ganancia_total = df_productora['revenue'].sum()
    return {'productora': productora, 'ganancia_total': ganancia_total, 'cantidad': cantidad}



@app.get('/retorno/{pelicula}')
def retorno(pelicula: str):
    '''Ingresas la película, retornando la inversión, la ganancia, el retorno y el año en el que se lanzó'''
    df_filtrado = movies[movies['title'].apply(lambda x: pelicula.strip() == str(x.strip()))]
    inversion = df_filtrado["budget"].sum()
    ganancia = df_filtrado["revenue"].sum()
    retorno = df_filtrado["return"].sum()
    anio = pd.to_datetime(df_filtrado["Year"]).dt.year

    return {'pelicula': pelicula, 'inversion': str(inversion), 'ganancia': str(ganancia), 'retorno': str(retorno), 'año':str(anio.item())}


@app.get('/titulo/{titulo}')
def recomendacion(titulo):
    pelicula_entrada = movies[movies['title'] == titulo]
    # Crear una matriz de características utilizando TF-IDF Vectorizer para 'genres'
    # Tokenización de los géneros utilizando nltk
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    movies['genres_tokens'] = movies['genres'].apply(lambda x: tokenizer.tokenize(x))
    # Unión de los tokens en una cadena de texto separada por espacios
    movies['genres_processed'] = movies['genres_tokens'].apply(lambda x: ' '.join(x))
    # Creación del vectorizador TF-IDF
    tfidf_genres = TfidfVectorizer(stop_words='english')
    # Creación de la matriz de características de los géneros
    matriz_caracteristicas_genres = tfidf_genres.fit_transform(movies['genres_processed'])
    # Encontrar el índice de la película de entrada
    indice_genres = pelicula_entrada.index[0]
    similitud_genres = cosine_similarity(matriz_caracteristicas_genres[indice_genres], matriz_caracteristicas_genres)
    indices_similares = similitud_genres.argsort()[0][-6:-1]
    titulos_similares = movies.iloc[indices_similares]['title'].tolist()
    return {'lista recomendada':titulos_similares}
