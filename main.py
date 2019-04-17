from flask import Flask
from flask import request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from google.cloud import firestore
from stop_words import get_stop_words

app = Flask(__name__)

class RecommenerHybrid:
    quests_data = ""
    ratings_data = ""
    quest_indices=""

    def __init__(self, q, ra):
        self.ratings_data = ra
        self.quests_data = q


    def printQuest(self):
        print(self.quests_data)

    """Recomendador Basado en contenido (BC)
    Descripcion: Se encarga de analizar la descripcion de los cuestionarios y encontrar similitud de acuerdo a las palabras encontradas. 
    Paramentro: Titulo del cuestionario"""

    def getResultBasedContent(self, id_quest):
        # Obtenemos los datos de la base de datos.
        #self.quests_data = pd.read_csv('cuestionarios.csv')
        # Analizamos las palabras, excepto palabras de relleno del idioma ingles.
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words=get_stop_words('spanish'))
        # Transformamos en una matriz de [palabras (filas), cuestionarios (columnas)]
        tfidf_matrix = tf.fit_transform(self.quests_data['title']+" "+self.quests_data['description']+ " "+self.quests_data['subject']+ " "+self.quests_data['keywords'])
        # Encontramos la simulitud mediante coseno, aplicando el producto vectorial
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        # Reseteamos los indices para evitar incovenientes.
        self.quests_data = self.quests_data.reset_index()
        # obtenemos una matriz donde el indice es el titulo del cuestionario
        indices = pd.Series(self.quests_data.index, index=self.quests_data['id_quest']).drop_duplicates()
        # obtenemos el indice del cuestionario solicitado dentro de todos los datos.
        idx = indices[id_quest]
        # obtenemos la lista de  similitud del coseno, para el cuestionario solicitado.
        sim_scores = list(enumerate(cosine_similarities[idx]))
        # ordenamos descendentemente.
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # obtenemos las similutdes de mayor valor.
        sim_scores = sim_scores[0:5]
        # almacenamos los indices en una variable
        self.quest_indices = [i[0] for i in sim_scores]
        # mostramos los resultados de este recomendador
        recomendaciones_bc = self.quests_data.iloc[self.quest_indices][['title', 'id_quest','subject']]
        # retornamos los indices para su posterior uso.
        return np.array(recomendaciones_bc)

    """Recomendador Filtro colaborativo (FC)
        Descripcion: Se encarga analizar las calificaciones de cada usuario y encontrar la simulitud entre usuarios. 
        """

    def getResultFilerCollaborative(self):
        # instaciamos un lector
        reader = Reader()
        # Instanciamos la descomposicion de valor singular
        svd = SVD()
        # Obtenemos los datos de la base de datos
        #self.ratings_data = pd.read_csv('calificaciones.csv')
        # obtenemos un dataset conformado por valores como id_usuario, id_cuestionario, calificacion.
        data = Dataset.load_from_df(self.ratings_data[['id_user', 'id_quest', 'calificacion']], reader)
        # Obtenemos un conjunto de prueba
        #trainset = data.build_full_trainset()
        # Entrenemos el algoritmo con el conjunto de pruebas
        #svd.fit(trainset)
        # evaluamos la predicciones con  RMSE Y MAE
        cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        # retornamos
        return svd

    """Recomendador Hibrido (BC y FC)
        Descripcion: Se encarga analizar las calificaciones de cada usuario y encontrar la simulitud entre usuarios.
        Paramentros: quest_indices(indices de similitud obtenidos por BC), svd(resultado de entramiento de FC)  
        """

    def getResultsHybrid(self,svd,id_user):
        # obtemos una matriz con columnas como titulo, numero de votos, promedio de votos, id cuesitonario
        quest = self.quests_data.iloc[self.quest_indices][['title', 'vote_count', 'vote_average', 'id_quest','subject']]
        # obtenemos los id de cuestionarios
        id_quest = quest['id_quest']
        # renombramos la columna id_quest
        id_quest = id_quest.rename(columns={'id_quest': 'ida'})
        # concatenemos la matriz quest con id_quest
        quest = pd.concat([quest, id_quest], axis=1)
        # establecemos como indices de la matriz el id de los cuestionarios
        quest = quest.set_index(0)
        # obtenemos las predicciones y las unimos a la matriz
        quest['est'] = quest['id_quest'].apply(lambda x: svd.predict(id_user, quest.loc[x]['id_quest'], quest.loc[x]['vote_average']).est)
        # ordenamos los vales en forma descendente
        quest = quest.sort_values('est', ascending=False)
        # imprimimos las recomendaciones
        print (quest)
        return np.array(quest)

@app.route('/',methods=['POST'])
def hello_world():

    print(request.json)
    id_user = request.json['id_user']
    id_questionnaire = request.json['id_questionnaire']
    db=firestore.Client()
    user_ref=db.collection(u'users')
    questionnaires_ref = db.collection(u'questionnaires')
    docs = questionnaires_ref.where(u'post', u'==',True).get()

    # CREACION DEL DATAFRAME
    ids=[]
    descriptions=[]
    titles=[]
    nums_ratings=[]
    ratings=[]
    subjects=[]
    keywords=[]

    ids_user=[]
    ids_quest=[]
    califaciones=[]
    cont=0


    #Se debe traer el titulo, la descripcion, la categoria y las palabras claves.

    for doc in docs:
        ids.append(doc.id)
        descriptions.append(doc.to_dict()['description'])
        titles.append(doc.to_dict()['title'])
        nums_ratings.append(doc.to_dict()['numAssessment'])
        ratings.append(doc.to_dict()['assessment'])
        subjects.append(doc.to_dict()['subject'])
        keywords.append(doc.to_dict()['keywords'])
        docs_ratings= db.collection(u'questionnaires').document(doc.id).collection(u'ratings').get()
        for ra in docs_ratings:
            cont=cont+1
            ids_user.append(ra.id)
            ids_quest.append(doc.id)
            califaciones.append(ra.to_dict()['value'])

    
    #Preparamos la estructura para crear un dataframe
    questionnaires_data={
        'id_quest':ids,
        'title':titles,
        'description':descriptions,
        'subject':subjects,
        'keywords':keywords,
        'vote_count':nums_ratings,
        'vote_average':ratings
    }

    #Preparamos la estructura de los ratings
    ratings_data={
        'id_user':ids_user,
        'id_quest':ids_quest,
        'calificacion':califaciones
    }

    #Creamos el dataframe de cuestionarios
    data_frame_questionnaires=pd.DataFrame(questionnaires_data)

    #Creamos el dataframe de ratings
    data_frame_ratings=pd.DataFrame(ratings_data)

    #print (data_frame_questionnaires)
    #print(data_frame_ratings)
    r=RecommenerHybrid(data_frame_questionnaires,data_frame_ratings)
    results_based_content=r.getResultBasedContent(id_questionnaire)
    
    docs =  user_ref.document(id_user).collection(u'recommendations').get()

    for doc in docs:
        doc.reference.delete()

    if len(califaciones)>=len(titles):
        print ("Suficientes calificaciones para obtener recomendaciones colaborativas: ",cont)
        results_based_hybrid=r.getResultsHybrid(r.getResultFilerCollaborative(),id_user)
        for quest in results_based_hybrid:
            user_ref.document(id_user).collection(u'recommendations').document(quest[3]).set({u'title':quest[0],u'subject':quest[4]})

    else:
        print ("Imposible obtener recomendaciones colaborativas")
        for quest in results_based_content:
            user_ref.document(id_user).collection(u'recommendations').document(quest[1]).set({u'title':quest[0],u'subject':quest[2]})

    return 'Recomendaciones Generadas'
    

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
