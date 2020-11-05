from py2neo import Graph
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import multiprocessing
import gensim.models.word2vec as w2v
import sklearn.manifold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

graph = None
descriptions_original = []
descriptions = []
corpus = []
stops = set(stopwords.words("english"))
word2vec_model = None

def data_prep():
    global graph
    port = input("Enter Neo4j DB Port: ")
    user = input("Enter Neo4j DB Username: ")
    pswd = input("Enter Neo4j DB Password: ")
    graph = Graph('bolt://localhost:'+port, auth=(user, pswd))
    results = graph.run("""MATCH (n:Person) RETURN n.description AS description""").data()

    global descriptions_original
    global descriptions
    global corpus
    global stops
    for r in results:
        descriptions_original.append(r['description']) #save original description for reference later
        token = word_tokenize(r['description'].lower()) #tokenize
        words = []
        for w in token:
            if not w in stops: #don't include stop words
                w = (re.sub('[^A-Za-z0-9]+', '', w).lower()).strip() #remove punc & special chars
                if w:
                    words.append(w)
        descriptions.append(" ".join(words))
        corpus.append(words)

def build_word2vec_model():
    global word2vec_model
    num_features = 300
    min_word_count = 20
    num_workers = multiprocessing.cpu_count()
    context_size = 10
    downsampling = 1e-4
    seed = 2

    word2vec_model = w2v.Word2Vec(
        sg=1, #skip-gram
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    print("BUILDING WORD2VEC_MODEL VOCAB")
    word2vec_model.build_vocab(corpus)

    print("TRAINING WORD2VEC_MODEL")
    word2vec_model.train(corpus, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)

def create_word_points_matrix():
    global word2vec_model
    #reduce vector matrix of all the words to 2 dimensions
    tsne = sklearn.manifold.TSNE(n_components = 2, early_exaggeration = 6, learning_rate = 500, n_iter = 2000, random_state = 2)
    vector_matrix = word2vec_model.wv.syn0
    vector_matrix_2d = tsne.fit_transform(vector_matrix)
    word_points = pd.DataFrame([(word, coords[0], coords[1]) for word, coords in [(word, vector_matrix_2d[word2vec_model.wv.vocab[word].index]) for word in word2vec_model.wv.vocab]], columns=["word", "x", "y"])
    return word_points

def word_points_viz(word_points):
    #scatterplot with all words
    plt.scatter(word_points['x'], word_points['y'], c='lightblue')
    for i, point in word_points.iterrows():
        plt.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    plt.savefig('word_points.png')
    plt.clf()

def target_word_point_viz(word_points, target):
    #scatterplot centered around a single word and the words with similarities closest to it 
    target_word_point = word_points[word_points.word == target]
    x = target_word_point.iloc[0]['x']
    y = target_word_point.iloc[0]['y']
    x_bounds=(x-50, x+50)
    y_bounds=(y-50, y+50)
    section = word_points[(x_bounds[0] <= word_points.x) & (word_points.x <= x_bounds[1]) & 
                    (y_bounds[0] <= word_points.y) & (word_points.y <= y_bounds[1])]
    plt.scatter(section['x'].tolist(), section['y'].tolist(), c='lightblue')
    for i, point in section.iterrows():
        plt.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    plt.scatter(x, y, c='coral')
    plt.text(x + 0.005, y + 0.005, target, fontsize=11)
    plt.savefig('target_word_point.png')
    plt.clf()

def avg_vector(words, num_features):
    global word2vec_model
    wordset = set(word2vec_model.wv.index2word)
    avg_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in wordset:
            n_words += 1
            avg_vec = np.add(avg_vec, word2vec_model[word])
    if (n_words > 0):
        avg_vec = np.divide(avg_vec, n_words)
    return avg_vec

def sanitize(desc):
    token = word_tokenize(desc.lower())
    words = []
    for w in token:
        if not w in stops: #don't include stop words
            w = (re.sub('[^A-Za-z0-9]+', '', w).lower()).strip() #remove punc & special chars
            if w:
                words.append(w)
    return " ".join(words)
    
#SIMILARITY BETWEEN 2 DESCRIPTIONS
    #https://datascience.stackexchange.com/questions/23969/sentence-similarity-prediction
    #https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt
    # get average vector for both descriptions you're comparing
    # get cosine similarity between vectors
def similarity(node1, node2):
    global graph
    #get average vector for each description
    results = graph.run("""MATCH (n:Person) WHERE n.name='{}' RETURN n.description AS description""".format(node1)).data()
    desc1 = results[0]['description']
    desc1 = sanitize(desc1)
    desc1_avg_vector = avg_vector(desc1.split(), num_features=300)

    results = graph.run("""MATCH (n:Person) WHERE n.name='{}' RETURN n.description AS description""".format(node2)).data()
    desc2 = results[0]['description']
    desc2 = sanitize(desc2)
    desc2_avg_vector = avg_vector(desc2.split(), num_features=300)

    sim = 1 - spatial.distance.cosine(desc1_avg_vector, desc2_avg_vector)
    return sim
    #TODO: create relationship between 2 nodes with similar > 0.75 (see if this even happens with any)

def create_desc_points_matrix():
    #create list of average description vectors
    desc_avg_vectors = []
    for desc in descriptions:
        desc_avg_vectors.append(avg_vector(desc.split(), num_features=300))
    #reduce vector matrix of all the words to 2 dimensions
    tsne = sklearn.manifold.TSNE(n_components = 2, early_exaggeration = 6, learning_rate = 500, n_iter = 2000, random_state = 2)
    desc_vector_matrix = desc_avg_vectors
    desc_vector_matrix_2d = tsne.fit_transform(desc_vector_matrix)
    desc_points = pd.DataFrame([(i, desc, desc_vector_matrix_2d[i][0], desc_vector_matrix_2d[i][1]) for i, desc in enumerate(descriptions)], columns=["num", "desc", "x", "y"])
    return desc_points

def desc_points_viz(desc_points):
    #scatterplot with all descriptions
    plt.scatter(desc_points['x'], desc_points['y'], c='lightblue')
    for i, point in desc_points.iterrows():
        label = graph.run("""MATCH (n:Person) WHERE n.description CONTAINS "{}" RETURN n.name AS name""".format(descriptions_original[point.num])).data()
        label = label[0]['name']
        plt.text(point.x + 0.005, point.y + 0.005, label, fontsize=11)
    plt.savefig('desc_points.png')
    plt.clf()
        # desc_scatterplot = desc_points.plot.scatter(x='x', y='y', c='DarkBlue')
        # desc_fig = desc_scatterplot.get_figure()
        # desc_fig.savefig('desc_scatterplot.png')
    #scatterplot centered around a single description (labeled with node title) and the nodes with description similarities closest to it 

def target_desc_point_viz(desc_points, target):
    results = graph.run("""MATCH (n:Person) WHERE n.name='{}' RETURN n.description AS description""".format(target)).data()
    desc = results[0]['description']
    desc = sanitize(desc)
    desc_tech_points = desc_points[desc_points.desc == desc]
    x = desc_tech_points.iloc[0]['x']
    y = desc_tech_points.iloc[0]['y']
    x_bounds=(x-5, x+5)
    y_bounds=(y-5, y+5)
    section = desc_points[(x_bounds[0] <= desc_points.x) & (desc_points.x <= x_bounds[1]) & 
                    (y_bounds[0] <= desc_points.y) & (desc_points.y <= y_bounds[1])]
    plt.scatter(section['x'].tolist(), section['y'].tolist(), c='lightblue')
    for i, point in section.iterrows():
        label = graph.run("""MATCH (n:Person) WHERE n.description CONTAINS "{}" RETURN n.name AS name""".format(descriptions_original[point.num])).data()
        label = label[0]['name']
        plt.text(point.x + 0.005, point.y + 0.005, label, fontsize=11)
    plt.scatter(x, y, c='coral')
    plt.text(x + 0.005, y + 0.005, target, fontsize=11)
    plt.savefig('target_desc_point.png')
    plt.clf()

if __name__ == "__main__":
    data_prep()
    build_word2vec_model()

    word_points = create_word_points_matrix()
    # word_points_viz(word_points)
    target_word_point_viz(word_points, 'education')

    sim = similarity('Grace Hopper', 'Ada Lovelace')
    print('Similarity between two nodes: ', str(sim*100))

    desc_points = create_desc_points_matrix()
    # desc_points_viz(desc_points)
    target_desc_point_viz(desc_points, 'Ada Lovelace')

    #TODO: # print(word2vec_model.most_similar("Ada") )
    #TODO: doc2vec
    #TODO: VISUALIZATIONS THAT COMPARE RESULTS FROM WORD2VEC & DOC2VEC
