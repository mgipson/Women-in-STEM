# make sure you have all of these imports
# will also need to do
    # python
    # import nltk
    # nltk.download('stopwords')
    # nltk.download('punkt')


from py2neo import Graph
# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import wikipedia
import re
import multiprocessing
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
import sklearn.manifold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import pickle
import gensim
import time


graph = None
vocabulary = []
descriptions_original = []
descriptions = []
word2vec_model = None
word2vec_desc_points = pd.DataFrame()
tagged_vocabulary = []
doc2vec_model = None


def connect_to_database():
    global graph
    port = '7687' #input("Enter Neo4j DB Port: ")
    user = 'neo4j' #input("Enter Neo4j DB Username: ")
    pswd = 'stem' #input("Enter Neo4j DB Password: ")
    graph = Graph('bolt://localhost:'+port, auth=(user, pswd))

def generate_vocabulary():
    connect_to_database()
    global graph
    #people nodes
    people = graph.run("""CALL db.index.fulltext.queryNodes("People", "*") YIELD node RETURN DISTINCT node.name AS name, node.description AS description""").to_data_frame()
    people_names = people['name'].to_list()
    failed_people_summaries = []
    global vocabulary
    for i, person in enumerate(people_names):
        if person:
            vocabulary.append(person)
            try:
                #people summaries
                search = wikipedia.search(person, results = 1, suggestion = True)
                summary = wikipedia.summary(search)
                clean_summary = clean(summary)
                if clean_summary:
                    vocabulary.append(clean_summary)
            except:
                print("SEARCH & SUMMARY FAILED FOR: ", person)
                failed_people_summaries.append(person)
    print('Failed to find summaries for ',str(len(failed_people_summaries)), ' people.')

    print("FINISHED ADDING PEOPLE NAMES TO VOCABULARY.")

    filename = 'failed_people_summaries.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(failed_people_summaries, fi)
    print("CREATED PICKLE FILE WITH NAMES OF PEOPLE FOR WHOM SUMMARIES COULD NOT BE FOUND.")

    #people descriptions
    global descriptions_original
    global descriptions
    people_descriptions = people['description'].to_list()
    for desc in people_descriptions:
        if desc is not None: 
            clean_desc = clean(desc)
            if clean_desc:
                descriptions_original.append(desc) 
                descriptions.append(" ".join(clean_desc))
                vocabulary.append(clean_desc)

    print("FINISHED ADDING PEOPLE DESCRIPTIONS TO VOCABULARY.")

    filename = 'descriptions_original.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(descriptions_original, fi)
    print("CREATED PICKLE FILE WITH ORIGINAL DESCRIPTIONS.")

    filename = 'descriptions.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(descriptions, fi)
    print("CREATED PICKLE FILE WITH DESCRIPTIONS.")

    #occupations, nations, awards
    titles = graph.run("""CALL db.index.fulltext.queryNodes("Others", "*") YIELD node RETURN DISTINCT node.title AS title""").to_data_frame()
    titles = titles['title'].to_list()
    for title in titles:
        if title is not None:
            clean_title = clean(title)
            if clean_title:
                vocabulary.append(clean_title)

    print("FINISHED ADDING OTHER TITLES TO VOCABULARY.")

    filename = 'vocabulary.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(vocabulary, fi)

    print("CREATED PICKLE FILE WITH VOCABULARY.")

def generate_descriptions():
    global descriptions_original
    global descriptions
    global vocabulary
    connect_to_database()
    global graph
    #people nodes
    people = graph.run("""CALL db.index.fulltext.queryNodes("People", "*") YIELD node RETURN DISTINCT node.name AS name, node.description AS description""").to_data_frame()
    #people descriptions
    people_descriptions = people['description'].to_list()
    for desc in people_descriptions:
        if desc is not None: 
            descriptions_original.append(desc) 
            clean_desc = clean(desc)
            if clean_desc:
                descriptions.append(" ".join(clean_desc))
    print("FINISHED CREATING PEOPLE DESCRIPTIONS.")

    filename = 'descriptions_original.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(descriptions_original, fi)
    print("CREATED PICKLE FILE WITH ORIGINAL DESCRIPTIONS.")

    filename = 'descriptions.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(descriptions, fi)
    print("CREATED PICKLE FILE WITH DESCRIPTIONS.")

#tokenize and sanitize
def clean(string):
    stops = set(stopwords.words("english"))
    token = word_tokenize(string.lower()) #tokenize
    words = []
    for w in token:
        if not w in stops: #don't include stop words
            w = (re.sub('[^A-Za-z0-9]+', '', w).lower()).strip() #remove punc & special chars
            if w:
                words.append(w)
    return words

############### WORD2VEC ###############

def build_word2vec_model():
    global word2vec_model
    global vocabulary
    num_features = 300
    min_word_count = 20
    num_workers = multiprocessing.cpu_count()
    context_size = 10
    downsampling = 1e-4
    seed = 2

    word2vec_model = Word2Vec(
        sg=1, #skip-gram
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    print("BUILDING WORD2VEC_MODEL VOCAB")
    word2vec_model.build_vocab(vocabulary)

    print("TRAINING WORD2VEC_MODEL")
    word2vec_model.train(vocabulary, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)

    print("SAVING WORD2VEC_MODEL")
    word2vec_model.save("word2vec.model")

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
    
#SIMILARITY BETWEEN 2 DESCRIPTIONS
    #https://datascience.stackexchange.com/questions/23969/sentence-similarity-prediction
    #https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt
    # get average vector for both descriptions you're comparing
    # get cosine similarity between vectors
def word2vec_similarity(node1, node2):
    global graph
    word2vec_similarity_start = time.perf_counter()
    results = graph.run("""MATCH (n:Person) WHERE n.name="{}" RETURN DISTINCT n.description AS description""".format(node1)).data()
    desc1 = results[0]['description']
    desc1 = " ".join(clean(desc1))
    desc1_avg_vector = avg_vector(desc1.split(), num_features=300)
    results = graph.run("""MATCH (n:Person) WHERE n.name="{}" RETURN DISTINCT n.description AS description""".format(node2)).data()
    desc2 = results[0]['description']
    desc2 = " ".join(clean(desc2))
    desc2_avg_vector = avg_vector(desc2.split(), num_features=300)
    sim = 1 - spatial.distance.cosine(desc1_avg_vector, desc2_avg_vector)
    word2vec_similarity_end = time.perf_counter()
    print(f"WORD2VEC SIMILARITY BETWEEN 2 DESCRIPTIONS TOOK {word2vec_similarity_end-word2vec_similarity_start:0.4f} seconds.")
    return sim, desc1, desc2
    #TODO: create relationship between 2 nodes with similar > 0.75 (see if this even happens with any)

#create list of average description vectors
#reduce vector matrix of all the words to 2 dimensions
def create_desc_points_matrix():
    global word2vec_desc_points
    desc_avg_vectors = []
    for desc in descriptions:
        desc_avg_vectors.append(avg_vector(desc.split(), num_features=300))
    tsne = sklearn.manifold.TSNE(n_components = 2, early_exaggeration = 6, learning_rate = 500, n_iter = 2000, random_state = 2)
    desc_vector_matrix = desc_avg_vectors
    desc_vector_matrix_2d = tsne.fit_transform(desc_vector_matrix)
    word2vec_desc_points = pd.DataFrame([(i, desc, desc_vector_matrix_2d[i][0], desc_vector_matrix_2d[i][1]) for i, desc in enumerate(descriptions)], columns=["num", "desc", "x", "y"])
    filename = 'word2vec_desc_points_matrix.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(word2vec_desc_points, fi)

#scatterplot centered around a single description (labeled with node title) and the nodes with description similarities closest to it 
def word2vec_most_similar(target):
    global word2vec_desc_points
    global word2vec_model
    word2vec_target_start = time.perf_counter()
    results = graph.run("""MATCH (n:Person) WHERE n.name="{}" RETURN DISTINCT n.description AS description""".format(target.title())).data()
    target_raw_desc = results[0]['description']
    target_desc = clean(target_raw_desc)
    target_desc_avg_vector = avg_vector(target_desc, num_features=300)
    target_desc_points = word2vec_desc_points[word2vec_desc_points.desc == " ".join(target_desc)]
    target_x = target_desc_points.iloc[0]['x']
    target_y = target_desc_points.iloc[0]['y']

    top_names = []
    top_desc = []
    top_sim = []

    nodes = graph.run("""CALL db.index.fulltext.queryNodes("People", "*") YIELD node RETURN node.name AS name, node.description AS description""").data()
    for node in nodes:
        desc = clean(node['description'])
        desc_avg_vector = avg_vector(desc, num_features=300)
        sim = 1 - spatial.distance.cosine(desc_avg_vector, target_desc_avg_vector)
        if sim >= 0.9:
            top_names.append(node['name'])
            top_desc.append(node['description'])
            top_sim.append(sim)
            desc_points = word2vec_desc_points[word2vec_desc_points.desc == " ".join(desc)]
            x = desc_points.iloc[0]['x']
            y = desc_points.iloc[0]['y']
            plt.scatter(x, y, c='lightblue')
            plt.text(x + 0.005, y + 0.005, node['name'], fontsize=11)
    plt.scatter(target_x, target_y, c='coral')
    plt.text(target_x + 0.005, target_y + 0.005, target, fontsize=11)
    plt.savefig('target_desc_point.png')
    plt.clf()

    node_similarities = pd.DataFrame([(name, top_sim[i], top_desc[i]) for i, name in enumerate(top_names)], columns=["Name", "Similarity", "Description (Cleaned)"])
    print(node_similarities)
    word2vec_target_end = time.perf_counter()
    print(f"WORD2VEC FINDING TOP SIMILAR NODES TOOK {word2vec_target_end-word2vec_target_start:0.4f} seconds.")

############### DOC2VEC ###############

def tag_vocabulary():
    global vocabulary #should already be cleaned
    global tagged_vocabulary
    for index, vocab in enumerate(vocabulary):
        tagged_vocabulary.append(gensim.models.doc2vec.TaggedDocument(vocab, [index]))
    filename = 'tagged_vocabulary.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(tagged_vocabulary, fi)
    print("CREATED PICKLE FILE WITH TAGGED VOCABULARY.")

def build_doc2vec_model():
    global doc2vec_model
    global tagged_vocabulary
    doc2vec_model = Doc2Vec(dm=0, vector_size=200, min_count=2, epochs=100, window=4, dbow_word=1)
    print("BUILDING DOC2VEC_MODEL VOCAB")
    doc2vec_model.build_vocab(tagged_vocabulary)
    print("TRAINING DOC2VEC_MODEL")
    doc2vec_model.train(tagged_vocabulary, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    print("SAVING DOC2VEC_MODEL")
    doc2vec_model.save("doc2vec.model")

def doc2vec_similarity(node1, node2):
    global doc2vec_model
    #get descriptions of nodes
    doc2vec_similarity_start = time.perf_counter()
    node1_results = graph.run("""MATCH (n:Person) WHERE n.name="{}" RETURN DISTINCT n.description AS description""".format(node1.title())).data()
    raw_desc1 = node1_results[0]['description']
    desc1 = clean(raw_desc1)
    node2_results = graph.run("""MATCH (n:Person) WHERE n.name="{}" RETURN DISTINCT n.description AS description""".format(node2.title())).data()
    raw_desc2 = node2_results[0]['description']
    desc2 = clean(raw_desc2)
    desc1_vector = doc2vec_model.infer_vector(desc1) 
    desc2_vector = doc2vec_model.infer_vector(desc2)
    sim = 1 - spatial.distance.cosine(desc1_vector, desc2_vector)
    doc2vec_similarity_end = time.perf_counter()
    print(f"DOC2VEC SIMILARITY BETWEEN 2 DESCRIPTIONS TOOK {doc2vec_similarity_end-doc2vec_similarity_start:0.4f} seconds.")
    return sim, desc1, desc2 

def doc2vec_most_similar(target):
    global doc2vec_model
    doc2vec_target_start = time.perf_counter()
    #get description of target node
    target_results = graph.run("""MATCH (n:Person) WHERE n.name="{}" RETURN DISTINCT n.description AS description""".format(target.title())).data()
    raw_desc = target_results[0]['description']
    target_desc = clean(raw_desc)
    target_desc_vector = doc2vec_model.infer_vector(target_desc)
    top = doc2vec_model.docvecs.most_similar([target_desc_vector], topn=5)
    top_names = []
    top_sims = []
    top_desc = []
    for n in top:
        if n[1] >= 0.9:
            sim = n[1]*100
            clean_desc = " ".join(tagged_vocabulary[n[0]][0])
            nodes = graph.run("""CALL db.index.fulltext.queryNodes("People", "{}") YIELD node, score RETURN DISTINCT node.name AS name, node.description AS description, score""".format(clean_desc)).data()
            for node in nodes:
                if node['name'] != target.title():
                    top_names.append(node['name'])
                    top_sims.append(sim)
                    top_desc.append(" ".join(clean(node['description'])))

    #TODO: think of how this could visualize

    #dataframe with nodes and their similarities to the target node
    print("Target Node: ", target.title())
    print("Description: ", raw_desc)
    node_similarities = pd.DataFrame([(name, top_sims[i], top_desc[i]) for i, name in enumerate(top_names)], columns=["Name", "Similarity", "Description (Cleaned)"])
    print(node_similarities)
    doc2vec_target_end = time.perf_counter()
    print(f"DOC2VEC FINDING TOP SIMILAR NODES TOOK {doc2vec_target_end-doc2vec_target_start:0.4f} seconds.")

############### MAIN ###############

if __name__ == "__main__":
    #load existing vocabulary
    filename = 'vocabulary.pk'
    try:
        with open(filename, 'rb') as fi:
            vocabulary = pickle.load(fi)
    except:
        print("COULD NOT OPEN VOCABULARY FILE, CREATING IT INSTEAD.")
    if not vocabulary:
        print('GENERATING VOCABULARY')
        generate_vocabulary() 
    else:
        print("USING VOCABULARY FROM vocabulary.pk")

    #load existing descriptions
    filename1 = 'descriptions_original.pk'
    filename2 = 'descriptions.pk'
    try:
        with open(filename1, 'rb') as fi1:
            descriptions_original = pickle.load(fi1)
        with open(filename2, 'rb') as fi2:
            descriptions = pickle.load(fi2)
    except:
        print("COULD NOT OPEN ORIGINAL DESCRIPTIONS AND/OR DESCRIPTIONS FILE, CREATING IT INSTEAD.")
    if not descriptions or not descriptions_original:
        print('GENERATING DESCRIPTIONS')
        generate_descriptions() 
    else:
        print("USING CLEANED AND ORIGINAL DESCRIPTIONS FROM descriptions.pk AND descriptions_original.pk, RESPECTIVELY.")

    #load existing word2vec model
    try:
        word2vec_model = Word2Vec.load("word2vec.model")
    except:
        print("COULD NOT LOAD WORD2VEC MODEL, BUILDING ONE INSTEAD.")
    if not word2vec_model:
        print("BUILDING WORD2VEC MODEL")
        build_word2vec_model()
    else:
        print("USING EXISITNG WORD2VEC MODEL")

    #load existing word2vec points matrix
    filename = 'word2vec_desc_points_matrix.pk'
    try:
        with open(filename, 'rb') as fi:
            word2vec_desc_points = pickle.load(fi)
    except:
        print("COULD NOT OPEN WORD2VEC POINTS MATRIX FILE, CREATING THE MATRIX INSTEAD.")
    if word2vec_desc_points.empty:
        print('GENERATING MATRIX')
        create_desc_points_matrix() 
    else:
        print("USING MATRIX FROM word2vec_desc_points_matrix.pk")

    #connect to database if needed
    if graph is None:
        connect_to_database()
    
    #word2vec
    if not word2vec_desc_points.empty:
        print("------------------------------------------------------------------------")
        node1 = 'Grace Hopper' #'Telle Whitney' #'Grace Hopper'
        node2 = 'Ada Lovelace'#'Susan B. Horwitz' #'Ada Lovelace'
        # Grace & Ada -> 72.78
        # Telle & Susan -> 96.16
        sim, desc1, desc2 = word2vec_similarity(node1, node2)
        print('Word2Vec similarity between ', node1, ' and ', node2, ': ', str(sim*100))
        print(node1, ' description: ', desc1)
        print(node2, ' description: ', desc2)
        print("------------------------------------------------------------------------")
        word2vec_most_similar('Ada Lovelace')
    else:
        print('NO WORD2VEC DESCRIPTION POINTS TO USE.')
    #TODO: get the top 5 similar nodes to a target node with word2vec method


    #doc2vec
    #load existing tagged vocab
    filename = 'tagged_vocabulary.pk'
    try:
        with open(filename, 'rb') as fi:
            tagged_vocabulary = pickle.load(fi)
    except:
        print("COULD NOT OPEN TAGGED VOCABULARY FILE, TAGGING THE VOCABULARY NOW.")
    if not tagged_vocabulary:
        print('TAGGING VOCABULARY')
        tag_vocabulary() 
    else:
        print("USING TAGGED VOCABULARY FROM tagged_vocabulary.pk")

    #load existing doc2vec model
    try:
        doc2vec_model = Doc2Vec.load("doc2vec.model")
    except:
        print("COULD NOT LOAD DOC2VEC MODEL, BUILDING ONE INSTEAD.")
    if not doc2vec_model:
        print("BUILDING DOC2VEC MODEL")
        build_doc2vec_model()
    else:
        print("USING EXISITNG DOC2VEC MODEL")


    if doc2vec_model:
        print("------------------------------------------------------------------------")
        node1 = 'Grace Hopper' #'Telle Whitney' #'Grace Hopper'
        node2 = 'Ada Lovelace'#'Susan B. Horwitz' #'Ada Lovelace'
        # Grace & Ada -> 55.78
        # Telle & Susan -> 84.65
        sim, desc1, desc2 = doc2vec_similarity(node1, node2)
        print('Doc2Vec similarity between ', node1, ' and ', node2, ': ', str(sim*100))
        print(node1, ' description: ', desc1)
        print(node2, ' description: ', desc2)
        print("------------------------------------------------------------------------")
        target = 'Ada Lovelace'
        doc2vec_most_similar(target)
