import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


with open('file.txt', 'r') as infile:
    weather_words = infile.readlines()

def getWordVecs(words):
    vecs = []
    for word in words:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word].reshape((1,300)))
        except KeyError:
            continue
    vecs = np.concatenate(vecs)
    return np.array(vecs, dtype='float') #TSNE expects float type values

# get words vector
# weather_vecs = getWordVecs(weather_words)

ts = TSNE(2)
#reduced_vecs = ts.fit_transform(np.concatenate((weather_vecs, [set of vectors])))

#color points by word group to see if Word2Vec can separate them
def printWordsClusters(words):
    for i in range(len(reduced_vecs)):
        if i < len(food_vecs):
            #food words colored blue
            color = 'b'
        elif i >= len(food_vecs) and i < (len(food_vecs) + len(sports_vecs)):
            #sports words colored red
            color = 'r'
        else:
            #weather words colored green
            color = 'g'
        plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color=color, markersize=8)