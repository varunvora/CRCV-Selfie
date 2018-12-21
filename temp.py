from common import *
from keras.models import load_model
model = load_model(best_models[4])
import pickle
from sklearn import tree
from dataset_loader import img_loader
# print(model.layers[0].get_weights()[0])

def decisionTreeClassifier():
    X, Y = list(img_loader(io = 6, batch_size = 1))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    with open('DecisionTreeClassifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    # dot_data = tree.export_graphviz(clf, out_file = None)
    # graph = graphviz.Source(dot_data)
    # graph.render('clf')

if __name__ == "__main__":
    decisionTreeClassifier()