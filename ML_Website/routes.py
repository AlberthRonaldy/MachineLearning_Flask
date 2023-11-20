from ML_Website import app
from flask import  render_template, request
from sklearn.metrics import  f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from random import randint
import os
from matplotlib import pyplot as plt


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        op_classificador = request.form.get('classifier')
        if op_classificador in ('1', '2', '3', '4'):
            op_classificador = {
                '1': "DecisionTreeClassifier",
                '2': "RandomForestClassifier",
                '3': "MLPClassifier",
                '4': "KNeighborsClassifier"
            }[op_classificador]
            return render_template('index.html', classificador=op_classificador)
    return render_template('index.html')
    

@app.route('/treinar/<int:classificador>', methods=['POST', 'GET'])
def treinar(classificador):
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pasta = os.path.join(app.root_path,'static', 'graficos')
    os.makedirs(pasta, exist_ok=True)

    # Remove arquivos existentes
    for arquivo in os.listdir(pasta):
        caminho_arquivo = os.path.join(pasta, arquivo)
        try:
            if os.path.isfile(caminho_arquivo):
                os.remove(caminho_arquivo)
                print(f'Arquivo {arquivo} deletado com sucesso.')
        except Exception as e:
            print(f"Erro ao deletar {arquivo}: {str(e)}")

    if classificador == 1:
        clf_name = 'DecisionTreeClassifier'
        clf = DecisionTreeClassifier(max_depth=int(request.form['req1']),
                                     random_state=int(request.form['req2']),
                                     max_leaf_nodes=int(request.form['req3']))
    elif classificador == 2:
        clf_name = 'RandomForestClassifier'
        clf = RandomForestClassifier(n_estimators=int(request.form['req1']),
                                     max_depth=int(request.form['req2']),
                                     max_leaf_nodes=int(request.form['req3']))
    elif classificador == 3:
        clf_name = 'MLPClassifier'
        clf = MLPClassifier(hidden_layer_sizes=int(request.form['req1']),
                            random_state=int(request.form['req2']),
                            max_iter=int(request.form['req3']))
    elif classificador == 4:
        clf_name = 'KNeighborsClassifier'
        clf = KNeighborsClassifier(n_neighbors=int(request.form['req1']),
                                   leaf_size=int(request.form['req2']),
                                   p=int(request.form['req3']))
    else:
        return render_template('index.html')

    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    classes = iris.target_names.tolist()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()

    id_img = randint(1, 1024)
    arquivo = f'meu_grafico_{id_img}.png'
    plt.savefig(os.path.join(pasta, arquivo))

    return render_template('resultados.html', accuracy=acc, f1_score=f1_macro, url_img=arquivo, clf_name=clf_name)