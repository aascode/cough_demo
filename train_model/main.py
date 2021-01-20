import itertools
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import SGDClassifier
import config
from preprocess import read_data, MyDataset
from model import LSTMClassifier
from sklearn.metrics import classification_report
from  matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def train(data, model, criterion, optimizer, device):
    model.train()
    sum_loss = 0
    num_batch = 0
    all_label = []
    all_pred = []

    for feat, lbl in data:
        feat, lbl = feat.to(device), lbl.to(device)
        y_hat = model(feat)
        y_pred = torch.argmax(y_hat, dim=1)
        all_label += lbl.cpu().numpy().tolist()
        all_pred += y_pred.cpu().numpy().tolist()
        optimizer.zero_grad()
        loss = criterion(y_hat, lbl)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        num_batch += 1

    return sum_loss / num_batch, accuracy_score(all_label, all_pred)


def test(data, model, criterion, device):
    model.eval()
    sum_loss = 0
    num_batch = 0
    all_label = []
    all_pred = []

    for feat, lbl in data:
        feat, lbl = feat.to(device), lbl.to(device)
        with torch.no_grad():
            y_hat = model(feat)

        y_pred = torch.argmax(y_hat, dim=1)
        all_label += lbl.cpu().numpy().tolist()
        all_pred += y_pred.cpu().numpy().tolist()

        loss = criterion(y_hat, lbl)
        sum_loss += loss.item()
        num_batch += 1

    return sum_loss / num_batch, accuracy_score(all_label, all_pred)


def train_lstm():
    device = torch.dtest.pyevice('cuda' if torch.cuda.is_available() else 'cpu')
    feature, label = read_data(config.DATA_DIR)
    train_feature, test_feature, train_label, test_label = train_test_split(feature, label)
    train_data = MyDataset(train_feature, train_label)
    test_data = MyDataset(train_feature, train_label)
    train_data = DataLoader(train_data, batch_size=16, shuffle=True)
    test_data = DataLoader(test_data, batch_size=16, shuffle=False)

    model = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(params=model.parameters(), lr=0.001)

    for e in range(20):
        train_loss, train_acc = train(train_data, model, criterion, optimizer, device)
        test_loss, test_acc = test(test_data, model, criterion, device)
        print(train_loss, train_acc)
        print(test_loss, test_acc)
        print("----------------------------------------")
        torch.save(model, os.path.join(config.MODEL_DIR, "model_{}.pt".format(e + 1)))


def train_svm():
    feature, label = read_data(config.DATA_DIR)
    train_feature, test_feature, train_label, test_label = train_test_split(feature, label)
    train_feature = train_feature.reshape(train_feature.shape[0], -1)
    test_feature = test_feature.reshape(test_feature.shape[0], -1)
    svm = SVC()
    svm.fit(train_feature.numpy(), train_label.numpy())
    y_pred = svm.predict(test_feature.numpy())
    #pickle.dump(svm, open("resource/gmm.pkl", 'wb'))
    print(accuracy_score(test_label.numpy(), y_pred))
    #print(confusion_matrix(test_label.numpy(), y_pred))
    print(y_pred)
    np.set_printoptions(precision=2)
    matrix = classification_report(test_label.numpy(), y_pred)
    print("Classification report: \n", matrix)

    # Plot non-normalized confusion matrix
    con_matrix = confusion_matrix(test_label.numpy(), y_pred)
    class_names = ["Non_cough", "Cough"]
    plt.figure()
    plot_confusion_matrix(con_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(con_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def train_gmm():
    feature, label = read_data(config.DATA_DIR)
    feature, label = feature.reshape(feature.shape[0], -1).numpy(), label.numpy()
    train_feature, test_feature, train_label, test_label = train_test_split(feature, label)
    model_0 = GaussianMixture(n_components=3, max_iter=100, weights_init=[1/3, 1/3, 1/3], random_state=42)
    model_1 = GaussianMixture(n_components=3, max_iter=100, weights_init=[1/3, 1/3, 1/3], random_state=42)
    # model.means_init = numpy.array([train_feture[train_label == i].mean(axis=0)
    #                                 for i in range(2)])

    model_0.fit(train_feature[train_label == 0], train_label[train_label == 0])
    model_1.fit(train_feature[train_label == 1], train_label[train_label == 1])
    # pred = model.predict(test_feature)
    # for feat in test_feature:
    y_pred = []
    score_0 = model_0.score_samples(test_feature)
    score_1 = model_1.score_samples(test_feature)
    for i in range(len(score_1)):
        if score_0[i] > score_1[i]:
            y_pred.append(0)
        else:
            y_pred.append(1)

    # print(model_0.score_samples(test_feature), model_1.score_samples(test_feature))
    print(accuracy_score(test_label, y_pred))

    # recall and precision
    matrix = classification_report(test_label, y_pred)
    print("Classification report: \n", matrix)

    # Plot non-normalized confusion matrix
    np.set_printoptions(precision=2)
    con_matrix = confusion_matrix(test_label, y_pred)
    class_names = ["Non_cough", "Cough"]
    plt.figure()
    plot_confusion_matrix(con_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(con_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    pickle.dump(model_0, open("resource/non_cough.pkl", 'wb'))
    pickle.dump(model_1, open("resource/cough.pkl", 'wb'))

    print(model_1.weights_)
    print(model_0.weights_)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    if config.model == "GMM":
        train_gmm()
    elif config.model == "LSTM":
        train_lstm()
    else:
        train_svm()
