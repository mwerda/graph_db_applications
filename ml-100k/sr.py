import numpy as np
import re
import math
import matplotlib
import matplotlib.pyplot as plt
from pylab import *  # z powodu map kolorow
import random
from multiprocessing import Process, Manager
import time
import copy
import os


# from ref_algorithms import prepareHooiRS, prepareHosvdRS, prepareMultiSvdRS, prepareAvarageBasedRS

# import MOPCA2System_Szwabe_19_07_2016 as MOPCA

def getUDataContent():
    plik = open('u.data.txt', 'r')
    zawartosc = plik.read()
    zawartosc2 = zawartosc.split('\n')
    zawartosc4 = []
    zbiorUsers = set()
    zbiorMovies = set()
    for wiersz in zawartosc2[:-1]:
        wierszPodzielon = wiersz.split('\t')
        zawartosc4.append(wierszPodzielon)
        zbiorUsers.add(wierszPodzielon[0])
        zbiorMovies.add(wierszPodzielon[1])
    return zawartosc4, zbiorUsers, zbiorMovies


def pictureMatrix(matrix, label):
    # wylaczenie interaktywnego okna (na rzecz zapisu wylacznie do pliku):
    plt.ioff()
    # funkcja imshow sluzy prezentacji macierzy numpy jako bitmapy (zapisywalnej do pliku PNG):
    plt.imshow(matrix, cmap=cm.Spectral, interpolation='none')
    plt.colorbar()
    plt.title(label)
    savefig(label + ".png", format="png", dpi=100)


def findTheFirstEVectorBySVD(matrix):
    U, S, VT = np.linalg.svd(matrix)
    return U[:, 0]


def getMatrixSpectrum(matrix):
    U, S, VT = np.linalg.svd(matrix)
    return S


def findTheFirstEVectorByPowerIteration(matrix):
    numberOfIterations = 0
    tempMatrix = np.dot(matrix, np.transpose(matrix))
    tempVector = np.ones(matrix.shape[0])
    oldTempVector = tempVector
    tempVector = np.dot(tempMatrix, tempVector)
    tempVector = tempVector / math.sqrt(np.sum(tempVector ** 2))
    while not (np.allclose(tempVector, oldTempVector)):
        numberOfIterations += 1
        oldTempVector = tempVector
        tempVector = np.dot(tempMatrix, tempVector)
        tempVector = tempVector / math.sqrt(np.sum(tempVector ** 2))
    return tempVector, numberOfIterations


def findTheFirstEVector(matrix):
    theFirstEVector, numberOfIterations = findTheFirstEVectorByPowerIteration(matrix)
    #    theFirstEVector=findTheFirstEVectorBySVD(matrix)
    return theFirstEVector


def findRowAveragesVector(matrix):
    theFirstEVector = np.sum(matrix, axis=0)
    return theFirstEVector


# algorytm funkcji wzorowany na PageRank:
def getNMostCentralRows(firstEVector, n):
    firstEVectorWithSquaredEntriesAsList = list(firstEVector ** 2)
    vectorToRank = []
    for squaredEntryNumber in range(len(firstEVectorWithSquaredEntriesAsList)):
        vectorToRank.append([squaredEntryNumber, firstEVectorWithSquaredEntriesAsList[squaredEntryNumber]])
    vectorToRankSorted = sorted(vectorToRank, key=lambda x: x[1], reverse=True)
    NMostCentralRows = [row[0] for row in vectorToRankSorted]
    return NMostCentralRows[:n]


def getSelectedRows(firstEVector, n, mode=0):
    #    firstEVectorWithSquaredEntriesAsList=list(firstEVector**2)
    firstEVectorWithSquaredEntriesAsList = list(firstEVector)
    vectorToRank = []
    for squaredEntryNumber in range(len(firstEVectorWithSquaredEntriesAsList)):
        vectorToRank.append([squaredEntryNumber, firstEVectorWithSquaredEntriesAsList[squaredEntryNumber]])
    if mode == 0:
        vectorToRankSorted = sorted(vectorToRank, key=lambda x: x[1], reverse=True)
    if mode == 1:
        #        #print('vectorToRank: ',vectorToRank)
        vectorToRankSorted = sorted(vectorToRank, key=lambda x: x[1], reverse=False)
    if mode == 2:
        #        #print('vectorToRank: ',vectorToRank)
        random.shuffle(vectorToRank)
        vectorToRankSorted = vectorToRank

    # vectorToRankSorted = sorted(vectorToRank, key=lambda x: x[1], reverse=True)
    NSelectedRows = [row[0] for row in vectorToRankSorted]
    return NSelectedRows[:n]


def getDenseMatrix(uDataPart, shape):
    denseMatrix = np.zeros(shape)
    for row in uDataPart:
        ##print(row)
        #       denseMatrix[NMostCentralUsers.index(int(row[0])),NMostCentralMovies.index(int(row[1]))]=1
        if int(row[2]) > 4.0:
            denseMatrix[NMostCentralUsers.index(int(row[0])), NMostCentralMovies.index(int(row[1]))] = 1
        else:
            denseMatrix[NMostCentralUsers.index(int(row[0])), NMostCentralMovies.index(int(row[1]))] = -1
    return denseMatrix


def getMatrixDensity(matrix):
    density = np.count_nonzero(matrix) / (matrix.shape[0] * matrix.shape[1])
    return density


# def OLDgetSubUDataSet(datasetSizeReductionRatio, mode=0):
#    UDataContent,zbiorUsers,zbiorMovies=getUDataContent()
##    tablica=np.array(UDataContent)
##    tablica2=np.zeros((len(zbiorUsers),len(zbiorMovies)))
#    tablica3=np.zeros((len(zbiorUsers),len(zbiorMovies)))
#    for wiersz in UDataContent:
#        tablica3[int(wiersz[0])-1,int(wiersz[1])-1]=1
##        if int(wiersz[2])>2.0:
##            tablica2[int(wiersz[0])-1,int(wiersz[1])-1]=1
##        else:
##            tablica2[int(wiersz[0])-1,int(wiersz[1])-1]=-1
#    U0=findTheFirstEVector(tablica3)
#    V0=findTheFirstEVector(np.transpose(tablica3))
#    numberOfNMostCentralUsers=round(len(zbiorUsers)*datasetSizeReductionRatio)
#    numberOfNMostCentralMovies=round(len(zbiorMovies)*datasetSizeReductionRatio)
#    NMostCentralUsers=getNMostCentralRows(U0,numberOfNMostCentralUsers)
#    NMostCentralMovies=getNMostCentralRows(V0,numberOfNMostCentralMovies)
#    NMostCentralUsersSet=set(NMostCentralUsers)
#    NMostCentralMoviesSet=set(NMostCentralMovies)
#    subdataset=[wiersz[:3] for wiersz in UDataContent if (int(wiersz[0]) in NMostCentralUsersSet)and(int(wiersz[1]) in NMostCentralMoviesSet)]
#
##    numberOfSelectedUsers=round(len(zbiorUsers)*datasetSizeReductionRatio)
##    numberOfSelectedMovies=round(len(zbiorMovies)*datasetSizeReductionRatio)
##    selectedUsers=getSelectedRows(U0,numberOfSelectedUsers)
##    selectedMovies=getSelectedRows(V0,numberOfSelectedMovies)
##    selectedUsersSet=set(selectedUsers)
##    selectedMoviesSet=set(selectedMovies)
##    subdataset=[wiersz[:3] for wiersz in UDataContent if (int(wiersz[0]) in selectedUsersSet)and(int(wiersz[1]) in selectedMoviesSet)]
#
#
#    return subdataset



def getSubUDataSet(datasetSizeReductionRatio, mode):
    UDataContent, zbiorUsers, zbiorMovies = getUDataContent()
    #    tablica=np.array(UDataContent)
    #    tablica2=np.zeros((len(zbiorUsers),len(zbiorMovies)))
    tablica3 = np.zeros((len(zbiorUsers), len(zbiorMovies)))
    for wiersz in UDataContent:
        tablica3[int(wiersz[0]) - 1, int(wiersz[1]) - 1] = 1
    # if int(wiersz[2])>2.0:
    #            tablica2[int(wiersz[0])-1,int(wiersz[1])-1]=1
    #        else:
    #            tablica2[int(wiersz[0])-1,int(wiersz[1])-1]=-1
    numberOfSelectedUsers = int(round(len(zbiorUsers) * datasetSizeReductionRatio))
    # print("numberOfSelectedUsers: ",numberOfSelectedUsers)
    numberOfSelectedMovies = int(round(len(zbiorMovies) * datasetSizeReductionRatio))

    if mode == 0:
        U0 = findTheFirstEVector(tablica3)
        V0 = findTheFirstEVector(np.transpose(tablica3))

        selectedUsers = getSelectedRows(U0 ** 2, numberOfSelectedUsers, mode)
        selectedMovies = getSelectedRows(V0 ** 2, numberOfSelectedMovies, mode)

    if mode == 1:
        rowAveragesVector = findRowAveragesVector(tablica3)
        columnAveragesVector = findRowAveragesVector(np.transpose(tablica3))
        selectedUsers = getSelectedRows(rowAveragesVector, numberOfSelectedUsers, 0)
        selectedMovies = getSelectedRows(columnAveragesVector, numberOfSelectedMovies, 0)

    selectedUsersSet = set(selectedUsers)
    selectedMoviesSet = set(selectedMovies)
    subdataset = [wiersz[:3] for wiersz in UDataContent if
                  (int(wiersz[0]) in selectedUsersSet) and (int(wiersz[1]) in selectedMoviesSet)]

    return subdataset


def convertSubUDataSetToInTuplesList(subUDataSet, trueToFalseThreshold):
    inTuplesList = []
    for subUDataSetRow in subUDataSet:
        inTupleRow = []
        if int(subUDataSetRow[2]) > trueToFalseThreshold:
            inTupleRow.append(1)
        else:
            inTupleRow.append(-1)
        inTupleRow.append(subUDataSetRow[0])
        inTupleRow.append(subUDataSetRow[1])
        inTuplesList.append(tuple(inTupleRow))
    return inTuplesList


class dimensionsIndexingDictionaries:
    def __init__(self, listaKrotekWejsciowych=[], numberOfDataAttributes=1):
        if not (listaKrotekWejsciowych == []):
            numberOfDataAttributes = len(listaKrotekWejsciowych[0])
        self.liczbaWymiarowIndeksowaniaKomorekTensoraWeWy = numberOfDataAttributes
        self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania = []
        self.listaSlownikowIndeksowania = []
        for i in range(self.liczbaWymiarowIndeksowaniaKomorekTensoraWeWy):
            self.listaSlownikowIndeksowania.append({})
            self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania.append([])
            self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i] = 0
        self.update(listaKrotekWejsciowych)

    def update(self, listaKrotekWejsciowych=[]):
        for aktualnaKrotka in listaKrotekWejsciowych:
            aktualnaKrotkaJakoLista = list(aktualnaKrotka)
            for indeksSlownika in range(self.liczbaWymiarowIndeksowaniaKomorekTensoraWeWy):
                aktualnyKlucz = aktualnaKrotkaJakoLista[indeksSlownika]
                if (aktualnyKlucz not in self.listaSlownikowIndeksowania[indeksSlownika].keys()):
                    self.listaSlownikowIndeksowania[indeksSlownika][aktualnyKlucz] = len(
                        self.listaSlownikowIndeksowania[indeksSlownika].keys()) + 1
                    self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[indeksSlownika] += 1
        return self.listaSlownikowIndeksowania

    def getIndex(self, numerSlownika, kluczAdresujacyIndeks):
        if (kluczAdresujacyIndeks not in self.listaSlownikowIndeksowania[numerSlownika].keys()):
            self.listaSlownikowIndeksowania[numerSlownika][kluczAdresujacyIndeks] = len(
                self.listaSlownikowIndeksowania[numerSlownika].keys()) + 1
            self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[numerSlownika] += 1
        return self.listaSlownikowIndeksowania[numerSlownika][kluczAdresujacyIndeks]


# uwaga programistyczna: "dlugosc" pierwszej krotki z listy listaKrotekWejsciowych determinuje wartosc liczbaWymiarowIndeksowaniaKomorekTensoraWeWy:
# liczbaWymiarowIndeksowaniaKomorekTensoraWeWy=len(listaKrotekWejsciowych[0])
# liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania=[]
#
# listaSlownikowIndeksowania=[]
# for i in range(liczbaWymiarowIndeksowaniaKomorekTensoraWeWy):
#    listaSlownikowIndeksowania.append({})
#    liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania.append([])
#    liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i]=0
#
# for aktualnaKrotka in listaKrotekWejsciowych:
#    aktualnaKrotkaJakoLista=list(aktualnaKrotka)
#    for indeksSlownika in range(liczbaWymiarowIndeksowaniaKomorekTensoraWeWy):
#        aktualnyKlucz=aktualnaKrotkaJakoLista[indeksSlownika]
#        if(aktualnyKlucz not in listaSlownikowIndeksowania[indeksSlownika].keys()):
#            listaSlownikowIndeksowania[indeksSlownika][aktualnyKlucz]=len(listaSlownikowIndeksowania[indeksSlownika].keys())+1
#            liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[indeksSlownika]+=1

class arrayBasedDataRepresentation:
    def __init__(self, dids, trainSetTuplesList=[]):
        dids.update(trainSetTuplesList)
        self.inputDataRepresentationShape = dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:]
        self.trainsetDataRepresentationShape = []
        for i in range(len(self.inputDataRepresentationShape)):
            self.trainsetDataRepresentationShape.append(self.inputDataRepresentationShape[i] + 1)
        self.trainsetDataRepresentation = np.zeros((tuple(self.trainsetDataRepresentationShape)))
        self.trainsetDataRepresentation, self.rowSumsVector, self.columnSumsVector = self.storeData(dids,
                                                                                                    trainSetTuplesList)

    def storeData(self, dids, trainSetTuplesList=[]):
        # w wersji z wyzerowywana srednia:
        # najpierw def storeDataInbalanced(self, dids, trainSetTuplesList=[]):
        # a potem:
        # najpierw def storeDataBalanced(self, dids, trainSetTuplesList=[]):
        #       numberOfTrueTuples=0
        #       numberOfFalseTuples=0
        #        for aktualnaKrotka in trainSetTuplesList:
        #           if aktualnaKrotka[0]==1
        #               numberOfTrueTuples+=1
        #           else:
        #               numberOfFalseTuples+=1

        # print("len(trainSetTuplesList): ",len(trainSetTuplesList))
        for aktualnaKrotka in trainSetTuplesList:
            macierzTymczasowa = np.zeros((1, 1))
            macierzTymczasowa[0, 0] = aktualnaKrotka[0]
            # ... a w wersji z wyzerowywana srednia:
            #            macierzTymczasowaForTrueTuples[0,0]=1/numberOfTrueTuples
            #            macierzTymczasowaForFalseTuples[0,0]=1/numberOfFalseTuples

            # ..................
            ksztaltMacierzyTymczasowej = ()
            for i in range(len(dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:])):
                wektorTymczasowy = np.zeros(
                    (dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i + 1] + 1, 1))
                wektorTymczasowy[0] = 1
                wektorTymczasowy[dids.listaSlownikowIndeksowania[i + 1][aktualnaKrotka[i + 1]]] = 1
                # w wersji z "distributed representation":
                # wektorTymczasowy=dids.listaSlownikowIndeksowania[i+1][aktualnaKrotka[i+1]]
                ksztaltMacierzyTymczasowej = ksztaltMacierzyTymczasowej + (
                dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i + 1] + 1,)

                macierzTymczasowa = (np.outer(macierzTymczasowa, wektorTymczasowy)).reshape(ksztaltMacierzyTymczasowej)
            self.trainsetDataRepresentation = np.add(self.trainsetDataRepresentation, macierzTymczasowa)
            # ... a w wersji z wyzerowywana srednia:
            #            self.trainsetDataRepresentation=np.add(self.trainsetDataRepresentation,macierzTymczasowa)
            self.rowSumsVector = self.trainsetDataRepresentation[0, :]
            # print("self.rowSumsVector: ",self.rowSumsVector)
            self.columnSumsVector = self.trainsetDataRepresentation[:, 0]
        return self.trainsetDataRepresentation, self.rowSumsVector, self.columnSumsVector


##def storeDataInbalanced(self, dids, trainSetTuplesList=[]):
#        for aktualnaKrotka in trainSetTuplesList:
#            macierzTymczasowa=np.zeros((1,1))
#            macierzTymczasowa[0,0]=aktualnaKrotka[0]
#            ksztaltMacierzyTymczasowej=()
#            for i in range(len(dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:])):
#                wektorTymczasowy=np.zeros((dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i+1]+1,1))
#                wektorTymczasowy[0]=1
#                wektorTymczasowy[dids.listaSlownikowIndeksowania[i+1][aktualnaKrotka[i+1]]]=1
#                # w wersji z "distributed representation":
#                # wektorTymczasowy=dids.listaSlownikowIndeksowania[i+1][aktualnaKrotka[i+1]]
#                ksztaltMacierzyTymczasowej=ksztaltMacierzyTymczasowej+(dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i+1]+1,)
#
#                macierzTymczasowa=(np.outer(macierzTymczasowa, wektorTymczasowy)).reshape(ksztaltMacierzyTymczasowej)
#            self.trainsetDataRepresentation=np.add(self.trainsetDataRepresentation,macierzTymczasowa)
#
#        return self.trainsetDataRepresentation
#




## def storeSingleTuple(self, dids, trainSetTuple):
##           ...........trainSetTuple[0]
#           potrzebny dostep do self..........
#           tempQueryTuple=trainSetTuple[1:]
#           tempQueryResult=.....getQueryFloatResult(tempQueryTuple) .....potrzebna tu funkcja zwracajaca wartosc zblizona do sredniej a nie sumy
#           correctionValue=trainSetTuple[0]-tempQueryResult
#            macierzTymczasowa=np.zeros((1,1))
#            macierzTymczasowa[0,0]=aktualnaKrotka[0]
## ... a w wersji z wyzerowywana srednia:
##            macierzTymczasowaForTrueTuples[0,0]=1/numberOfTrueTuples
##            macierzTymczasowaForFalseTuples[0,0]=1/numberOfFalseTuples
#
## ..................
#            ksztaltMacierzyTymczasowej=()
#            for i in range(len(dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:])):
#                wektorTymczasowy=np.zeros((dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i+1]+1,1))
#                wektorTymczasowy[0]=1
#                wektorTymczasowy[dids.listaSlownikowIndeksowania[i+1][aktualnaKrotka[i+1]]]=1
#                # w wersji z "distributed representation":
#                # wektorTymczasowy=dids.listaSlownikowIndeksowania[i+1][aktualnaKrotka[i+1]]
#                ksztaltMacierzyTymczasowej=ksztaltMacierzyTymczasowej+(dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i+1]+1,)
#
#                macierzTymczasowa=(np.outer(macierzTymczasowa, wektorTymczasowy)).reshape(ksztaltMacierzyTymczasowej)
#            self.trainsetDataRepresentation=np.add(self.trainsetDataRepresentation,macierzTymczasowa)
## ... a w wersji z wyzerowywana srednia:
##            self.trainsetDataRepresentation=np.add(self.trainsetDataRepresentation,macierzTymczasowa)
#
#        return self.trainsetDataRepresentation
#


class testSetObject:
    def __init__(self, testSetTuplesList):
        self.sciagaDlaWszechwiedzacego = {}
        self.queryTuplesList = []
        for currentTestTuple in testSetTuplesList:
            currentTestTupleAsList = list(currentTestTuple)
            whitenedCurrentTestTuple = currentTestTupleAsList
            # 'wybielamy' wartosc logiczna krotki testowej:
            whitenedCurrentTestTuple[0] = 0
            # jak nie kijem go to palka ;-) :
            whitenedCurrentTestTuple = tuple(whitenedCurrentTestTuple[1:])
            self.sciagaDlaWszechwiedzacego[whitenedCurrentTestTuple] = currentTestTuple[0]
            self.queryTuplesList.append(whitenedCurrentTestTuple)


class recSystem:
    def __init__(self, dids1, inputArray, systemType):
        self.typeOfSystem = systemType
        self.trainsetDataRepresentation = inputArray
        #   print(inputArray)
        self.dids1 = dids1
        self.numberOfDimensionsToBeLeft = min(
            dids1.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:])
        self.inputDataProcessed = False
        self.sciagaDlaWszechwiedzacego = {}

    def processInputArray(self):
        k = self.numberOfDimensionsToBeLeft
        if (self.typeOfSystem == 3):
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.rowAveragesVector = self.rowSumsVector / len(self.rowSumsVector)
            self.columnAveragesVector = self.columnSumsVector / len(self.columnSumsVector)
            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.processedDataRepresentation = self.columnAveragesMatrix + self.rowAveragesMatrix
        if (self.typeOfSystem == 2):
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)
            # k-=1
            #            #print("k: ",k)
            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]
            # self.Sk=np.zeros(k)
            # self.Sk=self.S[:k]
            self.Sk = np.diag(self.S[:k])
            #            #print("k: ",k)
            #            #print("Sk.shape: ",self.Sk.shape)
            #            #print("VTk.shape: ",self.VTk.shape)
            #            #print("k: ",k)

            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

        if (self.typeOfSystem == 4):
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.overallSumScalar = self.trainsetDataRepresentation[0, 0]
            self.rowAveragesVector = self.rowSumsVector / len(self.rowSumsVector)
            self.columnAveragesVector = self.columnSumsVector / len(self.columnSumsVector)
            self.overallAverageScalar = self.overallSumScalar / (len(self.rowSumsVector) * len(self.columnSumsVector))
            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.averagesBasedDataRepresentation = (
                                                   self.columnAveragesMatrix + self.rowAveragesMatrix) - self.overallAverageScalar

            # TYMCZASOWO:
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:] - self.averagesBasedDataRepresentation

            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)  # full_matrices=False?
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)
            # k-=1
            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]
            #            self.Sk=np.zeros(self.numberOfCoreDataRepresentationComponents)
            self.Sk = np.diag(self.S[:k])
            #            self.Sk=np.diag(self.Sk)
            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

            # TYMCZASOWO:
            self.processedDataRepresentation = self.processedDataRepresentation + self.averagesBasedDataRepresentation
            # self.processedDataRepresentation=self.averagesBasedDataRepresentation
            # self.processedDataRepresentation=self.processedDataRepresentation

        #            self.rowSumsVector=self.trainsetDataRepresentation[1:,0]
        #            self.columnSumsVector=self.trainsetDataRepresentation[0,1:]
        #            self.rowAveragesVector=self.rowSumsVector/len(self.rowSumsVector)
        #            self.columnAveragesVector=self.columnSumsVector/len(self.columnSumsVector)
        #            self.columnAveragesMatrix=np.outer(np.ones(len(self.rowSumsVector)),self.columnAveragesVector)
        #            self.rowAveragesMatrix=np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
        #            self.averagesDataRepresentation=self.columnAveragesMatrix+self.rowAveragesMatrix+self.wholematrixAverage

        #            self.coreDataRepresentation=self.trainsetDataRepresentation[1:,1:]
        #            self.numberOfCoreDataRepresentationComponents=min(self.coreDataRepresentation.shape)
        #            self.U,self.S,self.VT=np.linalg.svd(self.coreDataRepresentation)
        #            self.Uk=self.U[:self.numberOfCoreDataRepresentationComponents,:]
        #            self.VTk=self.VT[:self.numberOfCoreDataRepresentationComponents,:]
        #            self.Sk=np.zeros(self.numberOfCoreDataRepresentationComponents)
        #            self.Sk[:k]=self.S[:k]
        #            self.Sk=np.diag(self.Sk)
        #            self.processedDataRepresentation_=np.dot(self.Uk,self.Sk)
        #            self.processedDataRepresentation=np.dot(self.processedDataRepresentation_,(self.VTk))


        if (self.typeOfSystem == 5):
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.rowNonZeroElements = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowNonZeroElements)):
                self.rowNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[i, :])

            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.columnNonZeroElements = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnNonZeroElements)):
                self.columnNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[:, i])

            self.nonzeroElements = np.count_nonzero(self.coreDataRepresentation[:, :])
            self.overallSumScalar = self.trainsetDataRepresentation[0, 0]

            #    self.rowAveragesVector=self.rowSumsVector/len(self.rowSumsVector)
            self.rowAveragesVector = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowSumsVector)):
                self.rowAveragesVector[i] = self.rowSumsVector[i] / (self.rowNonZeroElements[i])
                #    print(self.rowAveragesVector[i])

                #    self.columnAveragesVector=self.columnSumsVector/len(self.columnSumsVector)
            self.columnAveragesVector = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnSumsVector)):
                self.columnAveragesVector[i] = self.columnSumsVector[i] / (self.columnNonZeroElements[i])
                #    print(self.columnAveragesVector[i])

                #     self.overallAverageScalar=self.overallSumScalar/(len(self.rowSumsVector)*len(self.columnSumsVector))
            self.overallAverageScalar = self.overallSumScalar / (self.nonzeroElements)

            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.averagesBasedDataRepresentation = (
                                                   self.columnAveragesMatrix + self.rowAveragesMatrix) - self.overallAverageScalar

            # TYMCZASOWO:
            for i in range(len(self.rowSumsVector)):
                for j in range(len(self.columnSumsVector)):
                    if self.trainsetDataRepresentation[i + 1, j + 1] != 0:
                        self.coreDataRepresentation[i, j] = self.trainsetDataRepresentation[i + 1, j + 1] - \
                                                            self.rowAveragesVector[i] - self.columnAveragesVector[
                                                                j] + self.overallAverageScalar

            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)  # full_matrices=False?
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)
            # k-=1
            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]
            #            self.Sk=np.zeros(self.numberOfCoreDataRepresentationComponents)
            self.Sk = np.diag(self.S[:k])
            #            self.Sk=np.diag(self.Sk)
            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

            # TYMCZASOWO:
            self.processedDataRepresentation = self.processedDataRepresentation + self.averagesBasedDataRepresentation
            # self.processedDataRepresentation=self.averagesBasedDataRepresentation
            # self.processedDataRepresentation=self.processedDataRepresentation

        #            self.rowSumsVector=self.trainsetDataRepresentation[1:,0]
        #            self.columnSumsVector=self.trainsetDataRepresentation[0,1:]
        #            self.rowAveragesVector=self.rowSumsVector/len(self.rowSumsVector)
        #            self.columnAveragesVector=self.columnSumsVector/len(self.columnSumsVector)
        #            self.columnAveragesMatrix=np.outer(np.ones(len(self.rowSumsVector)),self.columnAveragesVector)
        #            self.rowAveragesMatrix=np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
        #            self.averagesDataRepresentation=self.columnAveragesMatrix+self.rowAveragesMatrix+self.wholematrixAverage

        #            self.coreDataRepresentation=self.trainsetDataRepresentation[1:,1:]
        #            self.numberOfCoreDataRepresentationComponents=min(self.coreDataRepresentation.shape)
        #            self.U,self.S,self.VT=np.linalg.svd(self.coreDataRepresentation)
        #            self.Uk=self.U[:self.numberOfCoreDataRepresentationComponents,:]
        #            self.VTk=self.VT[:self.numberOfCoreDataRepresentationComponents,:]
        #            self.Sk=np.zeros(self.numberOfCoreDataRepresentationComponents)
        #            self.Sk[:k]=self.S[:k]
        #            self.Sk=np.diag(self.Sk)
        #            self.processedDataRepresentation_=np.dot(self.Uk,self.Sk)
        #            self.processedDataRepresentation=np.dot(self.processedDataRepresentation_,(self.VTk))
        if (self.typeOfSystem == 6):
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.rowNonZeroElements = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowNonZeroElements)):
                self.rowNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[i, :])

            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.columnNonZeroElements = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnNonZeroElements)):
                self.columnNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[:, i])

            self.nonzeroElements = np.count_nonzero(self.coreDataRepresentation[:, :])
            self.overallSumScalar = self.trainsetDataRepresentation[0, 0]

            #    self.rowAveragesVector=self.rowSumsVector/len(self.rowSumsVector)
            self.rowAveragesVector = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowSumsVector)):
                self.rowAveragesVector[i] = self.rowSumsVector[i] / (self.rowNonZeroElements[i])
                #    print(self.rowAveragesVector[i])

                #    self.columnAveragesVector=self.columnSumsVector/len(self.columnSumsVector)
            self.columnAveragesVector = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnSumsVector)):
                self.columnAveragesVector[i] = self.columnSumsVector[i] / (self.columnNonZeroElements[i])
                #    print(self.columnAveragesVector[i])

                #     self.overallAverageScalar=self.overallSumScalar/(len(self.rowSumsVector)*len(self.columnSumsVector))
            self.overallAverageScalar = self.overallSumScalar / (self.nonzeroElements)

            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.averagesBasedDataRepresentation = (
                                                   self.columnAveragesMatrix + self.rowAveragesMatrix) - self.overallAverageScalar

            # TYMCZASOWO:
            for i in range(len(self.rowSumsVector)):
                for j in range(len(self.columnSumsVector)):
                    if self.trainsetDataRepresentation[i + 1, j + 1] != 0:
                        self.coreDataRepresentation[i, j] = self.trainsetDataRepresentation[i + 1, j + 1] - \
                                                            self.rowAveragesVector[i] - self.columnAveragesVector[
                                                                j] + self.overallAverageScalar

            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)  # full_matrices=False?
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)
            # k-=1
            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]
            #            self.Sk=np.zeros(self.numberOfCoreDataRepresentationComponents)
            self.Sk = np.diag(self.S[:k])
            #            self.Sk=np.diag(self.Sk)
            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

            # TYMCZASOWO:
            # self.processedDataRepresentation=self.processedDataRepresentation+self.averagesBasedDataRepresentation
            self.processedDataRepresentation = self.averagesBasedDataRepresentation
            # self.processedDataRepresentation=self.processedDataRepresentation

        self.inputDataProcessed = True

    #        return self.processedDataRepresentation

    def getQueryFloatResult(self, queryTuple):
        #        queryResult={}
        # przypadek systemu najgorszego z mozliwych:
        if (self.typeOfSystem == 0):
            #            #tu widac korzysc z tego, ze typ queryTuple jest "hashable":
            #            queryResult[queryTuple]=random.random()
            queryResult = random.random()

            # przypadek idealnego systemu:
        if (self.typeOfSystem == 1):
            ##print(queryTuple)
            #            queryResult[queryTuple]=self.sciagaDlaWszechwiedzacego[tuple(queryTuple)]
            #            if queryTuple in set(self.sciagaDlaWszechwiedzacego):
            queryResult = self.sciagaDlaWszechwiedzacego[queryTuple]
        # else:
        #                queryResult=0
        ##print('queryResult: ', queryResult)
        if (self.typeOfSystem == 2) or (self.typeOfSystem == 3) or (self.typeOfSystem == 4) or (
            self.typeOfSystem == 5) or (self.typeOfSystem == 6):
            #            #print(self.dids.listaSlownikowIndeksowania[1][queryTuple[0]],self.dids.listaSlownikowIndeksowania[2][queryTuple[1]])
            #            #print(self.processedDataRepresentation[self.dids.listaSlownikowIndeksowania[1][queryTuple[0]]-1,self.dids.listaSlownikowIndeksowania[2][queryTuple[1]]-1])
            if (queryTuple[0] in self.dids1.listaSlownikowIndeksowania[1].keys()) and (
                queryTuple[1] in self.dids1.listaSlownikowIndeksowania[2].keys()):
                queryResult = self.processedDataRepresentation[
                    self.dids1.listaSlownikowIndeksowania[1][queryTuple[0]] - 1,
                    self.dids1.listaSlownikowIndeksowania[2][queryTuple[1]] - 1]
            else:
                queryResult = 0
                #                if not(queryTuple[0] in self.dids1.listaSlownikowIndeksowania[1].keys()):
                #                    #print('a new user case!  ',end="",flush=True)
                #                if not(queryTuple[1] in self.dids1.listaSlownikowIndeksowania[2].keys()):
                #                    #print('a new item case!  ',end="",flush=True)
        return queryResult

    def getMultiQueryFloatResults(self, queryTuplesList):
        if not (self.inputDataProcessed):
            self.processInputArray()
        multiQueryFloatResults = {}
        for tempQueryTuple in queryTuplesList:
            #            #print('tempQueryTuple: ',tempQueryTuple)
            multiQueryFloatResults[tempQueryTuple] = self.getQueryFloatResult(tempQueryTuple)
        return multiQueryFloatResults

    def spoilResults(self, sciaga):
        self.sciagaDlaWszechwiedzacego = sciaga


class MOPCARecSystem:
    def __init__(self, dids1, trainSet, systemType, defaultNumberOfDimensions):
        self.typeOfSystem = systemType
        self.trainSet = trainSet
        self.dids1 = dids1
        self.numberOfDimensionsToBeLeft = min(
            dids1.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:])
        self.numberOfModes = 2
        self.inputDataProcessed = False
        self.samplingDimensionalityVector = []
        self.defaultNumberOfDimensions = defaultNumberOfDimensions
        for i in range(self.numberOfModes):
            self.samplingDimensionalityVector.append(self.defaultNumberOfDimensions)
            # self.aMOPCASystem=MOPCA.MOPCASystem(self.trainSet,self.samplingDimensionalityVector)

    def getQueryFloatResult(self, queryTuple):
        if (self.typeOfSystem == 10):
            if (queryTuple[0] in self.dids1.listaSlownikowIndeksowania[1].keys()) and (
                queryTuple[1] in self.dids1.listaSlownikowIndeksowania[2].keys()):
                queryResult = self.aMOPCASystem.query(queryTuple)
            else:
                queryResult = 0
                #                if not(queryTuple[0] in self.dids1.listaSlownikowIndeksowania[1].keys()):
                #                    #print('a new user case!  ',end="",flush=True)
                #                if not(queryTuple[1] in self.dids1.listaSlownikowIndeksowania[2].keys()):
                #                    #print('a new item case!  ',end="",flush=True)
        return queryResult

    def getMultiQueryFloatResults(self, queryTuplesList):
        if not (self.inputDataProcessed):

            numberOfModes = len(inTuples[0]) - 1
            #      defaultNumberOfDimensions=10
            #      samplingDimensionalityVector=[]
            #      for i in range(numberOfModes):
            #          samplingDimensionalityVector.append(defaultNumberOfDimensions)

            aManager = Manager()
            YAMFDict = aManager.dict()
            monotensorOutputMicroTuples = aManager.dict()
            kolejka = aManager.Queue()
            DIDsDict = aManager.dict()
            kolejkaWS = aManager.Queue()
            DIDsQueue = aManager.Queue()
            # deathQueue=aManager.Queue()

            # tu zaglada co sekunde do kolejkaWS
            # a jak juz dostanie komunikat, ze sa juz krotki lub jesli dostanie krotki przez kolejke...
            # to moze utworzyc aMOPCASystem:
            self.aMOPCASystem = MOPCA.MOPCASystem(trainSet, self.samplingDimensionalityVector, kolejka, DIDsQueue,
                                                  monotensorOutputMicroTuples, aManager, YAMFDict, DIDsDict)
            # potem raczej: aMOPCASystem=MOPCASystem(inTuples,samplingDimensionalityVector,instabilityThreshold,maxNumberOfSHOOICycles,maxNumberOfSHOOICycles,maxNumberOfPerModeRegularizationCycles)
            self.aMOPCASystem.preprocessData()
            # print("uwaga.............")
            self.aMOPCASystem.processData()

            while not (self.aMOPCASystem.inputDataProcessed):
                time.sleep(0.1)
                # print("tutaj przysypiam")

        multiQueryFloatResults = {}
        for tempQueryTuple in queryTuplesList:
            # print('tempQueryTuple: ',tempQueryTuple)
            multiQueryFloatResults[tempQueryTuple] = self.getQueryFloatResult(tempQueryTuple)
            # print("MOPCARecSystem:multiQueryFloatResults: ",multiQueryFloatResults)
        return multiQueryFloatResults


#
# def getQueryFloatResult(recSysNumber, queryTuple):
#    queryResult={}
#    #przypadek systemu najgorszego z mozliwych:
#    if (recSysNumber==0):
#        #tu widac korzysc z tego, ze typ queryTuple jest "hashable":
#        queryResult[queryTuple]=random.random()
#
#    #przypadek idealnego systemu:
#    if (recSysNumber==1):
#        ##print(queryTuple)
#        queryResult[queryTuple]=sciagaDlaWszechwiedzacego[tuple(queryTuple)]
#    ##print('queryResult: ', queryResult)
#    return queryResult
#
# def getMultiQueryFloatResults(recSysNumber, queryTuplesList):
#    multiQueryFloatResults={}
#    for tempQueryTuple in queryTuplesList:
#        multiQueryFloatResults[tempQueryTuple]=getQueryFloatResult(recSysNumber, tempQueryTuple)[tempQueryTuple]
#    return multiQueryFloatResults

def getMultiQueryBinaryResults(recSys, tso, numberOfThresholdSteps):
    # ponizej implementacja prostego algorytmu
    # w numberOfThresholdSteps-krokach
    # zliczajacego licznosci 'positives' - z uwzglednieniem kolejnych wartosci zmiennej intermediateTargetThreshold
    # - determinujacej dokladnosc dzialania algorytmu i wynikajacej z wartosci numberOfTargetPositive2AllRatioSteps
    # oraz minimalnej i maksymalnej wartosci threshold.
    # Kazda kolejna wartosc progowa intermediateTargetThreshold jest wielokrotnoscia parametru thresholdStepSize
    # - bedacego ilorazem roznicy miedzy minimalna (minThreshold) a maksymalna (maxThreshold) wartoscia threshold -
    # oraz liczby numberOfThresholdSteps powiekszonym o wartosc minThreshold.
    # Z zalozenia parametr thresholdStepSize spelnia warunek 0<=thresholdStepSize<=1,
    # podobnie jak parametr intermediateTargetThreshold zawsze spelnia warunek 0<=intermediateTargetPositive2AllRatio<=1.

    multiQueryFloatResults = recSys.getMultiQueryFloatResults(tso.queryTuplesList)
    # print('multiQueryFloatResults: ',multiQueryFloatResults)
    numberOfQueryTuples = len(tso.queryTuplesList)
    minThreshold = min(list(multiQueryFloatResults.values()))
    maxThreshold = max(list(multiQueryFloatResults.values()))
    #    if recSys.typeOfSystem==1:
    # print('minThreshold: ',minThreshold)
    # print('maxThreshold: ',maxThreshold)
    targetPositive2AllRatioStep = (maxThreshold - minThreshold) / (numberOfThresholdSteps + 2)
    listOfmultiQueryBinaryResults = []
    for intermediateTargetPositive2AllRatioStepNumber in range(numberOfThresholdSteps):
        threshold = ((intermediateTargetPositive2AllRatioStepNumber + 1) * targetPositive2AllRatioStep) + minThreshold
        #        #print('threshold: ',threshold)
        multiQueryBinaryResults = []
        multiQueryFloatResults_ = []
        numberOfPositiveBinResults = 0
        for tempQueryTuple in tso.queryTuplesList:
            tempQueryFloatResult = multiQueryFloatResults[tempQueryTuple]
            #            tempQueryFloatResult=getQueryFloatResult(recSysNumber, tempQueryTuple)
            #            #print('tempQueryFloatResult: ',tempQueryFloatResult)
            #            #print('threshold: ', threshold)

            tempQueryBinResult = -1
            ##print('tempQueryFloatResult: ',tempQueryFloatResult)
            if (tempQueryFloatResult >= threshold):
                tempQueryBinResult = 1
                numberOfPositiveBinResults += 1

            ##print('tempQueryBinResult: ', tempQueryBinResult)

            multiQueryBinaryResults.append(tuple([tempQueryBinResult]) + tuple(tempQueryTuple))
            multiQueryFloatResults_.append(tuple([tempQueryFloatResult]) + tuple(tempQueryTuple))
        # uwaga na brak "+1" w rekomendacjach, bo to uniemozliwia wyznaczenie precision:
        #       czyli nie opuszczac nigdy petli while jesli brak "+1" w rekomendacjach
        ##print('numberOfPositiveBinResults: ', numberOfPositiveBinResults)
        ##print('targetPositive2AllRatio*numberOfQueryTuples: ',targetPositive2AllRatio*numberOfQueryTuples)
        #        #print('targetPositive2AllRatio: ',targetPositive2AllRatio)
        #        oldNumberOfPositiveBinResults=numberOfPositiveBinResults
        if numberOfPositiveBinResults > 0:
            listOfmultiQueryBinaryResults.append(multiQueryBinaryResults)
        else:
            print('cosik nie teges....')

    return listOfmultiQueryBinaryResults, multiQueryFloatResults


def getPrecisionVsRecallPointsForASingleCurve(multiQueryBinaryResults):
    recallValues = []
    precisionValues = []
    for currentPvsRCurvePoint in range(numberOfPvsRCurvePoints):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        # print "multiQueryBinaryResults: ",multiQueryBinaryResults
        for currentMultiQueryResultTupleNumber in range(len(multiQueryBinaryResults[currentPvsRCurvePoint])):
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == 1) and (
                testSet[currentMultiQueryResultTupleNumber][0] == 1)):
                TP += 1
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == -1) and (
                testSet[currentMultiQueryResultTupleNumber][0] == -1)):
                TN += 1
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == 1) and (
                testSet[currentMultiQueryResultTupleNumber][0] == -1)):
                FP += 1
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == -1) and (
                testSet[currentMultiQueryResultTupleNumber][0] == 1)):
                FN += 1
        if ((TP + FP) == 0):
            print("(TP+FP)==0")
            print(TP)
            print(FP)
        if ((TP + FN) == 0):
            print("(TP+FN)==0")
            print(TP)
            print(FN)
        precisionValues.append(float(TP) / (TP + FP))
        recallValues.append(float(TP) / (TP + FN))
    return precisionValues, recallValues


def altGetPrecisionVsRecallPointsForASingleCurve(multiqueryFloatResults, testSet, numberOfPointsPerCurve=0):
    multiqueryFloatResultsAsList = []
    altPrecisionPoints = []
    altRecallPoints = []

    for tempMultiqueryFloatResult in multiqueryFloatResults:
        #        tempMultiqueryFloatResultsAsListElement=list(tempMultiqueryFloatResult)
        tempMultiqueryFloatResultsAsListElement = []
        tempMultiqueryFloatResultsAsListElement.append(multiqueryFloatResults[tempMultiqueryFloatResult])
        tempMultiqueryFloatResultsAsListElement.append(tempMultiqueryFloatResult)
        # print "tempMultiqueryFloatResultsAsListElement: ",tempMultiqueryFloatResultsAsListElement
        multiqueryFloatResultsAsList.append(tempMultiqueryFloatResultsAsListElement)
    multiqueryFloatResultsAsListSorted = sorted(multiqueryFloatResultsAsList, key=lambda x: x[0], reverse=True)
    # print "multiqueryFloatResultsAsListSorted: ",multiqueryFloatResultsAsListSorted
    # testSetSorted=sorted(testSet, key=lambda x: x[0], reverse=True)
    # print "testSetSorted:",testSetSorted
    # testSetSelected=[]
    testSetSelected = set()
    testSetLen = len(testSet)
    for tempTestSetSortedElement in testSet:
        if (tempTestSetSortedElement[0] == 1):
            testSetSelected.add((tempTestSetSortedElement[1:]))
            # testSetSelected.append(tempTestSetSortedElement[1:])
    totalNumberOfHits = len(testSetSelected)
    if numberOfPointsPerCurve == 0:
        numberOfPointsPerCurve = totalNumberOfHits
    curveGrain = int(float(totalNumberOfHits) / numberOfPointsPerCurve)
    tempNumberOfHits = 0
    tempNumberOfShots = 0
    # print"_________________________"
    curveGrainCounter = 0
    for tempMultiqueryFloatResultsAsListSortedElement in multiqueryFloatResultsAsListSorted:
        #    for tempMultiqueryFloatResultsAsListSortedElement in multiqueryFloatResultsAsList:
        # print"tempMultiqueryFloatResultsAsListSortedElement[1]: ",tempMultiqueryFloatResultsAsListSortedElement[1]
        # print"testSetSelected: ",testSetSelected
        tempNumberOfShots += 1
        if tempMultiqueryFloatResultsAsListSortedElement[1] in testSetSelected:
            tempNumberOfHits += 1
            # print"tempNumberOfHits: ",tempNumberOfHits

            # print"tempNumberOfShots: ",tempNumberOfShots
            tempAltPrecisionPointsValue = float(tempNumberOfHits) / tempNumberOfShots
            # print"tempAltPrecisionPointsValue: ",tempAltPrecisionPointsValue
            curveGrainCounter = (curveGrainCounter + 1) % curveGrain
            if curveGrainCounter == 0:
                altPrecisionPoints.append(tempAltPrecisionPointsValue)
                # print"##########"
                altRecallPoints.append(float(tempNumberOfHits) / totalNumberOfHits)
    # pass
    return altPrecisionPoints, altRecallPoints


def getTPVsPPointsForASingleCurve(multiQueryBinaryResults):
    TPValues = []
    PValues = []
    numberOfCurvePoints = len(multiQueryBinaryResults)
    for curvePoint in range(numberOfCurvePoints):
        TP = 0
        P = 0
        for currentMultiQueryResultTupleNumber in range(len(multiQueryBinaryResults[curvePoint])):
            if ((multiQueryBinaryResults[curvePoint][currentMultiQueryResultTupleNumber][0] == 1) and (
                testSet[currentMultiQueryResultTupleNumber][0] == 1)):
                TP += 1
            if (multiQueryBinaryResults[curvePoint][currentMultiQueryResultTupleNumber][0] == 1):
                P += 1
        TPValues.append(TP)
        PValues.append(P)
    return TPValues, PValues


def makeSpectrumFigure(spctr, numberOfComponents, fileName):
    if (numberOfComponents == 0):
        numberOfComponents = len(list(spctr[0][0]))
    plt.ioff()
    plt.figure(figsize=(20, 10), linewidth=0.1)
    for row in range(len(spctr)):
        for column in range(len(spctr[row])):
            trainingRatio = round((column + 1) * trainingRatioStep, 10)
            plt.subplot(len(spctr), len(spctr[0]), len(spctr[0]) * row + column + 1)
            plt.plot(list(spctr[row][column])[:numberOfComponents])
            axes = plt.gca()
            axes.spines['left'].set_linewidth(0.2)
            axes.spines['right'].set_linewidth(0.2)
            axes.spines['bottom'].set_linewidth(0.2)
            axes.spines['top'].set_linewidth(0.2)
            title = "randomTrainAndTestSetSplitCase#" + str(row) + ", trainingRatio=" + str(round(trainingRatio, 2))
            plt.title(title, fontsize=6, y=1.03)
            leg.get_frame().set_linewidth(0.05)
    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 6})
    plt.savefig(fileName + ".png", format="png", dpi=100)
    plt.clf()


def makeCurveFigure(horizontalValues, verticalValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps,
                    numberOfRecSystems, recSystemsLabels, horizontalAxisLabel, verticalAxisLabel, fileName,
                    legendLocation):
    plt.ioff()
    plt.figure(figsize=(20, 10), linewidth=0.1)
    for randomTrainAndTestSetSplitCaseNumber in range(numberOfRandomTrainAndTestSetSplitCases):
        for currentTRStepNumber in range(numberOfTRSteps):
            trainingRatio = round((currentTRStepNumber + 1) * trainingRatioStep, 10)
            plt.subplot(numberOfRandomTrainAndTestSetSplitCases + 0, numberOfTRSteps + 0,
                        currentTRStepNumber + 1 + ((numberOfTRSteps + 0) * (randomTrainAndTestSetSplitCaseNumber + 0)))
            for systemNumber in range(numberOfRecSystems):
                # print "horizontalValues[randomTrainAndTestSetSplitCaseNumber]: ",horizontalValues[randomTrainAndTestSetSplitCaseNumber]
                plt.plot(horizontalValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         verticalValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         marker='o', markersize=1, label=recSystemsLabels[systemNumber])
                axes = plt.gca()
                axes.spines['left'].set_linewidth(0.2)
                axes.spines['right'].set_linewidth(0.2)
                axes.spines['bottom'].set_linewidth(0.2)
                axes.spines['top'].set_linewidth(0.2)
                #                axes.set_xlim([0,1])
                #                axes.set_ylim([0,1])
                title = "randomTrainAndTestSetSplitCase#" + str(
                    randomTrainAndTestSetSplitCaseNumber) + ", trainingRatio=" + str(round(trainingRatio, 2))
                #            plt.title(title)
                plt.title(title, fontsize=6, y=1.03)
                leg = plt.legend(loc=legendLocation, fontsize=6)
                leg.get_frame().set_linewidth(0.05)
                #            plt.xlabel(horizontalAxisLabel)
                #            plt.ylabel(verticalAxisLabel)
                plt.xlabel(horizontalAxisLabel, fontsize=6)
                plt.ylabel(verticalAxisLabel, fontsize=6)
                # plt.plot()

                # plt.plot()
    # plt.figlegend(loc=3)
    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 6})
    # matplotlib.rcParams.update({'edge': 9})
    # matplotlib.rcParams.update({'axes.linewidth': 0.01})
    # leg=plt.legend()
    # for legobj in leg.legendHandles:
    #    legobj.set_alpha(1)
    plt.savefig(fileName + ".png", format="png", dpi=400)
    plt.clf()


if __name__ == '__main__':

    # wspolczynnikRedukcjiWielkosciZbioruDanych=0.02
    # wspolczynnikRedukcjiWielkosciZbioruDanych=0.1
    wspolczynnikRedukcjiWielkosciZbioruDanych = 0.3
    # wspolczynnikRedukcjiWielkosciZbioruDanych=0.5

    SubUDataSet = getSubUDataSet(wspolczynnikRedukcjiWielkosciZbioruDanych, 1)
    # sprawdzic dla innych wartosci progu niz 3!
    inTuples = convertSubUDataSetToInTuplesList(SubUDataSet, 4)

    dataSetSize = len(inTuples)

    #
    precisionValues = []

    altPrecisionValues = []
    altRecallValues = []

    TPValues = []
    recallValues = []
    PValues = []
    TrainSets = []
    TestSets = []
    TestResults = []
    TestResults2 = []

    #    numberOfRecSystems=3
    #    numberOfRecSystems=8
    numberOfRecSystems = 7

    numberOfTRSteps = 4
    trainingRatioStep = 1.0 / (numberOfTRSteps + 1)

    spectrum = []
    sortedRowSums = []
    sortedColumnSums = []

    numberOfRandomTrainAndTestSetSplitCases = 3

    random.shuffle(inTuples)
    random.shuffle(inTuples)
    random.shuffle(inTuples)
    random.shuffle(inTuples)

    # fileForInTuples=open("Z:/fileForInTuples",'w')
    fileForInTuples = open("fileForInTuples", 'w')
    fileForInTuples.write(str(inTuples))
    fileForInTuples.close()

    for randomTrainAndTestSetSplitCaseNumber in range(numberOfRandomTrainAndTestSetSplitCases):
        random.shuffle(inTuples)
        spectrum.append([])
        sortedRowSums.append([])
        sortedColumnSums.append([])
        #    #print()
        # print('randomTrainAndTestSetSplitCaseNumber: ',randomTrainAndTestSetSplitCaseNumber)
        precisionValues.append([])
        TPValues.append([])
        recallValues.append([])

        altPrecisionValues.append([])
        altRecallValues.append([])

        PValues.append([])
        TrainSets.append([])
        TestSets.append([])
        TestResults.append([])
        TestResults2.append([])

        for currentTRStepNumber in range(numberOfTRSteps):
            random.shuffle(inTuples)
            trainingRatio = round((currentTRStepNumber + 1) * trainingRatioStep, 10)
            #        #print()
            print
            'currentTRStepNumber: ', currentTRStepNumber
            # print 'trainingRatioStep: ',trainingRatioStep
            # print 'trainingRatio: ',trainingRatio
            precisionValues[-1].append([])
            TPValues[-1].append([])
            recallValues[-1].append([])

            altPrecisionValues[-1].append([])
            altRecallValues[-1].append([])

            PValues[-1].append([])
            # TrainSets[-1].append([])
            # TestSets[-1].append([])
            TestResults[-1].append([])
            TestResults2[-1].append([])

            trainSetSize = int(trainingRatio * dataSetSize)
            print
            "trainSetSize: ", trainSetSize
            # uwaga na brak "+1" w testsecie, bo to uniemozliwia wyznaczenie recall
            # czyli losowac tak dlugo, az w testsecie bedzie chociaz jedna "+1".
            # ponadto uwaga na brak "-1" w testsecie, bo to uniemozliwia wyznaczenie precision:
            # w przypadku idealnego algorytmu wszystkie multiQueryFloatResults sa rowne 1,
            # co po progowaniu objawia sie tym, ze wszystkie multiQueryBinaryResults sa rowne -1,
            # a to z kolei oznacza brak "+1" w rekomendacjach, czyli brak positives, tj. TP=FP=0,
            # co z kolei uniemozliwia wyznaczenie precision.
            numberOfPositivesInTestset = 0
            numberOfNegativesInTestset = 0
            while ((numberOfPositivesInTestset == 0) or (numberOfNegativesInTestset == 0)):
                random.shuffle(inTuples)
                testSet = inTuples[trainSetSize:]
                numberOfPositivesInTestset = 0
                numberOfNegativesInTestset = 0

                for tempTestsetTuple in testSet:
                    if tempTestsetTuple[0] == 1:
                        numberOfPositivesInTestset += 1
                    if tempTestsetTuple[0] == -1:
                        numberOfNegativesInTestset += 1
            trainSet = inTuples[:trainSetSize]
            # TrainSets[-1][-1].append(trainSet)
            TestSets[-1].append(testSet)
            TrainSets[-1].append(trainSet)
            ##print("TrainSets: ",TrainSets)
            # TestSets[-1].append(testSet)


            #            TestResults[-1][-1].append([])

            dids1 = dimensionsIndexingDictionaries(trainSet)
            dids2 = dimensionsIndexingDictionaries(testSet)

            # print("trainSet: ",trainSet)
            abdr1 = arrayBasedDataRepresentation(dids1, trainSet)
            spectrum[-1].append(getMatrixSpectrum(abdr1.trainsetDataRepresentation))
            #            sortedRowSumsVector=sorted(abdr1.rowSumsVector, reverse=True)
            #            sortedColumnSumsVector=sorted(abdr1.columnSumsVector, reverse=True)
            #            sortedRowSums[-1].append(sortedRowSumsVector)
            #            sortedColumnSums[-1].append(sortedColumnSumsVector)

            abdr2 = arrayBasedDataRepresentation(dids2, testSet)

            #        abdr=arrayBasedDataRepresentation(dids,trainSet)
            # rs=recSystem(dids,abdr.trainsetDataRepresentation,2)
            tso = testSetObject(testSet)

            #

            recSystems = []

            # abdr1File=open('z:/abdr1File','w')

            # abdr1File.write(str(abdr1.trainsetDataRepresentation))
            # tempCheck=copy.deepcopy(abdr1.trainsetDataRepresentation)

            rs = recSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 0)
            recSystems.append(rs)

            # abdr1File.write(str(abdr1.trainsetDataRepresentation))

            rs = recSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 1)
            rs.spoilResults(tso.sciagaDlaWszechwiedzacego)
            recSystems.append(rs)

            # abdr1File.write(str(abdr1.trainsetDataRepresentation))

            rs = recSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 2)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)
            #       firstEVectors[-1].append(rs.Uk)

            # abdr1File.write(str(abdr1.trainsetDataRepresentation))

            rs = recSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 3)
            recSystems.append(rs)

            # abdr1File.write(str(abdr1.trainsetDataRepresentation))

            rs = recSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 4)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            rs = recSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 5)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            rs = recSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 6)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            # abdr1File.write(str(abdr1.trainsetDataRepresentation))

            #            defaultNumberOfDimensions=10
            #            rs=MOPCARecSystem(dids1,trainSet,10,defaultNumberOfDimensions)
            #            recSystems.append(rs)

            #   recSystems.append(prepareMultiSvdRS(10,trainSet, True,False))


            # abdr1File.close()

            numberOfPvsRCurvePoints = 20
            for recSysNumber in range(numberOfRecSystems):
                # print('recSysNumber: ',recSysNumber)
                currentRecSystem = recSystems[recSysNumber]
                #            currentRecSystem.provideTrainset(abdr)

                multiQueryBinaryResults, multiqueryFloatResults_ = getMultiQueryBinaryResults(currentRecSystem, tso,
                                                                                              numberOfPvsRCurvePoints)
                precisionPoints, recallPoints = getPrecisionVsRecallPointsForASingleCurve(multiQueryBinaryResults)

                altPrecisionPoints, altRecallPoints = altGetPrecisionVsRecallPointsForASingleCurve(
                    multiqueryFloatResults_, testSet, 20)

                TPPoints, PPoints = getTPVsPPointsForASingleCurve(multiQueryBinaryResults)

                # print "precisionValues: ",precisionValues


                precisionValues[-1][-1].append(precisionPoints)
                # print "precisionPoints: ",precisionPoints
                TPValues[-1][-1].append(TPPoints)
                #                if (recSysNumber==1):
                # print('precisionPoints: ',precisionPoints)
                # print('TPPoints: ',TPPoints)
                # print('recallPoints: ',recallPoints)
                # print('PPoints: ',PPoints)

                # print "recallPoints: ",recallPoints
                recallValues[-1][-1].append(recallPoints)

                # print "precisionPoints: ",precisionPoints
                # print "altPrecisionPoints: ",altPrecisionPoints

                # print "altPrecisionValues: ",altPrecisionValues


                altPrecisionValues[-1][-1].append(altPrecisionPoints)
                altRecallValues[-1][-1].append(altRecallPoints)

                PValues[-1][-1].append(PPoints)
                TestResults[-1][-1].append(multiQueryBinaryResults)
                TestResults2[-1][-1].append(multiqueryFloatResults_)

    recSystemsLabels = []
    recSystemsLabels.append('random')
    recSystemsLabels.append('ideal')
    recSystemsLabels.append('SVD-based')
    recSystemsLabels.append('classical-averages-based')
    recSystemsLabels.append('overallCentring-based_MPCA')
    recSystemsLabels.append('statisticalCentring-based_MPCA with decentring')

    recSystemsLabels.append('statistical-avaraged-based')
    #    recSystemsLabels.append('MOPCADraft_MPCA_aka_HOOI_OC')
    #  recSystemsLabels.append('MPCA-MC')
    #    recSystemsLabels.append('MOPCADraft_averages-based')
    #    recSystemsLabels.append('MOPCADraft_SVD-based')
    #    recSystemsLabels.append('MOPCADraft_random')

    #    recSystemsLabels.append('MOPCADraftSHOOI')


    # wylaczenie interaktywnego okna (na rzecz zapisu wylacznie do pliku):
    plt.ioff()
    plt.figure(figsize=(20, 10), linewidth=0.1)

    for randomTrainAndTestSetSplitCaseNumber in range(numberOfRandomTrainAndTestSetSplitCases):
        for currentTRStepNumber in range(numberOfTRSteps):
            trainingRatio = round((currentTRStepNumber + 1) * trainingRatioStep, 10)
            plt.subplot(numberOfRandomTrainAndTestSetSplitCases + 0, numberOfTRSteps + 0,
                        currentTRStepNumber + 1 + ((numberOfTRSteps + 0) * (randomTrainAndTestSetSplitCaseNumber + 0)))
            for systemNumber in range(numberOfRecSystems):
                plt.plot(recallValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         precisionValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         marker='o', markersize=3, label=recSystemsLabels[systemNumber])
                axes = plt.gca()
                axes.spines['left'].set_linewidth(0.2)
                axes.spines['right'].set_linewidth(0.2)
                axes.spines['bottom'].set_linewidth(0.2)
                axes.spines['top'].set_linewidth(0.2)
                axes.set_xlim([0, 1])
                axes.set_ylim([0, 1])
                title = "randomTrainAndTestSetSplitCase#" + str(
                    randomTrainAndTestSetSplitCaseNumber) + ", trainingRatio=" + str(round(trainingRatio, 2))
                #            plt.title(title)
                plt.title(title, fontsize=6, y=1.03)
                leg = plt.legend(loc=3, fontsize=6)
                leg.get_frame().set_linewidth(0.05)
                #            plt.xlabel("Recall")
                #            plt.ylabel("Precision")
                plt.xlabel("Recall", fontsize=6)
                plt.ylabel("Precision", fontsize=6)
                # plt.plot()

                # plt.plot()
    # plt.figlegend(loc=3)
    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 6})
    # matplotlib.rcParams.update({'edge': 9})
    # matplotlib.rcParams.update({'axes.linewidth': 0.01})
    # leg=plt.legend()
    # for legobj in leg.legendHandles:
    #    legobj.set_alpha(1)
    plt.savefig("wykresPvsR1.png", format="png", dpi=100)
    plt.clf()

    makeCurveFigure(PValues, TPValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps, numberOfRecSystems,
                    recSystemsLabels, 'number of Positives', 'number of True Positives', 'PositivesVsTruePositives', 0)
    PvsTPFigureData = (
    PValues, TPValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps, numberOfRecSystems, recSystemsLabels,
    'number of Positives', 'number of True Positives', 'PositivesVsTruePositives', 0)

    makeCurveFigure(recallValues, precisionValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps,
                    numberOfRecSystems, recSystemsLabels, 'Recall', 'Precision', 'PrecisionVsRecall_2', 0)

    PvsRFigureData = (
    recallValues, precisionValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps, numberOfRecSystems,
    recSystemsLabels, 'Recall', 'Precision', 'PrecisionVsRecall_3', 0)

    makeCurveFigure(altRecallValues, altPrecisionValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps,
                    numberOfRecSystems, recSystemsLabels, 'Recall', 'Precision', 'PrecisionVsRecall_4', 0)

    # legend location values:
    # best -- 0
    # upper right -- 1
    # upper left -- 2
    # lower left -- 3
    # lower right -- 4
    # right -- 5
    # center left -- 6
    # center right -- 7
    # lower center -- 8
    # upper center -- 9
    # center -- 10



    makeSpectrumFigure(spectrum, 0, 'full_trainsets_spectra')
    makeSpectrumFigure(spectrum, 5, 'principal_components_of_trainsets_spectra')

    # makeSpectrumFigure(sortedRowSums,0,'sortedRowSums')
    # makeSpectrumFigure(sortedColumnSums,0,'sortedColumnSums')


    # for row in range(len(spectrum)):
    #    for column in range(len(spectrum[row])):
    #        #print(len(spectrum)*2, len(spectrum[0]), len(spectrum[0])*2*row+column)
    #        #print(len(spectrum)*2, len(spectrum[0]), len(spectrum[0])*1*(row+0)+column)




#    import os
#
#    baseWDPathString=os.getcwd()
#    resultsWDPathString=baseWDPathString+"\\"+str(datetime.datetime.now().strftime('results_%m-%d__%H-%M-%S'))
#    os.mkdir(resultsWDPathString)
#    os.chdir(resultsWDPathString)
#
#    import pickle
#
#    fileWithTrainSets=open("fileWithTrainSets.p", "wb")
#    pickle.dump(TrainSets,fileWithTrainSets)
#    fileWithTrainSets.close()
#
#    fileWithTestSets=open("fileWithTestSets.p", "wb")
#    pickle.dump(TestSets,fileWithTestSets)
#    fileWithTestSets.close()
#
##    fileWithBinaryTestResults=open("fileWithBinaryTestResults.p", "wb")
##    pickle.dump(TestResults,fileWithBinaryTestResults)
##    fileWithBinaryTestResults.close()
#
##    fileWithFloatTestResults=open("fileWithFloatTestResults.p", "wb")
##    pickle.dump(TestResults2,fileWithFloatTestResults)
##    fileWithFloatTestResults.close()
#
#    fileWithPvsRFigureData=open("fileWithPvsRFigureData.p", "wb")
#    pickle.dump(PvsRFigureData,fileWithPvsRFigureData)
#    fileWithPvsRFigureData.close()
#
#    fileWithPvsTPFigureData=open("fileWithPvsTPFigureData.p", "wb")
#    pickle.dump(PvsTPFigureData,fileWithPvsTPFigureData)
#    fileWithPvsTPFigureData.close()
#
#
#
#
#
#    fileWithPvsRFigureData=open("fileWithPvsRFigureData.p", "rb")
#    PvsRFigureDataBis=pickle.load(fileWithPvsRFigureData)
#    fileWithPvsRFigureData.close()
#
#
#    fileWithPvsTPFigureData=open("fileWithPvsTPFigureData.p", "rb")
#    PvsTPFigureDataBis=pickle.load(fileWithPvsTPFigureData)
#    fileWithPvsTPFigureData.close()
#
#    makeCurveFigure(*PvsRFigureDataBis)
#    makeCurveFigure(*PvsTPFigureDataBis)
#
##    fileWithTestSets=open("fileWithTestSets.p", "rb")
##        trainSetFile=open("Z:/trainSetFile","wb")
##        pickle.dump(trainSet,trainSetFile)
##        trainSetFile.close()
#    #inTuples=pickle.load(plikSloikowy)
##    fileWithTestSets.close()
#
#
#
#
#    os.chdir(baseWDPathString)
