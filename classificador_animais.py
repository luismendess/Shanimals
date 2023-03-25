import sys
sys.path.append('C:\Python27\Lib\site-packages')
from pyAudioAnalysis import audioTrainTest as aT
import os
import pydub
pydub.AudioSegment.converter = r"C:\ffmpeg\bin"
import numpy as np

testData="testData"
classifierData=[
	"classifierData/cachorro"
	,"classifierData/gato"
	,"classifierData/onca"
]

# Fase de Treinamento
aT.featureAndTrain(classifierData, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

# Fase de Testes
fileout = open("result.txt","w")

for filename in os.listdir(testData):
	filename=testData+"/"+filename
	Result, P, classNames = aT.fileClassification(filename, "svmSMtemp", "svm")
	P = P * 100
	winner = np.argmax(P) #pega o valor com a maior taxa de probabilidade.
	print("Arquivo: " +filename + "; Animal: " + classNames[winner] + "; Probabilidade: " + str(P[winner]) + "%")
	fileout.write("Arquivo: " +filename + "; Animal: " + classNames[winner] + "; Probabilidade: " + str(P[winner]) + "%\n")
fileout.close()
