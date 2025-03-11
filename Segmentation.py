from csv import writer
import pandas as pd
import math

def segment(dataset, label, seg_size, overlap):
    myFile = open('trainData_Seg.csv', 'w', newline = '')
    print("Non-overlapping Region: %s" %overlap)
    print("Segment Size: %s" %seg_size)
    with myFile:
        csv_writer = writer(myFile)
        for j, row in enumerate(dataset):
            if(len(row) < 2001):
                pos = math.ceil(len(row)/overlap)
                if(pos < math.ceil(seg_size/overlap)):
                    pos = math.ceil(seg_size/overlap)
                for itr in range(pos - math.ceil(seg_size/overlap) + 1):
                    init = itr * overlap
                    segment = [ ]
                    if(len(row[init : init + seg_size]) > 40):
                        segment.append(row[init : init + seg_size])
                        for item in label[j]:
                            segment.append(item)
                        csv_writer.writerow(segment)
    myFile.close()

def main_fun(segSize, overLap):
  dataframe = pd.read_csv('/content/gdrive/My Drive/Fast_MCWS/Molecular Function/trainData.csv', header=None)
  dataset = dataframe.values
  print('Original Dataset Size : %s' %len(dataset))
  X = dataset[:,0]
  Y = dataset[:,1:len(dataset[0])]
  segment(X, Y, segSize, overLap)
