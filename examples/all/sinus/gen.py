#generates a simple sinus in csvTS form for testing timeserie predicions
import numpy as np
import os

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')

input = data[3:, :-1]
target = data[3:, 1:]  # target is sinus as timestep + 1 wrt input
test_input = data[:3, :-1]
test_target = data[:3, 1:]

if os.path.exists("./train"):
    for f in os.listdir("train"):
        os.remove("train/"+f)
    os.rmdir("train")
if os.path.exists("./test"):
    for f in os.listdir("test"):
        os.remove("test/"+f)
    os.rmdir("test")

if os.path.exists("./predict"):
    for f in os.listdir("predict"):
        os.remove("predict/"+f)
    os.rmdir("predict")


os.mkdir("train")
os.mkdir("test")
os.mkdir("predict")

for i in range(N-3):
    csvdat = np.empty((L-1,2), 'float64')
    csvdat[:,0] = input[i,:]
    csvdat[:,1] = target[i,:]
    filename = "train/seq_" + str(i) + ".csv"
    np.savetxt(filename, csvdat, delimiter=",")
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0,0)
        f.write("input,output\n" + content)

for i in range(2):
    csvdat = np.empty((L-1,2), 'float64')
    csvdat[:,0] = test_input[i,:]
    csvdat[:,1] = test_target[i,:]
    filename = "test/seq_" + str(i) + ".csv"
    np.savetxt(filename, csvdat, delimiter=",")
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0,0)
        f.write("input,output\n" + content)

csvdat = np.empty((L-1,2), 'float64')
csvdat[:,0] = test_input[2,:]
csvdat[:,1] = test_target[2,:]
filename = "predict/seq_" + str(2) + ".csv"
np.savetxt(filename, csvdat,  delimiter=",")
with open(filename, 'r+') as f:
    content = f.read()
    f.seek(0,0)
    f.write("input,output\n" + content)
