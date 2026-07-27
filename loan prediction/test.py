import pickle

with open("loan.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))