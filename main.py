import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
def main():
    data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv').fillna("")
    vectorizer = TfidfVectorizer()
    raw_x = vectorizer.fit_transform(data['Review Text'])
    x_train, x_test, y_train, y_test = train_test_split(
        raw_x, data['Rating'], train_size=.7, random_state=42
    )
    print("beginning to train the model")
    reg = LassoCV().fit(x_train,y_train)
    print("finished training the model")
    reg.predict(x_test)
    for a,b,c in x_test,y_test,funny_ys:
        print(a,b,c)

if __name__ == '__main__':
    main()

