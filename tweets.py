
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tdf = pd.read_csv('corona_tweets.csv', index_col=0)
from sklearn.feature_extraction.text import CountVectorizer
cvect = CountVectorizer(stop_words='english')
dtm = cvect.fit_transform(tdf['tweets'])
dtm_df = pd.DataFrame(dtm.toarray(),
                      columns=cvect.get_feature_names())
top_df=dtm_df.sum().nlargest(10)

plt.barh(top_df.index, top_df)
plt.savefig('top_tweets.png')

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('img.html')

if __name__ == '__main__':
    app.run(debug = True)

