import pickle
from flask import Flask, render_template, request

# Load movie data
movies = pickle.load(open(r'movies.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           original_title=list(movies['original_title'].values),
                           homepage=list(movies['homepage'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_movies', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    # Add your recommendation logic here based on user_input
    # For simplicity, let's assume we're recommending the top 4 similar movies
    recommended_movies = movies.head(4)

    # Convert recommended_movies DataFrame to a list of dictionaries
    recommended_movies_list = recommended_movies.to_dict(orient='records')

    return render_template('recommend.html', recommended_movies=recommended_movies_list)

if __name__ == '__main__':
    app.run(debug=True)
