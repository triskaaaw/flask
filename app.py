from json import dumps
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from pymongo import MongoClient
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from werkzeug.security import generate_password_hash, check_password_hash
import string
import nltk # type: ignore
nltk.download('stopwords')
from nltk.corpus import stopwords # type: ignore
import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
import torch.nn as nn # type: ignore
# from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
# from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from flask_session import Session
import os
import gdown
import threading

load_dotenv()

app = Flask(__name__)


# Setup Flask Session
app.config['SESSION_TYPE'] = 'filesystem'  # Penyimpanan session di filesystem
app.config['SECRET_KEY'] = os.getenv("SESSION_SECRET_KEY")  # Secret key untuk session
Session(app)  # Initialize Flask-Session

# app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
# jwt = JWTManager(app)
CORS(app)

MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client['DATA_SKRIPSI_API']
db_raw = client['DATA_SKRIPSI']
user_collection = db['data_users']
cbf_collection = db['data_cbf']
resep_collection = db['data_resep']
likes_collection = db["user_likes"]
data_komen = db_raw['part1_data_comment']

model_path = "hybrid_ncf_cbf_model_v3.pth"
google_drive_id = "16t7UtJxap44J2_hkqtTJtNd4duG40r7K"  
download_url = f"https://drive.google.com/uc?id={google_drive_id}"
model_ready = False

# Inisialisasi model dll sebagai None dulu
model = None
user_encoder = None
item_encoder = None
cbf_features = None
ncf_data = None

@app.route("/test")
def home():
    return "API is running!"


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get("name")
    username = data.get("username")
    password = data.get("password")  # Ambil password dari request

    if not name or not username or not password:
        return jsonify({"error": "All fields are required"}), 400

    # Cek apakah username sudah ada
    if user_collection.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400

    # Ambil user_id terakhir dari database
    last_user = user_collection.find_one(sort=[("user_id", -1)])
    new_user_id = last_user["user_id"] + 1 if last_user else 1

    # Hash password sebelum disimpan
    password_hash = generate_password_hash(password)

    # Simpan user baru ke MongoDB
    user_collection.insert_one({
        "name": name,
        "username": username,
        "user_id": new_user_id,
        "password_hash": password_hash
    })

    return jsonify({"message": "User registered successfully", "username": username, "name": name}), 201

# @app.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     username = data.get("username")
#     password = data.get("password")

#     if not username or not password:
#         return jsonify({"error": "Username and password are required"}), 400

#     user = user_collection.find_one({"username": username})
#     if user and check_password_hash(user["password_hash"], password):
#         return jsonify({"message": "Login successful", "username": user["username"], "name": user["name"]}), 200
#     else:
#         return jsonify({"error": "Invalid username or password"}), 401

# @app.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     username = data.get('username')
#     password = data.get('password')

#     user = user_collection.find_one({'username': username})

#     if user is None:
#         return jsonify({'message': 'Invalid credentials'}), 401

#     hashed = user.get('password_hash') or user.get('password')

#     if not check_password_hash(hashed, password):
#         return jsonify({'message': 'Invalid credentials'}), 401

#     # # access_token = create_access_token(identity=user['user_id'])
#     # access_token = create_access_token(identity=user['username'])

#     # return jsonify({
#     #     'token': access_token,
#     #     'username': user['username'],
#     #     'user_id': user['user_id']
#     # }), 200

#     # Simpan username dalam session
#     session['username'] = user['username']
#     session['user_id'] = user['user_id']

#     return jsonify({
#         'message': 'Login successful',
#         'username': user['username'],
#         'user_id': user['user_id']
#     }), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = user_collection.find_one({'username': username})

    if user is None:
        return jsonify({'message': 'Invalid credentials'}), 401

    # Verifikasi password dengan hash yang tersimpan
    hashed = user.get('password_hash') or user.get('password')

    if not check_password_hash(hashed, password):
        return jsonify({'message': 'Invalid credentials'}), 401

    # Set session untuk username dan user_id
    session['username'] = user['username']
    session['user_id'] = user['user_id']

    # Ambil sessionId dari Flask session (berupa ID sesi yang dihasilkan Flask)
    session_id = session.sid  # Ambil session ID

    return jsonify({
        'sessionId': session_id,
        'message': 'Login successful',
        'username': user['username'],
        'user_id': user['user_id']
    }), 200

@app.route('/logout', methods=['POST'])
def logout():
    # Hapus sesi
    session.pop('username', None)
    session.pop('user_id', None)

    return jsonify({"message": "Logout successful"}), 200



# function untuk tokenizer
stemmer = nltk.stem.PorterStemmer()
ENGLISH_STOP_WORDS = stopwords.words('english')

def recipe_tokenizer(sentence):
    # remove punctuation and set to lower case
    for punctuation_mark in string.punctuation:
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []

    # remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)

    return listofstemmed_words

# Load model embeddings dan vectorizer
def load_model():
    with open('CBF_MODEL_FIX_TA.pkl', 'rb') as f:
        combined_embeddings = pickle.load(f)
    with open('TFIDF_MODEL_FIX_TA.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return combined_embeddings, vectorizer

combined_embeddings, vectorizer = load_model()

# Load data resep dari MongoDB ke Pandas DataFrame
#sampled_data = pd.DataFrame(list(cbf_collection.find({}, {"_id": 0, "idRecipe": 1, "title": 1, "text_data": 1})))

# Pakai data lokal
import pandas as pd

# Baca data dari file lokal
sampled_data = pd.read_csv('data_for_cbf.csv')

# Buat kolom text_data seperti saat training
sampled_data['text_data'] = sampled_data[['title', 'category_final', 'description', 'ingredients_final']].astype(str).agg(' '.join, axis=1)
sampled_data['text_data'] = sampled_data['text_data'].str.lower()



def get_recipe_info(id_recipe):
    data = resep_collection.find_one({"idRecipe": id_recipe})
    if data:
        return{
            "idRecipe": id_recipe,
            "title": data.get("title", ""),
            "image_url": data.get("image_url", ""),
            "ratingValue": data.get("ratingValue", ""),
            "time": data.get("time", "")
        }
    else:
        return{
            "idRecipe": id_recipe,
            "title": "",
            "image_url": "",
            "ratingValue": "",
            "time": ""
        }

top_recipes_df = pd.read_csv('data_top_5.csv')

@app.route('/top_recipes', methods=['GET'])
def get_top_recipes():
    # Ambil kolom yang mau ditampilkan
    result = top_recipes_df[['idRecipe', 'title', 'image_url', 'ratingValue', 'time']].to_dict(orient='records')
    return jsonify({'recommendations': result})


@app.route('/recommend_cbf_context', methods=['POST'])
def recommend_cbf_context():
    """Mencari rekomendasi resep berdasarkan input bahan makanan."""
    data = request.get_json()
    ingredients = data.get("ingredients", [])

    if not ingredients:
        return jsonify({"error": "Ingredients input is required"}), 400

    user_text_data = " ".join(ingredients).lower()

    # Vectorize user input
    user_vectorized_data = vectorizer.transform([user_text_data])

    # Adjust shape if necessary
    num_missing_features = combined_embeddings.shape[1] - user_vectorized_data.shape[1]
    if num_missing_features > 0:
        user_vectorized_data = np.pad(user_vectorized_data.toarray(), ((0, 0), (0, num_missing_features)))

    # Compute similarity
    cosine_sim_matrix = cosine_similarity(user_vectorized_data, combined_embeddings)
    similar_recipe_indices = cosine_sim_matrix[0].argsort()[::-1][:10]
    similarity_scores = cosine_sim_matrix[0][similar_recipe_indices]

    result = []
    for i, idx in enumerate(similar_recipe_indices):
        id_recipe = int(sampled_data.iloc[idx]['idRecipe'])
        recipe_info = get_recipe_info(id_recipe)
        recipe_info["similarity_score"] = float(similarity_scores[i])
        result.append(recipe_info)
    
    return jsonify({"recommendations": result}), 200


@app.route('/recommend_cbf_by_recipe', methods=['POST'])
def recommend_cbf_by_recipe():
    """Mencari resep serupa berdasarkan idRecipe."""
    data = request.get_json()
    selected_recipe_id = data.get("idRecipe")

    if not selected_recipe_id:
        return jsonify({"error": "idRecipe is required"}), 400

    # Cek apakah idRecipe ada di database
    if selected_recipe_id not in sampled_data['idRecipe'].values:
        return jsonify({"error": "Recipe ID not found"}), 404

    # Ambil indeks resep yang dipilih
    selected_index = sampled_data[sampled_data['idRecipe'] == selected_recipe_id].index[0]
    selected_recipe_embedding = combined_embeddings[selected_index].reshape(1, -1)

    # Hitung similarity
    cosine_sim_matrix = cosine_similarity(selected_recipe_embedding, combined_embeddings)
    similar_recipe_indices = cosine_sim_matrix[0].argsort()[::-1][1:11]  # Exclude itself
    similarity_scores = cosine_sim_matrix[0][similar_recipe_indices]

    result = []
    for i, idx in enumerate(similar_recipe_indices):
        id_recipe = int(sampled_data.iloc[idx]['idRecipe'])
        recipe_info = get_recipe_info(id_recipe)
        recipe_info["similarity_score"] = float(similarity_scores[i])
        result.append(recipe_info)
    
    return jsonify({"recommendations": result}), 200

class HybridNCF(nn.Module):
            # Define the same HybridNCF model as in your script
            def __init__(self, num_users, num_items, cbf_features, embedding_dim=64):
                super(HybridNCF, self).__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.cbf_features = cbf_features
                self.fc1 = nn.Linear(embedding_dim * 2 + cbf_features.shape[1], 128)
                self.bn1 = nn.BatchNorm1d(128)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)
                self.relu = nn.ReLU()

            def forward(self, user_ids, item_ids):
                user_embeds = self.user_embedding(user_ids)  # [1, 64]
                item_embeds = self.item_embedding(item_ids)  # [9503, 64]
                cbf_embeds = self.cbf_features[item_ids]     # [9503, 3337]

                # Expand user_embeds agar ukurannya sama dengan item_embeds dan cbf_embeds
                user_embeds = user_embeds.expand(item_embeds.size(0), -1)  # [9503, 64]

                # Gabungkan tensor
                x = torch.cat([user_embeds, item_embeds, cbf_embeds], dim=1)  # [9503, 64 + 64 + 3337]
                x = self.relu(self.fc1(x))
                x = self.bn1(x)
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                output = torch.sigmoid(self.fc3(x))
                return output

# Pastikan model siap dengan fungsi ini
def ensure_model_ready():
    global model, user_encoder, item_encoder, cbf_features, model_ready, ncf_data

    if model_ready:
        return

    try:
        if not os.path.exists(model_path):
            print("Downloading model .pth from Google Drive...")
            gdown.download(download_url, model_path, quiet=False)

        required_files = [
            'all_data_ncf.csv',
            'data_for_cbf.csv',
            'CBF_MODEL_FIX_TA.pkl',
            'user_encoder.pkl',
            'item_encoder.pkl'
        ]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} tidak ditemukan!")

        ncf_data = pd.read_csv('all_data_ncf.csv')
        cbf_data = pd.read_csv('data_for_cbf.csv')

        import pickle
        with open("CBF_MODEL_FIX_TA.pkl", "rb") as f:
            cbf_embeddings = pickle.load(f)

        with open("user_encoder.pkl", "rb") as f:
            user_encoder = pickle.load(f)

        with open("item_encoder.pkl", "rb") as f:
            item_encoder = pickle.load(f)

        import numpy as np
        cbf_features_arr = np.zeros((len(item_encoder.classes_), cbf_embeddings.shape[1]))
        for idx, item_id in enumerate(item_encoder.classes_):
            if item_id in cbf_data['idRecipe'].values:
                item_index = cbf_data[cbf_data['idRecipe'] == item_id].index[0]
                cbf_features_arr[idx] = cbf_embeddings[item_index].toarray()

        import torch
        cbf_features = torch.tensor(cbf_features_arr, dtype=torch.float32)

        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()

        model_ready = True
        print("Model and data loaded successfully!")

    except Exception as e:
        print("Failed to load model or data:", e)

#Start loading in a separete thread
threading.Thread(target=ensure_model_ready, daemon=True).start()


@app.route("/recommend_ncf", methods=["POST"])
def recommend_ncf():
    # Pastikan model dan semua komponen sudah siap
    ensure_model_ready()

    if not model_ready:
        return jsonify({"error": "Model masih loading atau gagal load"}), 503
    
   # Cek apakah user sudah login berdasarkan session
    if 'username' not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    username = session['username']
    print("Logged in as:", username)
    
    # data = request.get_json()
    # # username = data.get("username")
    # username = get_jwt_identity()

    # Validasi apakah username ada dalam encoder
    if username not in user_encoder.classes_:
        return jsonify({"error": "User tidak ditemukan"}), 404

    print("Parsed username:", username)
    print("Tipe username setelah parsing:", type(username))

    if username is None:
        return jsonify({"error": "username tidak boleh kosong"}), 400

    if username not in user_encoder.classes_:
        return jsonify({"error": "User tidak ditemukan"}), 404

    # Konversi username ke indeks model
    user_idx = user_encoder.transform([username])[0]
    user_idx_tensor = torch.tensor([user_idx], dtype=torch.long)

    # Ambil semua ID resep
    all_idRecipe = item_encoder.classes_

    # Ambil daftar item yang sudah diinteraksikan user
    interactions_df = ncf_data 
    interacted_items = interactions_df[interactions_df['username'] == username]['idRecipe'].tolist()


    # Ambil daftar item yang belum diinteraksikan
    not_interacted_items = list(set(all_idRecipe) - set(interacted_items))

    if len(not_interacted_items) == 0:
        return jsonify({"error": "Tidak ada item yang bisa direkomendasikan"}), 400

    # Konversi ke tensor
    item_indices = torch.tensor([np.where(all_idRecipe == i)[0][0] for i in not_interacted_items], dtype=torch.long)

    # Hitung skor rekomendasi
    scores = model(user_idx_tensor, item_indices).detach().numpy()
    scores = scores.squeeze()  # Pastikan skor berbentuk array 1D
    print("Scores:", scores[:10])  # Print 10 skor pertama untuk debugging

    # Pilih Top-K indeks dengan skor tertinggi
    top_indices = np.argsort(scores)[::-1][:5]
    print("Top indices:", top_indices)

    # Ambil ID resep berdasarkan indeks teratas
    recommended_items = [not_interacted_items[i] for i in top_indices]
    print("Recommended Items:", recommended_items)

    result = [get_recipe_info(int(recipe_id)) for recipe_id in recommended_items]

    return jsonify({"recommendations": result}), 200



@app.route('/recipe_detail', methods=['GET'])
def get_recipe_detail():
    id_recipe = request.args.get('idRecipe')

    if not id_recipe:
        return jsonify({"error": "idRecipe is required"}), 400

    # Mencari resep berdasarkan idRecipe
    recipe = resep_collection.find_one({"idRecipe": int(id_recipe)}, {"_id": 0})  # Menghapus _id dari hasil

    if not recipe:
        return jsonify({"error": "Recipe not found"}), 404

    return jsonify(recipe)


@app.route("/like_recipe", methods=["POST"])
def like_recipe():
    data = request.json
    id_user = data.get("idUser")
    id_recipe = data.get("idRecipe")

    if not id_user or not id_recipe:
        return jsonify({"error": "idUser dan idRecipe harus diisi"}), 400

    # Cek apakah user sudah like resep ini
    existing_like = likes_collection.find_one({"idUser": id_user, "idRecipe": id_recipe})

    if existing_like:
        return jsonify({"message": "User sudah like resep ini"}), 200

    # Simpan data like ke database
    likes_collection.insert_one({"idUser": id_user, "idRecipe": id_recipe})

    return jsonify({"message": "Like berhasil disimpan"}), 201


@app.route("/unlike_recipe", methods=["POST"])
def unlike_recipe():
    data = request.json
    id_user = data.get("idUser")
    id_recipe = data.get("idRecipe")

    if not id_user or not id_recipe:
        return jsonify({"error": "idUser dan idRecipe harus diisi"}), 400

    # Hapus like jika ada
    result = likes_collection.delete_one({"idUser": id_user, "idRecipe": id_recipe})

    if result.deleted_count > 0:
        return jsonify({"message": "Unlike berhasil"}), 200
    else:
        return jsonify({"message": "User belum like resep ini"}), 200


@app.route("/check_like", methods=["GET"])
def check_like():
    id_user = request.args.get("idUser")
    id_recipe = request.args.get("idRecipe")

    if not id_user or not id_recipe:
        return jsonify({"error": "idUser dan idRecipe harus diisi"}), 400

    try:
        id_recipe = int(id_recipe)  # Pastikan idRecipe bertipe integer
    except ValueError:
        return jsonify({"error": "idRecipe harus berupa angka"}), 400

    # Cek apakah user sudah like resep ini
    existing_like = likes_collection.find_one({"idUser": id_user, "idRecipe": id_recipe})

    if existing_like:
        return jsonify({"liked": True}), 200
    else:
        return jsonify({"liked": False}), 200
    
@app.route("/liked_recipes", methods =["GET"])
def get_like_recipes():
    username = request.args.get("idUser")

    if not username:
        return jsonify({"error": "idUser is required"}), 400
    
    liked_docs = likes_collection.find({"idUser": username})
    liked_recipe_ids = [doc["idRecipe"] for doc in liked_docs]

    # Ambil field yang dibutuhkan saja dari resep_collection
    recipes = list(resep_collection.find(
        {"idRecipe": {"$in": liked_recipe_ids}},
        {"_id": 0, "idRecipe": 1, "title": 1, "image_url": 1}
    ))
    
    return jsonify({"liked_recipes": recipes}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


