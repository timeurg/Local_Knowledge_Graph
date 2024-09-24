from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
from chat.api import get_embedding
from db.annoy import build_annoy_index, find_similar
from db.embeddings import get_db, EmbeddingDB
from strategies.old import generate_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        user_query = request.json['query']
    else:  # GET
        user_query = request.args.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # conn = create_database()
    emb_db = get_db('embeddings.db')
    # Clear the database before processing the new query
    emb_db.delete_all()
    # Add user query to database
    query_embedding = get_embedding(user_query)
    emb_db.insert_embedding(user_query, query_embedding, True)

    def generate():
        yield from generate_response(user_query, emb_db)

        # Rebuild Annoy index after adding new data
        build_annoy_index(emb_db)

        # Find similar questions/answers
        similar_items = find_similar(emb_db, query_embedding, top_k=5)
        yield f"data: {json.dumps({'type': 'similar', 'items': similar_items})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)