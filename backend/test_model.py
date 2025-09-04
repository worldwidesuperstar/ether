import torch

print("loading production model...")
prod_model = torch.load('models/production_model.pth', weights_only=False)

print(f"production model loaded")
print(f"movie embeddings: {prod_model['movie_embeddings'].shape}")
print(f"movies: {len(prod_model['movies_metadata']):,}")
print(f"test RMSE: {prod_model['test_rmse']:.4f}")

print("\ntesting synthetic user with production model")

# simulate questionnaire ratings (movie_id, rating)
questionnaire_ratings = [
    {'movie_id': 1, 'rating': 4.5},      # Toy Story
    {'movie_id': 2, 'rating': 3.0},      # Jumanji  
    {'movie_id': 50, 'rating': 4.0},     # Usual Suspects
    {'movie_id': 260, 'rating': 5.0},    # Star Wars
    {'movie_id': 1198, 'rating': 2.0},   # Raiders of Lost Ark
]

print(f"questionnaire has {len(questionnaire_ratings)} ratings")

def create_synthetic_user(ratings, model_data):
    """create user embedding from questionnaire ratings"""
    movie_encoder = model_data['movie_encoder']
    movie_embeddings = model_data['movie_embeddings']
    
    user_embeddings = []
    
    for rating_data in ratings:
        try:
            movie_idx = movie_encoder.transform([rating_data['movie_id']])[0]
            movie_emb = movie_embeddings[movie_idx]
            
            # weight by how much they liked it (centered around 3)
            weight = (rating_data['rating'] - 3.0) / 2.0
            weighted_embedding = movie_emb * weight
            user_embeddings.append(weighted_embedding)
            
        except ValueError:
            print(f"movie {rating_data['movie_id']} not in dataset")
            continue
    
    if len(user_embeddings) == 0:
        return torch.zeros(model_data['n_factors']), torch.tensor(0.0)
    
    synthetic_embedding = torch.stack(user_embeddings).mean(dim=0)
    synthetic_bias = torch.tensor(0.0)  # start neutral
    
    return synthetic_embedding, synthetic_bias

def predict_for_synthetic_user(user_embedding, user_bias, movie_indices, model_data):
    """predict ratings for synthetic user"""
    movie_embeddings = model_data['movie_embeddings'][movie_indices]
    movie_biases = model_data['movie_biases'][movie_indices]
    global_bias = model_data['global_bias']
    
    collaborative_scores = torch.mm(
        user_embedding.unsqueeze(0), 
        movie_embeddings.t()
    ).squeeze()
    
    predictions = global_bias + user_bias + movie_biases + collaborative_scores
    return predictions

# create synthetic user
synthetic_embedding, synthetic_bias = create_synthetic_user(questionnaire_ratings, prod_model)

print(f"synthetic embedding: {synthetic_embedding.shape}")
print(f"synthetic bias: {synthetic_bias.item():.3f}")

# predict on all movies
all_movie_indices = torch.arange(prod_model['n_movies'])
all_predictions = predict_for_synthetic_user(
    synthetic_embedding, 
    synthetic_bias, 
    all_movie_indices, 
    prod_model
)

# get top recommendations
top_indices = torch.argsort(all_predictions, descending=True)[:15]

print(f"\ntop 15 recommendations for synthetic user:")
for i, idx in enumerate(top_indices):
    movie_idx = idx.item()
    movie_data = prod_model['movies_metadata'][movie_idx]
    predicted_rating = all_predictions[idx].item()
    
    print(f"{i+1:2d}. '{movie_data['clean_title']}' ({movie_data['year']}) - predicted: {predicted_rating:.2f}")
