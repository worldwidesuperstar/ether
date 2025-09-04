import torch
import os

print("loading training model...")
checkpoint = torch.load('models/trained_model_5M_75factors.pth', weights_only=False)

original_size = os.path.getsize('models/trained_model_5M_75factors.pth') / (1024**2)
print(f"original model size: {original_size:.1f} MB")

model_state_dict = checkpoint['model_state_dict']

movie_embeddings = model_state_dict['movie_factors.weight']
movie_biases = model_state_dict['movie_biases.weight'].squeeze()
global_bias = model_state_dict['global_bias'].item()

movies_features = checkpoint['movies_features']
essential_movie_data = movies_features[['movieId', 'clean_title', 'year']].copy()

genre_columns = [col for col in movies_features.columns if col.startswith('genre_')]
genre_data = []

for _, row in movies_features.iterrows():
    genres = []
    for genre_col in genre_columns:
        if row[genre_col] == 1:
            genre_name = genre_col.replace('genre_', '').replace('_', ' ').title()
            genres.append(genre_name)
    genre_data.append(genres)

essential_movie_data['genres'] = genre_data
movies_metadata = essential_movie_data.to_dict('records')

production_model = {
    'movie_embeddings': movie_embeddings,
    'movie_biases': movie_biases,
    'global_bias': global_bias,
    'movies_metadata': movies_metadata,
    'movie_encoder': checkpoint['movie_encoder'],
    'n_movies': checkpoint['n_movies'],
    'n_factors': checkpoint['n_factors'],
    'test_rmse': checkpoint['test_rmse'],
}

output_path = 'models/production_model.pth'
torch.save(production_model, output_path)

new_size = os.path.getsize(output_path) / (1024**2)
reduction_factor = original_size / new_size

print(f"original: {original_size:.1f} MB")
print(f"production: {new_size:.1f} MB") 
print(f"reduction: {reduction_factor:.1f}x smaller")
print(f"saved to: {output_path}")

print("testing production model...")
prod_model = torch.load(output_path, weights_only=False)

print(f"movie embeddings: {prod_model['movie_embeddings'].shape}")
print(f"movies: {len(prod_model['movies_metadata']):,}")
print(f"global bias: {prod_model['global_bias']:.3f}")
print(f"test RMSE: {prod_model['test_rmse']:.4f}")

sample_movie = prod_model['movies_metadata'][0]
print(f"sample: '{sample_movie['clean_title']}' ({sample_movie['year']})")