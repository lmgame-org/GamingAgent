import os
import json
import base64
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Import API provider functions - assuming they're in the parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tools.serving.api_providers import (
    openai_completion,
    anthropic_completion,
    gemini_completion,
    together_ai_completion,
    xai_grok_completion
)

# Model names from ablation study notebook
MODEL_NAMES = [
    "o4-mini-2025-04-16",
    "o3-2025-04-16",
    "gemini-2.5-pro-exp-03-25",
    "claude-3-7-sonnet-20250219",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "claude-3-5-sonnet-20241022"
]

# Prompt template for vision models
PROMPT = """You are given ONE RGB frame from an NES game.
Respond with EXACTLY these three lines—no extra text, no blank lines:

GameName: <full title or UNKNOWN>
LevelNumber: <world-stage, e.g. 1-1, or UNKNOWN>
LevelDetails: <semi-colon–separated list in this template>
              area=<area_type>;
              onscreen=<comma-separated facts>;
              upcoming=<comma-separated fine-grained events for rest of level>

Formatting rules
• <area_type> one of: overworld, underground, water, castle, bonus
• <onscreen> list ONLY objects & terrain entirely visible NOW
• <upcoming> list EVERY key event that will happen later in THIS level,
  expressed as lowercase snake-case tokens, in left→right order
  (see examples). If nothing, write [].
• No synonyms, plurals, or re-ordering: use the exact vocabulary below.
  ── allowed upcoming tokens ──
  six_block_triangle_q_bricks_mushroom, triple_pipes_goombas_between,
  bonus_pipe_19_coins_skip, hidden_1up_block, pit_after_pipes,
  question_block_item, falling_goombas_block_row, ten_coin_brick,
  starman_brick, second_q_triangle_fireflower, koopa_troopa,
  extra_goombas, brick_question_row, pyramid_hard_blocks_gap,
  double_pyramid_hard_blocks_pit, exit_pipe_from_bonus,
  two_goombas_four_block_row, inaccessible_pipe_end, staircase, flagpole
• If unsure of a field, write UNKNOWN (not empty).
• Do NOT break the three-line structure under any circumstance."""

def load_ground_truth():
    """Load the ground truth data from the JSON file."""
    with open("super_mario_bros_ground_truth.json", "r") as f:
        ground_truth = json.load(f)
    return ground_truth

def encode_image(image_path):
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_descriptions(image_paths, models=MODEL_NAMES):
    """Generate descriptions from different models for each image."""
    # Load existing results or create new results dictionary
    try:
        with open("mario_generated_texts.json", "r") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}
    
    # Process each model
    for model_name in models:
        # Skip if model results already exist
        if model_name in results:
            print(f"Skipping {model_name} - already exists in results")
            continue
        
        results[model_name] = {}
        system_prompt = "You are an expert at identifying NES game content."
        
        # Process each image
        for image_path in image_paths:
            level_id = Path(image_path).stem  # Use filename as level ID
            print(f"Generating for {model_name} - {level_id}")
            
            try:
                # Encode the image
                base64_image = encode_image(image_path)
                
                # Generate based on model provider
                if "o1-" in model_name or "o3-" in model_name or "o4-" in model_name or "gpt-" in model_name:
                    generated_text = openai_completion(
                        system_prompt=system_prompt,
                        model_name=model_name,
                        base64_image=base64_image,
                        prompt=PROMPT,
                        temperature=0
                    )
                
                elif "claude" in model_name:
                    generated_text = anthropic_completion(
                        system_prompt=system_prompt,
                        model_name=model_name,
                        base64_image=base64_image,
                        prompt=PROMPT,
                        thinking=False
                    )
                
                elif "gemini" in model_name:
                    generated_text = gemini_completion(
                        system_prompt=system_prompt,
                        model_name=model_name,
                        base64_image=base64_image,
                        prompt=PROMPT
                    )
                
                elif "llama-4-maverick" in model_name.lower():
                    generated_text = together_ai_completion(
                        system_prompt=system_prompt,
                        model_name=model_name,
                        base64_image=base64_image,
                        prompt=PROMPT
                    )
                
                elif "grok" in model_name.lower():
                    generated_text = xai_grok_completion(
                        system_prompt=system_prompt,
                        model_name=model_name,
                        prompt=PROMPT
                    )
                
                # Store the result
                results[model_name][level_id] = generated_text
                
                # Save after each successful generation
                with open("mario_generated_texts.json", "w") as f:
                    json.dump(results, f, indent=4)
                
            except Exception as e:
                print(f"Error generating for {model_name} - {level_id}: {str(e)}")
                # Save partial results even if there's an error
                with open("mario_generated_texts.json", "w") as f:
                    json.dump(results, f, indent=4)
    
    return results

def compute_similarity(generated_texts, ground_truth):
    """Compute similarity between generated texts and ground truth."""
    # Load SBERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    similarity_scores = {}
    
    # Process each model's generated descriptions
    for model_name, level_texts in generated_texts.items():
        similarity_scores[model_name] = {}
        
        print(f"Processing model: {model_name}")
        print(f"Available texts: {list(level_texts.keys())}")
        
        try:
            for level_id, generated_text in level_texts.items():
                if level_id in ground_truth:
                    # Get embeddings
                    emb_gen = model.encode(generated_text, show_progress_bar=False)
                    emb_truth = model.encode(ground_truth[level_id], show_progress_bar=False)
                    
                    # Calculate cosine similarity
                    sim_score = cosine_similarity([emb_gen], [emb_truth])[0][0]
                    similarity_scores[model_name][level_id] = sim_score
                    
                    print(f"Computed similarity for {level_id}: {sim_score:.4f}")
                else:
                    print(f"Warning: No ground truth for {level_id}")
                    similarity_scores[model_name][level_id] = 0.0
                
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue
    
    return similarity_scores

def visualize_results(similarity_scores, level_ids=None):
    """Create visualizations for the similarity scores."""
    # Prepare the data
    models = list(similarity_scores.keys())
    
    if level_ids is None:
        # Get all unique level IDs from all models
        level_ids = set()
        for model_scores in similarity_scores.values():
            level_ids.update(model_scores.keys())
        level_ids = sorted(list(level_ids))
    
    # Create a DataFrame for visualization
    data = []
    for model in models:
        model_scores = similarity_scores[model]
        for level in level_ids:
            if level in model_scores:
                data.append({
                    'Model': model,
                    'Level': level,
                    'Similarity': model_scores[level]
                })
    
    df = pd.DataFrame(data)
    
    # Plot 1: Heatmap of similarity scores
    plt.figure(figsize=(10, 8))
    pivot_df = df.pivot(index='Model', columns='Level', values='Similarity')
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Cosine Similarity Between Generated Text and Ground Truth')
    plt.tight_layout()
    plt.savefig('mario_similarity_heatmap.png', dpi=300)
    plt.close()
    
    # Plot 2: Bar chart of average similarity per model
    plt.figure(figsize=(12, 6))
    avg_similarity = df.groupby('Model')['Similarity'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_similarity.index, y=avg_similarity.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model')
    plt.ylabel('Average Similarity Score')
    plt.title('Average Similarity Score by Model')
    plt.tight_layout()
    plt.savefig('mario_avg_similarity.png', dpi=300)
    plt.close()
    
    return df

def main(image_dir_path):
    """Main function to run the ablation study."""
    # Get image paths from the directory
    image_paths = [os.path.join(image_dir_path, f) for f in os.listdir(image_dir_path) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"No images found in {image_dir_path}")
        return
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Generate descriptions from models
    generated_texts = generate_descriptions(image_paths)
    
    # Compute similarity scores
    similarity_scores = compute_similarity(generated_texts, ground_truth)
    
    # Save similarity scores
    with open("mario_similarity_scores.json", "w") as f:
        json.dump(similarity_scores, f, indent=4)
    
    # Visualize the results
    results_df = visualize_results(similarity_scores)
    
    print("Ablation study completed.")
    print("Generated texts saved to: mario_generated_texts.json")
    print("Similarity scores saved to: mario_similarity_scores.json")
    print("Visualizations saved as: mario_similarity_heatmap.png and mario_avg_similarity.png")
    
    return results_df, similarity_scores, generated_texts

if __name__ == "__main__":
    # Example usage:
    # Replace with your image directory path
    image_dir = "mario_screenshots"
    
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist. Please create it and add screenshot images.")
        print("Each image filename should match its level ID in the ground truth file.")
    else:
        main(image_dir) 