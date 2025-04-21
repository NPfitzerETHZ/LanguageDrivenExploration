import numpy as np
import random
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from google import genai
import time

# Set your Gemini API key
genai_client = genai.Client(api_key="AIzaSyC6WV7Xto20AETHSFKSTI0_gJlvJb-HKDY")

# Parameters
NUM_DATA_TARGETS = 5000  # <-- Set your target number here
API_DELAY_SECONDS = 1.
grid_size = 10
min_patch_size = 3
max_patch_size = 30
min_std = 0.5
max_std = 2.

def get_neighbors(cell, grid_size):
    r, c = cell
    neighbors = []
    if r > 0: neighbors.append((r-1, c))
    if r < grid_size-1: neighbors.append((r+1, c))
    if c > 0: neighbors.append((r, c-1))
    if c < grid_size-1: neighbors.append((r, c+1))
    return neighbors

def generate_gaussian_prob_map(center, std_x, std_y, grid_size):
    cx, cy = center
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    gauss = np.exp(-((xx - cx) ** 2) / (2 * std_x ** 2) - ((yy - cy) ** 2) / (2 * std_y ** 2))
    gauss /= gauss.sum()
    return gauss

def compute_patch_spread(patch):
    coords = np.array(list(patch))
    if len(coords) == 0:
        return 0.0, 0.0
    std_y = np.std(coords[:, 0])  # rows
    std_x = np.std(coords[:, 1])  # cols
    return std_x, std_y

def generate_grid_and_patch():
    grid = np.ones((grid_size, grid_size, 3))
    
    start_r = random.randint(0, grid_size - 1)
    start_c = random.randint(0, grid_size - 1)
    start = (start_r, start_c)

    init_std_x = random.uniform(min_std, max_std)
    init_std_y = random.uniform(min_std, max_std)

    prob_map = generate_gaussian_prob_map((start_r, start_c), init_std_x, init_std_y, grid_size)

    patch = set([start])
    frontier = set(get_neighbors(start, grid_size))
    target_size = random.randint(min_patch_size, max_patch_size)

    while len(patch) < target_size and frontier:
        probs = np.array([prob_map[r, c] for r, c in frontier])
        probs /= probs.sum()
        next_cell = random.choices(list(frontier), weights=probs, k=1)[0]

        patch.add(next_cell)
        frontier.remove(next_cell)

        for neighbor in get_neighbors(next_cell, grid_size):
            if neighbor not in patch:
                frontier.add(neighbor)

    for r, c in patch:
        grid[r, c] = [1, 0, 0]

    norm_start = (start_r / (grid_size - 1), start_c / (grid_size - 1))

    std_x_actual, std_y_actual = compute_patch_spread(patch)

    norm_spread_x = std_x_actual / (grid_size - 1)
    norm_spread_y = std_y_actual / (grid_size - 1)

    return grid, norm_start, norm_spread_x, norm_spread_y

def plot_grid(grid):
    fig, ax = plt.subplots()
    ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

def describe_image(image_buf):
    img = Image.open(image_buf)
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=["Your are leading a team of robots. Describe the location, size and spread of the target patch with respect to the environement. Use sentences and be precise. Do not mention color or the grid", img])
    return response.text

# Main loop to generate dataset
dataset = []

for _ in tqdm(range(NUM_DATA_TARGETS), desc="Generating Data"):
    grid, norm_start, norm_std_x, norm_std_y = generate_grid_and_patch()
    image_buf = plot_grid(grid)
    description = describe_image(image_buf)
    time.sleep(API_DELAY_SECONDS)

    data_point = {
        "normalized_start": norm_start,
        "normalized_std": (norm_std_x, norm_std_y),
        "gemini_response": description
    }
    dataset.append(data_point)

# Now `dataset` contains your structured data
# You can save it to JSON or use however needed
import json
with open("gemini_patch_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
