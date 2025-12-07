import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:57:29.624820
# Source Brief: brief_02337.md
# Brief Index: 2337
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    """
    A hexagonal tile-based puzzle game Gymnasium environment.

    The goal is to make the entire grid a single color by strategically placing
    colored tiles. Placing a tile can trigger a chain reaction that converts
    adjacent tiles, creating a dynamic and strategic puzzle experience.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Cursor Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - action[1]: Place Tile (0: released, 1: pressed)
    - action[2]: Cycle Color (0: released, 1: pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A strategic puzzle game where you place colored tiles on a hexagonal grid to convert "
        "adjacent tiles and make the entire board a single color."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a tile and "
        "shift to cycle through available colors."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 19
    GRID_ROWS = 11
    HEX_SIZE = 20
    MAX_STEPS = 500

    # Colors (Clean, high-contrast palette)
    COLOR_BG = (26, 26, 26)  # #1a1a1a
    COLOR_GRID = (51, 51, 51)  # #333333
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_CURSOR = (255, 255, 0)

    # Game Colors (0: empty, 1: red, 2: green, 3: blue, 4: wildcard)
    COLORS = [
        (0, 0, 0),         # 0: Empty (not used in grid)
        (255, 71, 87),     # 1: Red (#ff4757)
        (46, 213, 115),    # 2: Green (#2ed573)
        (55, 66, 250),     # 3: Blue (#3742fa)
        (255, 165, 2),     # 4: Wildcard/Gold (#ffa502)
    ]
    PRIMARY_COLORS = [1, 2, 3]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Game State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.pulse_effects = None
        self.completed_lines = None

        # Pre-calculate hex centers for performance
        self.hex_centers = self._precompute_hex_centers()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.np_random.integers(1, 4, size=(self.GRID_COLS, self.GRID_ROWS))
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_color_idx = 0  # Index into PRIMARY_COLORS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.pulse_effects = []
        self.completed_lines = {'cols': set(), 'rows': set()}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, shift_press = self._process_actions(action)
        reward = 0.0
        
        self._handle_movement(movement)
        
        if shift_press:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PRIMARY_COLORS)
            # sfx: color_cycle.wav

        if space_press:
            reward = self._place_tile()
            self.score += reward
            # sfx: place_tile.wav

        self.steps += 1
        self._update_pulse_effects()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            terminal_reward = self._calculate_terminal_reward()
            reward += terminal_reward
            self.score += terminal_reward
            self.game_over = True

        truncated = False # This game is terminated by reaching a goal or step limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _process_actions(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return movement, space_press, shift_press

    def _handle_movement(self, movement):
        col, row = self.cursor_pos
        if movement == 1:  # Up
            row -= 1
        elif movement == 2:  # Down
            row += 1
        elif movement == 3:  # Left
            col -= 1
        elif movement == 4:  # Right
            col += 1
        
        self.cursor_pos[0] = np.clip(col, 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(row, 0, self.GRID_ROWS - 1)

    def _place_tile(self):
        place_pos = tuple(self.cursor_pos)
        place_color = self.PRIMARY_COLORS[self.selected_color_idx]
        
        # --- Find contiguous area (BFS) ---
        q = deque([place_pos])
        visited = {place_pos}
        
        while q:
            col, row = q.popleft()
            for neighbor_pos in self._get_neighbors(col, row):
                if neighbor_pos not in visited:
                    neighbor_color = self.grid[neighbor_pos]
                    if neighbor_color == place_color or neighbor_color == 4: # Wildcard
                        visited.add(neighbor_pos)
                        q.append(neighbor_pos)
        
        # Change color of the whole contiguous area
        for pos in visited:
            self.grid[pos] = place_color

        # --- Trigger chain reaction on the border of the new area ---
        conversions = []
        for pos in visited:
            for neighbor_pos in self._get_neighbors(pos[0], pos[1]):
                if neighbor_pos not in visited:
                    conversions.append(neighbor_pos)
        
        # --- Calculate reward and apply conversions ---
        reward = 0
        _, majority_color = self._get_grid_color_stats()
        
        newly_converted = set()
        for pos in set(conversions): # Use set to avoid double-counting
            if self.np_random.random() < 0.2:
                self.grid[pos] = 4 # Wildcard
                reward -= 0.5
                # sfx: wildcard_create.wav
            else:
                if self.grid[pos] != place_color:
                    if place_color == majority_color:
                        reward += 1.0
                    self.grid[pos] = place_color
            
            newly_converted.add(pos)
        
        if newly_converted:
            # sfx: chain_reaction.wav
            for pos in newly_converted:
                self.pulse_effects.append({"pos": pos, "timer": 1.0, "size": 1.5})
        
        reward += self._check_line_completion()

        return reward

    def _get_grid_color_stats(self):
        # Exclude wildcards from stats
        primary_grid = self.grid[self.grid != 4]
        if primary_grid.size == 0:
            # Check if grid is all wildcards
            if np.all(self.grid == 4):
                return 100.0, 4 # Special case: all wildcards is a "win" state
            return 0, -1 # Empty grid
        
        colors, counts = np.unique(primary_grid, return_counts=True)
        
        total_tiles = self.GRID_COLS * self.GRID_ROWS
        majority_idx = np.argmax(counts)
        majority_color = colors[majority_idx]
        majority_count = counts[majority_idx]
        
        # Percentage of non-wildcard tiles that are the majority color
        majority_percent = (majority_count / primary_grid.size) * 100
        
        return majority_percent, majority_color

    def _check_line_completion(self):
        reward = 0
        # Check rows
        for r in range(self.GRID_ROWS):
            if r not in self.completed_lines['rows']:
                row_colors = self.grid[:, r]
                unique_colors = np.unique(row_colors[row_colors != 4])
                if len(unique_colors) == 1:
                    reward += 5
                    self.completed_lines['rows'].add(r)
                    # sfx: line_complete.wav
        # Check columns
        for c in range(self.GRID_COLS):
            if c not in self.completed_lines['cols']:
                col_colors = self.grid[c, :]
                unique_colors = np.unique(col_colors[col_colors != 4])
                if len(unique_colors) == 1:
                    reward += 5
                    self.completed_lines['cols'].add(c)
                    # sfx: line_complete.wav
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        
        # Grid is one color (ignoring wildcards) or all wildcards
        primary_grid = self.grid[self.grid != 4]
        if primary_grid.size > 0:
            if len(np.unique(primary_grid)) == 1:
                return True
        elif np.all(self.grid == 4): # Grid is entirely wildcards
            return True
            
        return False

    def _calculate_terminal_reward(self):
        primary_grid = self.grid[self.grid != 4]
        # Check if win condition is met
        if (primary_grid.size > 0 and len(np.unique(primary_grid)) == 1) or np.all(self.grid == 4):
            wildcard_count = np.sum(self.grid == 4)
            wildcard_penalty = wildcard_count / (self.GRID_COLS * self.GRID_ROWS)
            # sfx: victory.wav
            return 100 * (1 - wildcard_penalty)
        return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw hex grid
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                center = self.hex_centers[c, r]
                color_id = self.grid[c, r]
                self._draw_hexagon(self.screen, self.COLORS[color_id], center, self.HEX_SIZE)
                self._draw_hexagon(self.screen, self.COLOR_GRID, center, self.HEX_SIZE, width=1)
        
        # Draw pulse effects
        for effect in self.pulse_effects:
            center = self.hex_centers[effect["pos"]]
            size = self.HEX_SIZE * (1 + (effect["timer"] * (effect["size"] - 1)))
            alpha = int(255 * effect["timer"])
            color = self.COLORS[self.grid[effect["pos"]]]
            
            glow_color = (*color, alpha)
            self._draw_hexagon(self.screen, glow_color, center, size, is_glow=True)

        # Draw cursor
        cursor_center = self.hex_centers[tuple(self.cursor_pos)]
        self._draw_hexagon(self.screen, self.COLOR_CURSOR, cursor_center, self.HEX_SIZE + 2, width=3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 10))

        # Steps
        steps_text = self.font_small.render(f"MOVES: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH // 2 - steps_text.get_width() // 2, self.SCREEN_HEIGHT - 25))

        # Selected Color
        selected_color_label = self.font_small.render("TILE:", True, self.COLOR_UI_TEXT)
        self.screen.blit(selected_color_label, (15, 15))
        color_to_place = self.PRIMARY_COLORS[self.selected_color_idx]
        self._draw_hexagon(self.screen, self.COLORS[color_to_place], (80, 23), self.HEX_SIZE * 0.8)

    def _update_pulse_effects(self):
        # Update timers and remove finished effects
        for effect in self.pulse_effects:
            effect["timer"] -= 1.0 / 15.0  # Fades over 0.5 seconds at 30fps
        self.pulse_effects = [e for e in self.pulse_effects if e["timer"] > 0]

    # --- Hex Grid Helper Functions ---

    def _precompute_hex_centers(self):
        centers = np.zeros((self.GRID_COLS, self.GRID_ROWS, 2))
        hex_w = self.HEX_SIZE * 2
        hex_h = math.sqrt(3) * self.HEX_SIZE
        offset_x = (self.SCREEN_WIDTH - (self.GRID_COLS * hex_w * 0.75)) / 2 + self.HEX_SIZE
        offset_y = (self.SCREEN_HEIGHT - (self.GRID_ROWS * hex_h)) / 2 + self.HEX_SIZE

        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                px = offset_x + c * hex_w * 0.75
                py = offset_y + r * hex_h
                if c % 2 != 0:
                    py += hex_h / 2
                centers[c, r] = (int(px), int(py))
        return centers

    @staticmethod
    def _draw_hexagon(surface, color, center, size, width=0, is_glow=False):
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((center[0] + size * math.cos(angle_rad),
                           center[1] + size * math.sin(angle_rad)))
        
        points_int = [(int(p[0]), int(p[1])) for p in points]

        if is_glow and len(color) == 4:
            # For glows, we can't use aapolygon with alpha, so we draw a thick line
            pygame.draw.aalines(surface, color, True, points_int, blend=1)
        elif width == 0:
            pygame.gfxdraw.aapolygon(surface, points_int, color)
            pygame.gfxdraw.filled_polygon(surface, points_int, color)
        else:
            pygame.draw.aalines(surface, color, True, points_int)

    def _get_neighbors(self, col, row):
        """ Get valid neighbor coordinates using odd-q offset system. """
        neighbors = []
        # Parity determines vertical offsets
        parity = col % 2
        
        # N, S
        potential_neighbors = [(col, row - 1), (col, row + 1)]
        
        # Diagonal neighbors depend on column parity
        if parity == 1: # Odd columns (shifted down)
            potential_neighbors.extend([
                (col - 1, row), (col - 1, row + 1),
                (col + 1, row), (col + 1, row + 1)
            ])
        else: # Even columns
            potential_neighbors.extend([
                (col - 1, row - 1), (col - 1, row),
                (col + 1, row - 1), (col + 1, row)
            ])

        for c, r in potential_neighbors:
            if 0 <= c < self.GRID_COLS and 0 <= r < self.GRID_ROWS:
                neighbors.append((c, r))
        return neighbors

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Hex Color Chain")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while running:
        # Pygame event handling
        action = [0, 0, 0] # Default action: no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                    print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

            if done:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for smooth manual play

    env.close()
    pygame.quit()