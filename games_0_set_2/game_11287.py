import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:56:26.633100
# Source Brief: brief_01287.md
# Brief Index: 1287
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a serene underwater tile-matching game.

    The agent controls a small, agile creature in a bioluminescent reef.
    The goal is to cultivate a garden of glowing plants by matching pairs of
    runic tiles. The agent must achieve the target garden size before time
    runs out, while avoiding a dangerous predator that patrols the area.
    The agent can activate a time boost to accelerate plant growth, but this
    also speeds up the predator and consumes energy.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - `actions[1]`: Select Tile (0=released, 1=pressed)
    - `actions[2]`: Time Boost (0=released, 1=held)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +100 for winning (reaching target garden size).
    - -100 for losing (timeout or caught by predator).
    - +5 for each new plant grown.
    - +1 for each successful tile match.
    - -10 for being close to the predator.
    - -0.1 per time step.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Cultivate a garden of glowing plants by matching pairs of runic tiles. "
        "Avoid the predator and complete your garden before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to move. Press space to select a tile. Hold shift to use the time boost."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 2000
    GRID_ROWS, GRID_COLS = 5, 8
    TILE_SIZE = 40
    TILE_GAP = 5
    NUM_TILE_TYPES = 6

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLAYER = (200, 255, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_PREDATOR = (255, 80, 80)
    COLOR_PREDATOR_GLOW = (255, 100, 20)
    COLOR_PLANT = (100, 255, 150)
    COLOR_PLANT_GLOW = (50, 200, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_BOOST_BAR = (255, 220, 0)
    COLOR_BOOST_VIGNETTE = (255, 220, 0)
    TILE_COLORS = [
        (255, 150, 0), (255, 90, 90), (0, 200, 255),
        (200, 100, 255), (255, 255, 100), (100, 255, 200)
    ]
    GRID_COLOR = (30, 50, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        self.grid_width = self.GRID_COLS * (self.TILE_SIZE + self.TILE_GAP) - self.TILE_GAP
        self.grid_height = self.GRID_ROWS * (self.TILE_SIZE + self.TILE_GAP) - self.TILE_GAP
        self.grid_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_y = (self.SCREEN_HEIGHT - self.grid_height) // 2

        self.target_garden_size = 50
        self.render_mode = render_mode
        self._initialize_state()


    def _initialize_state(self):
        # This function is used to set initial values for attributes.
        # It's separated from reset() to allow for first-time setup.
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=np.float32)
        self.predator_pos = np.array([50.0, 50.0], dtype=np.float32)
        self.predator_speed = 1.0
        self.predator_dir = 1
        self.plants = []
        self.particles = []
        self.tile_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.selected_tiles = []
        self.last_space_state = 0
        self.time_boost_fuel = 100.0
        self.max_boost_fuel = 100.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()

        # Create background particles
        self.particles = [
            [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT), random.uniform(0.5, 2.0)]
            for _ in range(100)
        ]
        
        # Setup tile grid
        self._setup_tiles()
        while not self._has_available_matches():
            self._setup_tiles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.1  # Time penalty
        
        # --- Handle Input and State Updates ---
        speed_multiplier = self._handle_time_boost(shift_held)
        
        # We loop for the speed multiplier to make the game feel faster
        for _ in range(speed_multiplier):
            self._update_player(movement)
            self._update_predator()
        
        self._update_plants()
        
        # --- Game Logic ---
        match_made, plant_grown = self._handle_tile_selection(space_held)
        if match_made:
            reward += 1.0
            # SFX: Match success sound
        if plant_grown:
            reward += 5.0
            # SFX: Plant growth sound
        
        if not self._has_available_matches():
            self._reshuffle_board()
            self.selected_tiles.clear()
            # SFX: Reshuffle sound

        # Predator proximity penalty
        dist_to_predator = np.linalg.norm(self.player_pos - self.predator_pos)
        if dist_to_predator < 80:
            reward -= 10.0

        # --- Termination Check ---
        terminated, term_reward = self._check_termination(dist_to_predator)
        reward += term_reward
        self.game_over = terminated
        
        if terminated and term_reward > 0: # Win condition
            self.target_garden_size += 10

        self.score += reward
        self.steps += 1
        self.last_space_state = space_held
        
        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_time_boost(self, shift_held):
        if shift_held and self.time_boost_fuel > 0:
            self.time_boost_fuel = max(0, self.time_boost_fuel - 1.0)
            # SFX: Time boost active loop
            return 2
        else:
            self.time_boost_fuel = min(self.max_boost_fuel, self.time_boost_fuel + 0.25)
            return 1

    def _update_player(self, movement):
        player_speed = 4.0
        if movement == 1: self.player_pos[1] -= player_speed
        elif movement == 2: self.player_pos[1] += player_speed
        elif movement == 3: self.player_pos[0] -= player_speed
        elif movement == 4: self.player_pos[0] += player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)

    def _update_predator(self):
        base_speed = 1.0 + (self.steps // 100) * 0.1
        self.predator_pos[0] += self.predator_dir * base_speed
        if self.predator_pos[0] <= 20 or self.predator_pos[0] >= self.SCREEN_WIDTH - 20:
            self.predator_dir *= -1

    def _update_plants(self):
        for plant in self.plants:
            plant['size'] = min(plant['max_size'], plant['size'] + 0.1)

    def _handle_tile_selection(self, space_held):
        match_made, plant_grown = False, False
        is_press = space_held and not self.last_space_state
        if not is_press:
            return False, False

        # SFX: Selection click
        interaction_radius = 40
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.tile_grid[r, c] == 0: continue
                
                tile_center = self._get_tile_center(r, c)
                if np.linalg.norm(self.player_pos - tile_center) < interaction_radius:
                    if (r, c) not in self.selected_tiles:
                        self.selected_tiles.append((r, c))

                    if len(self.selected_tiles) == 2:
                        r1, c1 = self.selected_tiles[0]
                        r2, c2 = self.selected_tiles[1]
                        if self.tile_grid[r1, c1] == self.tile_grid[r2, c2]:
                            self.tile_grid[r1, c1] = 0
                            self.tile_grid[r2, c2] = 0
                            match_made = True
                            plant_grown = self._grow_plant(r1, c1, r2, c2)
                        self.selected_tiles.clear()
                    return match_made, plant_grown
        return False, False

    def _grow_plant(self, r1, c1, r2, c2):
        valid_spots = []
        for r_offset, c_offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for r_orig, c_orig in [(r1, c1), (r2, c2)]:
                nr, nc = r_orig + r_offset, c_orig + c_offset
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and self.tile_grid[nr, nc] == 0:
                    pos = self._get_tile_center(nr, nc)
                    is_occupied = any(np.linalg.norm(np.array(p['pos']) - pos) < 1 for p in self.plants)
                    if not is_occupied:
                        valid_spots.append(pos)
        
        if valid_spots:
            pos = random.choice(valid_spots)
            self.plants.append({'pos': pos, 'size': 5, 'max_size': random.uniform(15, 25)})
            return True
        return False

    def _check_termination(self, dist_to_predator):
        if len(self.plants) >= self.target_garden_size:
            # SFX: Win fanfare
            return True, 100.0
        if dist_to_predator < 20: # Collision
            # SFX: Player caught sound
            return True, -100.0
        if self.steps >= self.MAX_STEPS:
            # SFX: Loss/timeout sound
            return True, -100.0
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        speed_multiplier = 2 if self.time_boost_fuel < self.max_boost_fuel and self.time_boost_fuel > 0 else 1
        self._render_background_effects(speed_multiplier)
        self._render_tiles()
        self._render_plants()
        self._render_entities()
        self._render_ui(speed_multiplier > 1)
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "garden_size": len(self.plants),
            "target_size": self.target_garden_size,
            "predator_speed": 1.0 + (self.steps // 100) * 0.1,
        }

    # --- Rendering Methods ---

    def _render_background_effects(self, speed_multiplier):
        for p in self.particles:
            p[0] -= p[2] * speed_multiplier * 0.5 # Horizontal drift
            p[1] += p[2] * speed_multiplier * 0.1 # Slow sink
            if p[0] < 0: p[0] = self.SCREEN_WIDTH
            if p[1] > self.SCREEN_HEIGHT: p[1] = 0
            
            alpha = int(p[2] * 50)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), int(p[2]), (*self.COLOR_PLAYER_GLOW, alpha))

    def _render_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_val = self.tile_grid[r, c]
                rect = self._get_tile_rect(r, c)
                
                # Draw grid background
                pygame.draw.rect(self.screen, self.GRID_COLOR, rect, border_radius=5)

                if tile_val > 0:
                    color = self.TILE_COLORS[tile_val - 1]
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=4)
                    
                    # Draw symbol for accessibility
                    center = rect.center
                    if tile_val == 1: pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 10, (0,0,0))
                    if tile_val == 2: pygame.draw.rect(self.screen, (0,0,0), (center[0]-8, center[1]-8, 16, 16), 2)
                    if tile_val == 3: pygame.gfxdraw.aapolygon(self.screen, [(center[0], center[1]-8), (center[0]-8, center[1]+8), (center[0]+8, center[1]+8)], (0,0,0))
                    if tile_val == 4: pygame.draw.line(self.screen, (0,0,0), (center[0]-8, center[1]-8), (center[0]+8, center[1]+8), 2)
                    if tile_val == 5: pygame.draw.line(self.screen, (0,0,0), (center[0]-8, center[1]+8), (center[0]+8, center[1]-8), 2)
                    if tile_val == 6: pygame.gfxdraw.filled_trigon(self.screen, center[0], center[1]-8, center[0]-8, center[1]+8, center[0]+8, center[1]+8, (0,0,0))

                if (r, c) in self.selected_tiles:
                    pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, 4, border_radius=5)

    def _render_plants(self):
        for plant in self.plants:
            pos = (int(plant['pos'][0]), int(plant['pos'][1]))
            size = int(plant['size'])
            
            # Glow effect
            self._draw_glow(self.screen, self.COLOR_PLANT_GLOW, pos, size + 5, 10)
            
            # Stem
            stem_top = (pos[0], pos[1] - size)
            pygame.draw.line(self.screen, self.COLOR_PLANT, (pos[0], pos[1]+5), stem_top, 3)
            
            # Leaves
            num_leaves = 3
            for i in range(num_leaves):
                angle = math.pi / 2 + (i - (num_leaves-1)/2) * 0.8 + math.sin(self.steps * 0.05 + i) * 0.1
                leaf_len = size * 0.8
                end_pos = (stem_top[0] + leaf_len * math.cos(angle), stem_top[1] - leaf_len * math.sin(angle))
                pygame.draw.line(self.screen, self.COLOR_PLANT, stem_top, end_pos, 2)
                pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 3, self.COLOR_PLANT)

    def _render_entities(self):
        # Player
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        self._draw_glow(self.screen, self.COLOR_PLAYER_GLOW, pos, 20, 15)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_BG)
        
        # Predator
        pos = (int(self.predator_pos[0]), int(self.predator_pos[1]))
        self._draw_glow(self.screen, self.COLOR_PREDATOR_GLOW, pos, 30, 20)
        
        points = [
            (pos[0] - 15, pos[1]),
            (pos[0], pos[1] - 10),
            (pos[0] + 15, pos[1]),
            (pos[0], pos[1] + 10)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PREDATOR)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PREDATOR)
        
        eye_pos = (pos[0] + self.predator_dir * 5, pos[1])
        pygame.gfxdraw.filled_circle(self.screen, int(eye_pos[0]), int(eye_pos[1]), 3, (255, 255, 255))

    def _render_ui(self, boost_active):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Garden Size
        garden_text = self.font_small.render(f"GARDEN: {len(self.plants)} / {self.target_garden_size}", True, self.COLOR_UI_TEXT)
        self.screen.blit(garden_text, (self.SCREEN_WIDTH - garden_text.get_width() - 10, 10))
        
        # Time
        time_text = self.font_small.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 30))
        
        # Boost Bar
        bar_w, bar_h = 150, 15
        bar_x, bar_y = (self.SCREEN_WIDTH - bar_w) // 2, self.SCREEN_HEIGHT - bar_h - 10
        fill_w = int((self.time_boost_fuel / self.max_boost_fuel) * bar_w)
        pygame.draw.rect(self.screen, self.GRID_COLOR, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
        
        if boost_active:
            self._draw_glow(self.screen, self.COLOR_BOOST_VIGNETTE, (bar_x + bar_w/2, bar_y + bar_h/2), 25, 10)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        won = len(self.plants) >= self.target_garden_size
        message = "GARDEN COMPLETE" if won else "GAME OVER"
        color = self.COLOR_PLANT if won else self.COLOR_PREDATOR
        
        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text, text_rect)
        
    # --- Helper Methods ---

    def _setup_tiles(self):
        num_pairs = (self.GRID_ROWS * self.GRID_COLS) // 2
        tile_values = []
        for i in range(self.NUM_TILE_TYPES):
            tile_values.extend([i + 1] * (num_pairs // self.NUM_TILE_TYPES))
        
        # Fill remaining slots to ensure pairs
        while len(tile_values) < num_pairs:
            tile_values.append(self.np_random.integers(1, self.NUM_TILE_TYPES + 1))
            
        tile_values.extend(tile_values)
        self.np_random.shuffle(tile_values)
        self.tile_grid = np.array(tile_values).reshape((self.GRID_ROWS, self.GRID_COLS))

    def _has_available_matches(self):
        counts = {}
        for tile_val in self.tile_grid.flatten():
            if tile_val > 0:
                counts[tile_val] = counts.get(tile_val, 0) + 1
        return any(count >= 2 for count in counts.values())

    def _reshuffle_board(self):
        flat_tiles = [t for t in self.tile_grid.flatten() if t > 0]
        self.np_random.shuffle(flat_tiles)
        
        new_grid = np.zeros_like(self.tile_grid)
        idx = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.tile_grid[r, c] > 0:
                    new_grid[r, c] = flat_tiles[idx]
                    idx += 1
        self.tile_grid = new_grid

    def _get_tile_rect(self, r, c):
        x = self.grid_x + c * (self.TILE_SIZE + self.TILE_GAP)
        y = self.grid_y + r * (self.TILE_SIZE + self.TILE_GAP)
        return pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
        
    def _get_tile_center(self, r, c):
        rect = self._get_tile_rect(r, c)
        return np.array(rect.center, dtype=np.float32)

    def _draw_glow(self, surface, color, center, max_radius, steps):
        for i in range(steps):
            progress = i / steps
            radius = int(max_radius * (1 - progress))
            alpha = int(50 * (1 - progress)**2)
            if radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius, (*color, alpha))

if __name__ == '__main__':
    # This block allows you to play the game manually for testing.
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Underwater Garden")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}")
    pygame.quit()