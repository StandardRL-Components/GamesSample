import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:45:13.271947
# Source Brief: brief_01257.md
# Brief Index: 1257
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two robotic arms simultaneously in this fast-paced, dual-grid match-3 puzzle game. "
        "Race against the clock to clear tiles and achieve the target number of matches."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move both cursors. "
        "Press space to swap tiles in the left grid and shift to swap tiles in the right grid."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    GRID_SIZE = 3
    TILE_SIZE = 50
    TILE_GAP = 8
    GRID_LINE_WIDTH = 3
    
    TARGET_MATCHES = 20

    # --- Colors ---
    COLOR_BG = (13, 15, 33) # #0d0f21
    COLOR_GRID = (74, 78, 138) # #4a4e8a
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_MATCH_FLASH = (255, 255, 255)
    
    TILE_COLORS = [
        (255, 71, 102),   # #ff4766 Red
        (89, 255, 140),  # #59ff8c Green
        (71, 168, 255),  # #47a8ff Blue
        (255, 221, 71),  # #ffdd47 Yellow
        (179, 71, 255),   # #b347ff Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.grid_left = None
        self.grid_right = None
        self.cursor_left = None
        self.cursor_right = None
        self.last_move_direction = None
        self.steps = None
        self.score = None
        self.matches_made = None
        self.time_remaining = None
        self.game_over = None
        self.win_condition_met = None
        
        self.visual_effects = []
        self.stars = []

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is not needed in the final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.matches_made = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.win_condition_met = False
        
        self.grid_left = self._create_valid_grid()
        self.grid_right = self._create_valid_grid()
        
        self.cursor_left = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.cursor_right = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.last_move_direction = 4 # Default to right
        
        self.visual_effects = []
        self._initialize_stars()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        self.time_remaining -= 1
        
        # --- Update Game Logic ---
        # 1. Handle cursor movement
        self._handle_cursor_movement(movement)
        
        # 2. Handle swaps
        swap_made = False
        if space_held:
            # Sfx: Left arm swap
            if self._perform_swap(self.grid_left, self.cursor_left, self.last_move_direction):
                swap_made = True
        if shift_held:
            # Sfx: Right arm swap
            if self._perform_swap(self.grid_right, self.cursor_right, self.last_move_direction):
                swap_made = True
        
        # 3. Handle matches and cascades
        matches_this_step = 0
        if swap_made:
            matches_left = self._handle_cascades(self.grid_left)
            matches_right = self._handle_cascades(self.grid_right)
            matches_this_step = matches_left + matches_right

            if matches_this_step > 0:
                self.matches_made += matches_this_step
                # Sfx: Match success chime
        
        # 4. Check for deadlocks and shuffle if necessary
        if swap_made and matches_this_step == 0:
            if not self._check_for_possible_moves(self.grid_left):
                self._shuffle_grid(self.grid_left)
                # Sfx: Reshuffle sound
            if not self._check_for_possible_moves(self.grid_right):
                self._shuffle_grid(self.grid_right)
                # Sfx: Reshuffle sound
        
        # 5. Calculate reward and check termination
        terminated = self._check_termination()
        reward = self._calculate_reward(matches_this_step, terminated, self.win_condition_met)
        self.score += reward
        
        truncated = False # This environment does not have a truncation condition separate from termination
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "matches_made": self.matches_made,
            "time_remaining_seconds": self.time_remaining / self.FPS
        }

    # --- Game Logic Helpers ---
    def _create_valid_grid(self):
        while True:
            grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_matches(grid) and self._check_for_possible_moves(grid):
                return grid

    def _handle_cursor_movement(self, movement):
        if movement == 0: return
        self.last_move_direction = movement
        
        move_vec = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0,0))
        
        self.cursor_left = np.clip(self.cursor_left + move_vec, 0, self.GRID_SIZE - 1)
        self.cursor_right = np.clip(self.cursor_right + move_vec, 0, self.GRID_SIZE - 1)

    def _perform_swap(self, grid, cursor, direction):
        move_vec = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(direction)
        if not move_vec: return False
        
        p1 = cursor
        p2 = cursor + move_vec
        
        if not (0 <= p2[0] < self.GRID_SIZE and 0 <= p2[1] < self.GRID_SIZE):
            return False
        
        grid[p1[1], p1[0]], grid[p2[1], p2[0]] = grid[p2[1], p2[0]], grid[p1[1], p1[0]]
        return True

    def _find_matches(self, grid):
        matched_tiles = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Horizontal
                if c < self.GRID_SIZE - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matched_tiles.update([(c, r), (c+1, r), (c+2, r)])
                # Vertical
                if r < self.GRID_SIZE - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matched_tiles.update([(c, r), (c, r+1), (c, r+2)])
        return matched_tiles

    def _handle_cascades(self, grid):
        total_matches = 0
        while True:
            matches = self._find_matches(grid)
            if not matches:
                break
            
            total_matches += len(matches) // 3
            
            # Add visual effect for matched tiles
            is_left_grid = np.array_equal(grid, self.grid_left)
            offset_x = self.SCREEN_WIDTH // 4 if is_left_grid else 3 * self.SCREEN_WIDTH // 4
            for x, y in matches:
                self.visual_effects.append({
                    "type": "flash", 
                    "pos": (offset_x, self.SCREEN_HEIGHT // 2), 
                    "tile_pos": (x, y), 
                    "timer": 15
                })

            self._remove_matches_and_refill(grid, matches)
            # Sfx: Cascade sound
        return total_matches

    def _remove_matches_and_refill(self, grid, matched_tiles):
        cols_to_update = {c for c, r in matched_tiles}
        for c, r in matched_tiles:
            grid[r, c] = -1 # Mark for removal
        
        for c in cols_to_update:
            col = grid[:, c]
            empty_count = np.sum(col == -1)
            if empty_count > 0:
                new_col = col[col != -1]
                new_tiles = self.np_random.integers(0, len(self.TILE_COLORS), size=empty_count)
                grid[:, c] = np.concatenate((new_tiles, new_col))

    def _check_for_possible_moves(self, grid):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                        temp_grid = np.copy(grid)
                        temp_grid[r, c], temp_grid[nr, nc] = temp_grid[nr, nc], temp_grid[r, c]
                        if self._find_matches(temp_grid):
                            return True
        return False

    def _shuffle_grid(self, grid):
        temp_grid = self._create_valid_grid()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                grid[r, c] = temp_grid[r, c]

    def _check_termination(self):
        if self.matches_made >= self.TARGET_MATCHES:
            self.game_over = True
            self.win_condition_met = True
        elif self.time_remaining <= 0:
            self.game_over = True
        return self.game_over

    def _calculate_reward(self, matches_this_step, terminated, won):
        reward = 0
        if matches_this_step > 0:
            reward += matches_this_step * 10
            reward += (self.time_remaining / self.FPS) # Bonus for remaining time
        
        # Potential match reward
        def count_pairs(g):
            pairs = 0
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE - 1):
                    if g[r,c] == g[r,c+1]: pairs += 1
            for c in range(self.GRID_SIZE):
                for r in range(self.GRID_SIZE - 1):
                    if g[r,c] == g[r+1,c]: pairs += 1
            return pairs
        
        reward += (count_pairs(self.grid_left) + count_pairs(self.grid_right)) * 0.1

        if terminated:
            if won:
                reward += 100
            else: # Time ran out
                reward -= 100
        
        return reward

    # --- Rendering Helpers ---
    def _initialize_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                "pos": [random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)],
                "size": random.uniform(0.5, 1.5),
                "speed": random.uniform(0.1, 0.3)
            })

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Move and draw stars
        for star in self.stars:
            star["pos"][0] -= star["speed"]
            if star["pos"][0] < 0:
                star["pos"][0] = self.SCREEN_WIDTH
                star["pos"][1] = random.randint(0, self.SCREEN_HEIGHT)
            pygame.draw.circle(self.screen, (200, 200, 220), star["pos"], star["size"])

    def _render_game(self):
        # Grid positions
        left_grid_center_x = self.SCREEN_WIDTH // 4
        right_grid_center_x = 3 * self.SCREEN_WIDTH // 4
        grid_center_y = self.SCREEN_HEIGHT // 2

        # Render grids and tiles
        self._render_grid_and_tiles(self.grid_left, left_grid_center_x, grid_center_y)
        self._render_grid_and_tiles(self.grid_right, right_grid_center_x, grid_center_y)
        
        # Render cursors
        self._render_cursor(self.cursor_left, left_grid_center_x, grid_center_y)
        self._render_cursor(self.cursor_right, right_grid_center_x, grid_center_y)
        
        # Render robotic arms
        self._render_robotic_arm(self.cursor_left, (-50, grid_center_y), left_grid_center_x, grid_center_y)
        self._render_robotic_arm(self.cursor_right, (self.SCREEN_WIDTH + 50, grid_center_y), right_grid_center_x, grid_center_y)
        
        # Render and update visual effects
        self._render_effects()

    def _get_tile_screen_pos(self, tile_x, tile_y, grid_center_x, grid_center_y):
        total_grid_width = self.GRID_SIZE * self.TILE_SIZE + (self.GRID_SIZE - 1) * self.TILE_GAP
        top_left_x = grid_center_x - total_grid_width / 2
        top_left_y = grid_center_y - total_grid_width / 2
        
        screen_x = top_left_x + tile_x * (self.TILE_SIZE + self.TILE_GAP) + self.TILE_SIZE / 2
        screen_y = top_left_y + tile_y * (self.TILE_SIZE + self.TILE_GAP) + self.TILE_SIZE / 2
        return int(screen_x), int(screen_y)

    def _render_grid_and_tiles(self, grid, center_x, center_y):
        total_grid_width = self.GRID_SIZE * self.TILE_SIZE + (self.GRID_SIZE - 1) * self.TILE_GAP
        grid_rect = pygame.Rect(0, 0, total_grid_width + self.TILE_GAP, total_grid_width + self.TILE_GAP)
        grid_rect.center = (center_x, center_y)
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, self.GRID_LINE_WIDTH, border_radius=5)

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = grid[r, c]
                if color_index == -1: continue
                
                color = self.TILE_COLORS[color_index]
                glow_color = (*color, 30)
                
                sx, sy = self._get_tile_screen_pos(c, r, center_x, center_y)
                
                radius = self.TILE_SIZE // 2 - 4
                
                # Glow effect
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius + 4, glow_color)
                # Main tile
                pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, color)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, color)

    def _render_cursor(self, cursor, center_x, center_y):
        sx, sy = self._get_tile_screen_pos(cursor[0], cursor[1], center_x, center_y)
        size = self.TILE_SIZE // 2
        
        # Pulsating alpha for glow
        alpha = int(128 + 127 * math.sin(self.steps * 0.2))
        
        # Draw glowing brackets
        for i in range(1, 4):
            pygame.draw.rect(self.screen, (*self.COLOR_CURSOR, alpha//i), (sx - size - i, sy - size - i, (size*2)+i*2, (size*2)+i*2), 1, border_radius=5)

    def _render_robotic_arm(self, cursor, base_pos, grid_center_x, grid_center_y):
        cursor_sx, cursor_sy = self._get_tile_screen_pos(cursor[0], cursor[1], grid_center_x, grid_center_y)
        mid_pos = (base_pos[0] + (cursor_sx - base_pos[0]) * 0.6, base_pos[1] + (cursor_sy - base_pos[1]) * 0.3)
        
        pygame.draw.line(self.screen, self.COLOR_GRID, base_pos, mid_pos, 10)
        pygame.draw.line(self.screen, self.COLOR_GRID, mid_pos, (cursor_sx, cursor_sy), 6)

    def _render_effects(self):
        new_effects = []
        for effect in self.visual_effects:
            effect["timer"] -= 1
            if effect["timer"] > 0:
                if effect["type"] == "flash":
                    tx, ty = effect["tile_pos"]
                    grid_center_x = effect["pos"][0]
                    grid_center_y = effect["pos"][1]
                    sx, sy = self._get_tile_screen_pos(tx, ty, grid_center_x, grid_center_y)
                    
                    alpha = int(255 * (effect["timer"] / 15))
                    radius = int((self.TILE_SIZE / 2) * (1 + (15 - effect["timer"]) / 15))
                    pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, (*self.COLOR_MATCH_FLASH, alpha))
                
                new_effects.append(effect)
        self.visual_effects = new_effects

    def _render_ui(self):
        # Time
        time_text = f"TIME: {max(0, self.time_remaining / self.FPS):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH // 2 - time_surf.get_width() // 2, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Matches
        match_text = f"MATCHES: {self.matches_made} / {self.TARGET_MATCHES}"
        match_surf = self.font_ui.render(match_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(match_surf, (self.SCREEN_WIDTH - match_surf.get_width() - 20, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.win_condition_met:
            text = "MISSION COMPLETE"
            color = (100, 255, 150)
        else:
            text = "TIME UP"
            color = (255, 100, 100)
            
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging, and is not used by the evaluation system.
    # It will be ignored, so you can leave it as is.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible display driver
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Dual Synchro Swap")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Matches: {info['matches_made']}")
            # Wait for 'R' to reset
            reset_pressed = False
            while not reset_pressed and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        reset_pressed = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        reset_pressed = True
                clock.tick(env.FPS)

        clock.tick(env.FPS)

    pygame.quit()