import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:53:13.857859
# Source Brief: brief_00078.md
# Brief Index: 78
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    game_description = "Slide rows and columns to sort the numbered tiles in this physics-based puzzle against the clock."
    user_guide = "Use arrow keys (↑↓←→) to move the selector. Hold space and press an arrow key to slide the selected row or column."
    auto_advance = True

    def __init__(self, render_mode="rgb_array", grid_size=4):
        super().__init__()
        
        self.GRID_SIZE = grid_size
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
            self.font_medium = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 60)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 22)
        
        # Game constants
        self.MAX_STEPS = 180 * self.metadata["render_fps"] # 3 minutes
        
        # Visuals
        self.COLOR_BG = (26, 42, 58) # #1a2a3a
        self.COLOR_GRID = (42, 58, 74) # #2a3a4a
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_SELECTOR = (255, 200, 0)
        self.COLOR_COMPLETED = (0, 0, 0, 100)
        self.COLOR_TILE_START = np.array([160, 210, 235]) # light blue
        self.COLOR_TILE_END = np.array([240, 128, 128]) # light coral
        
        # Physics
        self.SPRING_CONSTANT = 0.03
        self.DRAG_FACTOR = 0.92
        self.MOMENTUM_IMPULSE = 15.0
        self.MOMENTUM_DECAY_INTERVAL = 5 # moves
        self.MOMENTUM_DECAY_FACTOR = 0.8
        
        # Rewards
        self.REWARD_WIN = 100.0
        self.REWARD_LOSS = -100.0
        self.REWARD_ROW_COMPLETE = 5.0
        self.REWARD_DISTANCE_SCALE = 0.1
        self.PENALTY_MOMENTUM_SCALE = -0.01

        # Grid layout
        self.GRID_DIM = 320
        self.TILE_SIZE = self.GRID_DIM // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_DIM) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_DIM) // 2

        # State variables are initialized in reset()
        self.grid = None
        self.tiles = None
        self.selector_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_state = None
        self.moves_since_decay = None
        self.completed_rows = None
        self.completed_cols = None
        self.game_speed_modifier = None
        self.last_total_dist = None
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is not needed in the final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_since_decay = 0
        self.completed_rows = set()
        self.completed_cols = set()
        self.game_speed_modifier = 1.0

        self.selector_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])

        # Initialize tiles
        tile_ids = list(range(1, self.GRID_SIZE * self.GRID_SIZE + 1))
        self.grid = np.array(tile_ids).reshape((self.GRID_SIZE, self.GRID_SIZE))
        
        self.tiles = {}
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_id = self.grid[r, c]
                pos = self._get_grid_center(r, c)
                self.tiles[tile_id] = {
                    "id": tile_id,
                    "pos": pos.copy(),
                    "momentum": np.array([0.0, 0.0]),
                    "grid_pos": np.array([r, c])
                }

        # Shuffle the board by making random moves
        for _ in range(50): # Ensure a solvable, shuffled state
            axis = self.np_random.integers(0, 2) # 0 for row, 1 for col
            index = self.np_random.integers(0, self.GRID_SIZE)
            direction = self.np_random.choice([-1, 1])
            self._slide(axis, index, direction)
        
        # Reset tile visual positions after shuffling
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_id = self.grid[r, c]
                self.tiles[tile_id]["grid_pos"] = np.array([r, c])
                self.tiles[tile_id]["pos"] = self._get_grid_center(r, c)
                self.tiles[tile_id]["momentum"] = np.array([0.0, 0.0])

        self.last_total_dist = self._get_total_manhattan_distance()

        return self._get_observation(), self._get_info()

    def _slide(self, axis, index, direction):
        """ Slides a row (axis=0) or column (axis=1). """
        if axis == 0: # Slide row
            self.grid[index, :] = np.roll(self.grid[index, :], direction)
        else: # Slide col
            self.grid[:, index] = np.roll(self.grid[:, index], direction)
        
        # Update tile grid positions
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_id = self.grid[r, c]
                self.tiles[tile_id]["grid_pos"] = np.array([r, c])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        action_taken = False
        reward = 0

        # 1. Handle Input
        if space_held == 1 and movement in [1, 2, 3, 4]:
            # sound: slide_whoosh.wav
            action_taken = True
            sel_r, sel_c = self.selector_pos
            
            if movement == 1: # Up
                self._slide(axis=1, index=sel_c, direction=-1)
                for r in range(self.GRID_SIZE):
                    self.tiles[self.grid[r, sel_c]]["momentum"][1] -= self.MOMENTUM_IMPULSE
            elif movement == 2: # Down
                self._slide(axis=1, index=sel_c, direction=1)
                for r in range(self.GRID_SIZE):
                    self.tiles[self.grid[r, sel_c]]["momentum"][1] += self.MOMENTUM_IMPULSE
            elif movement == 3: # Left
                self._slide(axis=0, index=sel_r, direction=-1)
                for c in range(self.GRID_SIZE):
                    self.tiles[self.grid[sel_r, c]]["momentum"][0] -= self.MOMENTUM_IMPULSE
            elif movement == 4: # Right
                self._slide(axis=0, index=sel_r, direction=1)
                for c in range(self.GRID_SIZE):
                    self.tiles[self.grid[sel_r, c]]["momentum"][0] += self.MOMENTUM_IMPULSE
            
            self.moves_since_decay += 1

        elif movement != 0: # Move selector
            # sound: selector_tick.wav
            if movement == 1: self.selector_pos[0] = (self.selector_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
            elif movement == 2: self.selector_pos[0] = (self.selector_pos[0] + 1) % self.GRID_SIZE
            elif movement == 3: self.selector_pos[1] = (self.selector_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
            elif movement == 4: self.selector_pos[1] = (self.selector_pos[1] + 1) % self.GRID_SIZE
        
        # 2. Update Physics
        total_momentum = 0
        for tile in self.tiles.values():
            target_pos = self._get_grid_center(tile["grid_pos"][0], tile["grid_pos"][1])
            spring_force = (target_pos - tile["pos"]) * self.SPRING_CONSTANT
            tile["momentum"] += spring_force
            tile["momentum"] *= self.DRAG_FACTOR
            tile["pos"] += tile["momentum"]
            total_momentum += np.linalg.norm(tile["momentum"])

        # 3. Momentum Decay Event
        if self.moves_since_decay >= self.MOMENTUM_DECAY_INTERVAL:
            for tile in self.tiles.values():
                tile["momentum"] *= self.MOMENTUM_DECAY_FACTOR
            self.moves_since_decay = 0

        # 4. Calculate Rewards
        if action_taken:
            # Distance-based reward
            new_total_dist = self._get_total_manhattan_distance()
            reward += self.REWARD_DISTANCE_SCALE * (self.last_total_dist - new_total_dist)
            self.last_total_dist = new_total_dist
            
            # Completion-based reward
            prev_completed_count = len(self.completed_rows) + len(self.completed_cols)
            self._check_completion_state()
            new_completed_count = len(self.completed_rows) + len(self.completed_cols)
            if new_completed_count > prev_completed_count:
                # sound: success_chime.wav
                reward += self.REWARD_ROW_COMPLETE * (new_completed_count - prev_completed_count)
            
            # Update game speed based on completion
            self.game_speed_modifier = max(0.5, 1.0 - 0.05 * new_completed_count)

        # Momentum penalty (applied every step to encourage stability)
        reward += self.PENALTY_MOMENTUM_SCALE * total_momentum
        
        self.score += reward
        self.steps += 1

        # 5. Check Termination
        terminated = False
        truncated = False
        if self.win_state:
            # sound: victory.wav
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # sound: failure_buzzer.wav
            reward += self.REWARD_LOSS
            self.score += self.REWARD_LOSS
            terminated = True # Per Gymnasium API, time limit is termination
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _check_completion_state(self):
        self.completed_rows.clear()
        self.completed_cols.clear()
        is_won = True
        
        # Check rows
        for r in range(self.GRID_SIZE):
            is_row_sorted = all(self.grid[r, c] == r * self.GRID_SIZE + c + 1 for c in range(self.GRID_SIZE))
            if is_row_sorted:
                self.completed_rows.add(r)
            else:
                is_won = False
        
        # Check columns (less common but possible)
        for c in range(self.GRID_SIZE):
            is_col_sorted = all(self.grid[r, c] == r * self.GRID_SIZE + c + 1 for r in range(self.GRID_SIZE))
            if is_col_sorted:
                self.completed_cols.add(c)
        
        # A full win requires all rows to be sorted
        self.win_state = is_won

    def _get_total_manhattan_distance(self):
        total_dist = 0
        for tile in self.tiles.values():
            correct_r = (tile["id"] - 1) // self.GRID_SIZE
            correct_c = (tile["id"] - 1) % self.GRID_SIZE
            current_r, current_c = tile["grid_pos"]
            total_dist += abs(correct_r - current_r) + abs(correct_c - current_c)
        return total_dist

    def _get_grid_center(self, r, c):
        x = self.GRID_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
        y = self.GRID_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
        return np.array([x, y])
    
    def _get_tile_color(self, tile_id, momentum_mag):
        # Interpolate base color
        t = (tile_id - 1) / max(1, self.GRID_SIZE * self.GRID_SIZE - 1)
        base_color = self.COLOR_TILE_START * (1 - t) + self.COLOR_TILE_END * t
        
        # Increase brightness/saturation with momentum
        momentum_boost = min(1.0, momentum_mag / (self.MOMENTUM_IMPULSE * 0.5))
        lightness_factor = 1.0 + 0.5 * momentum_boost
        
        color = np.clip(base_color * lightness_factor, 0, 255)
        return tuple(color.astype(int))
    
    def _render_game(self):
        # Draw grid background and completed highlights
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.TILE_SIZE,
                    self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
                
                if r in self.completed_rows or c in self.completed_cols:
                    s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    s.fill(self.COLOR_COMPLETED)
                    self.screen.blit(s, rect.topleft)

        # Draw tiles
        tile_radius = int(self.TILE_SIZE * 0.1)
        for tile_id in sorted(self.tiles.keys()):
            tile = self.tiles[tile_id]
            momentum_mag = np.linalg.norm(tile["momentum"])
            color = self._get_tile_color(tile["id"], momentum_mag)
            
            center_x, center_y = tile["pos"]
            tile_rect = pygame.Rect(
                int(center_x - self.TILE_SIZE / 2 + 5),
                int(center_y - self.TILE_SIZE / 2 + 5),
                self.TILE_SIZE - 10, self.TILE_SIZE - 10
            )
            pygame.draw.rect(self.screen, color, tile_rect, border_radius=tile_radius)

            text_surf = self.font_medium.render(str(tile["id"]), True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(int(center_x), int(center_y)))
            self.screen.blit(text_surf, text_rect)

        # Draw selector
        sel_r, sel_c = self.selector_pos
        selector_rect = pygame.Rect(
            self.GRID_OFFSET_X + sel_c * self.TILE_SIZE,
            self.GRID_OFFSET_Y + sel_r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, 3, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.metadata["render_fps"]
        time_left = max(0, time_left)
        mins, secs = divmod(time_left, 60)
        timer_text = self.font_medium.render(f"Time: {int(mins):02}:{int(secs):02}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(timer_text, timer_rect)

        # Game Speed
        speed_text = self.font_small.render(f"Speed Mod: {int(self.game_speed_modifier * 100)}%", True, self.COLOR_TEXT)
        speed_rect = speed_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 50))
        self.screen.blit(speed_text, speed_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_state else "TIME'S UP"
            color = (100, 255, 100) if self.win_state else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.metadata["render_fps"],
            "win": self.win_state
        }
    
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To play the game manually
    # Set the video driver to a real one for display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Tile Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        
        # Poll for events and keys once per frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle single key presses for selector movement
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_held = 1
            # If space is held, override selector movement with slide action
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Win: {info['win']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(env.metadata["render_fps"])

    env.close()