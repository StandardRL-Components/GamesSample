
# Generated: 2025-08-27T22:18:14.695865
# Source Brief: brief_03082.md
# Brief Index: 3082

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to flip a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced memory game. Match all the pairs of colored tiles before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 4, 4
    TILE_COUNT = GRID_ROWS * GRID_COLS
    
    # Timings (in frames, assuming 30 FPS)
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * 30
    MISMATCH_DELAY_FRAMES = 15 # How long to show a mismatch
    ANIMATION_SPEED = 0.1 # Progress per frame for flip animation
    
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (50, 55, 68)
    COLOR_TILE_HIDDEN = (71, 78, 93)
    COLOR_CURSOR = (255, 204, 34)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TIMER_GOOD = (66, 194, 119)
    COLOR_TIMER_WARN = (255, 204, 34)
    COLOR_TIMER_BAD = (224, 77, 86)
    COLOR_MISMATCH_FLASH = (224, 77, 86)
    
    TILE_PALETTE = [
        (41, 128, 185), (231, 76, 60), (46, 204, 113), (155, 89, 182),
        (241, 196, 15), (26, 188, 156), (52, 73, 94), (230, 126, 34)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Calculate grid layout
        self.grid_area_width = self.SCREEN_WIDTH - 100
        self.grid_area_height = self.SCREEN_HEIGHT - 100
        self.tile_size = min(self.grid_area_width // self.GRID_COLS, self.grid_area_height // self.GRID_ROWS) - 10
        self.grid_width = self.GRID_COLS * (self.tile_size + 10)
        self.grid_height = self.GRID_ROWS * (self.tile_size + 10)
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 30
        
        # Initialize state variables
        self.tiles = []
        self.cursor_pos = [0, 0]
        self.first_selection = None
        self.mismatched_pair = []
        self.mismatch_timer = 0
        self.previous_space_state = False
        self.matched_pairs_count = 0
        self.time_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.matched_pairs_count = 0
        self.first_selection = None
        self.mismatched_pair = []
        self.mismatch_timer = 0
        self.previous_space_state = False
        self.cursor_pos = [0, 0]

        # Generate tile grid
        tile_ids = list(range(self.TILE_COUNT // 2)) * 2
        self.np_random.shuffle(tile_ids)
        
        self.tiles = []
        for r in range(self.GRID_ROWS):
            row_tiles = []
            for c in range(self.GRID_COLS):
                tile_id = tile_ids[r * self.GRID_COLS + c]
                x = self.grid_offset_x + c * (self.tile_size + 10) + 5
                y = self.grid_offset_y + r * (self.tile_size + 10) + 5
                tile = {
                    "id": tile_id,
                    "state": "hidden", # hidden, revealing, revealed, hiding, matched
                    "anim_progress": 0.0,
                    "color": self.TILE_PALETTE[tile_id],
                    "rect": pygame.Rect(x, y, self.tile_size, self.tile_size),
                    "pos": (r, c)
                }
                row_tiles.append(tile)
            self.tiles.append(row_tiles)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        space_pressed = space_held and not self.previous_space_state
        self.previous_space_state = bool(space_held)
        
        reward = self._update_game_state(movement, space_pressed)
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.matched_pairs_count == self.TILE_COUNT // 2:
            reward += 50
            self.score += 50
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement, space_pressed):
        reward = 0.1 # Base reward for surviving a step

        # --- Handle Timers and Animations ---
        self.time_left = max(0, self.time_left - 1)

        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                for r, c in self.mismatched_pair:
                    self.tiles[r][c]["state"] = "hiding"
                    self.tiles[r][c]["anim_progress"] = 0.0
                self.mismatched_pair = []
                # Sound: tile_flip_down.wav

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.tiles[r][c]
                if tile["state"] in ["revealing", "hiding"]:
                    tile["anim_progress"] += self.ANIMATION_SPEED
                    if tile["anim_progress"] >= 1.0:
                        tile["anim_progress"] = 1.0
                        if tile["state"] == "revealing":
                            tile["state"] = "revealed"
                        elif tile["state"] == "hiding":
                            tile["state"] = "hidden"

        # --- Handle Input ---
        if self.mismatch_timer == 0: # Lock input during mismatch display
            # Move cursor
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
            
            # Select tile
            if space_pressed:
                r, c = self.cursor_pos
                selected_tile = self.tiles[r][c]

                if selected_tile["state"] == "hidden":
                    # Sound: tile_flip_up.wav
                    selected_tile["state"] = "revealing"
                    selected_tile["anim_progress"] = 0.0

                    if self.first_selection is None:
                        self.first_selection = (r, c)
                    else:
                        first_r, first_c = self.first_selection
                        first_tile = self.tiles[first_r][first_c]

                        if (r, c) != (first_r, first_c):
                            if selected_tile["id"] == first_tile["id"]: # Match
                                # Sound: match_success.wav
                                first_tile["state"] = "matched"
                                selected_tile["state"] = "matched"
                                self.matched_pairs_count += 1
                                reward += 10
                                self.score += 10
                                self.first_selection = None
                            else: # Mismatch
                                # Sound: mismatch_error.wav
                                self.mismatched_pair = [(first_r, first_c), (r, c)]
                                self.mismatch_timer = self.MISMATCH_DELAY_FRAMES
                                reward = -1 # Override base reward
                                self.score = max(0, self.score - 1)
                                self.first_selection = None

        return reward

    def _check_termination(self):
        if self.matched_pairs_count == self.TILE_COUNT // 2:
            self.game_over = True
            return True
        if self.time_left <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background rects
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.tiles[r][c]
                pygame.draw.rect(self.screen, self.COLOR_GRID, tile["rect"], border_radius=5)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.tiles[r][c]
                rect = tile["rect"]
                
                if tile["state"] == "hidden":
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect, border_radius=5)
                elif tile["state"] in ["revealing", "hiding"]:
                    self._render_flip_animation(tile)
                elif tile["state"] == "revealed":
                    pygame.draw.rect(self.screen, tile["color"], rect, border_radius=5)
                    if self.mismatch_timer > 0 and tile["pos"] in self.mismatched_pair:
                        s = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                        flash_alpha = 128 * (math.sin(self.steps * 0.5) * 0.5 + 0.5)
                        s.fill((*self.COLOR_MISMATCH_FLASH, flash_alpha))
                        self.screen.blit(s, rect.topleft)
                elif tile["state"] == "matched":
                    # Flash effect on match
                    s = pygame.Surface((self.tile_size, self.tile_size))
                    s.set_alpha(100)
                    s.fill((255,255,255))
                    self.screen.blit(s, rect.topleft)
        
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = self.tiles[cursor_r][cursor_c]["rect"].inflate(8, 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)

    def _render_flip_animation(self, tile):
        progress = tile["anim_progress"]
        rect = tile["rect"]
        
        is_revealing = tile["state"] == "revealing"
        is_hiding = tile["state"] == "hiding"

        # Determine which color to show based on animation progress
        show_front = (is_revealing and progress > 0.5) or (is_hiding and progress < 0.5)
        color = tile["color"] if show_front else self.COLOR_TILE_HIDDEN

        # Calculate width based on a cosine curve for smooth acceleration/deceleration
        scale = abs(math.cos(progress * math.pi))
        
        anim_width = int(rect.width * scale)
        anim_rect = pygame.Rect(
            rect.centerx - anim_width // 2,
            rect.y,
            anim_width,
            rect.height
        )
        pygame.draw.rect(self.screen, color, anim_rect, border_radius=5)


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Timer bar
        timer_pct = self.time_left / self.MAX_STEPS
        bar_width = self.SCREEN_WIDTH - 40
        current_width = int(bar_width * timer_pct)
        
        timer_color = self.COLOR_TIMER_GOOD
        if timer_pct < 0.5: timer_color = self.COLOR_TIMER_WARN
        if timer_pct < 0.2: timer_color = self.COLOR_TIMER_BAD

        pygame.draw.rect(self.screen, self.COLOR_GRID, (20, 50, bar_width, 10), border_radius=5)
        if current_width > 0:
            pygame.draw.rect(self.screen, timer_color, (20, 50, current_width, 10), border_radius=5)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "matched_pairs": self.matched_pairs_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display for human playing
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # Human input handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        keys = pygame.key.get_pressed()
        
        # Map pygame keys to the MultiDiscrete action space
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([movement, space_held, shift_held])
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))

        if terminated:
            # Display game over message
            s = pygame.Surface((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            display_screen.blit(s, (0,0))
            
            font = pygame.font.Font(None, 60)
            msg = "YOU WIN!" if info['matched_pairs'] == GameEnv.TILE_COUNT // 2 else "TIME'S UP!"
            text = font.render(msg, True, (255, 255, 255))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 20))
            display_screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 30)
            restart_text = font_small.render("Press 'R' to restart", True, (200, 200, 200))
            restart_rect = restart_text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 30))
            display_screen.blit(restart_text, restart_rect)

        pygame.display.flip()
        env.clock.tick(30)

    env.close()