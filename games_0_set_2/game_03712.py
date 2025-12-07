
# Generated: 2025-08-28T00:12:25.219601
# Source Brief: brief_03712.md
# Brief Index: 3712

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "then move to an adjacent tile and press Space again to swap. Press Shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Clear the entire board before you run out of moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.NUM_TILE_TYPES = 5
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000

        # Visuals
        self.TILE_SIZE = 52
        self.GRID_LINE_WIDTH = 2
        self.GRID_WIDTH = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 10

        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECTED = (0, 200, 255)
        self.COLOR_MATCH_FLASH = (255, 255, 255)
        self.TILE_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 120, 255),   # Blue
            (255, 255, 80),   # Yellow
            (200, 80, 255),   # Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_game_over = pygame.font.Font(None, 52)
        
        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.selected_tile = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_state = 0
        self.last_shift_state = 0
        
        # Visual feedback state (reset each step)
        self.matched_tiles_to_flash = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.last_space_state = 0
        self.last_shift_state = 0
        self.matched_tiles_to_flash = []

        self._create_initial_board()
        
        return self._get_observation(), self._get_info()

    def _create_initial_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_matches():
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.matched_tiles_to_flash = []
        reward = 0
        
        step_reward = self._handle_input(action)
        reward += step_reward

        self.steps += 1
        terminated = self.game_over

        if not terminated and (self.moves_left <= 0 or self.steps >= self.MAX_STEPS):
            terminated = True
            if not self._is_board_clear():
                reward += -100  # Loss penalty
            self.game_over = True
        
        if self._is_board_clear():
            terminated = True
            reward += 100 # Win Bonus
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_val, shift_val = action
        space_pressed = space_val == 1 and self.last_space_state == 0
        shift_pressed = shift_val == 1 and self.last_shift_state == 0
        self.last_space_state = space_val
        self.last_shift_state = shift_val

        if shift_pressed and self.selected_tile:
            self.selected_tile = None
            return 0

        if movement != 0:
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        
        if space_pressed:
            if not self.selected_tile:
                self.selected_tile = list(self.cursor_pos)
            else:
                if self.selected_tile == list(self.cursor_pos):
                    self.selected_tile = None # Deselect if pressing space on same tile
                elif self._is_adjacent(self.selected_tile, self.cursor_pos):
                    reward = self._attempt_swap(self.selected_tile, self.cursor_pos)
                    self.selected_tile = None
                    return reward
                else:
                    self.selected_tile = list(self.cursor_pos) # Select new, non-adjacent tile
        return 0

    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _attempt_swap(self, pos1, pos2):
        if self.moves_left <= 0: return 0
        self.moves_left -= 1
        # SFX: swap.wav

        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        total_cleared, _ = self._process_matches()
        
        if total_cleared == 0:
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # SFX: invalid_swap.wav
            return 0
        else:
            self.score += total_cleared
            return total_cleared

    def _process_matches(self):
        total_cleared = 0
        is_cascade = False
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            if is_cascade: # SFX: cascade_match.wav
                pass
            else: # SFX: match.wav
                pass

            num_cleared = len(matches)
            total_cleared += num_cleared
            self.matched_tiles_to_flash.extend(list(matches))
            
            for r, c in matches:
                self.grid[r, c] = -1

            if self._is_board_clear():
                break

            self._apply_gravity()
            self._refill_grid()
            is_cascade = True
            
        return total_cleared, is_cascade

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                tile_val = self.grid[r, c]
                if tile_val != -1 and self.grid[r, c+1] == tile_val and self.grid[r, c+2] == tile_val:
                    for i in range(3): matches.add((r, c + i))
                    # Check for longer matches
                    for i in range(3, self.GRID_SIZE - c):
                        if self.grid[r, c + i] == tile_val: matches.add((r, c + i))
                        else: break

        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                tile_val = self.grid[r, c]
                if tile_val != -1 and self.grid[r+1, c] == tile_val and self.grid[r+2, c] == tile_val:
                    for i in range(3): matches.add((r + i, c))
                    # Check for longer matches
                    for i in range(3, self.GRID_SIZE - r):
                        if self.grid[r + i, c] == tile_val: matches.add((r + i, c))
                        else: break
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            write_idx = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != write_idx:
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    write_idx -= 1

    def _refill_grid(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    def _is_board_clear(self):
        return np.all(self.grid == -1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_val = self.grid[r, c]
                if tile_val == -1: continue

                x = self.GRID_X + c * self.TILE_SIZE
                y = self.GRID_Y + r * self.TILE_SIZE
                tile_rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                
                # Use gfxdraw for antialiasing
                inset = 4
                color = self.TILE_COLORS[tile_val]
                
                # Main gem shape
                pygame.gfxdraw.box(self.screen, tile_rect.inflate(-inset, -inset), color)
                pygame.gfxdraw.rectangle(self.screen, tile_rect.inflate(-inset, -inset), tuple(min(255, v+20) for v in color))

                # Highlight
                highlight_rect = pygame.Rect(tile_rect.left + inset, tile_rect.top + inset, self.TILE_SIZE - 2 * inset, (self.TILE_SIZE - 2*inset)//3)
                pygame.draw.rect(self.screen, (255,255,255,50), highlight_rect)

        # Draw flash effect for matched tiles
        for r, c in self.matched_tiles_to_flash:
            x = self.GRID_X + c * self.TILE_SIZE
            y = self.GRID_Y + r * self.TILE_SIZE
            flash_rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE).inflate(-4, -4)
            pygame.draw.rect(self.screen, (255, 255, 255, 180), flash_rect, border_radius=4)
            # SFX: particle_burst.wav

        # Draw selected tile outline
        if self.selected_tile:
            r, c = self.selected_tile
            x = self.GRID_X + c * self.TILE_SIZE
            y = self.GRID_Y + r * self.TILE_SIZE
            sel_rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, sel_rect, 4, border_radius=4)

        # Draw cursor
        r, c = self.cursor_pos
        x = self.GRID_X + c * self.TILE_SIZE
        y = self.GRID_Y + r * self.TILE_SIZE
        cursor_rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
        
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        width = 2 + int(pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width, border_radius=4)
    
    def _draw_text(self, text, pos, color, font, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _render_ui(self):
        self._draw_text(f"SCORE: {self.score}", (20, 15), self.COLOR_TEXT, self.font_ui)
        self._draw_text(f"MOVES: {self.moves_left}", (self.WIDTH - 150, 15), self.COLOR_TEXT, self.font_ui)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "BOARD CLEARED!" if self._is_board_clear() else "GAME OVER"
            self._draw_text(win_text, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.COLOR_CURSOR, self.font_game_over, center=True)
            self._draw_text(f"Final Score: {self.score}", (self.WIDTH // 2, self.HEIGHT // 2 + 30), self.COLOR_TEXT, self.font_ui, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_board_clear": self._is_board_clear()
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    
    action = [0, 0, 0] # no-op, released, released
    
    print(env.user_guide)

    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                print("--- Game Reset ---")

        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step the environment ---
        # Since auto_advance is False, we only step on an action.
        # For a human player, this means we step every frame to register input.
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}, Moves Left: {info['moves_left']}")
            
        if terminated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")
            # The game state is now frozen. Wait for reset.

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()