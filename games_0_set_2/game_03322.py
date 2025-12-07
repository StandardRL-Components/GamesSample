
# Generated: 2025-08-27T23:00:40.603373
# Source Brief: brief_03322.md
# Brief Index: 3322

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then use arrows and Space again to swap. Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically match gems in a grid to reach a target score before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    GEM_SIZE = 40
    GRID_LINE_WIDTH = 2
    WIN_SCORE = 500
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        
        self.grid_start_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.grid_start_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.GEM_SIZE) // 2

        # State variables are initialized in reset()
        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = None
        self.selected_gem = None
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.rng = None

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem = None
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False

        self._create_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self._update_particles()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # 1. Handle deselect
        if shift_press and self.selected_gem is not None:
            self.selected_gem = None
            # sfx: deselect_sound

        # 2. Handle cursor movement
        if movement != 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1) # Down
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1) # Right

        # 3. Handle selection/swap
        if space_press:
            if self.selected_gem is None:
                self.selected_gem = tuple(self.cursor_pos)
                # sfx: select_sound
            else:
                # Attempt a swap
                if self._is_adjacent(self.cursor_pos, self.selected_gem):
                    self.moves_left -= 1
                    self._swap_gems(tuple(self.cursor_pos), self.selected_gem)
                    
                    match_reward, was_match = self._process_all_matches_and_refill()
                    reward += match_reward

                    if not was_match:
                        # Invalid swap, swap back, no move cost penalty (already deducted)
                        self._swap_gems(tuple(self.cursor_pos), self.selected_gem)
                        # sfx: invalid_swap_sound
                    
                    self.selected_gem = None # Deselect after any swap attempt
                else:
                    # Not adjacent, move selection to new cursor position
                    self.selected_gem = tuple(self.cursor_pos)
                    # sfx: select_sound
        
        # 4. Check for termination
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            reward += 100 # Win bonus
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
            reward -= 100 # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_gems()
        self._draw_cursor_and_selection()
        self._draw_particles()
        self._render_ui()
        if self.game_over:
            self._draw_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "moves_left": self.moves_left, "steps": self.steps}
    
    def close(self):
        pygame.quit()

    # --- Helper Methods ---

    def _create_initial_grid(self):
        while True:
            self.grid = self.rng.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            if not self._find_all_matches() and self._find_possible_moves():
                break

    def _process_all_matches_and_refill(self):
        total_reward = 0
        is_first_match = True
        
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            if is_first_match:
                # sfx: match_sound
                is_first_match = False
            else:
                # sfx: chain_reaction_sound
                pass

            # Calculate reward and clear gems
            num_cleared = len(matches)
            reward = num_cleared  # +1 per gem
            if num_cleared == 4: reward += 10
            elif num_cleared >= 5: reward += 20
            
            self.score += reward * 10 # Scale score for display
            total_reward += reward

            for x, y in matches:
                self._spawn_particles(x, y, self.grid[x, y])
                self.grid[x, y] = -1 # Mark as empty

            self._apply_gravity_and_refill()

        return total_reward, not is_first_match

    def _find_all_matches(self):
        matches = set()
        # Horizontal matches
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x, y] != -1 and self.grid[x, y] == self.grid[x + 1, y] == self.grid[x + 2, y]:
                    matches.update([(x, y), (x + 1, y), (x + 2, y)])
        # Vertical matches
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x, y] != -1 and self.grid[x, y] == self.grid[x, y + 1] == self.grid[x, y + 2]:
                    matches.update([(x, y), (x, y + 1), (x, y + 2)])
        return matches
    
    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self._swap_gems((x, y), (x + 1, y))
                    if self._find_all_matches():
                        self._swap_gems((x, y), (x + 1, y)) # Swap back
                        return True
                    self._swap_gems((x, y), (x + 1, y)) # Swap back
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_gems((x, y), (x, y + 1))
                    if self._find_all_matches():
                        self._swap_gems((x, y), (x, y + 1)) # Swap back
                        return True
                    self._swap_gems((x, y), (x, y + 1)) # Swap back
        return False

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[x, y + empty_count] = self.grid[x, y]
                    self.grid[x, y] = -1
            
            for y in range(empty_count):
                self.grid[x, y] = self.rng.integers(0, self.NUM_GEM_TYPES)

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _swap_gems(self, pos1, pos2):
        self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]

    def _spawn_particles(self, grid_x, grid_y, gem_type):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        center_x, center_y = px + self.GEM_SIZE // 2, py + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2.0 + 1.0
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.rng.integers(15, 30)
            self.particles.append([center_x, center_y, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 1    # life -= 1

    # --- Drawing Methods ---

    def _grid_to_pixel(self, x, y):
        return self.grid_start_x + x * self.GEM_SIZE, self.grid_start_y + y * self.GEM_SIZE

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.grid_start_x + x * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.grid_start_y), (px, self.grid_start_y + self.GRID_HEIGHT * self.GEM_SIZE), self.GRID_LINE_WIDTH)
        for y in range(self.GRID_HEIGHT + 1):
            py = self.grid_start_y + y * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_start_x, py), (self.grid_start_x + self.GRID_WIDTH * self.GEM_SIZE, py), self.GRID_LINE_WIDTH)

    def _draw_gems(self):
        radius = self.GEM_SIZE // 2 - 4
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[x, y]
                if gem_type != -1:
                    px, py = self._grid_to_pixel(x, y)
                    center_x, center_y = px + self.GEM_SIZE // 2, py + self.GEM_SIZE // 2
                    color = self.GEM_COLORS[gem_type]
                    
                    # 3D effect
                    dark_color = tuple(max(0, c - 50) for c in color)
                    light_color = tuple(min(255, c + 50) for c in color)
                    
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    
                    # Highlight
                    pygame.gfxdraw.filled_circle(self.screen, center_x - radius//3, center_y - radius//3, radius//3, light_color)

    def _draw_cursor_and_selection(self):
        # Draw selected gem highlight
        if self.selected_gem is not None:
            px, py = self._grid_to_pixel(self.selected_gem[0], self.selected_gem[1])
            rect = pygame.Rect(px, py, self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3)
            
        # Draw cursor
        px, py = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        rect = pygame.Rect(px, py, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _draw_particles(self):
        for x, y, vx, vy, life, color in self.particles:
            alpha = max(0, min(255, int(255 * (life / 30.0))))
            temp_surface = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, (*color, alpha), (2, 2), 2)
            self.screen.blit(temp_surface, (int(x) - 2, int(y) - 2))
    
    def _render_ui(self):
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

    def _draw_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
        text_surface = self.font_large.render(message, True, self.COLOR_CURSOR)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
        
        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for human play
    pygame.display.set_caption("Gem Matcher")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    # Game loop for human player
    action = [0, 0, 0] # no-op, no-press, no-press
    
    while not terminated:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Key state handling
        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0] # Movement, Space, Shift
        
        # This is a bit clunky for human play; we detect presses, not holds
        # But it matches the gym action space.
        # So we'll just register the key if it's down on this frame.
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # The internal screen surface is correct, so we'll just use that
        display_screen.blit(env.screen, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need a small delay for human playability
        env.clock.tick(15) # Limit to 15 FPS for human input

    env.close()
    print(f"Game Over! Final Score: {info['score']}")