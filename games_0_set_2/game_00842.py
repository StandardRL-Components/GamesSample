
# Generated: 2025-08-27T14:56:59.717331
# Source Brief: brief_00842.md
# Brief Index: 842

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your selection cursor. Press Shift to cycle through adjacent gems to swap with. Press Space to confirm the swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of three or more. Create combos to maximize your score and reach 1000 points before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visuals & Game Constants
        self.GRID_SIZE = 10
        self.NUM_GEM_TYPES = 3
        self.TARGET_SCORE = 1000
        self.MAX_STEPS = 1000
        
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_PIXEL_SIZE = 400
        self.CELL_SIZE = self.GRID_PIXEL_SIZE // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_PIXEL_SIZE) // 2
        self.GRID_OFFSET_Y = 0
        self.GEM_RADIUS = int(self.CELL_SIZE * 0.4)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SWAP_TARGET = (255, 255, 0)
        self.COLOR_FLASH = (255, 255, 200)
        self.GEM_COLORS = [
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 100, 220),  # Blue
        ]

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.swap_direction_idx = None
        self.score = None
        self.steps = None
        self.possible_moves = None
        self.game_over = None
        self.win = None
        
        # Animation state (cleared each step)
        self.animation_flash_coords = set()
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self._generate_grid()
        self.possible_moves = self._count_possible_moves()

        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.swap_direction_idx = 0 # 0:Up, 1:Right, 2:Down, 3:Left
        
        self.animation_flash_coords.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Clear previous step's animations
        self.animation_flash_coords.clear()

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        self.steps += 1

        # --- Handle Actions ---
        if shift_press:
            self.swap_direction_idx = (self.swap_direction_idx + 1) % 4
            # SFX: cursor_target_change.wav
        
        self._handle_movement(movement)

        if space_press:
            reward, terminated = self._attempt_swap()
        
        # --- Check Termination Conditions ---
        if not terminated:
            if self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True
                reward -= 10 # Penalty for running out of time
            elif self.score >= self.TARGET_SCORE:
                terminated = True
                self.game_over = True
                self.win = True
                reward += 100 # Win bonus
            elif self.possible_moves == 0:
                terminated = True
                self.game_over = True
                reward -= 10 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_movement(self, movement):
        r, c = self.cursor_pos
        if movement == 1: # Up
            r = (r - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: # Down
            r = (r + 1) % self.GRID_SIZE
        elif movement == 3: # Left
            c = (c - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: # Right
            c = (c + 1) % self.GRID_SIZE
        
        if (r,c) != self.cursor_pos:
            self.cursor_pos = (r, c)
            # SFX: cursor_move.wav

    def _attempt_swap(self):
        r1, c1 = self.cursor_pos
        r2, c2 = self._get_swap_target_pos()

        # Swap gems
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        # Check for matches
        all_matches = self._find_matches()

        if not all_matches:
            # Invalid swap, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # SFX: invalid_swap.wav
            return -0.1, False # Small penalty for invalid move
        
        # --- Successful Swap & Cascade ---
        # SFX: match_success.wav
        total_reward = 0
        combo_multiplier = 1.0

        while all_matches:
            # Add to score and reward
            num_matched = len(all_matches)
            total_reward += num_matched * combo_multiplier
            self.score += num_matched * int(combo_multiplier * 10)
            
            if combo_multiplier > 1.0:
                total_reward += 5 # Combo bonus
                # SFX: combo.wav
            
            # Store for animation
            self.animation_flash_coords.update(all_matches)

            # Remove matched gems (set to -1 to mark for removal)
            for r, c in all_matches:
                self.grid[r, c] = -1

            # Apply gravity
            self._apply_gravity()

            # Fill empty spaces from top
            self._fill_top_rows()

            # Check for new matches (cascade)
            all_matches = self._find_matches()
            combo_multiplier += 0.5
        
        # Update possible moves count for the new board state
        self.possible_moves = self._count_possible_moves()
        
        return total_reward, False

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem = self.grid[r, c]
                if gem == -1: continue

                # Horizontal check
                if c < self.GRID_SIZE - 2 and self.grid[r, c+1] == gem and self.grid[r, c+2] == gem:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
                
                # Vertical check
                if r < self.GRID_SIZE - 2 and self.grid[r+1, c] == gem and self.grid[r+2, c] == gem:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return matches
    
    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _fill_top_rows(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _generate_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            # Ensure no initial matches
            while self._find_matches():
                matches = self._find_matches()
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
            
            # Ensure there are possible moves
            if self._count_possible_moves() >= 3:
                break

    def _count_possible_moves(self):
        count = 0
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check swap with right neighbor
                if c < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches():
                        count += 1
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c] # Swap back
                
                # Check swap with down neighbor
                if r < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches():
                        count += 1
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c] # Swap back
        return count

    def _get_swap_target_pos(self):
        r, c = self.cursor_pos
        if self.swap_direction_idx == 0: # Up
            return (r - 1 + self.GRID_SIZE) % self.GRID_SIZE, c
        elif self.swap_direction_idx == 1: # Right
            return r, (c + 1) % self.GRID_SIZE
        elif self.swap_direction_idx == 2: # Down
            return (r + 1) % self.GRID_SIZE, c
        else: # Left
            return r, (c - 1 + self.GRID_SIZE) % self.GRID_SIZE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_PIXEL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_PIXEL_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        
        # Draw gems and animations
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type == -1: continue

                center_x = self.GRID_OFFSET_X + int((c + 0.5) * self.CELL_SIZE)
                center_y = self.GRID_OFFSET_Y + int((r + 0.5) * self.CELL_SIZE)
                
                # Draw flash effect for matched gems
                if (r, c) in self.animation_flash_coords:
                    flash_radius = int(self.CELL_SIZE * 0.5)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, flash_radius, self.COLOR_FLASH)
                    
                # Draw gem
                color = self.GEM_COLORS[gem_type]
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.GEM_RADIUS, color)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.GEM_RADIUS, color)
                
                # Simple highlight for 3D effect
                highlight_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
                pygame.gfxdraw.arc(self.screen, center_x, center_y, self.GEM_RADIUS - 2, 135, 225, highlight_color)


        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cursor_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=4)
        
        # Draw swap target highlight
        target_r, target_c = self._get_swap_target_pos()
        rect = pygame.Rect(
            self.GRID_OFFSET_X + target_c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + target_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SWAP_TARGET, rect, 2, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.GRID_OFFSET_X + self.GRID_PIXEL_SIZE + 20, 20))

        # Target Score
        target_text = self.font_small.render(f"TARGET: {self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (self.GRID_OFFSET_X + self.GRID_PIXEL_SIZE + 20, 60))

        # Possible Moves
        moves_text = self.font_main.render(f"MOVES: {self.possible_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.GRID_OFFSET_X + self.GRID_PIXEL_SIZE + 20, 100))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (20, 20))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "possible_moves": self.possible_moves,
            "cursor_pos": self.cursor_pos,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # This setup allows for human play testing.
    # It converts keyboard presses into the MultiDiscrete action format.
    
    pygame.display.set_caption("Gem Matcher")
    render_screen = pygame.display.set_mode((640, 400))
    
    running = True
    total_reward = 0
    
    # Game loop for human play
    while running:
        # Convert observation for display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"--- Game Reset ---")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}, Moves: {info['possible_moves']}")

            if terminated:
                print("--- Episode Finished ---")
                # Render final frame
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                render_screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(2000) # Wait 2 seconds before auto-reset
                
                obs, info = env.reset()
                total_reward = 0
                print(f"--- New Game Started ---")

    env.close()