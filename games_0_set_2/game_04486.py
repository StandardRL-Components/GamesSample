
# Generated: 2025-08-28T02:33:21.350914
# Source Brief: brief_04486.md
# Brief Index: 4486

        
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

    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to swap with the gem to the right. Shift to swap with the gem above."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Create chain reactions for big scores! Reach 1000 points before you run out of 50 moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.CELL_SIZE = 40
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.MAX_MOVES = 50
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 255, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 255, 50),  # Yellow
            (200, 50, 255),  # Purple
            (255, 150, 50),  # Orange
        ]
        self.GEM_HIGHLIGHTS = [pygame.Color(c).lerp((255, 255, 255), 0.5) for c in self.GEM_COLORS]

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.win_state = None
        self.last_action_effects = [] # For rendering visual effects

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        self.win_state = None
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_action_effects = []
        
        self._create_board()
        while not self._find_all_possible_moves():
             self._create_board()


        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        self.last_action_effects = []

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        action_taken = False
        if space_held or shift_held:
            action_taken = self._handle_swap(space_held, shift_held)
            if action_taken:
                # A swap attempt costs a move, regardless of outcome
                self.moves_left -= 1
                
                # Main game logic loop for processing matches
                chain_count = 1
                while True:
                    matches = self._find_matches()
                    if not matches:
                        break # No more matches, exit loop

                    # SFX: Play match sound, increasing pitch with chain_count
                    
                    num_cleared = len(matches)
                    reward += num_cleared # +1 per gem
                    if num_cleared == 4: reward += 10
                    if num_cleared >= 5: reward += 20
                    
                    self.score += num_cleared * 10 * chain_count

                    for r, c in matches:
                        self.last_action_effects.append(
                            {'type': 'burst', 'pos': (c, r), 'color': self.GEM_COLORS[self.grid[r, c] - 1]}
                        )
                    
                    self._remove_gems(matches)
                    self._apply_gravity()
                    self._refill_board()
                    
                    chain_count += 1
                
                # Reshuffle if no moves are left
                if not self.game_over and not self._find_all_possible_moves():
                    self._shuffle_board()
                    self.last_action_effects.append({'type': 'shuffle_text'})
                    # SFX: Play board reshuffle sound

        else: # No swap action, handle movement
            self._handle_movement(movement)
        
        # Check termination conditions
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            self.win_state = True
            reward += 100
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
            self.win_state = False
            reward -= 50
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_state = False

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    # --- Game Logic ---

    def _create_board(self):
        self.grid = self.rng.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        while self._find_matches():
            matches = self._find_matches()
            self._remove_gems(matches)
            self._apply_gravity()
            self._refill_board()

    def _shuffle_board(self):
        flat_grid = self.grid.flatten()
        self.rng.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_SIZE, self.GRID_SIZE))
        while self._find_matches():
            matches = self._find_matches()
            self._remove_gems(matches)
            self._apply_gravity()
            self._refill_board()

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)

    def _handle_swap(self, space_held, shift_held):
        r, c = self.cursor_pos
        
        if space_held and c < self.GRID_SIZE - 1:
            target_pos = (r, c + 1)
        elif shift_held and r > 0:
            target_pos = (r - 1, c)
        else:
            return False # No valid swap direction or at edge

        # Perform swap
        tr, tc = target_pos
        self.grid[r, c], self.grid[tr, tc] = self.grid[tr, tc], self.grid[r, c]
        
        # Check if swap creates a match
        if not self._find_matches():
            # If not, swap back. The move is still consumed.
            self.grid[r, c], self.grid[tr, tc] = self.grid[tr, tc], self.grid[r, c]
            # SFX: Play invalid move sound
        else:
            # SFX: Play valid swap sound
            pass
        
        return True

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if c < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != 0:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                if r < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != 0:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches
    
    def _find_all_possible_moves(self):
        moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check swap right
                if c < self.GRID_SIZE - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_matches():
                        moves.append(((r,c), (r,c+1)))
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c] # Swap back
                # Check swap down
                if r < self.GRID_SIZE - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_matches():
                        moves.append(((r,c), (r+1,c)))
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c] # Swap back
        return moves

    def _remove_gems(self, matches):
        for r, c in matches:
            self.grid[r, c] = 0

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[r, c], self.grid[empty_row, c] = self.grid[empty_row, c], self.grid[r, c]
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.rng.integers(1, self.NUM_GEM_TYPES + 1)

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_SIZE * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw gems
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._draw_gem(c, r, gem_type - 1)
        
        # Draw effects from last action
        for effect in self.last_action_effects:
            if effect['type'] == 'burst':
                self._draw_burst_effect(effect['pos'][0], effect['pos'][1], effect['color'])

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + cursor_c * self.CELL_SIZE,
            self.GRID_Y_OFFSET + cursor_r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)

    def _draw_gem(self, c, r, gem_type_idx):
        center_x = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
        radius = self.CELL_SIZE // 2 - 5
        color = self.GEM_COLORS[gem_type_idx]
        highlight_color = self.GEM_HIGHLIGHTS[gem_type_idx]
        
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        
        # Add a subtle highlight for 3D effect
        highlight_offset = radius // 3
        pygame.gfxdraw.filled_circle(self.screen, center_x - highlight_offset, center_y - highlight_offset, radius // 2, highlight_color)

    def _draw_burst_effect(self, c, r, color):
        center_x = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(10): # 10 particles per burst
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(5, self.CELL_SIZE * 0.75)
            end_x = int(center_x + radius * math.cos(angle))
            end_y = int(center_y + radius * math.sin(angle))
            start_pos = (center_x, center_y)
            end_pos = (end_x, end_y)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Moves
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(moves_text, moves_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

        # Shuffle message
        for effect in self.last_action_effects:
            if effect['type'] == 'shuffle_text':
                shuffle_text = self.font_medium.render("No more moves! Shuffling...", True, self.COLOR_CURSOR)
                shuffle_rect = shuffle_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
                self.screen.blit(shuffle_text, shuffle_rect)

    # --- Info and Validation ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "possible_moves": len(self._find_all_possible_moves())
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Swap")
    
    print(env.user_guide)
    
    while not done:
        # Default action is a no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
        
        # Only step if a key was pressed
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print("Game Over!")
            pygame.time.wait(3000) # Wait 3 seconds before closing

    env.close()