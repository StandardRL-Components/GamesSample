
# Generated: 2025-08-27T13:58:38.174798
# Source Brief: brief_00548.md
# Brief Index: 548

        
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
        "Controls: Arrows to move cursor. Space to reveal a square. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid, revealing safe squares while avoiding hidden mines to uncover the entire field."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_SIZE = (9, 9)
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_cell = pygame.font.SysFont("Consolas", 20, bold=True)

        self.COLOR_BG = (44, 62, 80) # #2c3e50
        self.COLOR_GRID_LINE = (52, 73, 94) # #34495e
        self.COLOR_UNREVEALED = (149, 165, 166) # #95a5a6
        self.COLOR_REVEALED = (189, 195, 199) # #bdc3c7
        self.COLOR_CURSOR = (241, 196, 15) # #f1c40f
        self.COLOR_FLAG = (52, 152, 219) # #3498db
        self.COLOR_MINE = (231, 76, 60) # #e74c3c
        self.COLOR_TEXT = (236, 240, 241) # #ecf0f1
        self.COLOR_NUM = {
            1: (52, 152, 219),  # Blue
            2: (46, 204, 113),  # Green
            3: (231, 76, 60),   # Red
            4: (142, 68, 173),  # Purple
            5: (243, 156, 18),  # Orange
            6: (26, 188, 156),  # Turquoise
            7: (44, 62, 80),    # Dark
            8: (127, 140, 141)  # Grey
        }

        # Initialize state variables
        self.grid_state = None
        self.mine_locations = None
        self.adjacent_counts = None
        self.revealed_mask = None
        self.flagged_mask = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.cursor_pos = np.array([self.GRID_SIZE[1] // 2, self.GRID_SIZE[0] // 2])
        
        # Grid state initialization
        grid_width, grid_height = self.GRID_SIZE
        self.mine_locations = np.zeros(self.GRID_SIZE, dtype=bool)
        self.adjacent_counts = np.zeros(self.GRID_SIZE, dtype=int)
        self.revealed_mask = np.zeros(self.GRID_SIZE, dtype=bool)
        self.flagged_mask = np.zeros(self.GRID_SIZE, dtype=bool)

        # Place mines
        mine_indices = self.np_random.choice(grid_width * grid_height, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(mine_indices, self.GRID_SIZE)
        self.mine_locations[mine_coords] = True
        
        # Calculate adjacent mine counts
        for r in range(grid_height):
            for c in range(grid_width):
                if not self.mine_locations[r, c]:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid_height and 0 <= nc < grid_width and self.mine_locations[nr, nc]:
                                count += 1
                    self.adjacent_counts[r, c] = count
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        movement = action[0]
        reveal_action = action[1] == 1
        flag_action = action[2] == 1
        
        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE[0]
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE[0]
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE[1]
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE[1]

        r, c = self.cursor_pos
        
        # 2. Handle actions if not game over
        if not self.game_over:
            if flag_action:
                # Toggle flag
                if not self.revealed_mask[r, c]:
                    if not self.flagged_mask[r, c]:
                        self.flagged_mask[r, c] = True
                        reward = -0.1 # Penalty for placing flag
                    else:
                        self.flagged_mask[r, c] = False
                        reward = 0.1 # Reward for removing flag
                    # sfx: flag_place.wav / flag_remove.wav

            elif reveal_action:
                # Reveal square
                if not self.revealed_mask[r, c] and not self.flagged_mask[r, c]:
                    self.revealed_mask[r, c] = True
                    # sfx: reveal.wav
                    if self.mine_locations[r, c]:
                        # Hit a mine
                        self.game_over = True
                        reward = -100
                        # sfx: explosion.wav
                    else:
                        # Safe square
                        if self.adjacent_counts[r, c] > 0:
                            reward = -0.2 # Penalty for revealing a numbered square (risky)
                        else:
                            reward = 1.0 # Base reward for revealing a safe square
                            reward += self._flood_fill(r, c) # Add reward for auto-revealed squares
        
        # 3. Check for termination
        terminated = self._check_termination()
        if not self.game_over and self.win:
            reward = 100 # Win bonus
            # sfx: win_jingle.wav
        
        self.score += reward
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _flood_fill(self, r, c):
        """Recursively reveals empty squares and returns count of newly revealed."""
        revealed_count = 0
        q = [(r, c)]
        visited = set([(r, c)])
        
        while q:
            curr_r, curr_c = q.pop(0)
            
            # Reveal this square if not already
            if not self.revealed_mask[curr_r, curr_c]:
                self.revealed_mask[curr_r, curr_c] = True
                revealed_count += 1
            
            # If it's an empty square (0 adjacent mines), check neighbors
            if self.adjacent_counts[curr_r, curr_c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = curr_r + dr, curr_c + dc
                        
                        if 0 <= nr < self.GRID_SIZE[0] and 0 <= nc < self.GRID_SIZE[1]:
                            if not self.revealed_mask[nr, nc] and not self.flagged_mask[nr, nc] and (nr, nc) not in visited:
                                q.append((nr, nc))
                                visited.add((nr, nc))
        return revealed_count

    def _check_termination(self):
        if self.game_over:
            return True
        
        num_safe_squares = self.GRID_SIZE[0] * self.GRID_SIZE[1] - self.NUM_MINES
        if np.sum(self.revealed_mask) == num_safe_squares:
            self.win = True
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
        grid_w, grid_h = self.GRID_SIZE
        cell_size = min((self.WIDTH - 100) // grid_w, (self.HEIGHT - 100) // grid_h)
        grid_pixel_w = grid_w * cell_size
        grid_pixel_h = grid_h * cell_size
        offset_x = (self.WIDTH - grid_pixel_w) // 2
        offset_y = (self.HEIGHT - grid_pixel_h) // 2

        for r in range(grid_h):
            for c in range(grid_w):
                rect = pygame.Rect(offset_x + c * cell_size, offset_y + r * cell_size, cell_size, cell_size)
                
                cell_color = self.COLOR_UNREVEALED
                if self.revealed_mask[r, c] or (self.game_over and self.mine_locations[r,c]):
                    cell_color = self.COLOR_REVEALED
                
                pygame.draw.rect(self.screen, cell_color, rect)

                if self.revealed_mask[r, c]:
                    if self.mine_locations[r, c]: # Should only happen on game over
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(cell_size * 0.3), self.COLOR_MINE)
                    else:
                        count = self.adjacent_counts[r, c]
                        if count > 0:
                            num_text = self.font_cell.render(str(count), True, self.COLOR_NUM[count])
                            text_rect = num_text.get_rect(center=rect.center)
                            self.screen.blit(num_text, text_rect)
                elif self.flagged_mask[r, c]:
                    # Draw a flag (triangle)
                    p1 = (rect.centerx, rect.top + int(cell_size * 0.2))
                    p2 = (rect.left + int(cell_size * 0.2), rect.centery)
                    p3 = (rect.right - int(cell_size * 0.2), rect.centery)
                    pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_FLAG)
                    pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_FLAG)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(offset_x + cursor_c * cell_size, offset_y + cursor_r * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Mines Left
        flags_placed = np.sum(self.flagged_mask)
        mines_left = self.NUM_MINES - flags_placed
        mines_text = self.font_main.render(f"MINES LEFT: {mines_left}", True, self.COLOR_TEXT)
        mines_rect = mines_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(mines_text, mines_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg_text = self.font_large.render("YOU WIN!", True, (46, 204, 113)) # Green
            else:
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_MINE)
            
            msg_rect = msg_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "mines_left": self.NUM_MINES - np.sum(self.flagged_mask),
            "win": self.win,
        }
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert np.sum(self.mine_locations) == self.NUM_MINES
        assert np.sum(self.revealed_mask) == 0
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test win condition
        self.reset()
        self.revealed_mask = ~self.mine_locations
        terminated = self._check_termination()
        assert self.win == True
        assert terminated == True

        # Test loss condition
        self.reset()
        mine_coord = np.argwhere(self.mine_locations)[0]
        self.cursor_pos = mine_coord
        self.step([0, 1, 0]) # Reveal action at mine location
        assert self.game_over == True
        assert self._check_termination() == True
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to actions
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT:[0, 0, 1],
        pygame.K_RSHIFT:[0, 0, 1],
    }
    
    # Pygame setup for human play
    pygame.display.set_caption("Minesweeper Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    while not done:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
        
        # Since auto_advance is False, we only step when there is an action
        # For human play, we step every frame to see cursor movement
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Win: {info['win']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
        
        clock.tick(30) # Limit frame rate

    pygame.quit()