
# Generated: 2025-08-27T20:55:46.188734
# Source Brief: brief_02622.md
# Brief Index: 2622

        
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
        "Controls: Use arrow keys to swap the selected block with its neighbor. Press space to move the selector. Clear the board before you run out of moves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manipulate a grid of falling colored blocks to create matches of 3 or more and clear the board before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_COLORS = 5
        self.MAX_MOVES = 10
        self.MAX_STEPS = 1000

        # Visual constants
        self.BLOCK_SIZE = 40
        self.GRID_LINE_WIDTH = 2
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        self.BLOCK_PADDING = 4
        
        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (0, 0, 0),  # 0 is empty
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 80),  # Yellow
            (160, 80, 255),  # Purple
        ]

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.match_animation_list = []

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [0, 0]
        self.match_animation_list = []
        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        self.match_animation_list = [] # Clear last turn's animations

        movement_action = action[0]  # 0-4: none/up/down/left/right
        select_action = action[1] == 1  # Boolean for space
        
        # Action 1: Move cursor (space bar)
        if select_action:
            self.cursor_pos[1] += 1
            if self.cursor_pos[1] >= self.GRID_WIDTH:
                self.cursor_pos[1] = 0
                self.cursor_pos[0] += 1
                if self.cursor_pos[0] >= self.GRID_HEIGHT:
                    self.cursor_pos[0] = 0
            # Sound: a soft 'tick' or 'click'

        # Action 0: Swap blocks (arrow keys)
        if movement_action != 0:
            self.moves_left -= 1
            
            r, c = self.cursor_pos
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement_action]
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                # Perform swap
                self.grid[r][c], self.grid[nr][nc] = self.grid[nr][nc], self.grid[r][c]
                # Sound: a 'swoosh' sound
                
                combo_multiplier = 1
                while True:
                    matches = self._find_matches()
                    if not matches:
                        break # No more matches, chain is over
                    
                    # Calculate reward for this wave of matches
                    num_cleared = len(matches)
                    reward += num_cleared * combo_multiplier
                    if num_cleared > 3:
                        reward += 5 # Bonus for larger matches
                    
                    # Add matched blocks to animation list
                    for pos in matches:
                        self.match_animation_list.append(pos)
                    # Sound: a 'pop' or 'chime' sound, pitch increases with combo

                    # Clear matched blocks
                    for r_m, c_m in matches:
                        self.grid[r_m][c_m] = 0

                    # Apply gravity and fill new blocks
                    self._apply_gravity()
                    self._fill_new_blocks()
                    
                    combo_multiplier += 1
            
        self.steps += 1
        
        # Check termination conditions
        if self._is_board_clear():
            reward += 100
            terminated = True
            # Sound: victory fanfare
        elif self.moves_left <= 0:
            reward -= 100
            terminated = True
            # Sound: failure buzzer
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(
            self.GRID_OFFSET_X, self.GRID_OFFSET_Y,
            self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw blocks and match animations
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_index = self.grid[r][c]
                if color_index > 0:
                    block_rect = pygame.Rect(
                        self.GRID_OFFSET_X + c * self.BLOCK_SIZE + self.BLOCK_PADDING,
                        self.GRID_OFFSET_Y + r * self.BLOCK_SIZE + self.BLOCK_PADDING,
                        self.BLOCK_SIZE - 2 * self.BLOCK_PADDING,
                        self.BLOCK_SIZE - 2 * self.BLOCK_PADDING
                    )
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_index], block_rect, border_radius=5)
                    
                    # Draw a subtle inner highlight for 3D effect
                    highlight_color = tuple(min(255, val + 50) for val in self.BLOCK_COLORS[color_index])
                    pygame.draw.rect(self.screen, highlight_color, block_rect.inflate(-8, -8), border_radius=4)

        # Draw match animations on top
        for r, c in self.match_animation_list:
            anim_rect = pygame.Rect(
                self.GRID_OFFSET_X + c * self.BLOCK_SIZE,
                self.GRID_OFFSET_Y + r * self.BLOCK_SIZE,
                self.BLOCK_SIZE,
                self.BLOCK_SIZE
            )
            # Draw a bright, star-like flash
            center = anim_rect.center
            pygame.draw.circle(self.screen, (255, 255, 255), center, self.BLOCK_SIZE // 2, width=3)
            pygame.draw.line(self.screen, (255, 255, 255), (center[0] - 15, center[1]), (center[0] + 15, center[1]), 3)
            pygame.draw.line(self.screen, (255, 255, 255), (center[0], center[1] - 15), (center[0], center[1] + 15), 3)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_c * self.BLOCK_SIZE,
            self.GRID_OFFSET_Y + cursor_r * self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            self.BLOCK_SIZE
        )
        pulse = abs(math.sin(self.steps * 0.3)) * 4
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect.inflate(pulse, pulse), width=3, border_radius=7)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
        }

    def _generate_grid(self):
        while True:
            self.grid = [[random.randint(1, self.NUM_COLORS) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
            
            # Ensure no initial matches
            while self._find_matches():
                matches = self._find_matches()
                for r, c in matches:
                    self.grid[r][c] = random.randint(1, self.NUM_COLORS)
            
            # Ensure at least one move is possible
            if self._is_move_possible():
                break

    def _is_move_possible(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                    if self._find_matches():
                        self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c] # Swap back
                
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                    if self._find_matches():
                        self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c] # Swap back
        return False

    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r][c] != 0 and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    for i in range(3): matches.add((r, c+i))
                    # Check for longer matches
                    for i in range(c + 3, self.GRID_WIDTH):
                        if self.grid[r][c] == self.grid[r][i]:
                            matches.add((r, i))
                        else:
                            break

        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r][c] != 0 and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    for i in range(3): matches.add((r+i, c))
                    # Check for longer matches
                    for i in range(r + 3, self.GRID_HEIGHT):
                         if self.grid[r][c] == self.grid[i][c]:
                            matches.add((i, c))
                         else:
                            break
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r][c] != 0:
                    self.grid[r][c], self.grid[empty_row][c] = self.grid[empty_row][c], self.grid[r][c]
                    empty_row -= 1

    def _fill_new_blocks(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] == 0:
                    self.grid[r][c] = random.randint(1, self.NUM_COLORS)

    def _is_board_clear(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] != 0:
                    return False
        return True

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
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Match-3 Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_q: # Quit
                    running = False
        
        if not terminated and (action[0] != 0 or action[1] != 0):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate
        
    env.close()