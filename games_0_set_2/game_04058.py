
# Generated: 2025-08-28T01:16:21.260106
# Source Brief: brief_04058.md
# Brief Index: 4058

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a tile. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic Minesweeper puzzle. Reveal all safe tiles without hitting a mine. "
        "Numbers on revealed tiles indicate the count of adjacent mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array", grid_size=9, num_mines=10):
        super().__init__()
        
        # Game constants
        self.GRID_SIZE = grid_size
        self.NUM_MINES = num_mines
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000

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
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_HIDDEN = (70, 80, 90)
        self.COLOR_REVEALED = (40, 50, 60)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_FLAG = (0, 200, 100)
        self.COLOR_MINE_BG = (200, 50, 50)
        self.COLOR_MINE_FG = (255, 100, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.NUM_COLORS = [
            (0, 0, 0, 0), # 0 is not drawn
            (100, 150, 255), # 1
            (100, 200, 100), # 2
            (255, 100, 100), # 3
            (150, 100, 255), # 4
            (255, 150, 50),  # 5
            (50, 200, 200),  # 6
            (220, 220, 220), # 7
            (150, 150, 150), # 8
        ]

        # Grid rendering properties
        self.CELL_SIZE = 38
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Etc...        
        
        # Initialize state variables
        self.solution_grid = None
        self.visible_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.mines_left = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.mines_left = self.NUM_MINES
        
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        
        # Create grids
        self.visible_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8) # 0: hidden, 1: revealed, 2: flagged
        self._generate_minefield()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _generate_minefield(self):
        self.solution_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        
        # Place mines
        mine_indices = self.np_random.choice(self.GRID_SIZE * self.GRID_SIZE, self.NUM_MINES, replace=False)
        mine_coords = [(i % self.GRID_SIZE, i // self.GRID_SIZE) for i in mine_indices]
        
        for x, y in mine_coords:
            self.solution_grid[y, x] = -1 # -1 represents a mine
            
        # Calculate adjacent mine counts
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.solution_grid[y, x] == -1:
                    continue
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.solution_grid[ny, nx] == -1:
                            count += 1
                self.solution_grid[y, x] = count

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean
        
        reward = 0
        self.steps += 1

        # 1. Handle movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_SIZE - 1)
        cx, cy = self.cursor_pos[0], self.cursor_pos[1]
        
        # 2. Handle actions (Reveal/Flag)
        if space_pressed:
            if self.visible_grid[cy, cx] == 0: 
                reward += self._reveal_tile(cx, cy)
        elif shift_pressed:
            if self.visible_grid[cy, cx] != 1:
                reward += self._toggle_flag(cx, cy)

        # 3. Check for termination
        terminated = self._check_termination()
        if self.win and terminated:
            reward += 100
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _reveal_tile(self, x, y):
        if self.solution_grid[y, x] == -1:
            self.visible_grid[y, x] = 1
            self.game_over = True
            self.win = False
            # Sound: Explosion
            return -100

        if self.visible_grid[y, x] == 0:
            self.visible_grid[y, x] = 1
            reward = 1
            # Sound: Click
            if self.solution_grid[y, x] == 0:
                reward += self._flood_fill(x, y)
            return reward
        return 0

    def _flood_fill(self, start_x, start_y):
        revealed_count = 0
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        if self.visible_grid[ny, nx] == 0:
                            self.visible_grid[ny, nx] = 1
                            revealed_count += 1
                            # Sound: Pop
                            if self.solution_grid[ny, nx] == 0:
                                q.append((nx, ny))
        return revealed_count

    def _toggle_flag(self, x, y):
        is_mine = self.solution_grid[y, x] == -1
        
        if self.visible_grid[y, x] == 2:
            self.visible_grid[y, x] = 0
            self.mines_left += 1
            # Sound: Unflag
            return -10 if is_mine else 5
        else:
            self.visible_grid[y, x] = 2
            self.mines_left -= 1
            # Sound: Flag
            reward = -0.1
            reward += 10 if is_mine else -5
            return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        revealed_count = np.sum(self.visible_grid == 1)
        non_mine_tiles = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
        if revealed_count == non_mine_tiles:
            self.game_over = True
            self.win = True
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + x * self.CELL_SIZE,
                    self.GRID_Y_OFFSET + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                state = self.visible_grid[y, x]
                
                if state == 0: # Hidden
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                elif state == 1: # Revealed
                    value = self.solution_grid[y, x]
                    if value == -1:
                        pygame.draw.rect(self.screen, self.COLOR_MINE_BG, rect)
                        pygame.draw.circle(self.screen, self.COLOR_MINE_FG, rect.center, self.CELL_SIZE // 3)
                        for i in range(3):
                             pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, int(self.CELL_SIZE/3 * (i+1) * 0.7), self.COLOR_MINE_FG)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                        if value > 0:
                            num_text = self.font_medium.render(str(value), True, self.NUM_COLORS[value])
                            text_rect = num_text.get_rect(center=rect.center)
                            self.screen.blit(num_text, text_rect)
                elif state == 2: # Flagged
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                    p1 = (rect.centerx, rect.top + 5)
                    p2 = (rect.left + 5, rect.centery)
                    p3 = (rect.centerx, rect.centery)
                    pygame.draw.polygon(self.screen, self.COLOR_FLAG, [p1,p2,p3])
                    pygame.draw.line(self.screen, self.COLOR_FLAG, (rect.centerx, rect.top + 5), (rect.centerx, rect.bottom - 5), 2)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        mines_text = self.font_small.render(f"MINES: {self.mines_left}", True, self.COLOR_TEXT)
        mines_rect = mines_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(mines_text, mines_rect)

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(midtop=(self.WIDTH // 2, 10))
        self.screen.blit(steps_text, steps_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_FLAG if self.win else self.COLOR_MINE_BG
                
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "mines_left": self.mines_left,
        }

    def close(self):
        pygame.font.quit()
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
    # This block allows you to play the game with keyboard controls
    import os
    # Set the video driver to a dummy one for headless execution if no display is available
    if 'DISPLAY' not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Only create a display if not in a headless environment
    use_display = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"
    if use_display:
        display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Minesweeper Gym Environment")

    terminated = False
    
    while not terminated:
        if use_display:
            # Human control
            current_action = [0, 0, 0] # Default to no-op
            should_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    continue
                if event.type == pygame.KEYDOWN:
                    should_step = True
                    if event.key == pygame.K_UP: current_action[0] = 1
                    elif event.key == pygame.K_DOWN: current_action[0] = 2
                    elif event.key == pygame.K_LEFT: current_action[0] = 3
                    elif event.key == pygame.K_RIGHT: current_action[0] = 4
                    elif event.key == pygame.K_SPACE: current_action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: current_action[2] = 1
                    elif event.key == pygame.K_r: # Reset game
                        obs, info = env.reset()
                        should_step = False
                    else:
                        should_step = False # Don't step on other key presses
            
            if should_step and not terminated:
                obs, reward, term, trunc, info = env.step(current_action)
                terminated = term
                print(f"Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

            # Render the observation to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display.blit(surf, (0, 0))
            pygame.display.flip()
        else:
            # Agent control (random actions)
            if not terminated:
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                terminated = term
                print(f"Action: {action.tolist()}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            else:
                print("Episode finished. Resetting.")
                obs, info = env.reset()
                terminated = False

    env.close()