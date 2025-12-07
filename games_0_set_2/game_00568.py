
# Generated: 2025-08-27T14:02:09.172852
# Source Brief: brief_00568.md
# Brief Index: 568

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to flag a square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist minesweeper-style puzzle game. Reveal all safe squares to win, but avoid the mines!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = 5
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
        self.font_large = pygame.font.SysFont("sans-serif", 36, bold=True)
        self.font_medium = pygame.font.SysFont("sans-serif", 24)
        self.font_small = pygame.font.SysFont("sans-serif", 18)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_HIDDEN = (70, 80, 90)
        self.COLOR_REVEALED = (110, 120, 130)
        self.COLOR_FLAG = (240, 200, 80)
        self.COLOR_MINE = (220, 50, 50)
        self.COLOR_CURSOR = (100, 200, 255)
        self.COLOR_TEXT = (230, 240, 250)
        self.NUMBER_COLORS = [
            (0, 0, 0),  # 0 is not used
            (80, 150, 255),  # 1
            (80, 200, 150),  # 2
            (255, 100, 100),  # 3
            (150, 80, 255),  # 4
            (255, 150, 50),   # 5+
        ]

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.num_safe_squares = 0
        self.revealed_count = 0
        self.flag_count = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.revealed_count = 0
        self.flag_count = 0

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self._initialize_grid()
        self._calculate_adjacent_mines()

        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = [[{
            "is_mine": False,
            "is_revealed": False,
            "is_flagged": False,
            "adjacent_mines": 0
        } for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        mine_positions = set()
        while len(mine_positions) < self.NUM_MINES:
            x = self.np_random.integers(0, self.GRID_SIZE)
            y = self.np_random.integers(0, self.GRID_SIZE)
            mine_positions.add((x, y))

        for x, y in mine_positions:
            self.grid[y][x]["is_mine"] = True
            
        self.num_safe_squares = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES

    def _calculate_adjacent_mines(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x]["is_mine"]:
                    continue
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny][nx]["is_mine"]:
                            count += 1
                self.grid[y][x]["adjacent_mines"] = count

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        if not self.game_over:
            # Handle cursor movement
            if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
            elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
            elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
            elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

            # Handle actions
            if space_pressed:
                reward += self._reveal_action()
            elif shift_pressed:
                reward += self._flag_action()

        self.steps += 1
        
        # Check for termination conditions
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
        
        if self.win:
            reward += 10 # Win bonus
            self.score += 10

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_action(self):
        x, y = self.cursor_pos
        cell = self.grid[y][x]

        if cell["is_revealed"] or cell["is_flagged"]:
            return 0 # No action, no reward

        if cell["is_mine"]:
            # Sound: Explosion
            self.game_over = True
            self._reveal_all_mines()
            return -100

        # Sound: Click/Reveal
        revealed_count = self._reveal_square(x, y)
        self.revealed_count += revealed_count
        
        if self.revealed_count == self.num_safe_squares:
            self.win = True
            self.game_over = True

        return revealed_count # +1 for each revealed safe square

    def _flag_action(self):
        x, y = self.cursor_pos
        cell = self.grid[y][x]

        if cell["is_revealed"]:
            return 0 # Can't flag revealed square

        # Sound: Flag place/remove
        cell["is_flagged"] = not cell["is_flagged"]
        self.flag_count += 1 if cell["is_flagged"] else -1
        return 0

    def _reveal_square(self, start_x, start_y):
        if self.grid[start_y][start_x]["is_revealed"]:
            return 0
            
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        count = 0

        while q:
            x, y = q.popleft()
            cell = self.grid[y][x]
            
            if cell["is_revealed"]: continue
            if cell["is_flagged"]:
                cell["is_flagged"] = False # Revealing unflags
                self.flag_count -= 1
                
            cell["is_revealed"] = True
            count += 1
            
            if cell["adjacent_mines"] == 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                            q.append((nx, ny))
                            visited.add((nx, ny))
        return count
        
    def _reveal_all_mines(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x]["is_mine"]:
                    self.grid[y][x]["is_revealed"] = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_pixel_size = 300
        cell_size = grid_pixel_size / self.GRID_SIZE
        grid_start_x = (self.WIDTH - grid_pixel_size) / 2
        grid_start_y = (self.HEIGHT - grid_pixel_size) / 2 + 30

        # Draw grid cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cell = self.grid[y][x]
                rect = pygame.Rect(
                    grid_start_x + x * cell_size,
                    grid_start_y + y * cell_size,
                    cell_size,
                    cell_size
                )
                
                # Draw cell background
                if cell["is_revealed"]:
                    color = self.COLOR_MINE if cell["is_mine"] else self.COLOR_REVEALED
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    color = self.COLOR_FLAG if cell["is_flagged"] else self.COLOR_HIDDEN
                    pygame.draw.rect(self.screen, color, rect)
                
                # Draw cell border
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2)

                # Draw cell content
                if cell["is_revealed"]:
                    if cell["is_mine"]:
                        # Draw explosion/mine icon
                        center = rect.center
                        pygame.draw.circle(self.screen, (50, 50, 50), center, int(cell_size * 0.3))
                        pygame.draw.circle(self.screen, (255, 255, 0), center, int(cell_size * 0.15))
                    elif cell["adjacent_mines"] > 0:
                        # Draw number
                        num_color = self.NUMBER_COLORS[min(cell["adjacent_mines"], len(self.NUMBER_COLORS)-1)]
                        text = self.font_large.render(str(cell["adjacent_mines"]), True, num_color)
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                elif cell["is_flagged"]:
                    # Draw flag icon
                    center_x, center_y = rect.center
                    pole_start = (center_x, center_y - cell_size * 0.25)
                    pole_end = (center_x, center_y + cell_size * 0.25)
                    pygame.draw.line(self.screen, self.COLOR_TEXT, pole_start, pole_end, 3)
                    flag_points = [
                        (center_x, center_y - cell_size * 0.25),
                        (center_x - cell_size * 0.3, center_y - cell_size * 0.1),
                        (center_x, center_y + cell_size * 0.05)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_TEXT, flag_points)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            grid_start_x + cursor_x * cell_size,
            grid_start_y + cursor_y * cell_size,
            cell_size,
            cell_size
        )
        
        # Pulsing effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2  # Varies between 0 and 1
        line_width = int(2 + pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=3)
        
    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Flags remaining display
        flags_text = self.font_medium.render(f"Flags: {self.flag_count} / {self.NUM_MINES}", True, self.COLOR_TEXT)
        flags_rect = flags_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(flags_text, flags_rect)
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else self.COLOR_MINE
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "win": self.win,
            "flags_placed": self.flag_count,
            "safe_squares_revealed": self.revealed_count,
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Manual play loop
    running = True
    
    # Create a display window for manual play
    pygame.display.set_caption("Minesweeper Gym Environment")
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop for human play
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                
                # Since auto_advance is False, we only need to register the action
                # and call step() once per key press.
                
                # Movement
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                # Actions
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                
                # Process the action
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

                if terminated:
                    print("Game Over! Press 'R' to restart.")


        # Update the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
    pygame.quit()