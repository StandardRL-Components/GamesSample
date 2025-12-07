
# Generated: 2025-08-27T17:48:12.099739
# Source Brief: brief_01646.md
# Brief Index: 1646

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. Reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, avoiding flashing red traps. Reach the green exit as quickly as possible for a higher score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and maze dimensions
        self.W, self.H = 640, 400
        self.MAZE_COLS, self.MAZE_ROWS = 32, 20
        self.CELL_SIZE = self.W // self.MAZE_COLS
        self.MAX_STEPS = 1000
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("sans", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (80, 80, 100)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_EXIT = (0, 255, 0)
        self.COLOR_TRAP = (255, 0, 0)
        self.COLOR_BREADCRUMB = (0, 100, 255)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Game state persistence across resets
        self.successful_escapes = 0
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Difficulty scaling
        num_traps = min(10 + (self.successful_escapes // 5) * 2, 50)
        
        # Generate maze and entities
        self.maze = self._generate_maze(self.MAZE_COLS, self.MAZE_ROWS)
        self.player_pos = [0, 0]
        self.exit_pos = [self.MAZE_COLS - 1, self.MAZE_ROWS - 1]
        self.traps = self._place_traps(num_traps)
        self.breadcrumbs = [self.player_pos[:]]
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self, width, height):
        # Create a grid with all walls up
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False} for _ in range(height)] for _ in range(width)]
        
        stack = []
        x, y = self.np_random.integers(0, width), self.np_random.integers(0, height)
        stack.append((x, y))
        maze[x][y]['visited'] = True
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            
            # Check neighbors
            if cy > 0 and not maze[cx][cy-1]['visited']: neighbors.append('N')
            if cy < height-1 and not maze[cx][cy+1]['visited']: neighbors.append('S')
            if cx < width-1 and not maze[cx+1][cy]['visited']: neighbors.append('E')
            if cx > 0 and not maze[cx-1][cy]['visited']: neighbors.append('W')
            
            if neighbors:
                direction = self.np_random.choice(neighbors)
                if direction == 'N':
                    nx, ny = cx, cy - 1
                    maze[cx][cy]['N'] = False
                    maze[nx][ny]['S'] = False
                elif direction == 'S':
                    nx, ny = cx, cy + 1
                    maze[cx][cy]['S'] = False
                    maze[nx][ny]['N'] = False
                elif direction == 'E':
                    nx, ny = cx + 1, cy
                    maze[cx][cy]['E'] = False
                    maze[nx][ny]['W'] = False
                elif direction == 'W':
                    nx, ny = cx - 1, cy
                    maze[cx][cy]['W'] = False
                    maze[nx][ny]['E'] = False
                
                maze[nx][ny]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _place_traps(self, num_traps):
        possible_cells = []
        for x in range(self.MAZE_COLS):
            for y in range(self.MAZE_ROWS):
                if [x, y] != self.player_pos and [x, y] != self.exit_pos:
                    possible_cells.append([x, y])
        
        self.np_random.shuffle(possible_cells)
        return possible_cells[:num_traps]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = -0.1  # Per-step penalty
        
        px, py = self.player_pos
        moved = False
        
        if movement == 1 and py > 0 and not self.maze[px][py]['N']: # Up
            self.player_pos[1] -= 1
            moved = True
        elif movement == 2 and py < self.MAZE_ROWS - 1 and not self.maze[px][py]['S']: # Down
            self.player_pos[1] += 1
            moved = True
        elif movement == 3 and px > 0 and not self.maze[px][py]['W']: # Left
            self.player_pos[0] -= 1
            moved = True
        elif movement == 4 and px < self.MAZE_COLS - 1 and not self.maze[px][py]['E']: # Right
            self.player_pos[0] += 1
            moved = True

        if moved:
            # sfx: player_step.wav
            if self.player_pos not in self.breadcrumbs:
                self.breadcrumbs.append(self.player_pos[:])
            else: # If backtracking, remove path
                idx = self.breadcrumbs.index(self.player_pos)
                self.breadcrumbs = self.breadcrumbs[:idx+1]

        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.player_pos in self.traps:
            # sfx: player_hit_trap.wav
            reward = -10.0
            terminated = True
            self.game_over = True
        elif self.player_pos == self.exit_pos:
            # sfx: win_level.wav
            reward = 100.0
            self.score = int(reward + (self.MAX_STEPS - self.steps) * 0.1) # Bonus for speed
            self.successful_escapes += 1
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # sfx: time_up.wav
            terminated = True
            self.game_over = True
        
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
        # Create a surface for alpha blending
        alpha_surface = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        # Render breadcrumbs
        for pos in self.breadcrumbs:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(alpha_surface, self.COLOR_BREADCRUMB + (100,), rect)

        # Render traps (flashing)
        pulse = (math.sin(self.steps * 0.4) + 1) / 2  # 0 to 1
        trap_alpha = 100 + 155 * pulse
        for pos in self.traps:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(alpha_surface, self.COLOR_TRAP + (int(trap_alpha),), rect)

        self.screen.blit(alpha_surface, (0, 0))

        # Render exit
        exit_rect = pygame.Rect(self.exit_pos[0] * self.CELL_SIZE, self.exit_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Render player
        player_center_x = int((self.player_pos[0] + 0.5) * self.CELL_SIZE)
        player_center_y = int((self.player_pos[1] + 0.5) * self.CELL_SIZE)
        player_radius = int(self.CELL_SIZE * 0.35)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

        # Render maze walls
        for x in range(self.MAZE_COLS):
            for y in range(self.MAZE_ROWS):
                cell = self.maze[x][y]
                px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
                if cell['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), 1)
                if cell['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), 1)
                if cell['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), 1)
                if cell['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), 1)

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 10, 10))

        # Render steps/timer
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "YOU ESCAPED!" if self.player_pos == self.exit_pos else "YOU WERE CAUGHT"
            if self.steps >= self.MAX_STEPS and self.player_pos != self.exit_pos:
                message = "OUT OF TIME"
            
            text_surf = self.font_game_over.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.W / 2, self.H / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "successful_escapes": self.successful_escapes,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This is a demonstration of how a human could play.
    # The environment itself is headless.
    
    play = True
    if play:
        pygame.display.set_caption("Maze Runner")
        screen = pygame.display.set_mode((env.W, env.H))
        obs, info = env.reset()
        terminated = False
        
        while True:
            action = np.array([0, 0, 0]) # Default no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        terminated = False
                    if terminated:
                        continue

                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
            
            # Since auto_advance is False, we only step when an action is taken.
            # In a manual play loop, this means we step on every key press.
            # For this demo, we'll step on every frame if a key is held, or if a reset happened.
            keys = pygame.key.get_pressed()
            if not terminated:
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                
                # Only step if an action is taken
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit FPS for manual play