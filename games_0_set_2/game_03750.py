
# Generated: 2025-08-28T00:18:30.132256
# Source Brief: brief_03750.md
# Brief Index: 3750

        
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
        "Controls: Use arrow keys (↑↓←→) to navigate the robot through the maze. Reach the green exit before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Guide your robot through a procedurally generated maze to find the exit. Collect yellow items for bonus points, but watch your limited move counter!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_WALL = (60, 80, 130)
    COLOR_WALL_TOP = (100, 120, 180)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 120, 120)
    COLOR_EXIT = (80, 255, 80)
    COLOR_EXIT_INNER = (150, 255, 150)
    COLOR_ITEM = (255, 220, 50)
    COLOR_ITEM_GLOW = (255, 240, 150)
    COLOR_HIGHLIGHT = (255, 255, 255)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.exit_pos = None
        self.items = None
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = None # "win", "lose", or None
        
        self.reset()
        
        # Critical self-check
        self.validate_implementation()
    
    def _generate_maze(self):
        """Generates a maze using randomized DFS, ensuring a solvable path."""
        grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
        visited = np.zeros_like(grid, dtype=bool)

        # Start DFS from a random position
        start_y, start_x = (
            self.np_random.integers(0, self.GRID_HEIGHT),
            self.np_random.integers(0, self.GRID_WIDTH),
        )
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        grid[start_y, start_x] = 0
        
        path = [(start_y, start_x)]
        longest_path = list(path)

        while stack:
            cy, cx = stack[-1]
            neighbors = []
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and not visited[ny, nx]:
                    neighbors.append((ny, nx))
            
            if neighbors:
                ny, nx = self.np_random.choice(neighbors, axis=0)
                visited[ny, nx] = True
                grid[ny, nx] = 0
                stack.append((ny, nx))
                path.append((ny,nx))
                if len(path) > len(longest_path):
                    longest_path = list(path)
            else:
                stack.pop()
                path.pop()
        
        # Ensure at least 30% of the map is open space
        if np.sum(grid == 0) < self.GRID_WIDTH * self.GRID_HEIGHT * 0.3:
            return self._generate_maze() # Retry if maze is too dense

        player_start = longest_path[0]
        exit_pos = longest_path[-1]

        # Path length check - must be solvable in 20 moves
        path_len = len(self._solve_bfs(grid, player_start, exit_pos))
        if path_len == 0 or path_len > 20:
             return self._generate_maze() # Retry if unsolvable or too long

        return grid, player_start, exit_pos

    def _solve_bfs(self, grid, start, end):
        q = [(start, [start])]
        visited = {start}
        while q:
            (y, x), path = q.pop(0)
            if (y, x) == end:
                return path
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and grid[ny, nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    q.append(((ny, nx), path + [(ny, nx)]))
        return [] # No path found

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid, self.player_pos, self.exit_pos = self._generate_maze()
        
        # Place items
        self.items = []
        open_cells = np.argwhere(self.grid == 0)
        self.np_random.shuffle(open_cells)
        num_items = self.np_random.integers(3, 6)
        for y, x in open_cells:
            if (y, x) != self.player_pos and (y, x) != self.exit_pos:
                self.items.append((y, x))
                if len(self.items) >= num_items:
                    break
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = 20
        self.game_over = False
        self.win_status = None
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        
        target_y, target_x = self.player_pos
        moved = False
        
        if movement > 0: # Any move action is taken
            self.moves_remaining -= 1
            reward -= 0.1 # Cost for taking a turn
            moved = True
            
            if movement == 1: target_y -= 1 # Up
            elif movement == 2: target_y += 1 # Down
            elif movement == 3: target_x -= 1 # Left
            elif movement == 4: target_x += 1 # Right

            # Check boundaries and walls
            if not (0 <= target_y < self.GRID_HEIGHT and 0 <= target_x < self.GRID_WIDTH and self.grid[target_y, target_x] == 0):
                # Invalid move (hit wall or boundary)
                target_y, target_x = self.player_pos # Reset to current position
                reward -= 0.5 # Penalty for bumping into a wall
            else:
                self.player_pos = (target_y, target_x)

        # Check for item collection
        if self.player_pos in self.items:
            self.items.remove(self.player_pos)
            item_reward = 5.0
            self.score += int(item_reward)
            reward += item_reward
            # sfx: item_pickup.wav

        # Check for termination conditions
        terminated = False
        if self.player_pos == self.exit_pos:
            win_reward = 100
            self.score += win_reward
            reward += win_reward
            self.game_over = True
            terminated = True
            self.win_status = "win"
            # sfx: level_complete.wav
        elif self.moves_remaining <= 0:
            lose_penalty = -100
            self.score += lose_penalty
            reward += lose_penalty
            self.game_over = True
            terminated = True
            self.win_status = "lose"
            # sfx: game_over.wav

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[y, x] == 1: # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, rect.inflate(-self.CELL_SIZE*0.7, -self.CELL_SIZE*0.7))
                else: # Empty space
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        # Draw valid move highlights
        if not self.game_over:
            py, px = self.player_pos
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and self.grid[ny, nx] == 0:
                    highlight_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    highlight_surf.fill((*self.COLOR_HIGHLIGHT, 30))
                    self.screen.blit(highlight_surf, (nx * self.CELL_SIZE, ny * self.CELL_SIZE))

        # Draw items
        for y, x in self.items:
            center_x = int(x * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(y * self.CELL_SIZE + self.CELL_SIZE / 2)
            glow_radius = int(self.CELL_SIZE * 0.3)
            item_radius = int(self.CELL_SIZE * 0.2)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_ITEM_GLOW, 80))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_ITEM_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, item_radius, self.COLOR_ITEM)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, item_radius, self.COLOR_ITEM)

        # Draw exit
        ex, ey = self.exit_pos[1], self.exit_pos[0]
        exit_rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-4, -4))
        pygame.draw.rect(self.screen, self.COLOR_EXIT_INNER, exit_rect.inflate(-12, -12))
        
        # Draw player
        py, px = self.player_pos
        player_center_x = int(px * self.CELL_SIZE + self.CELL_SIZE / 2)
        player_center_y = int(py * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        # Pulsating glow
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        glow_radius = int(self.CELL_SIZE * 0.4 * (1 + pulse * 0.2))
        glow_alpha = int(80 + pulse * 40)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, glow_radius, (*self.COLOR_PLAYER_GLOW, glow_alpha))
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, glow_radius, (*self.COLOR_PLAYER_GLOW, glow_alpha + 20))

        player_rect = pygame.Rect(0, 0, self.CELL_SIZE * 0.6, self.CELL_SIZE * 0.6)
        player_rect.center = (player_center_x, player_center_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
    
    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        # Draw Moves Remaining
        moves_text = f"MOVES: {self.moves_remaining}"
        draw_text(moves_text, self.font_medium, self.COLOR_TEXT, (10, 10))
        
        # Draw Score
        score_text = f"SCORE: {self.score}"
        score_size = self.font_medium.size(score_text)
        draw_text(score_text, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - score_size[0] - 10, 10))
        
        # Draw Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_status == "win":
                msg = "LEVEL COMPLETE"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            shadow_surf = self.font_large.render(msg, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, text_rect.move(3, 3))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Maze")
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping for human play ---
        action = [0, 0, 0] # Default: no-op, no buttons
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q: # Quit
                    done = True

        # Only step if a move was made
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
        if done:
            print("\nGame Over!")
            print(f"Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause before closing or resetting
            obs, info = env.reset() # Automatically reset for another game
            done = False


    env.close()