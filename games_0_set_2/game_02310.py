
# Generated: 2025-08-27T19:58:37.252285
# Source Brief: brief_02310.md
# Brief Index: 2310

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a procedurally generated maze game.

    The player must navigate a maze to collect all gems before running out of moves.
    The game is turn-based, with each movement action consuming one move.

    **State Variables:**
    - Player position (x, y)
    - List of gem positions [(x1, y1), ...]
    - Move counter
    - Collected gem count
    - Maze layout (2D numpy array)

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - A rendered RGB image of the game screen.

    **Rewards:**
    - +10 for collecting a gem.
    - +0.1 for moving closer to the nearest gem.
    - -0.2 for moving further from the nearest gem.
    - +100 for collecting all gems (win).
    - -50 for running out of moves (lose).
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Collect all yellow gems before you run out of moves!"
    )

    game_description = (
        "A minimalist puzzle game. Find the optimal path to collect all gems in a procedurally generated maze within a limited number of moves."
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAZE_COLS, MAZE_ROWS = 31, 19  # Must be odd numbers
    CELL_SIZE = 20
    OFFSET_X = (WIDTH - MAZE_COLS * CELL_SIZE) // 2
    OFFSET_Y = (HEIGHT - MAZE_ROWS * CELL_SIZE) // 2

    TOTAL_GEMS = 15
    MAX_MOVES = 50

    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (40, 40, 60)
    COLOR_PATH = (25, 25, 40)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_SPARKLE = (255, 255, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Game state variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.gems = None
        self.moves_left = 0
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_gem_dist = 0

        self.validate_implementation()


    def _generate_maze(self):
        # Using recursive backtracking for maze generation
        maze = np.ones((self.MAZE_ROWS, self.MAZE_COLS), dtype=np.uint8) # 1 = wall
        stack = collections.deque()

        # Start at a random odd-indexed cell
        start_x = self.np_random.integers(0, self.MAZE_COLS // 2) * 2 + 1
        start_y = self.np_random.integers(0, self.MAZE_ROWS // 2) * 2 + 1
        maze[start_y, start_x] = 0 # 0 = path
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_COLS and 0 < ny < self.MAZE_ROWS and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.maze = self._generate_maze()
        
        # Find all possible path locations
        path_indices = np.argwhere(self.maze == 0)
        
        # Place player
        player_idx = self.np_random.integers(0, len(path_indices))
        self.player_pos = path_indices[player_idx].tolist()[::-1] # [x, y] format
        
        # Place gems, ensuring no overlap with player or each other
        available_indices = np.delete(path_indices, player_idx, axis=0)
        gem_indices_idx = self.np_random.choice(len(available_indices), self.TOTAL_GEMS, replace=False)
        gem_indices = available_indices[gem_indices_idx]
        self.gems = [pos.tolist()[::-1] for pos in gem_indices] # List of [x, y]

        self.moves_left = self.MAX_MOVES
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_gem_dist = self._find_nearest_gem_dist(self.player_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Only process if a move is attempted
        if movement != 0:
            self.moves_left -= 1
            
            # Get proposed new position
            px, py = self.player_pos
            if movement == 1: # Up
                py -= 1
            elif movement == 2: # Down
                py += 1
            elif movement == 3: # Left
                px -= 1
            elif movement == 4: # Right
                px += 1

            # Wall collision check
            if self.maze[py, px] == 0: # 0 is path
                self.player_pos = [px, py]

            # Gem collection check
            if self.player_pos in self.gems:
                self.gems.remove(self.player_pos)
                self.gems_collected += 1
                reward += 10.0
                # placeholder: self.play_sound('collect_gem')

            # Distance-based reward
            current_dist = self._find_nearest_gem_dist(self.player_pos)
            if current_dist < self.last_gem_dist:
                reward += 0.1
            elif current_dist > self.last_gem_dist:
                reward -= 0.2
            self.last_gem_dist = current_dist

        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.gems_collected == self.TOTAL_GEMS:
                reward += 100 # Win bonus
                # placeholder: self.play_sound('win')
            else:
                reward -= 50 # Lose penalty
                # placeholder: self.play_sound('lose')
            self.score += reward # Add terminal reward to final score

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _check_termination(self):
        if self.gems_collected == self.TOTAL_GEMS:
            self.game_over = True
            return True
        if self.moves_left <= 0:
            self.game_over = True
            return True
        return False

    def _find_nearest_gem_dist(self, pos):
        if not self.gems:
            return 0
        px, py = pos
        min_dist = float('inf')
        for gx, gy in self.gems:
            dist = abs(px - gx) + abs(py - gy) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "gems_collected": self.gems_collected,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render maze paths and walls
        for y in range(self.MAZE_ROWS):
            for x in range(self.MAZE_COLS):
                rect = pygame.Rect(
                    self.OFFSET_X + x * self.CELL_SIZE,
                    self.OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                color = self.COLOR_PATH if self.maze[y, x] == 0 else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, rect)

        # Render gems with sparkle effect
        sparkle_size = 3 * (1 + math.sin(self.steps * 0.2)) / 2
        for gx, gy in self.gems:
            center_x = int(self.OFFSET_X + (gx + 0.5) * self.CELL_SIZE)
            center_y = int(self.OFFSET_Y + (gy + 0.5) * self.CELL_SIZE)
            radius = self.CELL_SIZE // 3
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GEM)
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius / 2 + sparkle_size), self.COLOR_GEM_SPARKLE)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius / 2 + sparkle_size), self.COLOR_GEM_SPARKLE)


        # Render player with glow effect
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.OFFSET_X + px * self.CELL_SIZE + self.CELL_SIZE // 4,
            self.OFFSET_Y + py * self.CELL_SIZE + self.CELL_SIZE // 4,
            self.CELL_SIZE // 2, self.CELL_SIZE // 2
        )
        
        # Glow effect
        glow_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surface, 
            self.COLOR_PLAYER_GLOW,
            (self.CELL_SIZE // 2, self.CELL_SIZE // 2),
            self.CELL_SIZE // 2
        )
        self.screen.blit(glow_surface, (player_rect.x - self.CELL_SIZE // 4, player_rect.y - self.CELL_SIZE // 4))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_text = font.render(text, True, self.COLOR_TEXT_SHADOW)
            main_text = font.render(text, True, color)
            self.screen.blit(shadow_text, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(main_text, pos)

        # Moves Left display
        moves_text = f"Moves Left: {self.moves_left}"
        draw_text(moves_text, self.font_large, self.COLOR_TEXT, (20, 10))

        # Gems collected display
        gems_text = f"Gems: {self.gems_collected}/{self.TOTAL_GEMS}"
        text_width = self.font_large.size(gems_text)[0]
        draw_text(gems_text, self.font_large, self.COLOR_GEM, (self.WIDTH - text_width - 20, 10))
        
        # Game Over message
        if self.game_over:
            if self.gems_collected == self.TOTAL_GEMS:
                msg = "YOU WIN!"
                color = self.COLOR_GEM
            else:
                msg = "GAME OVER"
                color = self.COLOR_PLAYER
            
            msg_width, msg_height = self.font_large.size(msg)
            draw_text(msg, self.font_large, color, 
                ((self.WIDTH - msg_width) // 2, (self.HEIGHT - msg_height) // 2))


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
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
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Use a separate window for rendering if not in a headless environment
    try:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Gem Collector")
        clock = pygame.time.Clock()
        
        print("\n" + "="*30)
        print("      MANUAL PLAY MODE")
        print("="*30)
        print(env.user_guide)
        print("Press ESC or close window to quit.")

        while not done:
            action = [0, 0, 0] # Default to no-op
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    # Since auto_advance is False, we only need to register one key press
                    # and then step the environment.
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_r: # Reset
                        obs, info = env.reset()
                        action = [0,0,0] # don't step after reset
                    
                    if action[0] != 0: # If a move was made
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                        if terminated:
                            print("Episode finished. Press 'R' to restart.")
            
            # Rendering
            render_obs = env._get_observation()
            surf = pygame.surfarray.make_surface(np.transpose(render_obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit frame rate
            
    except Exception as e:
        print(f"\nCould not create display. Error: {e}")
        print("This may happen in a headless environment. The environment itself is still functional.")
    finally:
        env.close()