
# Generated: 2025-08-27T16:15:41.729296
# Source Brief: brief_01172.md
# Brief Index: 1172

        
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
        "Controls: Use arrow keys (↑↓←→) to move your character through the maze."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze filled with deadly traps to reach the exit within a strict time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # Class attribute for persistent difficulty scaling
    global_total_steps = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_W, self.MAZE_H = 31, 19  # Must be odd numbers
        self.TILE_SIZE = 20
        self.RENDER_W = self.MAZE_W * self.TILE_SIZE
        self.RENDER_H = self.MAZE_H * self.TILE_SIZE
        self.OFFSET_X = (self.WIDTH - self.RENDER_W) // 2
        self.OFFSET_Y = (self.HEIGHT - self.RENDER_H) // 2
        
        self.INITIAL_TIME = 60
        self.MAX_EPISODE_STEPS = 1000
        self.INITIAL_TRAPS = 5

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_FLOOR = (80, 80, 100)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_EXIT = (0, 255, 0)
        self.COLOR_TRAP = (255, 0, 0)
        self.COLOR_TEXT = (240, 240, 240)

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.trap_pos = []
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.flash_counter = 0

        self.validate_implementation()

    def _generate_maze(self):
        maze = np.ones((self.MAZE_H, self.MAZE_W), dtype=np.uint8)
        stack = deque()
        
        start_x, start_y = (
            self.np_random.integers(0, self.MAZE_W // 2) * 2 + 1,
            self.np_random.integers(0, self.MAZE_H // 2) * 2 + 1,
        )
        
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.MAZE_W and 0 <= ny < self.MAZE_H and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                mx, my = (cx + nx) // 2, (cy + ny) // 2
                maze[ny, nx] = 0
                maze[my, mx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_furthest_point(self, start_pos):
        q = deque([(start_pos, 0)])
        visited = {start_pos}
        furthest_point = start_pos
        max_dist = 0

        while q:
            (x, y), dist = q.popleft()

            if dist > max_dist:
                max_dist = dist
                furthest_point = (x, y)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.MAZE_W and 0 <= ny < self.MAZE_H and
                        self.maze[ny, nx] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    q.append(((nx, ny), dist + 1))
        
        return furthest_point

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze = self._generate_maze()
        
        self.player_pos = (1, 1)
        self.exit_pos = self._find_furthest_point(self.player_pos)

        floor_tiles = []
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                if self.maze[y, x] == 0:
                    floor_tiles.append((x, y))

        floor_tiles.remove(self.player_pos)
        if self.exit_pos in floor_tiles:
            floor_tiles.remove(self.exit_pos)
        
        num_traps = self.INITIAL_TRAPS + (GameEnv.global_total_steps // 1000)
        num_traps = min(num_traps, len(floor_tiles))
        
        trap_indices = self.np_random.choice(len(floor_tiles), num_traps, replace=False)
        self.trap_pos = [floor_tiles[i] for i in trap_indices]
        
        self.time_remaining = self.INITIAL_TIME
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.flash_counter = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = -0.1  # Cost of time
        terminated = False
        
        px, py = self.player_pos
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if movement != 0:
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.MAZE_W and 0 <= ny < self.MAZE_H and self.maze[ny, nx] == 0:
                self.player_pos = (nx, ny)

        self.steps += 1
        GameEnv.global_total_steps += 1
        self.time_remaining -= 1
        
        # Check game state
        if self.player_pos == self.exit_pos:
            # sfx: victory fanfare
            goal_reward = 50.0 * (self.time_remaining / self.INITIAL_TIME)
            reward += goal_reward
            self.score += reward
            terminated = True
            self.game_over = True
        elif self.player_pos in self.trap_pos:
            # sfx: player falls into trap
            reward += -10.0
            self.score += reward
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            # sfx: timeout buzzer
            reward += -10.0 # Same penalty as a trap
            self.score += reward
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        if not terminated:
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _render_game(self):
        # Draw maze
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                rect = pygame.Rect(
                    self.OFFSET_X + x * self.TILE_SIZE,
                    self.OFFSET_Y + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw traps
        time_pressure = 1 - (max(0, self.time_remaining) / self.INITIAL_TIME)
        flash_speed = 0.1 + 0.4 * time_pressure
        self.flash_counter += flash_speed
        
        for tx, ty in self.trap_pos:
            rect = pygame.Rect(
                self.OFFSET_X + tx * self.TILE_SIZE,
                self.OFFSET_Y + ty * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            # Use a sine wave for a smooth flashing effect
            alpha = int(128 + 127 * math.sin(self.flash_counter))
            trap_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            trap_surface.fill((*self.COLOR_TRAP, alpha))
            self.screen.blit(trap_surface, rect.topleft)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.OFFSET_X + ex * self.TILE_SIZE,
            self.OFFSET_Y + ey * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        
        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.OFFSET_X + px * self.TILE_SIZE,
            self.OFFSET_Y + py * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Add a small border to make player pop
        pygame.draw.rect(self.screen, (0,0,0), player_rect, 1)

    def _render_ui(self):
        time_text = self.font_main.render(f"TIME: {self.time_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 15, 10))
        
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            status_text = ""
            if self.player_pos == self.exit_pos:
                status_text = "LEVEL COMPLETE!"
            elif self.player_pos in self.trap_pos:
                status_text = "TRAPPED!"
            elif self.time_remaining <= 0:
                status_text = "TIME'S UP!"
            
            text_surface = self.font_main.render(status_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
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
        # Need to reset to generate a valid state first
        self.reset()
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print(env.user_guide)

    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Episode Reset ---")
                if event.key == pygame.K_q: # Press 'Q' to quit
                    done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Since this is a turn-based game, we only step when a key is pressed
        # This differs from the main loop which would continuously poll.
        # For manual play, we need an event-driven step.
        # A simple way is to step only if a move key is pressed.
        
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}")
                # Wait for a moment before allowing reset
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                print("--- Episode Reset ---")


        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit frame rate for manual play

    env.close()