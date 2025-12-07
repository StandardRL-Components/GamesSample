
# Generated: 2025-08-28T00:48:35.770778
# Source Brief: brief_03902.md
# Brief Index: 3902

        
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

    user_guide = (
        "Controls: Use arrow keys to move. Push blocks to form a path from green to red."
    )

    game_description = (
        "A strategic block-pushing puzzle. Connect the start (green) and end (red) points "
        "by pushing blocks to form a continuous path. You have a limited number of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_BLOCK = (60, 120, 220)
        self.COLOR_BLOCK_OUTLINE = (100, 160, 255)
        self.COLOR_START = (0, 200, 100)
        self.COLOR_END = (220, 50, 50)
        self.COLOR_PATH_HIGHLIGHT = (100, 200, 255, 150) # RGBA for transparency
        self.COLOR_GOAL_LINE = (255, 255, 255, 20) # Faint hint for the goal
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WIN_TEXT = (150, 255, 150)
        self.COLOR_LOSE_TEXT = (255, 150, 150)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Grid layout calculation
        self.cell_size = min((self.WIDTH - 100) // self.GRID_SIZE, (self.HEIGHT - 100) // self.GRID_SIZE)
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_height) // 2

        # Initialize state variables
        self.player_pos = None
        self.block_positions = None
        self.start_pos = None
        self.end_pos = None
        self.steps = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.win_status = None
        self.particles = None
        self.connected_path = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Puzzle definition
        self.start_pos = (0, 2)
        self.end_pos = (4, 2)
        self.player_pos = [2, 4] # Use list for mutability
        self.block_positions = [[1, 1], [2, 1], [3, 3], [1, 3]]

        # Game state
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.win_status = None # None, "win", or "loss"
        
        # Visual effects state
        self.particles = []
        self.connected_path = set()
        
        # Initial check for win (in case of pre-solved puzzle)
        self._check_win_condition()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost for taking a turn
        
        moved = False
        if movement > 0:
            self.moves_remaining -= 1
            moved = True

        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            target_x, target_y = self.player_pos[0] + dx, self.player_pos[1] + dy

            # Check if target is a block
            target_is_block = False
            block_idx = -1
            for i, block_pos in enumerate(self.block_positions):
                if block_pos[0] == target_x and block_pos[1] == target_y:
                    target_is_block = True
                    block_idx = i
                    break
            
            if target_is_block:
                block_target_x, block_target_y = target_x + dx, target_y + dy
                
                # Check if block can be pushed
                can_push = True
                # 1. Check wall collision
                if not (0 <= block_target_x < self.GRID_SIZE and 0 <= block_target_y < self.GRID_SIZE):
                    can_push = False
                # 2. Check collision with other blocks
                if can_push:
                    for i, other_block in enumerate(self.block_positions):
                        if i != block_idx and other_block[0] == block_target_x and other_block[1] == block_target_y:
                            can_push = False
                            break
                
                if can_push:
                    old_block_pos = self.block_positions[block_idx][:]
                    
                    # Calculate reward for pushing block
                    dist_before = abs(old_block_pos[1] - self.start_pos[1])
                    dist_after = abs(block_target_y - self.start_pos[1])
                    if dist_after < dist_before:
                        reward += 1.0
                    
                    self.block_positions[block_idx] = [block_target_x, block_target_y]
                    self.player_pos = [target_x, target_y]
                    self._spawn_particles(old_block_pos, self.COLOR_BLOCK_OUTLINE)
                    # sfx: block_push.wav
            
            # If target is empty space
            elif 0 <= target_x < self.GRID_SIZE and 0 <= target_y < self.GRID_SIZE:
                self.player_pos = [target_x, target_y]

        self.steps += 1
        self.score += reward

        # Check game state after move
        is_path, path_nodes = self._check_win_condition()
        self.connected_path = path_nodes

        terminated = False
        if is_path:
            reward += 100.0
            self.score += 100.0
            self.game_over = True
            self.win_status = "win"
            terminated = True
            # sfx: win.wav
        elif moved and self.moves_remaining <= 0:
            reward -= 100.0
            self.score -= 100.0
            self.game_over = True
            self.win_status = "loss"
            terminated = True
            # sfx: lose.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if self.win_status is None:
                self.win_status = "loss" # Timeout is a loss

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_win_condition(self):
        block_set = {tuple(p) for p in self.block_positions}
        q = deque([self.start_pos])
        visited = {self.start_pos}
        
        while q:
            x, y = q.popleft()

            if (x, y) == self.end_pos:
                return True, visited

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if neighbor in visited:
                    continue
                
                # Path can go through start, end, or any block
                if neighbor == self.end_pos or neighbor in block_set:
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                        visited.add(neighbor)
                        q.append(neighbor)
        
        return False, set()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1 # life -= 1
            alpha = max(0, min(255, int(p[4] * 10)))
            radius = int(p[4] / 2)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, (*p[5], alpha))

        # Draw goal line hint
        goal_y_px = self.grid_offset_y + self.start_pos[1] * self.cell_size + self.cell_size // 2
        goal_line_rect = pygame.Rect(self.grid_offset_x, goal_y_px - self.cell_size // 4, self.grid_width, self.cell_size // 2)
        s = pygame.Surface((goal_line_rect.width, goal_line_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_GOAL_LINE)
        self.screen.blit(s, goal_line_rect.topleft)

        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)

        # Helper for grid to pixel conversion
        def to_pixel(grid_pos):
            x = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
            y = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
            return x, y

        # Draw start and end points
        start_px = to_pixel(self.start_pos)
        end_px = to_pixel(self.end_pos)
        radius = self.cell_size // 3
        pygame.gfxdraw.filled_circle(self.screen, start_px[0], start_px[1], radius, self.COLOR_START)
        pygame.gfxdraw.aacircle(self.screen, start_px[0], start_px[1], radius, self.COLOR_START)
        pygame.gfxdraw.filled_circle(self.screen, end_px[0], end_px[1], radius, self.COLOR_END)
        pygame.gfxdraw.aacircle(self.screen, end_px[0], end_px[1], radius, self.COLOR_END)
        
        # Draw blocks
        block_rad = self.cell_size // 2 - 5
        for pos in self.block_positions:
            px, py = to_pixel(pos)
            rect = pygame.Rect(px - block_rad, py - block_rad, block_rad * 2, block_rad * 2)
            
            # Highlight connected blocks
            if tuple(pos) in self.connected_path:
                s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, self.COLOR_PATH_HIGHLIGHT, s.get_rect(), border_radius=5)
                self.screen.blit(s, rect.topleft)
            
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_OUTLINE, rect, 2, border_radius=5)

        # Draw player
        player_px, player_py = to_pixel(self.player_pos)
        player_rad = self.cell_size // 4
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, player_rad, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_rad, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_rad+1, self.COLOR_PLAYER_OUTLINE)


    def _render_ui(self):
        # Draw moves remaining
        moves_text = self.font_small.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Draw score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))
        
        # Draw game over message
        if self.game_over:
            if self.win_status == "win":
                msg = "YOU WIN!"
                color = self.COLOR_WIN_TEXT
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE_TEXT
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "win": self.win_status == "win"
        }

    def _spawn_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel_center(grid_pos)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.randint(10, 25)
            self.particles.append([px, py, vx, vy, life, color])

    def _grid_to_pixel_center(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
        y = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
        return x, y

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
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
        
        # In manual play, we only step when a key is pressed
        if action[0] != 0 and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()