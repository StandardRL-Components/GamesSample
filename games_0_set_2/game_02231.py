
# Generated: 2025-08-28T04:10:01.078314
# Source Brief: brief_02231.md
# Brief Index: 2231

        
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

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. "
        "Collect all the blue gems and avoid the red traps."
    )

    game_description = (
        "A top-down maze puzzle. Collect all the gems to win, but watch out for "
        "dangerous traps. Plan your path carefully to maximize your score!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.MAZE_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.MAZE_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE
        
        self.NUM_GEMS = 25
        self.NUM_TRAPS = 15
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_GEM = (0, 150, 255)
        self.COLOR_TRAP = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.maze = None
        self.player_pos = None
        self.gem_positions = []
        self.trap_positions = []
        self.effects = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.effects = []

        self._generate_maze()
        
        open_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(open_cells)
        
        self.player_pos = self.np_random.choice(open_cells)
        
        # Define forbidden zones for traps (around player start and gems)
        forbidden_for_traps = set()
        forbidden_for_traps.add(tuple(self.player_pos))
        for neighbor in self._get_neighbors(self.player_pos):
            forbidden_for_traps.add(tuple(neighbor))

        # Place gems
        self.gem_positions = []
        possible_gem_cells = [cell for cell in open_cells if tuple(cell) != tuple(self.player_pos)]
        self.np_random.shuffle(possible_gem_cells)
        
        for _ in range(self.NUM_GEMS):
            if not possible_gem_cells: break
            gem_pos = possible_gem_cells.pop()
            self.gem_positions.append(gem_pos)
            forbidden_for_traps.add(tuple(gem_pos))
            for neighbor in self._get_neighbors(gem_pos):
                forbidden_for_traps.add(tuple(neighbor))

        # Place traps in safe locations
        self.trap_positions = []
        possible_trap_cells = [cell for cell in open_cells if tuple(cell) not in forbidden_for_traps]
        self.np_random.shuffle(possible_trap_cells)
        
        for _ in range(self.NUM_TRAPS):
            if not possible_trap_cells: break
            self.trap_positions.append(possible_trap_cells.pop())
            
        return self._get_observation(), self._get_info()

    def _get_neighbors(self, pos, distance=1):
        x, y = pos
        neighbors = []
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                if abs(dx) + abs(dy) == distance:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT:
                        neighbors.append([nx, ny])
        return neighbors

    def _generate_maze(self):
        self.maze = np.ones((self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.uint8)
        stack = deque()
        
        start_x, start_y = self.np_random.integers(0, self.MAZE_WIDTH), self.np_random.integers(0, self.MAZE_HEIGHT)
        self.maze[start_x, start_y] = 0
        stack.append((start_x, start_y))
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                self.maze[nx, ny] = 0
                self.maze[cx + (nx - cx) // 2, cy + (ny - cy) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = -0.1  # Cost of living
        
        px, py = self.player_pos
        nx, ny = px, py
        
        if movement == 1: ny -= 1 # Up
        elif movement == 2: ny += 1 # Down
        elif movement == 3: nx -= 1 # Left
        elif movement == 4: nx += 1 # Right

        # Wall collision check
        if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[nx, ny] == 0:
            self.player_pos = [nx, ny]
        
        self.steps += 1
        
        # Check for gem collection
        for i, gem_pos in enumerate(self.gem_positions):
            if np.array_equal(self.player_pos, gem_pos):
                reward += 10
                self.score += 10
                self.gems_collected += 1
                self.gem_positions.pop(i)
                # Add particle effect
                self.effects.append({'type': 'gem_collect', 'pos': gem_pos, 'timer': 10, 'radius': 0})
                # Sound effect placeholder: # sfx_gem_collect.play()
                break

        # Check for trap collision
        for trap_pos in self.trap_positions:
            if np.array_equal(self.player_pos, trap_pos):
                reward -= 100
                self.score -= 100
                self.game_over = True
                # Add particle effect
                self.effects.append({'type': 'trap_hit', 'pos': trap_pos, 'timer': 15, 'radius': 0})
                # Sound effect placeholder: # sfx_player_death.play()
                break
        
        # Proximity to trap reward
        if not self.game_over:
            min_dist = float('inf')
            for trap_pos in self.trap_positions:
                dist = abs(self.player_pos[0] - trap_pos[0]) + abs(self.player_pos[1] - trap_pos[1])
                min_dist = min(min_dist, dist)
            if 0 < min_dist < 3:
                reward += (1.0 / min_dist)
        
        # Update effects
        self.effects = [e for e in self.effects if e['timer'] > 0]
        for effect in self.effects:
            effect['timer'] -= 1
            effect['radius'] += 1
            
        terminated = self._check_termination()
        
        # Goal completion reward
        if self.gems_collected == self.NUM_GEMS and not self.game_over:
            reward += 100
            self.score += 100
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
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
        # Draw maze walls
        for x in range(self.MAZE_WIDTH):
            for y in range(self.MAZE_HEIGHT):
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, 
                                     (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw traps
        trap_pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        trap_size = int(self.CELL_SIZE * 0.3 + trap_pulse * self.CELL_SIZE * 0.2)
        for tx, ty in self.trap_positions:
            center_x = int((tx + 0.5) * self.CELL_SIZE)
            center_y = int((ty + 0.5) * self.CELL_SIZE)
            points = [
                (center_x, center_y - trap_size),
                (center_x - trap_size, center_y + trap_size),
                (center_x + trap_size, center_y + trap_size)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TRAP)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TRAP)

        # Draw gems
        gem_size = int(self.CELL_SIZE * 0.5)
        for gx, gy in self.gem_positions:
            center_x = int((gx + 0.5) * self.CELL_SIZE)
            center_y = int((gy + 0.5) * self.CELL_SIZE)
            rect = (center_x - gem_size // 2, center_y - gem_size // 2, gem_size, gem_size)
            pygame.draw.rect(self.screen, self.COLOR_GEM, rect, border_radius=2)
            
            # Sparkle effect
            for i in range(4):
                angle = self.steps * 0.1 + i * math.pi / 2
                sparkle_dist = gem_size * 0.7
                sx = center_x + int(math.cos(angle) * sparkle_dist)
                sy = center_y + int(math.sin(angle) * sparkle_dist)
                pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 1)

        # Draw player
        px, py = self.player_pos
        player_center_x = int((px + 0.5) * self.CELL_SIZE)
        player_center_y = int((py + 0.5) * self.CELL_SIZE)
        player_radius = int(self.CELL_SIZE * 0.35)
        
        # Glow effect
        glow_radius = int(player_radius * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_center_x - glow_radius, player_center_y - glow_radius))

        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        
        # Draw effects
        for effect in self.effects:
            ex, ey = effect['pos']
            center_x = int((ex + 0.5) * self.CELL_SIZE)
            center_y = int((ey + 0.5) * self.CELL_SIZE)
            alpha = max(0, 255 * (effect['timer'] / 10))
            if effect['type'] == 'gem_collect':
                color = (*self.COLOR_GEM, alpha)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(effect['radius'] * 2), color)
            elif effect['type'] == 'trap_hit':
                color = (*self.COLOR_TRAP, alpha)
                num_lines = 8
                for i in range(num_lines):
                    angle = i * (2 * math.pi / num_lines)
                    start_pos = (center_x, center_y)
                    end_pos = (center_x + math.cos(angle) * effect['radius'] * 3, 
                               center_y + math.sin(angle) * effect['radius'] * 3)
                    pygame.draw.aaline(self.screen, color, start_pos, end_pos, True)

    def _render_ui(self):
        # Gem count
        gem_text = f"Gems: {self.gems_collected}/{self.NUM_GEMS}"
        gem_surf = self.font_large.render(gem_text, True, self.COLOR_TEXT)
        self.screen.blit(gem_surf, (10, 5))
        
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 5))

        # Game Over / Win message
        if self.game_over:
            if self.gems_collected == self.NUM_GEMS:
                msg = "YOU WIN!"
                color = self.COLOR_GEM
            else:
                msg = "GAME OVER"
                color = self.COLOR_TRAP
            
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Simple background for text
            bg_rect = msg_rect.inflate(20, 10)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((20, 20, 30, 200))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This part requires a window and is for testing/demonstration.
    # It will not run in a headless environment.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy" # Force headless for servers
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Maze Gem Collector")
    except pygame.error:
        print("Pygame display unavailable. Skipping manual play example.")
        env.close()
        exit()

    obs, info = env.reset()
    done = False
    
    # Game loop for manual play
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'r' key
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Only step if a move action is taken, as auto_advance is False
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Wait a bit to make it playable
        pygame.time.wait(100)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()