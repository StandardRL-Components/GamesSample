
# Generated: 2025-08-27T20:51:54.746732
# Source Brief: brief_02598.md
# Brief Index: 2598

        
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
        "Controls: Arrow keys to move. Avoid the red horrors and reach the glowing exit before time runs out."
    )

    game_description = (
        "Navigate a dark, procedurally generated maze, evading lurking horrors to reach the exit. You have three lives and a limited time."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 32, 20 # Dims for maze grid
        self.CELL_SIZE = self.HEIGHT // self.MAZE_HEIGHT
        self.GAME_WIDTH = self.MAZE_WIDTH * self.CELL_SIZE
        self.GAME_HEIGHT = self.MAZE_HEIGHT * self.CELL_SIZE
        self.X_OFFSET = (self.WIDTH - self.GAME_WIDTH) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GAME_HEIGHT) // 2
        
        self.MAX_STEPS = 1200  # 120 seconds at 10 steps/sec
        self.START_LIVES = 3
        self.NUM_ENEMIES = 2
        self.INVINCIBILITY_FRAMES = 20

        # Colors
        self.COLOR_BG = (10, 5, 15)
        self.COLOR_WALL = (40, 30, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_ENEMY = (200, 0, 0)
        self.COLOR_ENEMY_GLOW = (150, 0, 0)
        self.COLOR_EXIT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)

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
        self.font_large = pygame.font.Font(None, 48)

        # Create vignette surface
        self.vignette = self._create_vignette()

        # State variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.enemies = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.invincibility_timer = 0
        self.game_over = False
        self.particles = []
        self.screen_shake = 0
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def _create_vignette(self):
        vignette = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        radius = self.HEIGHT // 1.5
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        for i in range(int(radius), 0, -1):
            alpha = int(255 * (1 - (i / radius))**2)
            alpha = min(255, max(0, alpha))
            pygame.draw.circle(vignette, (0, 0, 0, alpha), (center_x, center_y), i)
        return vignette

    def _generate_maze(self):
        maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        start_node = (0, 0)
        stack = [start_node]
        maze[start_node[0], start_node[1]] = 0

        while stack:
            current_y, current_x = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current_x + dx * 2, current_y + dy * 2
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                next_x, next_y = random.choice(neighbors)
                wall_x, wall_y = current_x + (next_x - current_x) // 2, current_y + (next_y - current_y) // 2
                maze[next_y, next_x] = 0
                maze[wall_y, wall_x] = 0
                stack.append((next_y, next_x))
            else:
                stack.pop()
        
        # Set exit at the furthest point from start using BFS
        q = deque([((0,0), 0)])
        visited = {(0,0)}
        farthest_pos, max_dist = (0,0), 0
        while q:
            (y,x), dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_pos = (y,x)

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.MAZE_HEIGHT and 0 <= nx < self.MAZE_WIDTH and maze[ny,nx] == 0 and (ny,nx) not in visited:
                    visited.add((ny,nx))
                    q.append(((ny,nx), dist+1))
        
        self.exit_pos = farthest_pos
        return maze

    def _find_path_bfs(self, start, end):
        q = deque([(start, [start])])
        visited = {start}
        while q:
            (y, x), path = q.popleft()
            if (y, x) == end:
                return path
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.MAZE_HEIGHT and 0 <= nx < self.MAZE_WIDTH and self.maze[ny, nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    new_path = list(path)
                    new_path.append((ny, nx))
                    q.append(((ny, nx), new_path))
        return None

    def _generate_enemy_patrol_path(self):
        while True:
            path_len = 0
            while path_len < 10: # Ensure path is not trivial
                start_y, start_x = random.randint(0, self.MAZE_HEIGHT-1), random.randint(0, self.MAZE_WIDTH-1)
                end_y, end_x = random.randint(0, self.MAZE_HEIGHT-1), random.randint(0, self.MAZE_WIDTH-1)
                if self.maze[start_y, start_x] == 0 and self.maze[end_y, end_x] == 0 and (start_y, start_x) != (end_y, end_x):
                    path_a_to_b = self._find_path_bfs((start_y, start_x), (end_y, end_x))
                    if path_a_to_b:
                        path_b_to_a = self._find_path_bfs((end_y, end_x), (start_y, start_x))
                        if path_b_to_a:
                            full_path = path_a_to_b + path_b_to_a[1:-1]
                            path_len = len(full_path)
                            if path_len >= 10:
                                return full_path
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.START_LIVES
        self.game_over = False
        self.invincibility_timer = 0
        self.screen_shake = 0
        self.particles.clear()
        
        self.maze = self._generate_maze()
        self.player_pos = (0, 0)
        
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            path = self._generate_enemy_patrol_path()
            self.enemies.append({
                "path": path,
                "path_index": 0,
                "pos": path[0]
            })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.1 # Survival reward
        
        # Update player position
        py, px = self.player_pos
        if movement == 1: # Up
            ny, nx = py - 1, px
        elif movement == 2: # Down
            ny, nx = py + 1, px
        elif movement == 3: # Left
            ny, nx = py, px - 1
        elif movement == 4: # Right
            ny, nx = py, px + 1
        else: # No-op
            ny, nx = py, px

        if 0 <= ny < self.MAZE_HEIGHT and 0 <= nx < self.MAZE_WIDTH and self.maze[ny, nx] == 0:
            self.player_pos = (ny, nx)

        # Update enemies
        for enemy in self.enemies:
            enemy["path_index"] = (enemy["path_index"] + 1) % len(enemy["path"])
            enemy["pos"] = enemy["path"][enemy["path_index"]]

        # Update timers
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        self.steps += 1
        
        # Check collisions
        if self.invincibility_timer == 0:
            for enemy in self.enemies:
                if self.player_pos == enemy["pos"]:
                    self.lives -= 1
                    reward -= 5
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    self.screen_shake = 10
                    self._create_particles(self.player_pos)
                    # sfx: player_hit.wav
                    break

        # Check for termination conditions
        terminated = False
        if self.player_pos == self.exit_pos:
            reward = 50
            self.score += 50
            terminated = True
            self.game_over = True
            # sfx: victory.wav
        elif self.lives <= 0:
            reward = -50
            terminated = True
            self.game_over = True
            # sfx: game_over.wav
        elif self.steps >= self.MAX_STEPS:
            reward = -50
            terminated = True
            self.game_over = True
            # sfx: timeout.wav

        self.score += reward # Score tracks rewards

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos):
        y, x = pos
        cx = self.X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
        cy = self.Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': cx, 'y': cy,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': random.randint(10, 20),
                'color': random.choice([self.COLOR_PLAYER, self.COLOR_ENEMY, (255, 100, 100)])
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        shake_offset = (0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            shake_offset = (random.randint(-4, 4), random.randint(-4, 4))
        
        # Render all game elements
        self._render_game(shake_offset)
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, shake_offset):
        self._render_maze(shake_offset)
        self._render_exit(shake_offset)
        self._render_enemies(shake_offset)
        self._render_player(shake_offset)
        self._update_and_render_particles(shake_offset)
        self.screen.blit(self.vignette, (0, 0))

    def _render_maze(self, offset):
        ox, oy = offset
        for r in range(self.MAZE_HEIGHT):
            for c in range(self.MAZE_WIDTH):
                if self.maze[r, c] == 1:
                    pygame.draw.rect(
                        self.screen, self.COLOR_WALL,
                        (self.X_OFFSET + c * self.CELL_SIZE + ox,
                         self.Y_OFFSET + r * self.CELL_SIZE + oy,
                         self.CELL_SIZE, self.CELL_SIZE)
                    )

    def _render_player(self, offset):
        ox, oy = offset
        y, x = self.player_pos
        center_x = self.X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2 + ox
        center_y = self.Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2 + oy
        radius = self.CELL_SIZE // 3

        # Invincibility flicker
        if self.invincibility_timer > 0 and self.steps % 4 < 2:
            return

        # Glow effect
        glow_radius = int(radius * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (center_x - glow_radius, center_y - glow_radius))

        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (center_x, center_y), radius)

    def _render_enemies(self, offset):
        ox, oy = offset
        for enemy in self.enemies:
            y, x = enemy["pos"]
            center_x = self.X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2 + ox
            center_y = self.Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2 + oy
            
            # Flicker effect
            base_radius = self.CELL_SIZE // 4
            flicker = math.sin(self.steps * 0.4 + id(enemy)) * 2
            radius = int(base_radius + flicker)

            # Glow
            glow_radius = int(radius * 2.5)
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*self.COLOR_ENEMY_GLOW, 70), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surface, (center_x - glow_radius, center_y - glow_radius))

            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (center_x, center_y), radius)

    def _render_exit(self, offset):
        ox, oy = offset
        y, x = self.exit_pos
        rect = pygame.Rect(
            self.X_OFFSET + x * self.CELL_SIZE + ox,
            self.Y_OFFSET + y * self.CELL_SIZE + oy,
            self.CELL_SIZE, self.CELL_SIZE
        )
        
        # Glow effect
        for i in range(5, 0, -1):
            glow_rect = rect.inflate(i * 4, i * 4)
            alpha = 100 - i * 20
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_EXIT, alpha), s.get_rect(), border_radius=i*2)
            self.screen.blit(s, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect, border_radius=3)

    def _update_and_render_particles(self, offset):
        ox, oy = offset
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 20))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
                temp_surf.fill(color)
                self.screen.blit(temp_surf, (p['x'] + ox, p['y'] + oy))

    def _render_ui(self):
        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 10.0
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 30))

        if self.game_over:
            if self.lives > 0 and self.player_pos == self.exit_pos:
                end_text = "ESCAPE SUCCESSFUL"
            else:
                end_text = "YOU DIED"
            end_surf = self.font_large.render(end_text, True, self.COLOR_EXIT if self.lives > 0 else self.COLOR_ENEMY)
            self.screen.blit(end_surf, (self.WIDTH // 2 - end_surf.get_width() // 2, self.HEIGHT // 2 - end_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    import time
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Maze Horror")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    done = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle keydown for single-press actions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                if event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    done = False
                    total_reward = 0
                    continue

                if not done:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated

        # Render the observation to the display window
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose the observation back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Limit FPS to not burn CPU
        
    env.close()