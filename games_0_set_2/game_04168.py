
# Generated: 2025-08-28T01:37:48.967090
# Source Brief: brief_04168.md
# Brief Index: 4168

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press space while on a treasure to collect it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated maze and collect all the treasures before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    
    FPS = 30
    TOTAL_TIME_SECONDS = 180
    MAX_STEPS = TOTAL_TIME_SECONDS * FPS
    
    NUM_TREASURES = 25
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_WALL = (80, 90, 110)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_TREASURE = (255, 220, 50)
    COLOR_TREASURE_SPARKLE = (255, 255, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (0, 0, 0, 128)

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.player_pos = np.array([0.0, 0.0])
        self.player_grid_pos = np.array([0, 0])
        self.treasures = []
        self.maze = np.array([[]])
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.win = False
        self.movement_cooldown = 0
        
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.TOTAL_TIME_SECONDS
        self.particles = []
        self.movement_cooldown = 0

        # Generate maze and place items
        self._generate_maze()
        floor_tiles = np.argwhere(self.maze == 0)
        self.np_random.shuffle(floor_tiles)

        # Place player
        player_start_idx = self.np_random.integers(len(floor_tiles))
        self.player_grid_pos = floor_tiles[player_start_idx]
        self.player_pos = self.player_grid_pos.astype(float) * self.CELL_SIZE + self.CELL_SIZE / 2
        
        # Place treasures
        treasure_indices = (np.arange(self.NUM_TREASURES) + player_start_idx + 1) % len(floor_tiles)
        self.treasures = [tuple(floor_tiles[i]) for i in treasure_indices]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _generate_maze(self):
        # Using randomized DFS (Recursive Backtracker)
        w, h = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        visited = np.zeros((w, h), dtype=bool)
        self.maze = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.uint8)
        
        stack = []
        
        # Start at a random cell
        start_x, start_y = self.np_random.integers(0, w), self.np_random.integers(0, h)
        stack.append((start_x, start_y))
        visited[start_x, start_y] = True
        self.maze[start_x*2+1, start_y*2+1] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[nx, ny]:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                
                # Carve path
                self.maze[cx*2+1 + dx, cy*2+1 + dy] = 0
                self.maze[nx*2+1, ny*2+1] = 0
                
                visited[nx, ny] = True
                stack.append((nx, ny))
            else:
                stack.pop()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- Game Logic ---
        self.steps += 1
        self.time_remaining -= 1 / self.FPS
        
        # Find distance to nearest treasure before moving
        dist_before = self._get_dist_to_nearest_treasure()

        # Player Movement
        if self.movement_cooldown <= 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1  # Right
            
            if dx != 0 or dy != 0:
                next_grid_pos = self.player_grid_pos + np.array([dx, dy])
                if self._is_walkable(next_grid_pos):
                    self.player_grid_pos = next_grid_pos
                    self.movement_cooldown = 4 # 4 frames to move one cell
        
        if self.movement_cooldown > 0:
            self.movement_cooldown -= 1
        
        # Smooth interpolation of visual position
        target_pos = self.player_grid_pos.astype(float) * self.CELL_SIZE + self.CELL_SIZE / 2
        self.player_pos += (target_pos - self.player_pos) * 0.5

        # Find distance to nearest treasure after moving
        dist_after = self._get_dist_to_nearest_treasure()
        
        # Distance-based reward
        if dist_after < dist_before:
            reward += 0.1
        elif dist_after > dist_before:
            reward -= 0.01

        # Treasure Collection
        if space_held:
            player_grid_tuple = tuple(self.player_grid_pos)
            if player_grid_tuple in self.treasures:
                self.treasures.remove(player_grid_tuple)
                self.score += 1
                reward += 10
                # sfx: treasure collect
                self._spawn_particles(self.player_pos[0], self.player_pos[1], 30, self.COLOR_TREASURE)

        # Update particles
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if self.time_remaining <= 0:
            self.game_over = True
            terminated = True
            self.win = False
            reward -= 50
        
        if len(self.treasures) == 0:
            self.game_over = True
            terminated = True
            self.win = True
            reward += 100
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _is_walkable(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        return self.maze[x, y] == 0

    def _get_dist_to_nearest_treasure(self):
        if not self.treasures:
            return 0
        player_p = self.player_grid_pos
        min_dist = float('inf')
        for t_pos in self.treasures:
            dist = np.linalg.norm(player_p - np.array(t_pos))
            if dist < min_dist:
                min_dist = dist
        return min_dist

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "treasures_left": len(self.treasures),
        }

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw maze walls
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.maze[c, r] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw treasures
        sparkle_size_mod = (math.sin(self.steps * 0.2) + 1) / 2 * 4
        for tx, ty in self.treasures:
            center_x = tx * self.CELL_SIZE + self.CELL_SIZE // 2
            center_y = ty * self.CELL_SIZE + self.CELL_SIZE // 2
            pygame.draw.rect(self.screen, self.COLOR_TREASURE, (center_x - 6, center_y - 6, 12, 12))
            pygame.draw.rect(self.screen, self.COLOR_TREASURE_SPARKLE, (center_x - 2 - sparkle_size_mod/2, center_y - 2 - sparkle_size_mod/2, 4 + sparkle_size_mod, 4 + sparkle_size_mod))

        # Draw particles
        for p in self.particles:
            p_color = list(p['color'])
            p_color.append(int(255 * (p['life'] / p['max_life']))) # Alpha
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p_color)

        # Draw player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_size = self.CELL_SIZE * 0.7
        glow_size = player_size * 2.5
        
        # Player Glow (using gfxdraw for alpha)
        pygame.gfxdraw.filled_circle(self.screen, px, py, int(glow_size / 2), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, int(glow_size / 2), self.COLOR_PLAYER_GLOW)
        
        # Player Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px - player_size/2, py - player_size/2, player_size, player_size), border_radius=2)
        
    def _render_ui(self):
        # UI Background panels
        s = pygame.Surface((180, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (10, 5))
        self.screen.blit(s, (self.SCREEN_WIDTH - 190, 5))
        
        # Score display
        score_text = self.font_ui.render(f"TREASURE: {self.score} / {self.NUM_TREASURES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Timer display
        time_str = f"TIME: {max(0, int(self.time_remaining)):03}"
        timer_text = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 15))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER
            else:
                msg = "TIME UP"
                color = (255, 80, 80)
                
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': np.array([x, y], dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            p['size'] *= 0.98
            if p['life'] > 0 and p['size'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run in a headless environment
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game with Pygame ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render the observation to the display ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        obs_transposed = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_transposed)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()