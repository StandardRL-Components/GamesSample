import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Collect fruits to score. Avoid the red obstacles. "
        "A timer counts down from 60 seconds. Collect 25 fruits to win."
    )

    game_description = (
        "Navigate a maze, collecting fruits and avoiding moving obstacles. "
        "Score bonus points for risky fruit grabs near enemies. Win by collecting 25 fruits before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.W, self.H = 640, 400
        self.FPS = 30

        self.observation_space = Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # --- Game Constants ---
        self.CELL_SIZE = 20
        self.GRID_W, self.GRID_H = self.W // self.CELL_SIZE, self.H // self.CELL_SIZE
        self.PLAYER_SPEED = 3.0
        self.PLAYER_RADIUS = self.CELL_SIZE * 0.4
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.FRUITS_TO_WIN = 25
        self.NUM_OBSTACLES = 5
        self.NUM_FRUITS = 3
        self.RISKY_DISTANCE = self.CELL_SIZE * 3

        # --- Colors ---
        self.COLOR_BG = (15, 25, 35)
        self.COLOR_WALL = (40, 60, 80)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 100, 30)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 50, 50, 40)
        self.COLOR_FRUITS = [(50, 255, 50), (255, 50, 255), (50, 255, 255)]
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # --- Fonts ---
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_float = pygame.font.SysFont("Arial", 16, bold=True)

        # --- State Variables (initialized in reset) ---
        self.maze = None
        self.player_pos = None
        self.fruits = []
        self.obstacles = []
        self.particles = []
        self.floating_texts = []
        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False
        self.base_obstacle_speed = 0
        self.obstacle_speed_increase = 0

    def _generate_maze(self):
        maze = np.ones((self.GRID_W, self.GRID_H), dtype=np.uint8)
        stack = deque()
        
        start_x = self.rng.integers(1, self.GRID_W - 1) | 1
        start_y = self.rng.integers(1, self.GRID_H - 1) | 1
        maze[start_x, start_y] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_W - 1 and 0 < ny < self.GRID_H - 1 and maze[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                maze[nx, ny] = 0
                maze[cx + (nx - cx) // 2, cy + (ny - cy) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _get_random_empty_cell_pos(self):
        while True:
            x = self.rng.integers(0, self.GRID_W)
            y = self.rng.integers(0, self.GRID_H)
            if self.maze[x, y] == 0:
                return pygame.Vector2(x * self.CELL_SIZE + self.CELL_SIZE / 2, 
                                      y * self.CELL_SIZE + self.CELL_SIZE / 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False
        self.particles.clear()
        self.floating_texts.clear()
        
        self.maze = self._generate_maze()
        self.player_pos = self._get_random_empty_cell_pos()
        
        self.fruits = []
        for i in range(self.NUM_FRUITS):
            self._spawn_fruit(i)

        self.obstacles = []
        self.base_obstacle_speed = 0.5 * self.CELL_SIZE / self.FPS
        self.obstacle_speed_increase = 0.05 * self.CELL_SIZE / self.FPS
        
        for _ in range(self.NUM_OBSTACLES):
            pos = self._get_random_empty_cell_pos()
            self.obstacles.append({
                "pos": pos,
                "angle": self.rng.uniform(0, 2 * math.pi),
                "turn_speed": self.rng.uniform(-0.05, 0.05)
            })

        return self._get_observation(), self._get_info()

    def _spawn_fruit(self, fruit_index):
        pos = self._get_random_empty_cell_pos()
        color = self.COLOR_FRUITS[fruit_index % len(self.COLOR_FRUITS)]
        
        # Ensure it doesn't spawn on the player
        while pos.distance_to(self.player_pos) < self.CELL_SIZE * 2:
            pos = self._get_random_empty_cell_pos()

        if fruit_index < len(self.fruits):
            self.fruits[fruit_index] = {"pos": pos, "color": color, "pulse": self.rng.uniform(0, math.pi)}
        else:
            self.fruits.append({"pos": pos, "color": color, "pulse": self.rng.uniform(0, math.pi)})

    def _get_dist_to_nearest(self, pos, entities, key='pos'):
        if not entities:
            return float('inf')
        return min(pos.distance_to(e[key]) for e in entities)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Continuous Rewards ---
        dist_fruit_before = self._get_dist_to_nearest(self.player_pos, self.fruits)
        dist_obstacle_before = self._get_dist_to_nearest(self.player_pos, self.obstacles)

        # --- 1. Handle Player Input & Movement ---
        movement = action[0]
        vel = pygame.Vector2(0, 0)
        if movement == 1: vel.y = -1 # Up
        elif movement == 2: vel.y = 1  # Down
        elif movement == 3: vel.x = -1 # Left
        elif movement == 4: vel.x = 1  # Right
        if vel.length() > 0:
            vel.normalize_ip()
            vel *= self.PLAYER_SPEED

        # --- 2. Update Positions & Handle Wall Collisions ---
        new_pos = self.player_pos + vel
        
        # Check X collision
        gx = int((new_pos.x + math.copysign(self.PLAYER_RADIUS, vel.x)) / self.CELL_SIZE)
        gy = int(self.player_pos.y / self.CELL_SIZE)
        if not (0 <= gx < self.GRID_W and self.maze[gx, gy] == 0):
            new_pos.x = self.player_pos.x
        
        # Check Y collision
        gx = int(self.player_pos.x / self.CELL_SIZE)
        gy = int((new_pos.y + math.copysign(self.PLAYER_RADIUS, vel.y)) / self.CELL_SIZE)
        if not (0 <= gy < self.GRID_H and self.maze[gx, gy] == 0):
            new_pos.y = self.player_pos.y

        self.player_pos = new_pos
        
        # --- Update Obstacles ---
        current_obstacle_speed = self.base_obstacle_speed + self.obstacle_speed_increase * (self.steps // (10 * self.FPS))
        for obs in self.obstacles:
            obs['angle'] += obs['turn_speed']
            obs_vel = pygame.Vector2(math.cos(obs['angle']), math.sin(obs['angle'])) * current_obstacle_speed
            
            obs_new_pos = obs['pos'] + obs_vel
            gx = int(obs_new_pos.x / self.CELL_SIZE)
            gy = int(obs_new_pos.y / self.CELL_SIZE)

            if not (0 <= gx < self.GRID_W and 0 <= gy < self.GRID_H and self.maze[gx, gy] == 0):
                obs['angle'] += math.pi + self.rng.uniform(-0.5, 0.5) # Bounce
            else:
                obs['pos'] = obs_new_pos
        
        # --- Post-movement Reward Calculation ---
        dist_fruit_after = self._get_dist_to_nearest(self.player_pos, self.fruits)
        dist_obstacle_after = self._get_dist_to_nearest(self.player_pos, self.obstacles)
        
        if dist_fruit_after < dist_fruit_before: reward += 0.01 # Small reward for moving towards fruit
        if dist_obstacle_after < dist_obstacle_before: reward -= 0.02 # Small penalty for moving towards obstacle

        # --- 3. Handle Entity Collisions ---
        # Obstacle collision (takes precedence)
        for obs in self.obstacles:
            if self.player_pos.distance_to(obs['pos']) < self.PLAYER_RADIUS + self.CELL_SIZE * 0.4:
                reward = -10
                self.game_over = True
                break
        
        # Fruit collection
        if not self.game_over:
            for i, fruit in enumerate(self.fruits):
                if self.player_pos.distance_to(fruit['pos']) < self.PLAYER_RADIUS + self.CELL_SIZE * 0.3:
                    base_reward = 1
                    dist_to_nearest_obs = self._get_dist_to_nearest(self.player_pos, self.obstacles)
                    
                    bonus_reward = 0
                    if dist_to_nearest_obs < self.RISKY_DISTANCE:
                        bonus_reward = 2 # Risky bonus
                    
                    total_fruit_reward = base_reward + bonus_reward
                    reward += total_fruit_reward
                    self.score += total_fruit_reward
                    self.fruits_collected += 1
                    
                    self._spawn_particles(fruit['pos'], fruit['color'], 20)
                    self._spawn_floating_text(f"+{total_fruit_reward}", fruit['pos'], (255,255,100))
                    
                    self._spawn_fruit(i)
                    dist_fruit_after = self._get_dist_to_nearest(self.player_pos, self.fruits)
                    break 

        # --- 4. Update Animations ---
        self._update_particles()
        self._update_floating_texts()

        # --- 5. Check Termination Conditions ---
        terminated = self.game_over
        if self.fruits_collected >= self.FRUITS_TO_WIN:
            reward += 50
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": self.rng.integers(15, 30), "color": color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_floating_text(self, text, pos, color):
        self.floating_texts.append({"pos": pos.copy(), "text": text, "life": 45, "color": color})

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['pos'].y -= 0.5
            ft['life'] -= 1
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw maze walls
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw fruits
        for fruit in self.fruits:
            fruit['pulse'] += 0.1
            pulse_size = (math.sin(fruit['pulse']) + 1) / 2 * 3
            radius = int(self.CELL_SIZE * 0.3 + pulse_size)
            pygame.gfxdraw.filled_circle(self.screen, int(fruit['pos'].x), int(fruit['pos'].y), radius, fruit['color'])
            pygame.gfxdraw.aacircle(self.screen, int(fruit['pos'].x), int(fruit['pos'].y), radius, fruit['color'])

        # Draw obstacles
        for obs in self.obstacles:
            size = int(self.CELL_SIZE * 0.8)
            glow_size = int(size * 1.5)
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, self.COLOR_OBSTACLE_GLOW)
            self.screen.blit(glow_surf, (int(obs['pos'].x - glow_size), int(obs['pos'].y - glow_size)))
            rect = pygame.Rect(obs['pos'].x - size/2, obs['pos'].y - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            color_with_alpha = (*p['color'], alpha)
            size = int(p['life'] / 6)
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (size, size), size)
                self.screen.blit(temp_surf, (p['pos'].x - size, p['pos'].y - size), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw player
        glow_radius = int(self.PLAYER_RADIUS * 1.8 + (math.sin(self.steps * 0.1) + 1) * 3)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (int(self.player_pos.x - glow_radius), int(self.player_pos.y - glow_radius)))
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(self.PLAYER_RADIUS), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(self.PLAYER_RADIUS), self.COLOR_PLAYER)
        
        # Draw floating texts
        for ft in self.floating_texts:
            alpha = max(0, min(255, int(255 * (ft['life'] / 45))))
            text_surf = self.font_float.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, text_surf.get_rect(center=ft['pos']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Fruits
        fruit_text = self.font_ui.render(f"FRUITS: {self.fruits_collected}/{self.FRUITS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(fruit_text, (self.W // 2 - fruit_text.get_width() // 2, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.W - time_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_collected": self.fruits_collected,
            "time_remaining_seconds": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not work in a headless environment but is useful for local testing
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        import pygame
        
        env = GameEnv()
        obs, info = env.reset()
        
        running = True
        terminated = False
        
        display_screen = pygame.display.set_mode((env.W, env.H))
        pygame.display.set_caption("Maze Runner")

        while running:
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                obs, info = env.reset()
                terminated = False

            movement = 0 # no-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            action = [movement, 0, 0] # Dummy actions for [1] and [2]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            env.clock.tick(env.FPS)

        env.close()
    except pygame.error as e:
        print("\nCould not create display for interactive mode. Pygame might be in headless mode.")
        print("This is expected if you're running in a server environment.")
        print(f"Pygame error: {e}")