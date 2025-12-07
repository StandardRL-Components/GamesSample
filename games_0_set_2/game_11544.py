import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:08:19.766043
# Source Brief: brief_01544.md
# Brief Index: 1544
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent transforms between a square and a circle
    to collect gems on a grid while avoiding spawning obstacles, all under a time limit.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) - No effect in this game.
    - actions[2]: Shift button (0=released, 1=held) - Toggles player shape.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +1.0 for each gem collected.
    - +50.0 for collecting all gems (win).
    - -50.0 if time runs out (lose).
    - +0.1 for moving closer to the nearest gem.
    - -0.01 for moving away from the nearest gem.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    game_description = (
        "Transform between a fast circle and a sturdy square to collect all the gems on the grid. "
        "Avoid spawning obstacles and beat the clock to win!"
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move. Press Shift to transform between shapes."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = self.HEIGHT // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.HEIGHT) // 2

        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        
        self.TOTAL_GEMS = 20
        self.OBSTACLE_SPAWN_RATE = 5 * self.FPS  # Spawn one every 5 seconds
        self.OBSTACLE_LIFETIME = 10 * self.FPS # Lasts for 10 seconds

        # --- Colors ---
        self.COLOR_BG = (15, 25, 35)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_grid_pos = None
        self.player_visual_pos = None
        self.player_shape = "square" # "square" or "circle"
        self.gems = []
        self.gems_collected = 0
        self.obstacles = []
        self.particles = []
        self.obstacle_spawn_timer = 0
        self.prev_shift_held = False
        self.last_gem_distance = 0.0
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Shape Shifter")

        # Initialize state variables
        # self.reset() # reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        
        self.player_grid_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.player_visual_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_shape = "square"
        
        self.gems = self._spawn_gems()
        self.obstacles = []
        self.particles = []
        self.obstacle_spawn_timer = self.OBSTACLE_SPAWN_RATE
        self.prev_shift_held = False

        self.last_gem_distance = self._get_closest_gem_dist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Update Game Logic ---
        self.steps += 1
        self._update_player(movement, shift_held)
        self._update_obstacles()
        self._update_particles()
        
        # --- Handle Interactions & Rewards ---
        collected_gem = self._handle_gem_collection()
        if collected_gem:
            reward += 1.0
            # SFX: Gem collect sound

        # Distance-based reward
        new_gem_distance = self._get_closest_gem_dist()
        if self.last_gem_distance is not None:
            if new_gem_distance < self.last_gem_distance:
                reward += 0.1
            elif new_gem_distance > self.last_gem_distance:
                reward -= 0.01
        self.last_gem_distance = new_gem_distance
        
        self.score += reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = False
        if terminated:
            self.game_over = True
            if self.gems_collected == self.TOTAL_GEMS:
                self.score += 50.0 # Win bonus
                reward += 50.0
            else: # Time ran out
                self.score -= 50.0 # Lose penalty
                reward -= 50.0
        
        # MUST return exactly this 5-tuple
        obs = self._get_observation()
        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return obs, reward, terminated, truncated, self._get_info()

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
            "gems_collected": self.gems_collected,
            "time_left": self.TIME_LIMIT_SECONDS - (self.steps / self.FPS),
        }

    # --- Update Methods ---
    def _update_player(self, movement, shift_held):
        # Shape toggling on press (rising edge)
        if shift_held and not self.prev_shift_held:
            self.player_shape = "circle" if self.player_shape == "square" else "square"
            # SFX: Transform sound
        self.prev_shift_held = shift_held

        # Movement
        px, py = self.player_grid_pos
        if movement == 1 and py > 0: py -= 1  # Up
        if movement == 2 and py < self.GRID_SIZE - 1: py += 1  # Down
        if movement == 3 and px > 0: px -= 1  # Left
        if movement == 4 and px < self.GRID_SIZE - 1: px += 1  # Right
        self.player_grid_pos = (px, py)

        # Smooth visual movement
        target_pos = self._grid_to_pixel(self.player_grid_pos)
        speed = 0.25 if self.player_shape == "circle" else 0.15 # Circle is faster
        self.player_visual_pos = self.player_visual_pos.lerp(target_pos, speed)

    def _update_obstacles(self):
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self.obstacle_spawn_timer = self.OBSTACLE_SPAWN_RATE
            
            occupied_cells = set(self.gems) | {self.player_grid_pos}
            for obs in self.obstacles:
                occupied_cells.add(obs['pos'])
            
            possible_cells = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE) if (x, y) not in occupied_cells]
            if possible_cells:
                pos = random.choice(possible_cells)
                self.obstacles.append({'pos': pos, 'age': 0})
                # SFX: Obstacle spawn sound

        # Update and remove old obstacles
        for obs in self.obstacles:
            obs['age'] += 1
        self.obstacles = [obs for obs in self.obstacles if obs['age'] < self.OBSTACLE_LIFETIME]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['age'] += 1
        self.particles = [p for p in self.particles if p['age'] < p['lifetime']]

    def _handle_gem_collection(self):
        player_center = self._grid_to_pixel(self.player_grid_pos)
        if self.player_visual_pos.distance_to(player_center) < self.CELL_SIZE * 0.1:
            if self.player_grid_pos in self.gems:
                self.gems.remove(self.player_grid_pos)
                self.gems_collected += 1
                self._spawn_particles(player_center, self.COLOR_GEM, 20)
                return True
        return False

    def _check_termination(self):
        if self.gems_collected == self.TOTAL_GEMS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    # --- Spawning Methods ---
    def _spawn_gems(self):
        gems = set()
        while len(gems) < self.TOTAL_GEMS:
            pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            if pos != self.player_grid_pos:
                gems.add(pos)
        return list(gems)

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'age': 0,
                'lifetime': random.randint(15, 30),
                'color': color,
                'radius': random.uniform(2, 5)
            })

    # --- Rendering Methods ---
    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            y = i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, y))

        # Draw obstacles
        for obs in self.obstacles:
            pixel_pos = self._grid_to_pixel(obs['pos'])
            size = self.CELL_SIZE * 0.5
            rect = pygame.Rect(pixel_pos.x - size / 2, pixel_pos.y - size / 2, size, size)
            
            # Fade-in/out effect
            age_ratio = min(1.0, obs['age'] / (self.OBSTACLE_LIFETIME * 0.2)) # Fade in
            if obs['age'] > self.OBSTACLE_LIFETIME * 0.8: # Fade out
                age_ratio = (self.OBSTACLE_LIFETIME - obs['age']) / (self.OBSTACLE_LIFETIME * 0.2)
            
            self._render_glow_rect(self.screen, self.COLOR_OBSTACLE, rect, max_alpha=int(200 * age_ratio))

        # Draw gems
        for gem_pos in self.gems:
            pixel_pos = self._grid_to_pixel(gem_pos)
            self._render_gem(pixel_pos)

        # Draw particles
        for p in self.particles:
            alpha = 1 - (p['age'] / p['lifetime'])
            color = (*p['color'], int(255 * alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius'] * alpha), color)

        # Draw player
        size = self.CELL_SIZE * 0.7
        if self.player_shape == "square":
            rect = pygame.Rect(self.player_visual_pos.x - size / 2, self.player_visual_pos.y - size / 2, size, size)
            self._render_glow_rect(self.screen, self.COLOR_PLAYER, rect)
        else: # circle
            self._render_glow_circle(self.screen, self.COLOR_PLAYER, self.player_visual_pos, size / 2)

    def _render_ui(self):
        # Gem counter
        gem_text = self.font_main.render(f"GEMS: {self.gems_collected}/{self.TOTAL_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 10))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_timer.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 5))
        self.screen.blit(timer_text, timer_rect)

    # --- Helper & Utility Methods ---
    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return pygame.Vector2(x, y)

    def _get_closest_gem_dist(self):
        if not self.gems:
            return 0
        player_pos_vec = pygame.Vector2(self.player_grid_pos)
        min_dist = float('inf')
        for gem_pos in self.gems:
            dist = player_pos_vec.distance_to(pygame.Vector2(gem_pos))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _render_glow_circle(self, surface, color, center, radius, max_alpha=150):
        center_int = (int(center.x), int(center.y))
        radius = int(radius)
        for i in range(4):
            alpha = max_alpha * (1 - i / 4)**2
            c = (*color, int(alpha))
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius + i * 3, c)
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius + i * 3, c)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def _render_glow_rect(self, surface, color, rect, max_alpha=150):
        for i in range(4):
            alpha = max_alpha * (1 - i / 4)**2
            c = (*color, int(alpha))
            glow_rect = rect.inflate(i * 6, i * 6)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, c, (0, 0, *glow_rect.size), border_radius=5)
            surface.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def _render_gem(self, center_pos):
        size = self.CELL_SIZE * 0.3
        points = [
            (center_pos.x, center_pos.y - size),
            (center_pos.x + size, center_pos.y),
            (center_pos.x, center_pos.y + size),
            (center_pos.x - size, center_pos.y),
        ]
        # Glow
        for i in range(4):
            alpha = 100 * (1 - i / 4)**2
            c = (*self.COLOR_GEM, int(alpha))
            s = size + i * 3
            pts = [ (center_pos.x, center_pos.y - s), (center_pos.x + s, center_pos.y), (center_pos.x, center_pos.y + s), (center_pos.x - s, center_pos.y) ]
            pygame.gfxdraw.aapolygon(self.screen, pts, c)
        
        # Main shape
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play & Demonstration ---
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    print("----------------\n")

    while not done:
        # Manual control mapping
        movement = 0 # none
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        
        if keys[pygame.K_q]:
            break
            
        action = [movement, 0, shift_held] # space is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Gems: {info['gems_collected']}")

        if done:
            print("\n--- Game Over ---")
            print(f"Final Score: {info['score']:.2f}")
            print(f"Gems Collected: {info['gems_collected']}/{env.TOTAL_GEMS}")
            print(f"Time Left: {info['time_left']:.1f}s")
            
            # Wait for a moment before closing
            pygame.time.wait(3000)

    env.close()