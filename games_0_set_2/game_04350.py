import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑ and ↓ to move the cart up and down the track to dodge obstacles and collect gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Steer a mine cart through a perilous, procedurally generated track. Dodge rocks, collect valuable gems, and try to reach the end before you crash!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000
    
    # Colors
    COLOR_BG_TOP = (25, 28, 50)
    COLOR_BG_BOTTOM = (45, 50, 80)
    COLOR_TRACK = (92, 64, 51)
    COLOR_SLEEPER = (61, 42, 34)
    COLOR_CART = (220, 50, 50)
    COLOR_CART_ACCENT = (255, 100, 100)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_OBSTACLE_ACCENT = (110, 110, 120)
    COLOR_GEM = (255, 220, 50)
    COLOR_GEM_GLOW = (255, 240, 150, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEART = (255, 80, 80)

    # Physics & Gameplay
    CART_SPEED = 8
    CART_WIDTH, CART_HEIGHT = 30, 20
    TRACK_Y_CENTER = HEIGHT // 2
    TRACK_HEIGHT = 80
    TRACK_AMPLITUDE = 60
    TRACK_WAVELENGTH = 300
    GRAVITY = 1.5
    LIFT = -2.5
    DRAG = 0.9

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 0
        self.gems_collected = 0
        self.cart_pos = pygame.math.Vector2(0, 0)
        self.cart_vel = pygame.math.Vector2(0, 0)
        self.track_points = []
        self.obstacles = []
        self.gems = []
        self.particles = []
        self.obstacle_spawn_prob = 0.0
        self.gem_spawn_prob = 0.0
        self.track_offset = 0
        self.invincibility_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 3
        self.gems_collected = 0
        
        self.cart_pos = pygame.math.Vector2(self.WIDTH * 0.2, self.TRACK_Y_CENTER)
        self.cart_vel = pygame.math.Vector2(0, 0)
        
        self.track_offset = 0
        self._generate_track()
        
        self.obstacles = []
        self.gems = []
        self.particles = []
        
        self.obstacle_spawn_prob = 0.02
        self.gem_spawn_prob = 0.015
        self.invincibility_timer = self.FPS # Brief invincibility at start
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        
        reward = 0.1  # Survival reward

        # --- Update Game Logic ---
        self.steps += 1
        self.track_offset += self.CART_SPEED

        # Handle player input
        if movement == 1:  # Up
            self.cart_vel.y += self.LIFT
        elif movement == 2:  # Down
            self.cart_vel.y += self.GRAVITY
        
        # Update cart physics
        self.cart_vel.y *= self.DRAG
        self.cart_pos.y += self.cart_vel.y

        # Keep cart on track
        track_y_center = self._get_track_y_at(self.cart_pos.x + self.track_offset)
        track_top = track_y_center - self.TRACK_HEIGHT // 2 + self.CART_HEIGHT // 2
        track_bottom = track_y_center + self.TRACK_HEIGHT // 2 - self.CART_HEIGHT // 2
        self.cart_pos.y = np.clip(self.cart_pos.y, track_top, track_bottom)
        
        # Update invincibility
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # Update entities
        self._update_particles()
        reward += self._update_obstacles()
        reward += self._update_gems()

        # Spawn new entities
        self._spawn_entities()

        # Difficulty scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.obstacle_spawn_prob = min(0.1, self.obstacle_spawn_prob + 0.001)

        # Check for termination
        if self.lives <= 0:
            self.game_over = True
            reward -= 20 # Extra penalty for losing
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            reward += 50 # Win bonus
        
        terminated = self.game_over
        truncated = False # Not using truncation based on time limit
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_track(self):
        self.track_points = []
        total_length = self.MAX_STEPS * self.CART_SPEED + self.WIDTH * 2
        for i in range(total_length):
            offset1 = math.sin(i / self.TRACK_WAVELENGTH) * self.TRACK_AMPLITUDE
            offset2 = math.sin(i / (self.TRACK_WAVELENGTH * 0.4)) * (self.TRACK_AMPLITUDE * 0.3)
            self.track_points.append(self.TRACK_Y_CENTER + offset1 + offset2)

    def _get_track_y_at(self, x):
        idx = int(x)
        if 0 <= idx < len(self.track_points):
            return self.track_points[idx]
        return self.TRACK_Y_CENTER

    def _update_obstacles(self):
        reward_change = 0
        cart_rect = pygame.Rect(self.cart_pos.x - self.CART_WIDTH / 2, self.cart_pos.y - self.CART_HEIGHT / 2, self.CART_WIDTH, self.CART_HEIGHT)
        
        for obstacle in self.obstacles[:]:
            obstacle['pos'].x -= self.CART_SPEED
            if obstacle['pos'].x < -20:
                self.obstacles.remove(obstacle)
                continue
            
            obstacle_rect = pygame.Rect(obstacle['pos'].x - obstacle['size']/2, obstacle['pos'].y - obstacle['size']/2, obstacle['size'], obstacle['size'])
            if cart_rect.colliderect(obstacle_rect) and self.invincibility_timer == 0:
                self.lives -= 1
                reward_change -= 5
                self.invincibility_timer = self.FPS * 2 # 2 seconds invincibility
                self._create_sparks(self.cart_pos)
                # Sound: "clank.wav"
                if obstacle in self.obstacles: # Check if not already removed
                    self.obstacles.remove(obstacle)
        return reward_change

    def _update_gems(self):
        reward_change = 0
        cart_rect = pygame.Rect(self.cart_pos.x - self.CART_WIDTH / 2, self.cart_pos.y - self.CART_HEIGHT / 2, self.CART_WIDTH, self.CART_HEIGHT)

        for gem in self.gems[:]:
            gem['pos'].x -= self.CART_SPEED
            if gem['pos'].x < -20:
                self.gems.remove(gem)
                continue

            gem_rect = pygame.Rect(gem['pos'].x - 10, gem['pos'].y - 10, 20, 20)
            if cart_rect.colliderect(gem_rect):
                self.gems_collected += 1
                self.score += 10
                reward_change += 1
                self._create_gleam(gem['pos'])
                # Sound: "gem_collect.wav"
                self.gems.remove(gem)
        return reward_change

    def _spawn_entities(self):
        spawn_x = self.WIDTH + 20
        track_y = self._get_track_y_at(spawn_x + self.track_offset)
        
        # Spawn obstacles
        if self.np_random.random() < self.obstacle_spawn_prob:
            size = self.np_random.integers(20, 40)
            y_offset = self.np_random.uniform(-self.TRACK_HEIGHT/2, self.TRACK_HEIGHT/2)
            pos = pygame.math.Vector2(spawn_x, track_y + y_offset)
            self.obstacles.append({'pos': pos, 'size': size})

        # Spawn gems
        if self.np_random.random() < self.gem_spawn_prob:
            y_offset = self.np_random.uniform(-self.TRACK_HEIGHT/2, self.TRACK_HEIGHT/2)
            pos = pygame.math.Vector2(spawn_x, track_y + y_offset)
            self.gems.append({'pos': pos})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_sparks(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            colors = [(255, 255, 100), (255, 180, 50)]
            color_index = self.np_random.integers(0, len(colors))
            self.particles.append({
                'pos': pygame.math.Vector2(pos), 'vel': vel, 'lifespan': self.np_random.integers(10, 20),
                'color': colors[color_index], 'size': self.np_random.integers(2, 4)
            })

    def _create_gleam(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.math.Vector2(pos), 'vel': vel, 'lifespan': self.np_random.integers(8, 15),
                'color': self.COLOR_GEM_GLOW[:3], 'size': self.np_random.integers(3, 6)
            })

    def _get_observation(self):
        self._render_background()
        self._render_track()
        self._render_entities()
        self._render_cart()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_track(self):
        start_idx = int(self.track_offset)
        points_top, points_bottom = [], []
        
        for i in range(self.WIDTH + 1):
            world_x = i + self.track_offset
            center_y = self._get_track_y_at(world_x)
            
            # Sleepers
            if int(world_x) % 40 == 0:
                p1 = (i, center_y - self.TRACK_HEIGHT / 2 - 5)
                p2 = (i, center_y + self.TRACK_HEIGHT / 2 + 5)
                pygame.draw.line(self.screen, self.COLOR_SLEEPER, p1, p2, 10)

            points_top.append((i, center_y - self.TRACK_HEIGHT / 2))
            points_bottom.append((i, center_y + self.TRACK_HEIGHT / 2))
        
        pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points_top, 3)
        pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points_bottom, 3)

    def _render_entities(self):
        # Particles
        for p in self.particles:
            size = int(p['size'] * (p['lifespan'] / 10))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

        # Obstacles
        for o in self.obstacles:
            pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, (int(o['pos'].x), int(o['pos'].y)), int(o['size']/2))
            pygame.draw.circle(self.screen, self.COLOR_OBSTACLE_ACCENT, (int(o['pos'].x), int(o['pos'].y)), int(o['size']/2), 2)

        # Gems
        for g in self.gems:
            points = [
                (g['pos'].x, g['pos'].y - 8), (g['pos'].x + 6, g['pos'].y),
                (g['pos'].x, g['pos'].y + 8), (g['pos'].x - 6, g['pos'].y)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_GEM)
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_GEM_GLOW)


    def _render_cart(self):
        # Flash when invincible
        if self.invincibility_timer > 0 and (self.steps // 3) % 2 == 0:
            return

        cart_rect = pygame.Rect(
            self.cart_pos.x - self.CART_WIDTH / 2, 
            self.cart_pos.y - self.CART_HEIGHT / 2, 
            self.CART_WIDTH, self.CART_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_CART, cart_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_CART_ACCENT, cart_rect.inflate(-6, -6), border_radius=3)

        # Wheels
        wheel_y = self.cart_pos.y + self.CART_HEIGHT / 2
        pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, (self.cart_pos.x - self.CART_WIDTH/3, wheel_y), 5)
        pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, (self.cart_pos.x + self.CART_WIDTH/3, wheel_y), 5)

    def _render_ui(self):
        # Lives
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, 30 + i * 30, 30, 10, self.COLOR_HEART)
            pygame.gfxdraw.aacircle(self.screen, 30 + i * 30, 30, 10, self.COLOR_HEART)

        # Gems collected
        gem_text = self.font_small.render(f"x {self.gems_collected}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (self.WIDTH - 100, 22))
        points = [
            (self.WIDTH - 120, 30 - 8), (self.WIDTH - 120 + 6, 30),
            (self.WIDTH - 120, 30 + 8), (self.WIDTH - 120 - 6, 30)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_GEM)

        # Progress bar
        progress_ratio = self.steps / self.MAX_STEPS
        bar_width = self.WIDTH - 40
        pygame.draw.rect(self.screen, (50, 50, 70), (20, self.HEIGHT - 30, bar_width, 15), border_radius=4)
        pygame.draw.rect(self.screen, (100, 200, 255), (20, self.HEIGHT - 30, bar_width * progress_ratio, 15), border_radius=4)

        # Game Over / Win Text
        if self.game_over:
            text = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            rendered_text = self.font_large.render(text, True, color)
            text_rect = rendered_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(rendered_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "gems_collected": self.gems_collected,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    # This part runs without a display, just to ensure the class works.
    print("Running headless test...")
    env = GameEnv()
    obs, info = env.reset(seed=42)
    print(f"Initial info: {info}")
    total_reward = 0
    for i in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 500 == 0:
            print(f"Step {i+1}, Info: {info}, Total Reward: {total_reward:.2f}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset(seed=42)
            total_reward = 0
    env.close()
    print("Headless test completed.")