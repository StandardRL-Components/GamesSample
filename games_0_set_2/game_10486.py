import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:33:13.655660
# Source Brief: brief_00486.md
# Brief Index: 486
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Maneuver a growing crystal through a field of obstacles.
    The crystal's growth rate is tied to its momentum. Colliding with
    obstacles reduces momentum and can shatter the crystal if it's too small.
    The goal is to reach a target size.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Maneuver a crystal through an obstacle field. Gain momentum to grow larger, "
        "but be careful—collisions can shatter a small crystal."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move the crystal. Maintain high speed to grow."
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (20, 40, 60)
    COLOR_CRYSTAL = (0, 150, 255)
    COLOR_CRYSTAL_GLOW = (0, 100, 200)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (200, 40, 40)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PARTICLE_COLLISION = (255, 100, 80)
    COLOR_PARTICLE_TRAIL = (50, 150, 255)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STEPS = 2000
    TARGET_CRYSTAL_SIZE = 50
    CRYSTAL_BREAK_THRESHOLD = TARGET_CRYSTAL_SIZE * 0.2
    NUM_OBSTACLES = 5
    OBSTACLE_SIZE = 40
    CRYSTAL_INITIAL_SIZE = 10
    
    # Physics
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.96
    MAX_SPEED = 6.0
    BASE_GROWTH_RATE = 0.02
    MOMENTUM_GROWTH_MULTIPLIER = 0.05
    COLLISION_MOMENTUM_LOSS = 0.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- State Variables ---
        self.crystal = None
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.screen_shake = 0
        
        # Initialize state variables
        # A seed is required for the first reset call.
        self.reset(seed=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.screen_shake = 0

        # Create crystal
        self.crystal = self._Crystal(
            x=self.SCREEN_WIDTH / 2,
            y=self.SCREEN_HEIGHT / 2,
            size=self.CRYSTAL_INITIAL_SIZE
        )

        # Create obstacles
        self.obstacles = []
        spawn_safety_radius = 150
        while len(self.obstacles) < self.NUM_OBSTACLES:
            new_obstacle = pygame.Rect(
                self.np_random.integers(0, self.SCREEN_WIDTH - self.OBSTACLE_SIZE),
                self.np_random.integers(0, self.SCREEN_HEIGHT - self.OBSTACLE_SIZE),
                self.OBSTACLE_SIZE,
                self.OBSTACLE_SIZE
            )
            # Ensure it's not too close to the start
            dist_to_start = math.hypot(new_obstacle.centerx - self.crystal.x, new_obstacle.centery - self.crystal.y)
            if dist_to_start < spawn_safety_radius:
                continue
            # Ensure it doesn't overlap with other obstacles
            if any(new_obstacle.colliderect(obs) for obs in self.obstacles):
                continue
            self.obstacles.append(new_obstacle)

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        # --- Update Game Logic ---
        self.steps += 1
        reward = 0.0
        
        prev_momentum = self.crystal.get_momentum()

        # Update crystal
        self.crystal.update(movement)
        self.crystal.apply_friction()
        self.crystal.move()
        self.crystal.handle_boundaries(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        
        # Handle growth
        momentum = self.crystal.get_momentum()
        growth = self.BASE_GROWTH_RATE + (momentum / self.MAX_SPEED) * self.MOMENTUM_GROWTH_MULTIPLIER
        self.crystal.size += growth

        # Update score for UI
        self.score = self.crystal.size

        # Reward for surviving and momentum gain
        reward += 0.1 # Survival reward
        if momentum > prev_momentum:
            reward += 1.0

        # Handle collisions
        collided = False
        for obstacle in self.obstacles:
            if self.crystal.get_bounding_box().colliderect(obstacle):
                collided = True
                # Sound: Collision_Impact.wav
                self._create_particles(self.crystal.x, self.crystal.y, 20, self.COLOR_PARTICLE_COLLISION, 5, 20)
                self.screen_shake = 10
                
                # Penalties
                reward -= 5.0 # Collision event penalty
                
                # Physics response
                self.crystal.vx *= -self.COLLISION_MOMENTUM_LOSS
                self.crystal.vy *= -self.COLLISION_MOMENTUM_LOSS
                
                # Check for breakage
                if self.crystal.size < self.CRYSTAL_BREAK_THRESHOLD:
                    self.game_over = True
                    reward = -100.0 # Breaking penalty
                    # Sound: Crystal_Shatter.wav
                    break
        
        if collided:
             reward -= 0.5 # Continuous collision penalty

        # Update particles
        self._update_particles()
        if self.crystal.get_momentum() > 0.5:
             self._create_particles(self.crystal.x, self.crystal.y, 1, self.COLOR_PARTICLE_TRAIL, 1, 10, trail=True)

        # --- Check Termination Conditions ---
        terminated = False
        if self.game_over:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        elif self.crystal.size >= self.TARGET_CRYSTAL_SIZE:
            # Sound: Victory_Chime.wav
            terminated = True
            self.game_over = True
            reward = 100.0 # Victory reward
            self._create_particles(self.crystal.x, self.crystal.y, 50, self.COLOR_CRYSTAL, 8, 60)

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset_x = self.np_random.integers(-4, 5)
            render_offset_y = self.np_random.integers(-4, 5)

        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid(render_offset_x, render_offset_y)
        self._render_particles(render_offset_x, render_offset_y)
        self._render_obstacles(render_offset_x, render_offset_y)
        self._render_crystal(render_offset_x, render_offset_y)
        
        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "momentum": self.crystal.get_momentum() if self.crystal else 0,
            "crystal_size": self.crystal.size if self.crystal else 0,
        }

    # --- Rendering Helpers ---
    def _render_grid(self, ox, oy):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i + ox, oy), (i + ox, self.SCREEN_HEIGHT + oy))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (ox, i + oy), (self.SCREEN_WIDTH + ox, i + oy))

    def _render_obstacles(self, ox, oy):
        for obs in self.obstacles:
            glow_rect = obs.inflate(10, 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_OBSTACLE_GLOW, 100), glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, (glow_rect.x + ox, glow_rect.y + oy))
            
            r = obs.move(ox, oy)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, r, border_radius=3)

    def _render_crystal(self, ox, oy):
        if not self.crystal: return
        
        x, y, size = self.crystal.x + ox, self.crystal.y + oy, self.crystal.size
        
        # Glow effect
        self._draw_glowing_hexagon(self.screen, x, y, size * 1.5, self.COLOR_CRYSTAL_GLOW, 128)
        self._draw_glowing_hexagon(self.screen, x, y, size * 1.2, self.COLOR_CRYSTAL_GLOW, 128)
        
        # Main crystal
        self._draw_hexagon(self.screen, self.COLOR_CRYSTAL, (x, y), size)
        # Inner highlight
        self._draw_hexagon(self.screen, (200, 220, 255), (x, y), size * 0.7)

    def _render_particles(self, ox, oy):
        for p in self.particles:
            p.draw(self.screen, ox, oy)

    def _render_ui(self):
        # Crystal Size
        size_text = self.font_ui.render(f"SIZE: {self.score:.1f}/{self.TARGET_CRYSTAL_SIZE}", True, self.COLOR_TEXT)
        self.screen.blit(size_text, (10, 10))

        # Momentum
        momentum_percent = (self.crystal.get_momentum() / self.MAX_SPEED) * 100
        momentum_text = self.font_ui.render(f"MOMENTUM: {momentum_percent:.0f}%", True, self.COLOR_TEXT)
        self.screen.blit(momentum_text, (self.SCREEN_WIDTH - momentum_text.get_width() - 10, 10))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))

        msg = "TARGET REACHED" if self.crystal.size >= self.TARGET_CRYSTAL_SIZE else "CRYSTAL SHATTERED"
        color = (100, 255, 100) if self.crystal.size >= self.TARGET_CRYSTAL_SIZE else (255, 100, 100)
        
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    # --- Particle System ---
    def _create_particles(self, x, y, count, color, speed_max, lifetime, trail=False):
        for _ in range(count):
            if trail:
                angle = math.atan2(self.crystal.vy, self.crystal.vx) + math.pi + self.np_random.uniform(-0.5, 0.5)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
            
            speed = self.np_random.uniform(1, speed_max)
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            self.particles.append(self._Particle(x, y, vel_x, vel_y, color, lifetime, self.np_random))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    # --- Drawing Helpers ---
    @staticmethod
    def _draw_hexagon(surface, color, center, size):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6 # Rotated for point-up
            x = center[0] + size * math.cos(angle)
            y = center[1] + size * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    @staticmethod
    def _draw_glowing_hexagon(surface, x, y, size, color, alpha):
        temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            px = size + size * math.cos(angle)
            py = size + size * math.sin(angle)
            points.append((int(px), int(py)))
        
        final_color = (*color, alpha)
        pygame.gfxdraw.filled_polygon(temp_surf, points, final_color)
        pygame.gfxdraw.aapolygon(temp_surf, points, final_color)
        surface.blit(temp_surf, (int(x - size), int(y - size)), special_flags=pygame.BLEND_RGBA_ADD)

    # --- Inner Classes ---
    class _Crystal:
        def __init__(self, x, y, size):
            self.x, self.y = x, y
            self.vx, self.vy = 0.0, 0.0
            self.size = size

        def update(self, movement):
            # 0=none, 1=up, 2=down, 3=left, 4=right
            if movement == 1: self.vy -= GameEnv.PLAYER_ACCELERATION
            if movement == 2: self.vy += GameEnv.PLAYER_ACCELERATION
            if movement == 3: self.vx -= GameEnv.PLAYER_ACCELERATION
            if movement == 4: self.vx += GameEnv.PLAYER_ACCELERATION
            
            speed = self.get_momentum()
            if speed > GameEnv.MAX_SPEED:
                self.vx = (self.vx / speed) * GameEnv.MAX_SPEED
                self.vy = (self.vy / speed) * GameEnv.MAX_SPEED

        def apply_friction(self):
            self.vx *= GameEnv.PLAYER_FRICTION
            self.vy *= GameEnv.PLAYER_FRICTION

        def move(self):
            self.x += self.vx
            self.y += self.vy

        def get_momentum(self):
            return math.hypot(self.vx, self.vy)

        def handle_boundaries(self, width, height):
            if self.x - self.size < 0:
                self.x = self.size
                self.vx *= -0.5
            if self.x + self.size > width:
                self.x = width - self.size
                self.vx *= -0.5
            if self.y - self.size < 0:
                self.y = self.size
                self.vy *= -0.5
            if self.y + self.size > height:
                self.y = height - self.size
                self.vy *= -0.5
        
        def get_bounding_box(self):
            return pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)

    class _Particle:
        def __init__(self, x, y, vx, vy, color, lifetime, np_random):
            self.x, self.y = x, y
            self.vx, self.vy = vx, vy
            self.color = color
            self.lifetime = lifetime
            self.max_lifetime = lifetime
            self.np_random = np_random

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.vx *= 0.95
            self.vy *= 0.95
            self.lifetime -= 1
            return self.lifetime > 0

        def draw(self, surface, ox, oy):
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            radius = int(5 * (self.lifetime / self.max_lifetime))
            if radius < 1: return
            
            color = (*self.color, alpha)
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            surface.blit(temp_surf, (int(self.x - radius + ox), int(self.y - radius + oy)), special_flags=pygame.BLEND_RGBA_ADD)


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run in headless mode. Unset SDL_VIDEODRIVER to play.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # Pygame setup for human play
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Crystal Growth Environment")
        clock = pygame.time.Clock()
        
        total_reward = 0
        
        while not done:
            # Action mapping for human input
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Rendering
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")

            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Size: {info['crystal_size']:.2f}")

            clock.tick(30) # Run at 30 FPS

        pygame.quit()
        print("Game Over!")