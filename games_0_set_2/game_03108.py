
# Generated: 2025-08-27T22:24:22.877937
# Source Brief: brief_03108.md
# Brief Index: 3108

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to pilot your ship. Survive for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship through a dense asteroid field. Dodge incoming rocks to survive as long as possible. The asteroid shower intensifies over time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (255, 255, 255)
        self.COLOR_SHIP_GLOW = (180, 220, 255)
        self.COLOR_ASTEROID = (120, 120, 130)
        self.COLOR_ASTEROID_SHADOW = (80, 80, 90)
        self.COLOR_THRUSTER = (255, 180, 80)
        self.COLOR_EXPLOSION = (255, 80, 50)
        self.COLOR_UI = (220, 220, 220)

        # Ship properties
        self.SHIP_SIZE = 12
        self.SHIP_SPEED = 6.0
        
        # Asteroid properties
        self.INITIAL_ASTEROID_SPEED = 2.5
        self.ASTEROID_SPEED_INCREASE_INTERVAL = 10 * self.FPS # Every 10 seconds
        self.ASTEROID_SPEED_INCREMENT = 0.2
        self.ASTEROID_SPAWN_PROB = 0.08
        self.MAX_ASTEROIDS = 25
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.ship_pos = None
        self.asteroids = None
        self.particles = None
        self.current_asteroid_speed = self.INITIAL_ASTEROID_SPEED
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.ship_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float64)
        
        self.asteroids = []
        for _ in range(10): # Start with a few asteroids
            self._spawn_asteroid(initial_spawn=True)

        self.particles = []
        self.current_asteroid_speed = self.INITIAL_ASTEROID_SPEED
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        if not self.game_over:
            self.steps += 1
            reward = 0.01  # Reward for surviving one more step

            # --- Update Game Logic ---
            self._handle_input(action)
            self._update_asteroids()
            self._update_particles()
            self._spawn_new_asteroids()
            self._check_collisions()
            self._update_difficulty()
            self._check_termination_conditions()

            if self.victory:
                reward += 100.0 # Large reward for winning

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        # Movement
        if movement == 1:  # Up
            self.ship_pos[1] -= self.SHIP_SPEED
        elif movement == 2:  # Down
            self.ship_pos[1] += self.SHIP_SPEED
        elif movement == 3:  # Left
            self.ship_pos[0] -= self.SHIP_SPEED
        elif movement == 4:  # Right
            self.ship_pos[0] += self.SHIP_SPEED

        # Clamp ship position to screen bounds
        self.ship_pos[0] = np.clip(self.ship_pos[0], self.SHIP_SIZE, self.WIDTH - self.SHIP_SIZE)
        self.ship_pos[1] = np.clip(self.ship_pos[1], self.SHIP_SIZE, self.HEIGHT - self.SHIP_SIZE)
        
        # Spawn thruster particles
        if movement != 0:
            # SFX: Ship thruster sound
            self._spawn_thruster_particles(movement)

    def _update_asteroids(self):
        asteroids_to_keep = []
        for asteroid in self.asteroids:
            asteroid["pos"][1] += asteroid["speed"]
            if asteroid["pos"][1] - asteroid["size"] < self.HEIGHT:
                asteroids_to_keep.append(asteroid)
        self.asteroids = asteroids_to_keep

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _spawn_new_asteroids(self):
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.np_random.random() < self.ASTEROID_SPAWN_PROB:
            self._spawn_asteroid()

    def _spawn_asteroid(self, initial_spawn=False):
        size = self.np_random.integers(8, 25)
        pos = np.array([
            self.np_random.uniform(size, self.WIDTH - size),
            self.np_random.uniform(-self.HEIGHT, -size) if initial_spawn else -float(size)
        ], dtype=np.float64)
        
        speed = self.current_asteroid_speed * self.np_random.uniform(0.8, 1.2)
        
        self.asteroids.append({"pos": pos, "size": size, "speed": speed})
        
    def _spawn_thruster_particles(self, movement_dir):
        # Spawn 1-2 particles per frame
        for _ in range(self.np_random.integers(1, 3)):
            if movement_dir == 1: # Up
                vel = np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(2, 4)])
            elif movement_dir == 2: # Down
                 vel = np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-4, -2)])
            elif movement_dir == 3: # Left
                 vel = np.array([self.np_random.uniform(2, 4), self.np_random.uniform(-0.5, 0.5)])
            elif movement_dir == 4: # Right
                 vel = np.array([self.np_random.uniform(-4, -2), self.np_random.uniform(-0.5, 0.5)])
            else:
                return

            self.particles.append({
                "pos": self.ship_pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(8, 15),
                "color": self.COLOR_THRUSTER,
                "type": "thruster"
            })

    def _spawn_explosion(self):
        # SFX: Explosion sound
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": self.ship_pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(20, 40),
                "color": self.COLOR_EXPLOSION,
                "type": "explosion"
            })

    def _check_collisions(self):
        ship_center = self.ship_pos
        for asteroid in self.asteroids:
            dist = np.linalg.norm(ship_center - asteroid["pos"])
            if dist < self.SHIP_SIZE * 0.8 + asteroid["size"]: # Using 80% of ship size for more forgiving hitbox
                self.game_over = True
                self._spawn_explosion()
                break

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
            self.current_asteroid_speed += self.ASTEROID_SPEED_INCREMENT

    def _check_termination_conditions(self):
        if not self.game_over and self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.victory = True

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_particles()
        self._render_asteroids()
        if not self.game_over or self.victory:
             self._render_ship()

    def _render_ship(self):
        x, y = int(self.ship_pos[0]), int(self.ship_pos[1])
        s = self.SHIP_SIZE
        
        points = [
            (x, y - s),
            (x - s // 2, y + s // 2),
            (x + s // 2, y + s // 2)
        ]

        # Glow effect
        glow_s = int(s * 2.5)
        glow_points = [
            (x, y - glow_s),
            (x - glow_s // 2, y + glow_s // 2),
            (x + glow_s // 2, y + glow_s // 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, (*self.COLOR_SHIP_GLOW, 50))
        pygame.gfxdraw.aapolygon(self.screen, glow_points, (*self.COLOR_SHIP_GLOW, 50))

        # Main ship body
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SHIP)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            x, y = int(asteroid["pos"][0]), int(asteroid["pos"][1])
            size = int(asteroid["size"])
            
            # Shadow for 3D effect
            shadow_offset = int(size * 0.1)
            pygame.gfxdraw.filled_circle(self.screen, x + shadow_offset, y + shadow_offset, size, self.COLOR_ASTEROID_SHADOW)
            pygame.gfxdraw.aacircle(self.screen, x + shadow_offset, y + shadow_offset, size, self.COLOR_ASTEROID_SHADOW)
            
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            x, y = int(p["pos"][0]), int(p["pos"][1])
            
            # Fade color based on lifespan
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 20))))
            color = (*p["color"], alpha)

            if p["type"] == "thruster":
                size = max(1, int(p["lifespan"] * 0.3))
                pygame.draw.circle(self.screen, color, (x, y), size)
            elif p["type"] == "explosion":
                size = max(1, int(p["lifespan"] * 0.2))
                pygame.draw.circle(self.screen, color, (x, y), size)


    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        text_surface = self.font_ui.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 10))

        # Game Over / Victory Message
        if self.game_over:
            if self.victory:
                msg = "SURVIVAL COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_surface = self.font_game_over.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        self.score = self.steps / self.FPS
        return {
            "score": self.score,
            "steps": self.steps,
            "victory": self.victory,
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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a window for manual play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Dodger")
    clock = pygame.time.Clock()

    while not terminated:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Press 'R' to reset

        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()