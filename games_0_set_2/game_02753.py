
# Generated: 2025-08-28T05:52:08.488603
# Source Brief: brief_02753.md
# Brief Index: 2753

        
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
    user_guide = "Controls: Use arrow keys (↑↓←→) to apply thrust to your ship."

    # Must be a short, user-facing description of the game:
    game_description = "Pilot a spaceship in a dense asteroid field. Dodge the floating rocks and survive for 60 seconds to win."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 3600  # 60 seconds * 60 FPS

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_ASTEROID_OUTLINE = (180, 190, 200)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 200, 100)

    # Player Physics
    PLAYER_THRUST = 0.15
    PLAYER_MAX_SPEED = 4.0
    PLAYER_FRICTION = 0.02
    PLAYER_RADIUS = 8  # For collision

    # Asteroid Physics
    NUM_ASTEROIDS = 10
    ASTEROID_MIN_RADIUS = 10
    ASTEROID_MAX_RADIUS = 25
    ASTEROID_MAX_SPEED = 2.0
    ASTEROID_MIN_SPEED = 0.5
    ASTEROID_MAX_ROT_SPEED = 0.03 # radians per frame

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.SysFont("Consolas", 24)
        
        # Use numpy's random generator, initialized in reset
        self.np_random = None
        
        # State variables (initialized in reset)
        self.steps = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.asteroids = []
        self.particles = []
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False

        # Initialize player
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)

        # Initialize asteroids
        self.asteroids = []
        while len(self.asteroids) < self.NUM_ASTEROIDS:
            self._spawn_asteroid()

        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If the game is over, do nothing and return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Player ---
        thrust = pygame.math.Vector2(0, 0)
        if movement == 1: thrust.y = -self.PLAYER_THRUST
        elif movement == 2: thrust.y = self.PLAYER_THRUST
        elif movement == 3: thrust.x = -self.PLAYER_THRUST
        elif movement == 4: thrust.x = self.PLAYER_THRUST

        self.player_vel += thrust
        self.player_vel *= (1 - self.PLAYER_FRICTION)

        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        self.player_pos += self.player_vel
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT
        
        if thrust.length() > 0:
            # SFX: Thruster sound
            self._create_particles(self.player_pos, -thrust, count=2)

        # --- Update Asteroids ---
        for a in self.asteroids:
            a["pos"] += a["vel"]
            a["angle"] += a["rot_speed"]
            a["pos"].x %= self.WIDTH
            a["pos"].y %= self.HEIGHT

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

        # --- Collision Detection ---
        collision = self._check_collision()
        
        self.steps += 1
        
        # --- Calculate Reward and Termination ---
        reward = 0.01  # Survival reward
        terminated = False

        if collision:
            self.game_over = True
            terminated = True
            reward = -10
            # SFX: Explosion
            self._create_particles(self.player_pos, count=50, speed_mult=3.0)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = 100  # Victory reward
            # SFX: Victory fanfare
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_collision(self):
        for a in self.asteroids:
            # Check collision with screen wrapping
            for dx in [-self.WIDTH, 0, self.WIDTH]:
                for dy in [-self.HEIGHT, 0, self.HEIGHT]:
                    ast_pos = a["pos"] + pygame.math.Vector2(dx, dy)
                    if self.player_pos.distance_to(ast_pos) < self.PLAYER_RADIUS + a["radius"]:
                        return True
        return False

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
            "score": self.steps * 0.01,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 30))))
            size = max(1, int(3 * (p["lifespan"] / 30)))
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p["pos"].x), int(p["pos"].y)), size)
        
        # Render asteroids with seamless wrapping
        for a in self.asteroids:
            for dx_offset in [-self.WIDTH, 0, self.WIDTH]:
                for dy_offset in [-self.HEIGHT, 0, self.HEIGHT]:
                    draw_pos = a["pos"] + pygame.math.Vector2(dx_offset, dy_offset)
                    if not (-a['radius'] < draw_pos.x < self.WIDTH + a['radius'] and \
                            -a['radius'] < draw_pos.y < self.HEIGHT + a['radius']):
                        continue
                    
                    rotated_shape = []
                    for x, y in a["shape"]:
                        rx = x * math.cos(a["angle"]) - y * math.sin(a["angle"])
                        ry = x * math.sin(a["angle"]) + y * math.cos(a["angle"])
                        rotated_shape.append((int(draw_pos.x + rx), int(draw_pos.y + ry)))
                    
                    if len(rotated_shape) > 2:
                        pygame.gfxdraw.filled_polygon(self.screen, rotated_shape, self.COLOR_ASTEROID)
                        pygame.gfxdraw.aapolygon(self.screen, rotated_shape, self.COLOR_ASTEROID_OUTLINE)

        # Render player if not game over
        if not self.game_over:
            p1 = pygame.math.Vector2(0, -self.PLAYER_RADIUS * 1.5)
            p2 = pygame.math.Vector2(-self.PLAYER_RADIUS, self.PLAYER_RADIUS * 0.8)
            p3 = pygame.math.Vector2(self.PLAYER_RADIUS, self.PLAYER_RADIUS * 0.8)
            
            angle = self.player_vel.angle_to(pygame.math.Vector2(0, -1)) if self.player_vel.length() > 0.1 else 0
            points = [p.rotate(angle) for p in [p1, p2, p3]]
            
            for dx_offset in [-self.WIDTH, 0, self.WIDTH]:
                for dy_offset in [-self.HEIGHT, 0, self.HEIGHT]:
                    draw_pos = self.player_pos + pygame.math.Vector2(dx_offset, dy_offset)
                    screen_points = [(int(draw_pos.x + p.x), int(draw_pos.y + p.y)) for p in points]
                    pygame.gfxdraw.aapolygon(self.screen, screen_points, self.COLOR_PLAYER)
                    pygame.gfxdraw.filled_polygon(self.screen, screen_points, self.COLOR_PLAYER)

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.2f}"
        text_surface = self.font.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            end_text_str = "SURVIVED!" if self.steps >= self.MAX_STEPS else "GAME OVER"
            end_text_surface = self.font.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text_surface, text_rect)

    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ASTEROID_MAX_RADIUS)
            vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(0.1, 1))
        elif edge == 1: # Bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_MAX_RADIUS)
            vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, -0.1))
        elif edge == 2: # Left
            pos = pygame.math.Vector2(-self.ASTEROID_MAX_RADIUS, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.math.Vector2(self.np_random.uniform(0.1, 1), self.np_random.uniform(-1, 1))
        else: # Right
            pos = pygame.math.Vector2(self.WIDTH + self.ASTEROID_MAX_RADIUS, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.math.Vector2(self.np_random.uniform(-1, -0.1), self.np_random.uniform(-1, 1))
        
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        vel.scale_to_length(speed)
        radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
        
        if self.player_pos and pos.distance_to(self.player_pos) < 150:
            return

        self.asteroids.append({
            "pos": pos, "vel": vel, "radius": radius,
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-self.ASTEROID_MAX_ROT_SPEED, self.ASTEROID_MAX_ROT_SPEED),
            "shape": self._generate_asteroid_shape(radius),
        })

    def _generate_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        return [(r * math.cos(a), r * math.sin(a)) for i in range(num_vertices)
                for a in [2 * math.pi * i / num_vertices]
                for r in [radius + self.np_random.uniform(-radius * 0.3, radius * 0.3)]]

    def _create_particles(self, pos, base_vel=None, count=5, speed_mult=1.0):
        for _ in range(count):
            angle_offset = self.np_random.uniform(-30, 30)
            speed_factor = self.np_random.uniform(0.5, 1.5)
            vel = base_vel.rotate(angle_offset) * speed_factor if base_vel else pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            vel *= speed_mult
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifespan": self.np_random.integers(15, 30)})

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset(seed=0)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=0)
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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # --- For Manual Play: Uncomment this block ---
    # pygame.display.set_caption("Asteroid Survival")
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    # running = True
    # while running:
    #     action = [0, 0, 0] # No-op default
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: action[0] = 1
    #     elif keys[pygame.K_DOWN]: action[0] = 2
    #     if keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4
        
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    #         pygame.time.wait(2000)
    #         obs, info = env.reset(seed=random.randint(0, 10000))

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    # env.close()
    
    # --- For Agent Simulation (Default) ---
    print("Running agent simulation...")
    total_reward = 0
    terminated = False
    for step_count in range(1, 10001):
        if terminated:
            print(f"Episode finished after {info['steps']} steps. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step_count % 1000 == 0:
            print(f"Step {step_count}...")
    
    print("Simulation finished.")
    env.close()