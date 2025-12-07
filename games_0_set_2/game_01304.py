
# Generated: 2025-08-27T16:42:26.913354
# Source Brief: brief_01304.md
# Brief Index: 1304

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to pilot your ship and dodge the asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a ship through a procedurally generated asteroid field. Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_TIME = 60.0  # seconds
    MAX_STEPS = int(MAX_TIME * FPS)

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 50)
    COLOR_ASTEROID = (120, 120, 130)
    COLOR_ASTEROID_OUTLINE = (160, 160, 170)
    COLOR_EXPLOSION = (255, 180, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_STARS = [(200, 200, 200), (150, 150, 150), (100, 100, 100)]

    # Game parameters
    PLAYER_SPEED = 5.0
    PLAYER_RADIUS = 10
    INITIAL_ASTEROID_SPAWN_INTERVAL = 2.0  # seconds
    FINAL_ASTEROID_SPAWN_INTERVAL = 1.0 # at 30 seconds
    
    # Data structures
    Asteroid = namedtuple("Asteroid", ["pos", "velocity", "radius", "angle", "angular_velocity", "shape_points"])
    Particle = namedtuple("Particle", ["pos", "velocity", "radius", "lifetime", "color"])


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # These will be initialized in reset()
        self.player_pos = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_timer = 0.0
        self.next_asteroid_spawn_time = 0.0
        self.np_random = None

        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.asteroids = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_timer = 0.0
        self.next_asteroid_spawn_time = 0.0
        
        # Generate a static starfield for the background
        self.stars = []
        for _ in range(150):
            pos = (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
            size = self.np_random.integers(1, 3)
            color = random.choice(self.COLOR_STARS)
            self.stars.append((pos, size, color))

        # Spawn initial asteroids away from the player
        for _ in range(3):
            self._spawn_asteroid(avoid_pos=self.player_pos, min_dist=150)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, no more actions can be taken.
            # We simply return the final state.
            reward = 0.0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update game logic ---
        self.steps += 1
        self.game_timer += 1 / self.FPS
        
        self._handle_player_movement(movement)
        self._update_asteroids()
        self._update_particles()
        self._spawn_new_asteroids()
        
        # --- Check for termination and calculate reward ---
        reward = 0.1  # Reward for surviving one step
        terminated = False

        if self._check_player_collision():
            self.game_over = True
            terminated = True
            reward = -10.0
            self._create_explosion(self.player_pos)
            # sfx: player_explosion.wav
        elif self.game_timer >= self.MAX_TIME:
            self.game_over = True
            terminated = True
            reward = 100.0
            # sfx: victory_fanfare.wav

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Screen wrap-around
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT

    def _update_asteroids(self):
        for i, asteroid in enumerate(self.asteroids):
            new_pos = asteroid.pos + asteroid.velocity
            new_pos[0] %= self.SCREEN_WIDTH
            new_pos[1] %= self.SCREEN_HEIGHT
            new_angle = (asteroid.angle + asteroid.angular_velocity) % (2 * math.pi)
            self.asteroids[i] = asteroid._replace(pos=new_pos, angle=new_angle)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for i, p in enumerate(self.particles):
            new_pos = p.pos + p.velocity
            new_lifetime = p.lifetime - 1
            new_radius = max(0, p.radius * (new_lifetime / (1.0 * self.FPS)))
            self.particles[i] = p._replace(pos=new_pos, lifetime=new_lifetime, radius=new_radius)

    def _get_current_spawn_interval(self):
        # Linearly decrease spawn interval from initial to final over 30 seconds
        progress = min(1.0, self.game_timer / 30.0)
        interval = self.INITIAL_ASTEROID_SPAWN_INTERVAL + progress * (self.FINAL_ASTEROID_SPAWN_INTERVAL - self.INITIAL_ASTEROID_SPAWN_INTERVAL)
        return interval

    def _spawn_new_asteroids(self):
        if self.game_timer >= self.next_asteroid_spawn_time:
            self._spawn_asteroid()
            self.next_asteroid_spawn_time = self.game_timer + self._get_current_spawn_interval()
            # sfx: asteroid_spawn.wav

    def _spawn_asteroid(self, avoid_pos=None, min_dist=0):
        while True:
            edge = self.np_random.integers(4)
            if edge == 0:  # Top
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -30.0], dtype=np.float32)
            elif edge == 1:  # Bottom
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30.0], dtype=np.float32)
            elif edge == 2:  # Left
                pos = np.array([-30.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float32)
            else:  # Right
                pos = np.array([self.SCREEN_WIDTH + 30.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float32)

            if avoid_pos is None or np.linalg.norm(pos - avoid_pos) > min_dist:
                break
        
        target = np.array([
            self.np_random.uniform(self.SCREEN_WIDTH * 0.2, self.SCREEN_WIDTH * 0.8),
            self.np_random.uniform(self.SCREEN_HEIGHT * 0.2, self.SCREEN_HEIGHT * 0.8)
        ], dtype=np.float32)
        
        direction = (target - pos) / np.linalg.norm(target - pos)
        speed = self.np_random.uniform(1.0, 2.5)
        velocity = direction * speed
        
        radius = self.np_random.uniform(15, 35)
        angular_velocity = self.np_random.uniform(-0.05, 0.05)
        
        # Generate irregular polygon shape
        num_vertices = self.np_random.integers(6, 10)
        shape_points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist_factor = self.np_random.uniform(0.7, 1.1)
            point = (math.cos(angle) * radius * dist_factor, math.sin(angle) * radius * dist_factor)
            shape_points.append(point)

        self.asteroids.append(self.Asteroid(pos, velocity, radius, 0, angular_velocity, shape_points))

    def _check_player_collision(self):
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid.pos)
            if dist < self.PLAYER_RADIUS + asteroid.radius * 0.8: # Use 80% of radius for more forgiving hitbox
                return True
        return False

    def _create_explosion(self, pos):
        num_particles = 50
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            radius = self.np_random.uniform(2, 5)
            lifetime = int(self.np_random.uniform(0.5, 1.0) * self.FPS)
            self.particles.append(self.Particle(pos.copy(), velocity, radius, lifetime, self.COLOR_EXPLOSION))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw stars
        for pos, size, color in self.stars:
            pygame.draw.circle(self.screen, color, pos, size)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifetime / (1.0 * self.FPS)))
            color = (*p.color, alpha)
            if p.radius > 0:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((p.radius*2, p.radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p.radius, p.radius), p.radius)
                self.screen.blit(temp_surf, (int(p.pos[0] - p.radius), int(p.pos[1] - p.radius)))

        # Draw asteroids
        for a in self.asteroids:
            # Calculate rotated points
            points = []
            for p in a.shape_points:
                x_rot = p[0] * math.cos(a.angle) - p[1] * math.sin(a.angle)
                y_rot = p[0] * math.sin(a.angle) + p[1] * math.cos(a.angle)
                points.append((int(a.pos[0] + x_rot), int(a.pos[1] + y_rot)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Draw player ship if not game over
        if not self.game_over:
            # Player ship is a triangle
            p1 = (self.player_pos[0], self.player_pos[1] - self.PLAYER_RADIUS)
            p2 = (self.player_pos[0] - self.PLAYER_RADIUS * 0.7, self.player_pos[1] + self.PLAYER_RADIUS * 0.7)
            p3 = (self.player_pos[0] + self.PLAYER_RADIUS * 0.7, self.player_pos[1] + self.PLAYER_RADIUS * 0.7)
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            
            # Draw glow effect
            glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2), self.PLAYER_RADIUS * 1.5)
            self.screen.blit(glow_surf, (int(self.player_pos[0] - self.PLAYER_RADIUS*2), int(self.player_pos[1] - self.PLAYER_RADIUS*2)))

            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        time_left = max(0, self.MAX_TIME - self.game_timer)
        timer_text = f"TIME: {time_left:.1f}"
        text_surface = self.font.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            if self.game_timer >= self.MAX_TIME:
                end_text = "SURVIVED!"
            else:
                end_text = "GAME OVER"
            
            end_surface = self.font.render(end_text, True, self.COLOR_UI_TEXT)
            text_rect = end_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_survived": self.game_timer,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Asteroid Dodger")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        # --- Action mapping for human play ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are unused

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over. Score: {info['score']:.2f}, Time: {info['time_survived']:.2f}s. Press 'R' to restart.")
        
        # --- Render to screen ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()