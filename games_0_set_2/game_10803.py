import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:05:20.510432
# Source Brief: brief_00803.md
# Brief Index: 803
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Asteroid:
    def __init__(self, pos, vel, radius):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = int(radius)
        self.mass = math.pi * self.radius**2
        self.is_captured = False

class Particle:
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.color = np.array(color, dtype=float)
        self.lifetime = lifetime
        self.max_lifetime = lifetime

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control the strength of gravity wells to guide asteroids into collisions for points, "
        "while preventing them from escaping into deep space."
    )
    user_guide = (
        "Use ↑↓ arrow keys to adjust the power of the left gravity well and ←→ for the middle one. "
        "The right well's power balances automatically."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Game rules
        self.WIN_SCORE = 500
        self.MAX_ESCAPED = 10
        
        # Physics
        self.GRAVITY_CONSTANT = 7000
        self.STRENGTH_INCREMENT = 0.05
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_WELL_1 = (50, 150, 255)
        self.COLOR_WELL_2 = (50, 255, 150)
        self.COLOR_WELL_3 = (255, 100, 100)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_UI_BAR_BG = (50, 50, 70)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 30)
            self.font_medium = pygame.font.SysFont("Consolas", 20)
            self.font_small = pygame.font.SysFont("Consolas", 14)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 20)

        # --- Game State Initialization ---
        self.well_positions = [
            np.array([self.WIDTH * 0.2, self.HEIGHT / 2]),
            np.array([self.WIDTH * 0.5, self.HEIGHT / 2]),
            np.array([self.WIDTH * 0.8, self.HEIGHT / 2]),
        ]
        self.well_colors = [self.COLOR_WELL_1, self.COLOR_WELL_2, self.COLOR_WELL_3]
        self.well_capture_radii = [60, 80, 100] # Visual radius for capture

        # These will be reset in self.reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.escaped_count = 0
        self.well_strengths = []
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.spawn_timer = 0.0
        self.base_spawn_rate = 0.5
        self.current_spawn_rate = 0.5
        
        # self.reset() # No need to call reset in init
        # self.validate_implementation() # Not needed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.escaped_count = 0
        
        self.well_strengths = np.array([0.33, 0.34, 0.33])
        
        self.asteroids.clear()
        self.particles.clear()

        self.spawn_timer = 0.0
        self.current_spawn_rate = self.base_spawn_rate

        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5),
                self.np_random.uniform(50, 100)
            )
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        step_reward = 0.0
        self.steps += 1
        
        # 1. Handle Action
        self._handle_action(action)
        
        # 2. Update Game Logic
        step_reward += self._update_spawner()
        step_reward += self._update_asteroids()
        step_reward += self._handle_collisions()
        self._update_particles()
        
        # 3. Check for Termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                step_reward += 100  # Win reward
            elif self.escaped_count >= self.MAX_ESCAPED:
                step_reward -= 100  # Loss penalty
            elif truncated:
                step_reward -= 10 # Timeout penalty

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        
        strengths = self.well_strengths.copy()
        
        # up/down for well 1
        if movement == 1: strengths[0] += self.STRENGTH_INCREMENT
        elif movement == 2: strengths[0] -= self.STRENGTH_INCREMENT
        
        # left/right for well 2
        if movement == 4: strengths[1] += self.STRENGTH_INCREMENT
        elif movement == 3: strengths[1] -= self.STRENGTH_INCREMENT
            
        # Clamp individual strengths
        strengths[0] = np.clip(strengths[0], 0, 1)
        strengths[1] = np.clip(strengths[1], 0, 1)

        # Normalize if sum > 1 to maintain total power
        if strengths[0] + strengths[1] > 1.0:
            total = strengths[0] + strengths[1]
            strengths[0] /= total
            strengths[1] /= total
            
        # Well 3 gets the remainder
        strengths[2] = 1.0 - strengths[0] - strengths[1]
        
        self.well_strengths = strengths

    def _update_spawner(self):
        # Difficulty scaling
        self.current_spawn_rate = self.base_spawn_rate + 0.01 * (self.steps / self.FPS)
        
        self.spawn_timer += self.current_spawn_rate / self.FPS
        if self.spawn_timer >= 1.0:
            self.spawn_timer -= 1.0
            
            # Spawn on left or right edge
            edge = self.np_random.choice([0, 1])
            x = -20 if edge == 0 else self.WIDTH + 20
            y = self.np_random.uniform(0, self.HEIGHT)
            
            # Velocity towards the center
            angle_to_center = math.atan2(self.HEIGHT/2 - y, self.WIDTH/2 - x)
            angle = self.np_random.uniform(angle_to_center - 0.3, angle_to_center + 0.3)
            speed = self.np_random.uniform(50, 100)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            
            # Larger asteroids are rarer
            size_roll = self.np_random.random()
            if size_roll > 0.95: radius = self.np_random.uniform(12, 15) # Large
            elif size_roll > 0.8: radius = self.np_random.uniform(8, 11) # Medium
            else: radius = self.np_random.uniform(4, 7) # Small
            
            self.asteroids.append(Asteroid([x, y], vel, radius))
        return 0.0

    def _update_asteroids(self):
        reward = 0.0
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            # Apply gravity from wells
            total_accel = np.zeros(2, dtype=float)
            for j, well_pos in enumerate(self.well_positions):
                vec_to_well = well_pos - asteroid.pos
                dist_sq = np.dot(vec_to_well, vec_to_well)
                if dist_sq < 1: dist_sq = 1 # Avoid division by zero
                
                force_mag = self.GRAVITY_CONSTANT * self.well_strengths[j] / dist_sq
                accel = (vec_to_well / math.sqrt(dist_sq)) * force_mag
                total_accel += accel
            
            # Update velocity and position
            asteroid.vel += total_accel / self.FPS
            asteroid.pos += asteroid.vel / self.FPS
            
            # Check for capture reward
            if not asteroid.is_captured:
                for j, well_pos in enumerate(self.well_positions):
                    dist_to_well = np.linalg.norm(asteroid.pos - well_pos)
                    if dist_to_well < self.well_capture_radii[j]:
                        asteroid.is_captured = True
                        reward += 0.1
                        # sound: low-pitched hum
                        break
            
            # Check for escape
            if not ((-30 < asteroid.pos[0] < self.WIDTH + 30) and \
                    (-30 < asteroid.pos[1] < self.HEIGHT + 30)):
                asteroids_to_remove.append(i)
                self.escaped_count += 1

        # Remove escaped asteroids
        for i in sorted(asteroids_to_remove, reverse=True):
            del self.asteroids[i]
            
        return reward

    def _handle_collisions(self):
        reward = 0.0
        collided_pairs = set()

        for i in range(len(self.asteroids)):
            for j in range(i + 1, len(self.asteroids)):
                if (i,j) in collided_pairs: continue

                a1 = self.asteroids[i]
                a2 = self.asteroids[j]
                
                vec = a1.pos - a2.pos
                dist = np.linalg.norm(vec)
                
                if dist < a1.radius + a2.radius:
                    collided_pairs.add((i,j))
                    # Collision response
                    normal = vec / dist
                    tangent = np.array([-normal[1], normal[0]])
                    
                    v1n = np.dot(a1.vel, normal)
                    v1t = np.dot(a1.vel, tangent)
                    v2n = np.dot(a2.vel, normal)
                    v2t = np.dot(a2.vel, tangent)
                    
                    m1, m2 = a1.mass, a2.mass
                    v1n_new = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
                    v2n_new = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)
                    
                    a1.vel = v1n_new * normal + v1t * tangent
                    a2.vel = v2n_new * normal + v2t * tangent
                    
                    # Prevent sticking
                    overlap = a1.radius + a2.radius - dist
                    a1.pos += normal * overlap * 0.5
                    a2.pos -= normal * overlap * 0.5
                    
                    # Add score and reward
                    size_multiplier = (a1.radius + a2.radius) / 10.0
                    score_gain = int(10 * size_multiplier)
                    self.score += score_gain
                    reward += 1.0 * size_multiplier
                    
                    # Create particles
                    self._create_collision_particles( (a1.pos + a2.pos) / 2, score_gain)
                    # sound: sharp crack or pop
        return reward

    def _create_collision_particles(self, pos, num_particles):
        for _ in range(min(30, num_particles)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(20, 100)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            radius = self.np_random.uniform(1, 3)
            color = self.np_random.choice([
                (255, 255, 100), (255, 200, 50), (255, 150, 0)
            ])
            lifetime = self.np_random.uniform(0.3, 0.8)
            self.particles.append(Particle(pos.copy(), vel, radius, color, lifetime))

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p.pos += p.vel / self.FPS
            p.vel *= 0.95 # Damping
            p.lifetime -= 1 / self.FPS
            if p.lifetime <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or
            self.escaped_count >= self.MAX_ESCAPED
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size, alpha in self.stars:
            color = (alpha, alpha, alpha * 1.1)
            pygame.draw.circle(self.screen, color, (x, y), size, 0)

    def _render_game(self):
        # Render Wells with glow
        for i, pos in enumerate(self.well_positions):
            strength = self.well_strengths[i]
            base_radius = self.well_capture_radii[i]
            color = self.well_colors[i]
            
            for j in range(10, 0, -1):
                alpha = 10 + (strength * 100) * (j / 10)**2
                radius = base_radius * (1 + strength * 0.5) * (j / 10)
                glow_color = (color[0], color[1], color[2], max(0, min(255, alpha)))
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), glow_color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(base_radius), color)

        # Render Asteroids
        for asteroid in self.asteroids:
            brightness = 180 + asteroid.radius * 5
            color = (min(255, brightness), min(255, brightness), min(255, brightness + 10))
            pos_int = (int(asteroid.pos[0]), int(asteroid.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], asteroid.radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], asteroid.radius, (100,100,100))

        # Render Particles
        for p in self.particles:
            alpha = (p.lifetime / p.max_lifetime)
            radius = p.radius * alpha
            color = p.color * alpha
            pos_int = (int(p.pos[0]), int(p.pos[1]))
            pygame.draw.circle(self.screen, color, pos_int, max(1, int(radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 5))
        
        # Escaped Count
        escaped_color = (255, 80, 80) if self.escaped_count > 5 else self.COLOR_TEXT
        escaped_text = self.font_medium.render(f"ESCAPED: {self.escaped_count} / {self.MAX_ESCAPED}", True, escaped_color)
        self.screen.blit(escaped_text, (self.WIDTH/2 - escaped_text.get_width()/2, self.HEIGHT - 30))

        # Well Strength Bars
        bar_width, bar_height = 100, 15
        for i in range(3):
            pos_x = self.well_positions[i][0] - bar_width/2
            pos_y = self.HEIGHT - 60
            
            strength = self.well_strengths[i]
            fill_width = bar_width * strength
            
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (pos_x, pos_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.well_colors[i], (pos_x, pos_y, fill_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (pos_x, pos_y, bar_width, bar_height), 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "escaped_count": self.escaped_count,
            "well_strengths": self.well_strengths.tolist(),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gravity Wells")

    # Game loop
    total_reward = 0
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1 # up
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2 # down
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3 # left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4 # right
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        pygame.surfarray.blit_array(env.screen, np.transpose(obs, (1, 0, 2)))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()