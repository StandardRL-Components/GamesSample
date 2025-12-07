
# Generated: 2025-08-27T14:39:42.911500
# Source Brief: brief_00750.md
# Brief Index: 750

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects to keep the main environment class clean
class Player:
    """Represents the player's ship."""
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.angle = -90  # Pointing up
        self.radius = 12
        self.max_speed = 4
        self.acceleration = 0.2
        self.turn_speed = 4
        self.friction = 0.98

    def update(self, movement, world_width, world_height):
        # Turning
        if movement == 3:  # Left
            self.angle -= self.turn_speed
        if movement == 4:  # Right
            self.angle += self.turn_speed

        # Acceleration/Braking
        if movement == 1:  # Up (Thrust)
            accel_vec = pygame.Vector2(self.acceleration, 0).rotate(-self.angle)
            self.vel += accel_vec
        elif movement == 2:  # Down (Brake)
            self.vel *= 0.95 # Stronger friction for braking

        # Speed limit
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
            
        # Apply friction and update position
        self.vel *= self.friction
        self.pos += self.vel

        # World wrapping
        self.pos.x = self.pos.x % world_width
        self.pos.y = self.pos.y % world_height

    def draw(self, surface, camera_offset):
        # Ship body points
        points = [
            pygame.Vector2(self.radius, 0),
            pygame.Vector2(-self.radius, self.radius * 0.7),
            pygame.Vector2(-self.radius * 0.5, 0),
            pygame.Vector2(-self.radius, -self.radius * 0.7)
        ]
        
        # Rotate and translate points to world space, then to camera space
        rotated_points = [p.rotate(-self.angle) + self.pos - camera_offset for p in points]
        
        # Draw with antialiasing for a smooth look
        pygame.gfxdraw.aapolygon(surface, rotated_points, (150, 255, 150))
        pygame.gfxdraw.filled_polygon(surface, rotated_points, (50, 200, 50))
        
        # Add a glow effect for visibility
        pygame.gfxdraw.aapolygon(surface, rotated_points, (100, 255, 100, 50))


class Asteroid:
    """Represents a mineable asteroid."""
    def __init__(self, x, y, rng):
        self.pos = pygame.Vector2(x, y)
        self.minerals = rng.integers(1, 6)
        self.radius = 15 + self.minerals * 2
        self.angle = 0
        self.rotation_speed = rng.uniform(-1, 1)
        
        # Create a unique, jagged shape for the asteroid
        self.shape_points = []
        num_points = rng.integers(7, 12)
        for i in range(num_points):
            angle = i * (360 / num_points)
            dist = rng.uniform(self.radius * 0.8, self.radius * 1.2)
            self.shape_points.append(pygame.Vector2(dist, 0).rotate(angle))
            
    def update(self):
        self.angle += self.rotation_speed

    def draw(self, surface, camera_offset):
        rotated_points = [p.rotate(self.angle) + self.pos - camera_offset for p in self.shape_points]
        pygame.gfxdraw.aapolygon(surface, rotated_points, (150, 150, 150))
        pygame.gfxdraw.filled_polygon(surface, rotated_points, (100, 100, 100))


class Enemy:
    """Represents a hostile ship."""
    def __init__(self, x, y, type, rng):
        self.pos = pygame.Vector2(x, y)
        self.type = type
        self.radius = 10
        self.base_speed = rng.uniform(1.5, 2.5)
        self.speed_multiplier = 1.0

        if self.type == 'red': # Horizontal patrol
            self.vel = pygame.Vector2(self.base_speed, 0)
            self.color = (255, 50, 50)
        elif self.type == 'blue': # Vertical patrol
            self.vel = pygame.Vector2(0, self.base_speed)
            self.color = (50, 150, 255)
        else: # Purple (Diagonal patrol)
            self.vel = pygame.Vector2(self.base_speed * 0.707, self.base_speed * 0.707)
            self.color = (200, 50, 255)

    def update(self, world_width, world_height):
        self.pos += self.vel * self.speed_multiplier

        # Bounce off world edges
        if self.type == 'red':
            if self.pos.x <= 0 or self.pos.x >= world_width:
                self.vel.x *= -1
        elif self.type == 'blue':
            if self.pos.y <= 0 or self.pos.y >= world_height:
                self.vel.y *= -1
        else: # Purple
            if self.pos.x <= 0 or self.pos.x >= world_width:
                self.vel.x *= -1
            if self.pos.y <= 0 or self.pos.y >= world_height:
                self.vel.y *= -1
        
        # Clamp position to prevent going out of bounds
        self.pos.x = np.clip(self.pos.x, 0, world_width)
        self.pos.y = np.clip(self.pos.y, 0, world_height)

    def draw(self, surface, camera_offset):
        draw_pos = self.pos - camera_offset
        # Draw antialiased circle with a glow
        pygame.gfxdraw.aacircle(surface, int(draw_pos.x), int(draw_pos.y), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(draw_pos.x), int(draw_pos.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(draw_pos.x), int(draw_pos.y), self.radius + 2, self.color + (100,))


class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, vel, lifetime, color, size):
        self.pos = pygame.Vector2(x, y)
        self.vel = vel
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color
        self.size = size

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1

    def draw(self, surface, camera_offset):
        # Fade out over time
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        color_with_alpha = self.color + (alpha,)
        draw_pos = self.pos - camera_offset
        pygame.draw.circle(surface, color_with_alpha, (int(draw_pos.x), int(draw_pos.y)), int(self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to mine nearby asteroids."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and World Dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 1280, 800

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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_MINERAL = (255, 220, 50)
        self.COLOR_EXPLOSION = (255, 100, 0)
        
        # Game constants
        self.WIN_SCORE = 50
        self.MAX_STEPS = 2000
        self.MINING_RADIUS = 60
        self.INITIAL_ASTEROIDS = 8
        self.INITIAL_ENEMIES = 4
        self.MAX_ASTEROIDS = 12

        # Initialize state variables
        self.player = None
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.stars = []
        
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._create_stars()
        
        # Initialize state variables by calling reset
        self.reset()
        
        self.validate_implementation()

    def _create_stars(self):
        """Create a starfield for the parallax background."""
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)),
                'depth': random.uniform(0.1, 0.6) # For parallax speed
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player = Player(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2)
        
        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid()

        self.enemies = []
        enemy_types = ['red', 'blue', 'purple']
        for i in range(self.INITIAL_ENEMIES):
            self._spawn_enemy(enemy_types[i % len(enemy_types)])

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def _spawn_asteroid(self):
        """Spawns an asteroid at a random location away from the player."""
        while True:
            x = self.rng.uniform(0, self.WORLD_WIDTH)
            y = self.rng.uniform(0, self.WORLD_HEIGHT)
            if self.player.pos.distance_to((x, y)) > 200:
                self.asteroids.append(Asteroid(x, y, self.rng))
                break

    def _spawn_enemy(self, enemy_type):
        """Spawns an enemy at a random location away from the player."""
        while True:
            x = self.rng.uniform(0, self.WORLD_WIDTH)
            y = self.rng.uniform(0, self.WORLD_HEIGHT)
            if self.player.pos.distance_to((x, y)) > 300:
                self.enemies.append(Enemy(x, y, enemy_type, self.rng))
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.02 # Time penalty per step

        # === 1. Handle Input & Player Update ===
        self.player.update(movement, self.WORLD_WIDTH, self.WORLD_HEIGHT)

        # Engine exhaust particles for visual feedback
        if movement == 1:
            for _ in range(2):
                vel = pygame.Vector2(-2, 0).rotate(-self.player.angle + self.rng.uniform(-15, 15))
                pos = self.player.pos + pygame.Vector2(-self.player.radius*0.8, 0).rotate(-self.player.angle)
                self.particles.append(Particle(pos.x, pos.y, vel, 15, (200, 200, 255), self.rng.uniform(1, 3)))

        # === 2. Mining Logic ===
        if space_held:
            closest_asteroid, min_dist = self._find_closest_asteroid()
            
            if closest_asteroid and min_dist < self.MINING_RADIUS + closest_asteroid.radius:
                # # Sound: Mining laser
                closest_asteroid.minerals -= 1
                self.score += 1
                reward += 0.1 # Reward for collecting a mineral

                # Create mineral particles flying from asteroid
                for _ in range(3):
                    angle_to_player = (self.player.pos - closest_asteroid.pos).angle_to(pygame.Vector2(1,0))
                    vel = pygame.Vector2(self.rng.uniform(1, 3), 0).rotate(-angle_to_player + self.rng.uniform(-30, 30))
                    self.particles.append(Particle(closest_asteroid.pos.x, closest_asteroid.pos.y, vel, 40, self.COLOR_MINERAL, 2))
                
                if closest_asteroid.minerals <= 0:
                    reward += 1.0 # Bonus reward for depleting an asteroid
                    self.asteroids.remove(closest_asteroid)
                    # # Sound: Asteroid destroyed

        # === 3. Update Game State ===
        for enemy in self.enemies:
            enemy.speed_multiplier = 1.0 + (0.05 * (self.score // 50)) # Difficulty scaling
            enemy.update(self.WORLD_WIDTH, self.WORLD_HEIGHT)
        for asteroid in self.asteroids:
            asteroid.update()
        
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

        # Respawn asteroids to prevent softlocks
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.rng.random() < 0.01:
            self._spawn_asteroid()

        # === 4. Collision Detection ===
        for enemy in self.enemies:
            if self.player.pos.distance_to(enemy.pos) < self.player.radius + enemy.radius:
                self.game_over = True
                reward = -100 # Large penalty for losing
                # # Sound: Explosion
                # Create explosion effect
                for _ in range(50):
                    speed = self.rng.uniform(1, 5)
                    angle = self.rng.uniform(0, 360)
                    vel = pygame.Vector2(speed, 0).rotate(angle)
                    self.particles.append(Particle(self.player.pos.x, self.player.pos.y, vel, self.rng.integers(30, 60), self.COLOR_EXPLOSION, self.rng.uniform(1, 4)))
                break
        
        # === 5. Check Win/Termination Conditions ===
        self.steps += 1
        terminated = self.game_over
        
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward = 100 # Large reward for winning
            self.game_over = True

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
    
    def _find_closest_asteroid(self):
        """Helper to find the nearest asteroid to the player."""
        closest_asteroid = None
        min_dist = float('inf')
        if not self.asteroids:
            return None, min_dist
        for asteroid in self.asteroids:
            dist = self.player.pos.distance_to(asteroid.pos)
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        return closest_asteroid, min_dist

    def _get_observation(self):
        # Camera follows player, keeping them centered
        camera_offset = self.player.pos - pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self.screen.fill(self.COLOR_BG)
        self._draw_stars(camera_offset)
        
        # Render all game elements
        self._render_game(camera_offset)
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, camera_offset):
        """Renders all game entities."""
        for asteroid in self.asteroids:
            asteroid.draw(self.screen, camera_offset)
        for enemy in self.enemies:
            enemy.draw(self.screen, camera_offset)
        
        for p in self.particles:
            p.draw(self.screen, camera_offset)
        
        # Don't draw player if they just exploded
        if not (self.game_over and self.score < self.WIN_SCORE):
             self.player.draw(self.screen, camera_offset)

    def _draw_stars(self, camera_offset):
        """Draws the parallax scrolling starfield."""
        for star in self.stars:
            star_pos_on_screen_x = (star['pos'].x - camera_offset.x * star['depth']) % self.SCREEN_WIDTH
            star_pos_on_screen_y = (star['pos'].y - camera_offset.y * star['depth']) % self.SCREEN_HEIGHT
            
            size = int(star['depth'] * 3)
            brightness = int(100 + 155 * star['depth'])
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (star_pos_on_screen_x, star_pos_on_screen_y), max(1, size))

    def _render_ui(self):
        """Renders the UI elements like score and timers."""
        score_text = self.font.render(f"MINERALS: {self.score} / {self.WIN_SCORE}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        steps_text = self.small_font.render(f"TIME: {self.MAX_STEPS - self.steps}", True, (200, 200, 200))
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                end_text = self.font.render("MISSION COMPLETE", True, (100, 255, 100))
            else:
                end_text = self.font.render("SHIP DESTROYED", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
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
        
        # Test reset to get initial obs
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test info
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

# Example of how to run the environment for interactive play
if __name__ == '__main__':
    # Set this to "dummy" for headless execution for training
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()

    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        else:
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                total_reward = 0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)
        
    env.close()