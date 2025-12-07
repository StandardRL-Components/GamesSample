
# Generated: 2025-08-27T16:23:28.284247
# Source Brief: brief_00023.md
# Brief Index: 23

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, color, lifetime, radius):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.lifetime = lifetime
        self.radius = radius

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1
        self.radius = max(0, self.radius - 0.1)

class Gem:
    def __init__(self, pos, creation_time):
        self.pos = np.array(pos, dtype=float)
        self.radius = 8
        self.creation_time = creation_time

    def update(self, current_time):
        # Pulsating effect
        self.pulse = 1 + 0.2 * math.sin((current_time - self.creation_time) * 0.1)

    def draw(self, surface):
        r = int(self.radius * self.pulse)
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        color = (150, 255, 150)
        glow_color = (50, 150, 50)
        
        # Draw glow
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], r + 3, glow_color)
        # Draw gem
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], r, color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], r, color)


class Laser:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.lifetime = 30  # Frames

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1

    def draw(self, surface):
        start_pos = (int(self.pos[0]), int(self.pos[1]))
        end_pos = (int(self.pos[0] - self.vel[0] * 2), int(self.pos[1] - self.vel[1] * 2))
        pygame.draw.line(surface, (255, 255, 100), start_pos, end_pos, 4)


class Asteroid:
    def __init__(self, pos, vel, size, screen_width, screen_height):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.size = size  # 3: large, 2: medium, 1: small
        self.radius = size * 12
        self.rotation = random.uniform(0, 360)
        self.rot_speed = random.uniform(-1, 1)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.points = self._generate_points()

    def _generate_points(self):
        num_points = random.randint(7, 12)
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = self.radius * random.uniform(0.7, 1.1)
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
        return points

    def update(self):
        self.pos += self.vel
        self.rotation += self.rot_speed
        self.pos[0] %= self.screen_width
        self.pos[1] %= self.screen_height

    def draw(self, surface):
        rotated_points = []
        rad = math.radians(self.rotation)
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)
        for x, y in self.points:
            new_x = (x * cos_rad - y * sin_rad) + self.pos[0]
            new_y = (x * sin_rad + y * cos_rad) + self.pos[1]
            rotated_points.append((int(new_x), int(new_y)))

        color = (120, 120, 130)
        pygame.gfxdraw.filled_polygon(surface, rotated_points, color)
        pygame.gfxdraw.aapolygon(surface, rotated_points, (150, 150, 160))


class Meteor:
    def __init__(self, pos, vel, screen_width, screen_height):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = 10
        self.screen_width = screen_width
        self.screen_height = screen_height

    def update(self):
        self.pos += self.vel
        # No wrap-around for meteors, they fly across
        
    def is_out_of_bounds(self):
        return (self.pos[0] < -self.radius or self.pos[0] > self.screen_width + self.radius or
                self.pos[1] < -self.radius or self.pos[1] > self.screen_height + self.radius)

    def draw(self, surface):
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        color = (255, 100, 80)
        glow_color = (180, 50, 30)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.radius + 4, glow_color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.radius, color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.radius, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your ship. Hold space to fire your mining laser."
    )

    game_description = (
        "Pilot a mining ship, blast asteroids for gems, and dodge fiery meteors to survive."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 50
        self.MAX_STEPS = 3000 # 100 seconds at 30fps
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (80, 150, 255)
        self.COLOR_PLAYER_GLOW = (30, 60, 120)
        self.COLOR_TEXT = (220, 220, 240)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_radius = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.invincibility_timer = None
        self.laser_cooldown = None

        self.asteroids = []
        self.meteors = []
        self.gems = []
        self.lasers = []
        self.particles = []
        self.stars = []

        self.reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_radius = 12
        self.lives = self.INITIAL_LIVES
        self.invincibility_timer = 0
        self.laser_cooldown = 0
        
        # Game state
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # Entity lists
        self.asteroids.clear()
        self.meteors.clear()
        self.gems.clear()
        self.lasers.clear()
        self.particles.clear()
        
        # Initial population
        for _ in range(8):
            self._spawn_asteroid(size=random.randint(2, 3))
        
        # Starfield
        if not self.stars:
            for _ in range(150):
                self.stars.append({
                    "pos": (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                    "brightness": random.randint(50, 150)
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # -- 1. Action Handling --
        movement = action[0]
        space_held = action[1] == 1
        
        player_speed = 4.0
        if movement == 1: self.player_vel[1] = -player_speed
        elif movement == 2: self.player_vel[1] = player_speed
        elif movement == 3: self.player_vel[0] = -player_speed
        elif movement == 4: self.player_vel[0] = player_speed
        else: self.player_vel *= 0.8 # Deceleration

        if space_held and self.laser_cooldown <= 0:
            self._fire_laser()
        
        # -- 2. Reward Shaping (Distance to Gem) --
        dist_before = self._get_closest_gem_dist()

        # -- 3. Update Player --
        self.player_pos += self.player_vel
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
        
        if self.laser_cooldown > 0: self.laser_cooldown -= 1
        if self.invincibility_timer > 0: self.invincibility_timer -= 1

        dist_after = self._get_closest_gem_dist()
        
        if dist_before < float('inf') and dist_after < float('inf'):
            if dist_after < dist_before:
                reward += 0.01  # Scaled down from brief for better balance
            else:
                reward -= 0.002 # Scaled down

        # -- 4. Update Entities --
        for laser in self.lasers[:]:
            laser.update()
            if laser.lifetime <= 0: self.lasers.remove(laser)
        
        for asteroid in self.asteroids:
            asteroid.update()
            
        for meteor in self.meteors[:]:
            meteor.update()
            if meteor.is_out_of_bounds(): self.meteors.remove(meteor)

        for gem in self.gems:
            gem.update(self.steps)

        for particle in self.particles[:]:
            particle.update()
            if particle.lifetime <= 0: self.particles.remove(particle)
            
        # -- 5. Spawning --
        self._spawn_meteors()
        if len(self.asteroids) < 5 + (self.score // 5):
            self._spawn_asteroid()

        # -- 6. Collision Detection --
        gem_collected_this_step = self._handle_collisions()
        if gem_collected_this_step:
            reward += 1

        # -- 7. Termination Check --
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
        elif self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_closest_gem_dist(self):
        if not self.gems:
            return float('inf')
        
        min_dist = float('inf')
        for gem in self.gems:
            dist = np.linalg.norm(self.player_pos - gem.pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _fire_laser(self):
        # # SFX: Laser fire
        self.laser_cooldown = 8 # Cooldown in frames
        # Fire based on last movement direction, or straight up if stationary
        fire_vel = np.array([0.0, -1.0]) # Default up
        if np.linalg.norm(self.player_vel) > 0.1:
            fire_vel = self.player_vel / np.linalg.norm(self.player_vel)
        
        laser_vel = fire_vel * 15.0
        laser_pos = self.player_pos + fire_vel * (self.player_radius + 5)
        self.lasers.append(Laser(laser_pos, laser_vel))

    def _handle_collisions(self):
        gem_collected = False
        # Player vs Gems
        for gem in self.gems[:]:
            if np.linalg.norm(self.player_pos - gem.pos) < self.player_radius + gem.radius:
                self.gems.remove(gem)
                self.score = min(self.WIN_SCORE, self.score + 1)
                gem_collected = True
                # # SFX: Gem collect
                self._create_particles(self.player_pos, 15, (100, 255, 100), 2)
        
        # Player vs Meteors
        if self.invincibility_timer <= 0:
            for meteor in self.meteors:
                if np.linalg.norm(self.player_pos - meteor.pos) < self.player_radius + meteor.radius:
                    self.lives -= 1
                    self.invincibility_timer = 90 # 3 seconds invincibility
                    # # SFX: Player hit/explosion
                    self._create_particles(self.player_pos, 50, (255, 100, 80), 4)
                    if self.lives > 0:
                        # Reset position to center after hit
                        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
                        self.player_vel = np.array([0.0, 0.0], dtype=float)
                    break # Only one hit per frame
        
        # Lasers vs Asteroids
        for laser in self.lasers[:]:
            for asteroid in self.asteroids[:]:
                if np.linalg.norm(laser.pos - asteroid.pos) < asteroid.radius:
                    if laser in self.lasers: self.lasers.remove(laser)
                    self._break_asteroid(asteroid)
                    break

        return gem_collected

    def _break_asteroid(self, asteroid):
        # # SFX: Asteroid explosion
        self.asteroids.remove(asteroid)
        self._create_particles(asteroid.pos, int(asteroid.radius * 2), (150, 150, 150), asteroid.size)

        if asteroid.size > 1:
            # Break into smaller asteroids
            for _ in range(2):
                self._spawn_asteroid(size=asteroid.size - 1, pos=asteroid.pos)
        else:
            # Spawn gems
            num_gems = random.randint(1, 3)
            for _ in range(num_gems):
                offset = np.random.rand(2) * 20 - 10
                self.gems.append(Gem(asteroid.pos + offset, self.steps))

    def _spawn_asteroid(self, size=None, pos=None):
        if size is None:
            size = random.randint(1, 3)
        if pos is None:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': pos = [random.uniform(0, self.WIDTH), -30]
            elif edge == 'bottom': pos = [random.uniform(0, self.WIDTH), self.HEIGHT + 30]
            elif edge == 'left': pos = [-30, random.uniform(0, self.HEIGHT)]
            else: pos = [self.WIDTH + 30, random.uniform(0, self.HEIGHT)]
        
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 1.5)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        
        self.asteroids.append(Asteroid(pos, vel, size, self.WIDTH, self.HEIGHT))

    def _spawn_meteors(self):
        # Meteor frequency increases by 0.01 per second (0.01 / 30 per step)
        meteor_chance = 0.01 + (self.steps * (0.01 / 30))
        if random.random() < meteor_chance:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                pos = [random.uniform(0, self.WIDTH), -20]
                vel = [random.uniform(-2, 2), random.uniform(4, 8)]
            elif edge == 'bottom':
                pos = [random.uniform(0, self.WIDTH), self.HEIGHT + 20]
                vel = [random.uniform(-2, 2), random.uniform(-8, -4)]
            elif edge == 'left':
                pos = [-20, random.uniform(0, self.HEIGHT)]
                vel = [random.uniform(4, 8), random.uniform(-2, 2)]
            else: # right
                pos = [self.WIDTH + 20, random.uniform(0, self.HEIGHT)]
                vel = [random.uniform(-8, -4), random.uniform(-2, 2)]
            self.meteors.append(Meteor(pos, vel, self.WIDTH, self.HEIGHT))

    def _create_particles(self, pos, count, color, speed_multiplier):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_multiplier
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(15, 30)
            radius = random.uniform(2, 4)
            self.particles.append(Particle(pos, vel, color, lifetime, radius))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for star in self.stars:
            c = star["brightness"]
            self.screen.set_at(star["pos"], (c, c, c))
            
        # Meteors
        for meteor in self.meteors:
            meteor.draw(self.screen)
            
        # Asteroids
        for asteroid in self.asteroids:
            asteroid.draw(self.screen)
            
        # Gems
        for gem in self.gems:
            gem.draw(self.screen)

        # Lasers
        for laser in self.lasers:
            laser.draw(self.screen)

        # Player
        if self.lives > 0:
            is_invincible_flash = self.invincibility_timer > 0 and (self.steps // 3) % 2 == 0
            if not is_invincible_flash:
                pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
                # Thruster particles
                if np.linalg.norm(self.player_vel) > 0.5:
                    thruster_pos = self.player_pos - self.player_vel / np.linalg.norm(self.player_vel) * self.player_radius
                    self._create_particles(thruster_pos, 2, (255, 180, 50), 0.5)

                # Player ship
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.player_radius + 4, self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.player_radius, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.player_radius, self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.lifetime / 30))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((p.radius*2, p.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p.radius, p.radius), p.radius)
            self.screen.blit(temp_surf, (p.pos[0] - p.radius, p.pos[1] - p.radius))

    def _render_ui(self):
        # Score (Gems)
        score_text = self.font_small.render(f"GEMS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            pos = (self.WIDTH - 80 + i * 25, 18)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This requires a display. If running headless, this part will fail.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        # Only run display if not in a headless environment
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Asteroid Miner")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Action mapping for human play
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            clock.tick(30) # Run at 30 FPS

        env.close()
        print("Game Over!")
        print(f"Final Score: {info['score']}, Steps: {info['steps']}")