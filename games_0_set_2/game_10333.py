import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:12:28.145406
# Source Brief: brief_00333.md
# Brief Index: 333
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, color, size, lifetime, vx, vy, gravity=0.1):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(vx, vy)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.gravity = gravity

    def update(self):
        self.pos += self.vel
        self.vel.y += self.gravity
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0 and self.size > 0:
            pygame.draw.circle(surface, self.color, self.pos, int(self.size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Navigate a spaceship to collect asteroids while dodging falling meteors. "
        "Use your limited boosts to maneuver through space and reach the target score."
    )
    user_guide = (
        "Controls: Use the ← and → arrow keys to turn your ship. Press space to use a boost."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30 # For visual interpolation and game speed

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_BOOST = (100, 150, 255)
    COLOR_ASTEROID = (150, 255, 150)
    COLOR_ASTEROID_OUTLINE = (80, 180, 80)
    COLOR_METEOR = (255, 100, 100)
    COLOR_METEOR_OUTLINE = (200, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    
    # Player settings
    PLAYER_SIZE = 12
    PLAYER_TURN_SPEED = 6
    PLAYER_BOOST_FORCE = 2.0
    PLAYER_FRICTION = 0.985

    # Game settings
    WIN_SCORE = 100
    MAX_STEPS = 1500 # Increased slightly to allow more time
    INITIAL_BOOSTS = 3
    INITIAL_ASTEROIDS = 5
    INITIAL_METEOR_SPAWN_RATE = 0.05
    METEOR_SPAWN_RATE_INCREASE = 0.01
    METEOR_SPEED_MIN, METEOR_SPEED_MAX = 2, 4
    ASTEROID_MIN_SIZE, ASTEROID_MAX_SIZE = 15, 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0
        self.boosts_remaining = 0
        self.prev_space_held = False
        self.asteroids = []
        self.meteors = []
        self.particles = []
        self.stars = []
        self.meteor_spawn_rate = self.INITIAL_METEOR_SPAWN_RATE

        self._generate_stars()
        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.boosts_remaining = self.INITIAL_BOOSTS
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90  # Pointing up

        self.prev_space_held = False
        
        self.asteroids.clear()
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid()

        self.meteors.clear()
        self.particles.clear()
        self.meteor_spawn_rate = self.INITIAL_METEOR_SPAWN_RATE

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Action Handling ---
        self._handle_input(movement, space_held)

        # --- Physics & Game Logic Update ---
        self._update_player()
        self._update_meteors()
        self._update_particles()
        
        # --- Continuous Rewards ---
        reward += self._calculate_continuous_reward()

        # --- Collision Detection ---
        reward += self._handle_collisions()

        # --- Spawning ---
        self._spawn_entities()

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 100 == 0:
            self.meteor_spawn_rate += self.METEOR_SPAWN_RATE_INCREASE

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.boosts_remaining < 0: # Hit a meteor with 0 boosts
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time/step limits
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _handle_input(self, movement, space_held):
        # Rotation
        if movement == 3:  # Left
            self.player_angle -= self.PLAYER_TURN_SPEED
        elif movement == 4:  # Right
            self.player_angle += self.PLAYER_TURN_SPEED

        # Boost
        if space_held and not self.prev_space_held and self.boosts_remaining > 0:
            self.boosts_remaining -= 1
            boost_vec = pygame.math.Vector2(1, 0).rotate(self.player_angle) * self.PLAYER_BOOST_FORCE
            self.player_vel += boost_vec
            # SFX: Boost sound
            self._spawn_particles_at(self.player_pos, self.COLOR_BOOST, 15, -boost_vec * 0.5)
        self.prev_space_held = space_held

    def _update_player(self):
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel

        # Screen wrapping for player
        if self.player_pos.x < 0: self.player_pos.x = self.WIDTH
        if self.player_pos.x > self.WIDTH: self.player_pos.x = 0
        if self.player_pos.y < 0: self.player_pos.y = self.HEIGHT
        if self.player_pos.y > self.HEIGHT: self.player_pos.y = 0

    def _update_meteors(self):
        for meteor in self.meteors:
            meteor['pos'].y += meteor['speed']
        self.meteors = [m for m in self.meteors if m['pos'].y < self.HEIGHT + m['size']]

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _calculate_continuous_reward(self):
        reward = 0
        # Small penalty for existing to encourage speed
        reward -= 0.001 
        
        if self.player_vel.length() > 0.1:
            # Reward for moving towards nearest asteroid
            if self.asteroids:
                closest_ast = min(self.asteroids, key=lambda a: self.player_pos.distance_to(a['pos']))
                dir_to_ast = (closest_ast['pos'] - self.player_pos).normalize()
                if self.player_vel.dot(dir_to_ast) > 0:
                    reward += 0.1
            # Penalty for moving towards nearest meteor
            if self.meteors:
                closest_met = min(self.meteors, key=lambda m: self.player_pos.distance_to(m['pos']))
                dir_to_met = (closest_met['pos'] - self.player_pos).normalize()
                if self.player_vel.dot(dir_to_met) > 0:
                    reward -= 0.1
        return reward

    def _handle_collisions(self):
        reward = 0
        # Player vs Asteroids
        for asteroid in self.asteroids[:]:
            if self.player_pos.distance_to(asteroid['pos']) < self.PLAYER_SIZE + asteroid['size']:
                self.asteroids.remove(asteroid)
                self.score += 10
                reward += 10
                # SFX: Asteroid collection sound
                self._spawn_particles_at(asteroid['pos'], self.COLOR_ASTEROID, 20)
                self._spawn_asteroid() # Keep the number of asteroids constant
        
        # Player vs Meteors
        for meteor in self.meteors[:]:
            if self.player_pos.distance_to(meteor['pos']) < self.PLAYER_SIZE + meteor['size']:
                self.meteors.remove(meteor)
                self.score -= 20
                reward -= 20
                self.boosts_remaining -= 1
                # SFX: Meteor collision sound
                self._spawn_particles_at(self.player_pos, self.COLOR_METEOR, 30)
        return reward
    
    def _spawn_entities(self):
        # Spawn meteors
        if self.np_random.random() < self.meteor_spawn_rate:
            self._spawn_meteor()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "boosts": self.boosts_remaining}

    def _render_game(self):
        self._render_stars()
        for p in self.particles: p.draw(self.screen)
        for asteroid in self.asteroids: self._render_asteroid(asteroid)
        for meteor in self.meteors: self._render_meteor(meteor)
        self._render_player()

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Boosts
        boost_text = self.font_ui.render("BOOSTS:", True, self.COLOR_TEXT)
        self.screen.blit(boost_text, (self.WIDTH - 180, 10))
        for i in range(max(0, self.boosts_remaining)):
            pygame.draw.circle(self.screen, self.COLOR_BOOST, (self.WIDTH - 70 + i * 25, 22), 8)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, (self.WIDTH - 70 + i * 25, 22), 8, 1)

        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_ASTEROID if self.score >= self.WIN_SCORE else self.COLOR_METEOR
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': pygame.math.Vector2(random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)),
                'size': random.randint(1, 2),
                'parallax': random.uniform(0.1, 0.5)
            })

    def _render_stars(self):
        # Move stars based on player velocity for parallax effect
        if self.player_vel.length() > 0:
            for star in self.stars:
                star['pos'] -= self.player_vel * star['parallax']
                star['pos'].x %= self.WIDTH
                star['pos'].y %= self.HEIGHT
            
        for star in self.stars:
            color_val = int(100 + 155 * star['parallax'])
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, star['pos'], star['size'])
            
    def _render_player(self):
        # Create rotated points for the triangle
        angle_rad = math.radians(self.player_angle)
        p1 = self.player_pos + pygame.math.Vector2(self.PLAYER_SIZE, 0).rotate(self.player_angle)
        p2 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE / 2, self.PLAYER_SIZE * 0.75).rotate(self.player_angle)
        p3 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE / 2, -self.PLAYER_SIZE * 0.75).rotate(self.player_angle)
        points = [p1, p2, p3]
        
        # Draw boost flame if moving fast
        if self.player_vel.length() > 1.5:
            flame_p1 = p2 + (p3 - p2) / 2
            flame_p2 = flame_p1 - pygame.math.Vector2(self.PLAYER_SIZE * self.player_vel.length() * 0.3, 0).rotate(self.player_angle)
            flame_p3 = p2 + (p3 - p2) / 4
            flame_p4 = p3 - (p3 - p2) / 4
            pygame.gfxdraw.aapolygon(self.screen, [flame_p2, flame_p3, flame_p4], self.COLOR_BOOST)
            pygame.gfxdraw.filled_polygon(self.screen, [flame_p2, flame_p3, flame_p4], self.COLOR_BOOST)

        # Draw glow effect
        glow_radius = int(self.PLAYER_SIZE * 1.8)
        for i in range(glow_radius, 0, -2):
            alpha = int(50 * (1 - i / glow_radius))
            color = (*self.COLOR_PLAYER_GLOW, alpha)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), i, color)

        # Draw ship
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_asteroid(self, asteroid):
        points = [(asteroid['pos'] + v.rotate(asteroid['angle'])) for v in asteroid['vertices']]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
        asteroid['angle'] += asteroid['rot_speed']

    def _render_meteor(self, meteor):
        pos = (int(meteor['pos'].x), int(meteor['pos'].y))
        size = int(meteor['size'])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_METEOR_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_METEOR)

    def _spawn_asteroid(self):
        # Ensure asteroids don't spawn too close to the player
        while True:
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
            if self.player_pos.distance_to(pos) > 100:
                break
        
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        num_vertices = self.np_random.integers(6, 10)
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = size * self.np_random.uniform(0.8, 1.2)
            vertices.append(pygame.math.Vector2(radius * math.cos(angle), radius * math.sin(angle)))

        self.asteroids.append({
            'pos': pos,
            'size': size,
            'vertices': vertices,
            'angle': 0,
            'rot_speed': self.np_random.uniform(-1.5, 1.5)
        })

    def _spawn_meteor(self):
        self.meteors.append({
            'pos': pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -20),
            'size': self.np_random.uniform(8, 15),
            'speed': self.np_random.uniform(self.METEOR_SPEED_MIN, self.METEOR_SPEED_MAX)
        })

    def _spawn_particles_at(self, pos, color, count, base_vel=None):
        if base_vel is None:
            base_vel = pygame.math.Vector2(0, 0)
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4)
            vel = base_vel + pygame.math.Vector2(speed, 0).rotate(angle)
            p = Particle(pos.x, pos.y, color, self.np_random.uniform(3, 7), self.np_random.integers(15, 30), vel.x, vel.y)
            self.particles.append(p)

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run with SDL_VIDEODRIVER="dummy"
    # To play, comment out the os.environ line at the top of the file
    if os.getenv("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run interactive game with SDL_VIDEODRIVER=dummy.")
        print("Comment out the os.environ.setdefault line to run the game interactively.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Asteroid Collector")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        total_reward = 0
        
        # --- Control Mapping for Manual Play ---
        # Arrow Keys: Rotate Left/Right
        # Space: Boost
        
        while not terminated and not truncated:
            movement_action = 0 # No-op
            space_action = 0
            shift_action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement_action = 3
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement_action = 4
            if keys[pygame.K_SPACE]:
                space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_action = 1
                
            action = [movement_action, space_action, shift_action]
            
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term
            truncated = trunc

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        
        # Display final screen for a moment
        pygame.time.wait(2000)
        
        env.close()