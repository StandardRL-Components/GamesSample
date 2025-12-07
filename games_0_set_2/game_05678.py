import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to mine asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship through an asteroid field, mining ore while dodging collisions to collect 100 units of ore."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (60, 180, 255)
        self.COLOR_SHIP_GLOW = (30, 90, 180)
        self.COLOR_ASTEROID = (120, 110, 100)
        self.COLOR_ORE = (255, 215, 0)
        self.COLOR_LASER = (50, 255, 50)
        self.COLOR_TEXT = (230, 230, 240)
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # Game constants
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 100
        self.STARTING_LIVES = 3
        self.SHIP_ACCELERATION = 0.2
        self.SHIP_TURN_SPEED = 4.0
        self.SHIP_FRICTION = 0.98
        self.SHIP_BRAKE_FRICTION = 0.92
        self.SHIP_MAX_SPEED = 5
        self.SHIP_RADIUS = 12
        self.ASTEROID_ORE_MIN = 15
        self.ASTEROID_ORE_MAX = 40
        self.MIN_ASTEROIDS = 10
        self.MAX_ASTEROIDS = 15
        self.LASER_RANGE = 150
        self.LASER_DPS = 0.5 # Ore per step

        # Initialize state variables
        self.ship_pos = None
        self.ship_vel = None
        self.ship_angle = None
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.game_over = False
        self.game_won = False
        self.screen_shake = 0
        self.asteroid_base_speed = 0.5
        
        self._generate_stars()
        # self.reset() is called by the test harness, no need to call it here.
        # self.validate_implementation() is also a test-related utility.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.ship_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.ship_vel = pygame.math.Vector2(0, 0)
        self.ship_angle = -90.0
        
        self.lives = self.STARTING_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.screen_shake = 0
        self.asteroid_base_speed = 0.5

        self.asteroids = []
        self.particles = []
        # Use the seeded RNG for deterministic asteroid placement
        num_asteroids = self.np_random.integers(self.MIN_ASTEROIDS, self.MAX_ASTEROIDS + 1)
        for _ in range(num_asteroids):
            self._spawn_asteroid(on_edge=False)
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Find distance to nearest asteroid before acting
        dist_before = self._get_dist_to_nearest_asteroid()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update game logic
        self._update_ship(movement)
        reward += self._handle_mining(space_held)
        self._update_asteroids()
        self._update_particles()
        reward += self._handle_collisions()
        self._replenish_asteroids()

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.asteroid_base_speed = min(2.0, self.asteroid_base_speed + 0.05)

        # Calculate continuous rewards
        dist_after = self._get_dist_to_nearest_asteroid()
        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += -0.05  # Approaching is risky
            elif dist_after > dist_before and self.score < self.WIN_SCORE:
                reward += -0.1 # Moving away from objective

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated: # Game ended due to win/loss, not timeout
            if self.game_won:
                reward += 100
            # Life loss penalty is handled in _handle_collisions

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Apply screen shake
        render_offset = pygame.math.Vector2(0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset.x = self.np_random.integers(-4, 5)
            render_offset.y = self.np_random.integers(-4, 5)

        # Render all game elements
        self._render_stars(render_offset)
        self._render_asteroids(render_offset)
        self._render_particles(render_offset)
        self._render_ship(render_offset)
        
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
            "lives": self.lives,
        }

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.game_won = True
            return True
        if self.lives <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _update_ship(self, movement):
        direction_vec = pygame.math.Vector2(1, 0).rotate(-self.ship_angle)
        
        if movement == 1:  # Up
            self.ship_vel += direction_vec * self.SHIP_ACCELERATION
            # Thruster particles
            if self.steps % 2 == 0:
                self._spawn_particle(
                    pos=self.ship_pos - direction_vec * 15,
                    vel=-direction_vec * 2 + (self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)),
                    lifespan=15,
                    start_color=(255, 150, 50),
                    end_color=self.COLOR_BG,
                    radius=4
                )
        elif movement == 2:  # Down (Brake)
            self.ship_vel *= self.SHIP_BRAKE_FRICTION
        
        if movement == 3:  # Left
            self.ship_angle -= self.SHIP_TURN_SPEED
        elif movement == 4:  # Right
            self.ship_angle += self.SHIP_TURN_SPEED

        # Limit speed and apply friction
        if self.ship_vel.length() > self.SHIP_MAX_SPEED:
            self.ship_vel.scale_to_length(self.SHIP_MAX_SPEED)
        self.ship_vel *= self.SHIP_FRICTION
        
        self.ship_pos += self.ship_vel
        
        # World wrapping
        self.ship_pos.x %= self.WIDTH
        self.ship_pos.y %= self.HEIGHT

    def _handle_mining(self, space_held):
        reward = 0
        if not space_held:
            return 0
            
        nearest_asteroid, dist = self._get_nearest_asteroid_in_range()
        
        if nearest_asteroid and dist <= self.LASER_RANGE:
            # Laser effect
            self._spawn_particle(
                pos=pygame.math.Vector2(nearest_asteroid['pos']),
                vel=(0,0), lifespan=2, start_color=self.COLOR_LASER, end_color=(200, 255, 200), radius=3
            )
            # Ore collection
            mined_amount = min(nearest_asteroid['ore'], self.LASER_DPS)
            nearest_asteroid['ore'] -= mined_amount
            self.score += mined_amount
            reward += mined_amount * 0.1

            # Ore particles flying to ship
            if self.steps % 3 == 0:
                ore_p_vel = (self.ship_pos - nearest_asteroid['pos']).normalize() * 3
                self._spawn_particle(
                    pos=pygame.math.Vector2(nearest_asteroid['pos']),
                    vel=ore_p_vel,
                    lifespan=int(dist / 3),
                    start_color=self.COLOR_ORE,
                    end_color=(150, 120, 0),
                    radius=3,
                    is_ore=True
                )
            
            if nearest_asteroid['ore'] <= 0:
                # Asteroid destroyed
                reward += 10
                self._destroy_asteroid(nearest_asteroid)
        return reward

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            
            # Wrap around screen
            asteroid['pos'].x %= self.WIDTH
            asteroid['pos'].y %= self.HEIGHT

    def _handle_collisions(self):
        for asteroid in self.asteroids:
            if self.ship_pos.distance_to(asteroid['pos']) < self.SHIP_RADIUS + asteroid['radius']:
                self.lives -= 1
                self.screen_shake = 15
                
                # Explosion effect
                for _ in range(30):
                    self._spawn_particle(
                        pos=pygame.math.Vector2(self.ship_pos),
                        vel=pygame.math.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4)),
                        lifespan=self.np_random.integers(20, 41),
                        start_color=(255, 100, 0),
                        end_color=(255, 200, 0),
                        radius=self.np_random.integers(2, 6)
                    )
                
                # Reset ship position and velocity
                self.ship_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
                self.ship_vel = pygame.math.Vector2(0, 0)
                
                # Destroy colliding asteroid
                self._destroy_asteroid(asteroid)
                
                return -10  # Collision penalty
        return 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['lifespan'] -= 1
            p['pos'] += p['vel']
            p['current_radius'] = p['radius'] * (p['lifespan'] / p['max_lifespan'])
            
            # Color interpolation
            t = 1.0 - (p['lifespan'] / p['max_lifespan'])
            p['current_color'] = (
                int(p['start_color'][0] * (1 - t) + p['end_color'][0] * t),
                int(p['start_color'][1] * (1 - t) + p['end_color'][1] * t),
                int(p['start_color'][2] * (1 - t) + p['end_color'][2] * t),
            )
            
    def _render_ship(self, offset):
        pos = self.ship_pos + offset
        
        # Glow effect
        glow_radius = int(self.SHIP_RADIUS * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_SHIP_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        angle_rad = math.radians(self.ship_angle)
        p1 = pos + pygame.math.Vector2(self.SHIP_RADIUS, 0).rotate(-self.ship_angle)
        p2 = pos + pygame.math.Vector2(-self.SHIP_RADIUS * 0.7, self.SHIP_RADIUS * 0.8).rotate(-self.ship_angle)
        p3 = pos + pygame.math.Vector2(-self.SHIP_RADIUS * 0.7, -self.SHIP_RADIUS * 0.8).rotate(-self.ship_angle)
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SHIP)

    def _render_asteroids(self, offset):
        for asteroid in self.asteroids:
            pos = asteroid['pos'] + offset
            points = []
            for p in asteroid['shape']:
                rotated_p = p.rotate(asteroid['angle'])
                points.append((pos.x + rotated_p.x, pos.y + rotated_p.y))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = p['pos'] + offset
            if p['current_radius'] > 0:
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(p['current_radius']), p['current_color'])
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(p['current_radius']), p['current_color'])

    def _render_stars(self, offset):
        for star in self.stars:
            pos = (star['pos'] + offset * star['depth'])
            pos.x %= self.WIDTH
            pos.y %= self.HEIGHT
            pygame.draw.circle(self.screen, star['color'], (int(pos.x), int(pos.y)), star['radius'])
    
    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Render lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Render mining laser if applicable
        if not self.game_over:
            # This part is tricky in a headless env. We can't query keys.
            # A better way is to check the action passed to step(), but step() has already finished.
            # For pure visualization, this is okay, but it's not state-correct.
            # Let's assume this is for the interactive __main__ block.
            nearest_asteroid, dist = self._get_nearest_asteroid_in_range()
            if nearest_asteroid and dist <= self.LASER_RANGE:
                # We can't know if space was held. Let's not draw the laser here
                # to avoid showing it when it's not active. The particle effects
                # from _handle_mining are a better indicator.
                pass


    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        msg = "MISSION COMPLETE" if self.game_won else "GAME OVER"
        text = self.font_game_over.render(msg, True, self.COLOR_ORE if self.game_won else self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _spawn_asteroid(self, on_edge=True):
        radius = self.np_random.uniform(15, 30)
        shape = self._create_asteroid_shape(radius)
        ore = self.np_random.integers(self.ASTEROID_ORE_MIN, self.ASTEROID_ORE_MAX + 1)
        
        if on_edge:
            edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -radius)
            elif edge == 'bottom': pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + radius)
            elif edge == 'left': pos = pygame.math.Vector2(-radius, self.np_random.uniform(0, self.HEIGHT))
            else: pos = pygame.math.Vector2(self.WIDTH + radius, self.np_random.uniform(0, self.HEIGHT))
        else:
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))

        vel_magnitude = self.np_random.uniform(0.5, 1.5) * self.asteroid_base_speed
        vel_angle = self.np_random.uniform(0, 360)
        vel = pygame.math.Vector2(vel_magnitude, 0).rotate(vel_angle)
        
        rot_speed = self.np_random.uniform(-1.5, 1.5)
        
        self.asteroids.append({
            'pos': pos, 'vel': vel, 'radius': radius, 'shape': shape, 'angle': 0,
            'rot_speed': rot_speed, 'ore': ore
        })

    def _create_asteroid_shape(self, radius):
        points = []
        num_vertices = self.np_random.integers(7, 13)
        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * math.pi
            dist = radius * self.np_random.uniform(0.7, 1.1)
            points.append(pygame.math.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
        return points

    def _destroy_asteroid(self, asteroid_to_remove):
        self.asteroids = [a for a in self.asteroids if a is not asteroid_to_remove]

    def _replenish_asteroids(self):
        if len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroid(on_edge=True)

    def _spawn_particle(self, pos, vel, lifespan, start_color, end_color, radius, is_ore=False):
        self.particles.append({
            'pos': pos, 'vel': pygame.math.Vector2(vel), 'lifespan': lifespan, 'max_lifespan': lifespan,
            'start_color': start_color, 'end_color': end_color, 'radius': radius, 'is_ore': is_ore,
            'current_radius': radius, 'current_color': start_color
        })

    def _generate_stars(self):
        self.stars = []
        # Use a fixed seed for stars so they don't change on reset
        rng = np.random.default_rng(42)
        for _ in range(150):
            depth = rng.uniform(0.1, 0.6)
            self.stars.append({
                'pos': pygame.math.Vector2(rng.integers(0, self.WIDTH), rng.integers(0, self.HEIGHT)),
                'radius': int(depth * 2),
                'color': (int(100 + 155 * depth), int(100 + 155 * depth), int(100 + 155 * depth)),
                'depth': depth
            })

    def _get_nearest_asteroid_in_range(self):
        nearest = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = self.ship_pos.distance_to(asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest = asteroid
        return nearest, min_dist

    def _get_dist_to_nearest_asteroid(self):
        if not self.asteroids:
            return None
        return min(self.ship_pos.distance_to(a['pos']) for a in self.asteroids)

    def close(self):
        pygame.quit()


# Example usage to run and visualize the game
if __name__ == '__main__':
    # The main block is for interactive play and visualization, not for headless testing
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset(seed=random.randint(0, 1_000_000))
    
    # Set up Pygame window for rendering
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Map keyboard inputs to the action space
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
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        # Draw the laser beam for interactive mode
        if space_held:
            nearest, dist = env._get_nearest_asteroid_in_range()
            if nearest and dist <= env.LASER_RANGE:
                pygame.draw.aaline(surf, env.COLOR_LASER, env.ship_pos, nearest['pos'], 2)

        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(3000)
            obs, info = env.reset(seed=random.randint(0, 1_000_000))
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()