
# Generated: 2025-08-27T14:42:52.824109
# Source Brief: brief_00768.md
# Brief Index: 768

        
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
    """
    An arcade-style, top-down spaceship game where the player pilots a mining ship
    to collect crystals from asteroids while dodging hazardous lasers. The game
    features vector graphics, particle effects, and a progressive difficulty curve.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to thrust, ←→ to turn, and ↓ to brake. "
        "Hold Space to activate the mining laser. Collect 50 crystals to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, dodge enemy lasers, and collect crystals in a "
        "procedurally generated asteroid field. A fast-paced arcade shooter."
    )

    # Frames auto-advance for smooth real-time gameplay.
    auto_advance = True
    
    # --- Constants ---
    # Game world
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_SCORE = 50
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_SHIP = (255, 255, 255)
    COLOR_ASTEROID = (120, 120, 130)
    COLOR_LASER = (255, 50, 50)
    COLOR_CRYSTAL = (50, 150, 255)
    COLOR_EXPLOSION = (255, 150, 50)
    COLOR_THRUSTER = (255, 200, 100)
    COLOR_MINING_BEAM = (100, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)

    # Player ship
    SHIP_SIZE = 12
    SHIP_THRUST = 0.4
    SHIP_TURN_RATE = 6
    SHIP_DRAG = 0.97
    SHIP_BRAKE_DRAG = 0.92
    
    # Lasers
    INITIAL_LASER_FREQUENCY = 0.1 # Lasers per second
    LASER_FREQUENCY_INCREASE = 0.005 # Increase per second
    LASER_SPEED = 5
    LASER_PROXIMITY_THRESHOLD = 50

    # Asteroids
    ASTEROID_SPAWN_INTERVAL = 7 # in seconds
    INITIAL_ASTEROIDS = 8
    
    # Mining
    MINING_BEAM_LENGTH = 150
    MINING_DAMAGE = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 64)
        
        self.render_mode = render_mode
        self.np_random = None

        # These attributes are defined in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.is_mining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.laser_frequency = None
        self.laser_spawn_timer = None
        self.asteroid_spawn_timer = None
        self.stars = None
        self.asteroids = None
        self.lasers = None
        self.crystals = None
        self.particles = None
        self.step_reward = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90  # Pointing up
        self.is_mining = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.laser_frequency = self.INITIAL_LASER_FREQUENCY
        self.laser_spawn_timer = 0
        self.asteroid_spawn_timer = 0

        self.stars = self._generate_stars()
        self.asteroids = [self._create_asteroid() for _ in range(self.INITIAL_ASTEROIDS)]
        self.lasers = []
        self.crystals = []
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.step_reward = 0
        
        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_spawners()
            self._update_lasers()
            self._update_asteroids()
            self._update_crystals()
        
        self._update_particles()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        if terminated and not self.game_over: # Win condition
            self.win = True
            reward += 100
        elif terminated and self.game_over: # Lose condition
            reward -= 100

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Turning
        if movement == 3:  # Left
            self.player_angle -= self.SHIP_TURN_RATE
        if movement == 4:  # Right
            self.player_angle += self.SHIP_TURN_RATE

        # Thrust / Brake
        if movement == 1:  # Up
            thrust_vec = pygame.math.Vector2(math.cos(math.radians(self.player_angle)), 
                                             math.sin(math.radians(self.player_angle)))
            self.player_vel += thrust_vec * self.SHIP_THRUST
            self._create_thruster_particles()
        elif movement == 2: # Down
            self.player_vel *= self.SHIP_BRAKE_DRAG
        
        # Mining
        self.is_mining = (space_held == 1)

    def _update_player(self):
        self.player_vel *= self.SHIP_DRAG
        self.player_pos += self.player_vel

        # Screen wrapping
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT

    def _update_spawners(self):
        # Laser spawner
        time_delta = 1 / self.FPS
        self.laser_spawn_timer += time_delta
        self.laser_frequency += self.LASER_FREQUENCY_INCREASE * time_delta
        
        if self.laser_spawn_timer * self.laser_frequency >= 1.0:
            self.lasers.append(self._create_laser())
            self.laser_spawn_timer = 0
            
        # Asteroid spawner
        self.asteroid_spawn_timer += time_delta
        if self.asteroid_spawn_timer >= self.ASTEROID_SPAWN_INTERVAL:
            self.asteroids.append(self._create_asteroid(on_edge=True))
            self.asteroid_spawn_timer = 0

    def _update_lasers(self):
        for laser in self.lasers[:]:
            laser['pos'] += laser['vel']
            if not self.screen.get_rect().colliderect(pygame.Rect(laser['pos'].x, laser['pos'].y, 1, 1)):
                self.lasers.remove(laser)
                continue
            
            # Check collision with player
            if self.player_pos.distance_to(laser['pos']) < self.SHIP_SIZE:
                self.game_over = True
                # sfx: player_explosion
                self._create_explosion(self.player_pos, self.COLOR_EXPLOSION, 50)
                break

    def _update_asteroids(self):
        if not self.is_mining:
            return
            
        beam_start = self.player_pos
        beam_end = self.player_pos + pygame.math.Vector2(self.MINING_BEAM_LENGTH, 0).rotate(self.player_angle)

        for asteroid in self.asteroids[:]:
            # Simple line-circle intersection
            dist_to_beam = self._dist_point_to_segment(asteroid['pos'], beam_start, beam_end)
            
            if dist_to_beam < asteroid['radius']:
                asteroid['health'] -= self.MINING_DAMAGE
                # sfx: mining_hit
                self._create_explosion(asteroid['pos'], self.COLOR_ASTEROID, 3, 0.5)
                if asteroid['health'] <= 0:
                    self.step_reward += 1 # Reward for destroying asteroid
                    # sfx: asteroid_destroyed
                    self._create_explosion(asteroid['pos'], self.COLOR_ASTEROID, 20)
                    for _ in range(self.np_random.integers(2, 5)):
                        self.crystals.append(self._create_crystal(asteroid['pos']))
                    self.asteroids.remove(asteroid)

    def _update_crystals(self):
        for crystal in self.crystals[:]:
            crystal['anim_timer'] += 1
            if self.player_pos.distance_to(crystal['pos']) < self.SHIP_SIZE + 5:
                # sfx: crystal_pickup
                self.crystals.remove(crystal)
                self.score = min(self.WIN_SCORE, self.score + 1)
                self.step_reward += 0.1 # Reward for collecting crystal

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        # Proximity penalty
        is_near_laser = False
        for laser in self.lasers:
            if self.player_pos.distance_to(laser['pos']) < self.LASER_PROXIMITY_THRESHOLD:
                is_near_laser = True
                break
        if is_near_laser:
            self.step_reward -= 0.01
            
        return self.step_reward

    def _check_termination(self):
        return self.game_over or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_crystals()
        self._render_asteroids()
        self._render_lasers()
        if not self.game_over:
            self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Rendering Methods ---

    def _render_stars(self):
        for star in self.stars:
            star['pos'].x = (star['pos'].x - star['speed']) % self.WIDTH
            size = int(star['speed'])
            pygame.draw.rect(self.screen, star['color'], (int(star['pos'].x), int(star['pos'].y), size, size))

    def _render_player(self):
        # Ship body
        angle_rad = math.radians(self.player_angle)
        p1 = self.player_pos + pygame.math.Vector2(self.SHIP_SIZE, 0).rotate(self.player_angle)
        p2 = self.player_pos + pygame.math.Vector2(-self.SHIP_SIZE / 2, self.SHIP_SIZE * 0.75).rotate(self.player_angle)
        p3 = self.player_pos + pygame.math.Vector2(-self.SHIP_SIZE / 2, -self.SHIP_SIZE * 0.75).rotate(self.player_angle)
        points = [(int(p.x), int(p.y)) for p in [p1, p2, p3]]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SHIP)
        
        # Mining beam
        if self.is_mining:
            beam_end = self.player_pos + pygame.math.Vector2(self.MINING_BEAM_LENGTH, 0).rotate(self.player_angle)
            color = self.COLOR_MINING_BEAM + (100 + self.np_random.integers(0, 100),) # Alpha
            pygame.draw.line(self.screen, color, self.player_pos, beam_end, 3)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [asteroid['pos'] + p.rotate(asteroid['angle']) for p in asteroid['shape']]
            int_points = [(int(p.x), int(p.y)) for p in points]
            if len(int_points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ASTEROID)

    def _render_lasers(self):
        for laser in self.lasers:
            end_pos = laser['pos'] - laser['vel'] * 3
            pygame.draw.line(self.screen, self.COLOR_LASER, laser['pos'], end_pos, 3)

    def _render_crystals(self):
        for crystal in self.crystals:
            size = 6 + math.sin(crystal['anim_timer'] * 0.2) * 2
            p1 = (crystal['pos'].x, crystal['pos'].y - size)
            p2 = (crystal['pos'].x + size, crystal['pos'].y)
            p3 = (crystal['pos'].x, crystal['pos'].y + size)
            p4 = (crystal['pos'].x - size, crystal['pos'].y)
            points = [p1, p2, p3, p4]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        score_text = self.font.render(f"Crystals: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            if self.win:
                end_text = self.font_large.render("VICTORY", True, self.COLOR_CRYSTAL)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_LASER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    # --- Entity Creation & Helpers ---

    def _generate_stars(self):
        stars = []
        for _ in range(100):
            speed = self.np_random.uniform(0.2, 1.5)
            color_val = int(100 + 100 * (speed / 1.5))
            stars.append({
                'pos': pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'speed': speed,
                'color': (color_val, color_val, color_val)
            })
        return stars

    def _create_asteroid(self, on_edge=False):
        if on_edge:
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = pygame.math.Vector2(-30, self.np_random.uniform(0, self.HEIGHT))
            elif edge == 1: pos = pygame.math.Vector2(self.WIDTH + 30, self.np_random.uniform(0, self.HEIGHT))
            elif edge == 2: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -30)
            else: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 30)
        else:
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))

        radius = self.np_random.uniform(15, 30)
        num_points = self.np_random.integers(7, 12)
        shape = []
        for i in range(num_points):
            angle = (360 / num_points) * i
            dist = self.np_random.uniform(radius * 0.7, radius * 1.3)
            shape.append(pygame.math.Vector2(dist, 0).rotate(angle))
            
        return {'pos': pos, 'radius': radius, 'health': radius * 5, 'shape': shape, 'angle': self.np_random.uniform(0, 360)}

    def _create_laser(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Left
            pos = pygame.math.Vector2(-10, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.math.Vector2(self.LASER_SPEED, 0)
        elif edge == 1: # Right
            pos = pygame.math.Vector2(self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.math.Vector2(-self.LASER_SPEED, 0)
        elif edge == 2: # Top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -10)
            vel = pygame.math.Vector2(0, self.LASER_SPEED)
        else: # Bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10)
            vel = pygame.math.Vector2(0, -self.LASER_SPEED)
        return {'pos': pos, 'vel': vel}

    def _create_crystal(self, pos):
        offset = pygame.math.Vector2(self.np_random.uniform(-10, 10), self.np_random.uniform(-10, 10))
        return {'pos': pos + offset, 'anim_timer': self.np_random.uniform(0, 10)}

    def _create_explosion(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.math.Vector2(speed, 0).rotate(angle)
            lifespan = self.np_random.integers(15, 40)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan,
                'max_lifespan': lifespan, 'color': color, 'size': self.np_random.uniform(2, 5)
            })

    def _create_thruster_particles(self):
        if self.np_random.random() < 0.8: # Don't spawn every frame
            angle = self.player_angle + 180 + self.np_random.uniform(-15, 15)
            speed = self.np_random.uniform(1, 2)
            vel = pygame.math.Vector2(speed, 0).rotate(angle) + self.player_vel * 0.5
            lifespan = self.np_random.integers(10, 20)
            start_pos = self.player_pos - pygame.math.Vector2(self.SHIP_SIZE, 0).rotate(self.player_angle)
            self.particles.append({
                'pos': start_pos, 'vel': vel, 'lifespan': lifespan,
                'max_lifespan': lifespan, 'color': self.COLOR_THRUSTER, 'size': self.np_random.uniform(1, 3)
            })

    def _dist_point_to_segment(self, p, a, b):
        # p, a, b are pygame.math.Vector2
        ab = b - a
        ap = p - a
        l2 = ab.length_squared()
        if l2 == 0:
            return ap.length()
        t = max(0, min(1, ap.dot(ab) / l2))
        projection = a + t * ab
        return p.distance_to(projection)
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This allows a human to play the game.
    # The agent's actions are determined by keyboard inputs.
    
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()

    env.close()