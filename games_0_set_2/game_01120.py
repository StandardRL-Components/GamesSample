
# Generated: 2025-08-27T16:06:25.475653
# Source Brief: brief_01120.md
# Brief Index: 1120

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to move. Hold space to fire your mining laser. Avoid asteroids!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship, mine asteroids for valuable ore, and survive the dangers of a dense asteroid field."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 100
        self.STARTING_LIVES = 3

        # Player settings
        self.PLAYER_SPEED = 6
        self.PLAYER_SIZE = 12
        self.PLAYER_FRICTION = 0.95

        # Asteroid settings
        self.INITIAL_SPAWN_RATE = 0.01
        self.MAX_SPAWN_RATE = 0.1
        self.SPAWN_RATE_INCREASE_INTERVAL = 500
        self.SPAWN_RATE_INCREASE_AMOUNT = 0.01

        # Mining laser settings
        self.BEAM_LENGTH = 150
        self.BEAM_DURATION = 3 # frames

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_BEAM = (100, 255, 255)
        self.COLOR_ASTEROID_LOW = (120, 120, 130)
        self.COLOR_ASTEROID_MED = (210, 180, 100)
        self.COLOR_ASTEROID_HIGH = (255, 100, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_EXPLOSION = (255, 150, 0)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.player_lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.asteroids = None
        self.particles = None
        self.stars = None
        self.mining_beam_timer = None
        self.spawn_rate = None
        self.np_random = None

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90  # Pointing up
        self.player_lives = self.STARTING_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.mining_beam_timer = 0
        self.spawn_rate = self.INITIAL_SPAWN_RATE

        self.asteroids = []
        self.particles = []
        self._generate_stars(200)

        # Initial asteroid population
        for _ in range(5):
            self._spawn_asteroid(random_pos=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused in this design

        reward = 0
        ore_collected_this_step = False

        # --- Handle Player Input and Movement ---
        self._handle_player_input(movement, space_held)
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel
        self._wrap_around_screen(self.player_pos)

        # --- Update Game State ---
        self._update_asteroids()
        self._update_particles()
        
        # --- Handle Collisions and Mining ---
        collision_reward, ore_collected = self._handle_collisions_and_mining()
        reward += collision_reward
        if ore_collected:
            ore_collected_this_step = True
            
        # --- Spawn new asteroids ---
        if self.np_random.random() < self.spawn_rate:
            self._spawn_asteroid()

        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % self.SPAWN_RATE_INCREASE_INTERVAL == 0:
            self.spawn_rate = min(self.MAX_SPAWN_RATE, self.spawn_rate + self.SPAWN_RATE_INCREASE_AMOUNT)

        # --- Calculate Step Reward ---
        if not ore_collected_this_step:
            reward -= 0.1

        # --- Check for Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.player_lives <= 0:
                reward -= 50 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement: 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up
            acceleration = pygame.Vector2(0, -1).rotate(-self.player_angle)
            self.player_vel += acceleration * (self.PLAYER_SPEED / self.FPS)
            # SFX: Thruster noise
            self._create_thruster_particles()
        if movement == 2: # Down (brakes)
             self.player_vel *= 0.90
        if movement == 3: # Left
            self.player_angle -= 5
        if movement == 4: # Right
            self.player_angle += 5
        
        if space_held:
            self.mining_beam_timer = self.BEAM_DURATION
            # SFX: Laser humming sound start
        
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            self._wrap_around_screen(asteroid['pos'])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if 'radius' in p:
                p['radius'] = max(0, p['radius'] - p['decay'])

    def _handle_collisions_and_mining(self):
        reward = 0
        ore_collected = False
        
        beam_active = self.mining_beam_timer > 0
        if beam_active:
            self.mining_beam_timer -= 1
        
        # Use a copy for safe removal during iteration
        for asteroid in self.asteroids[:]:
            # 1. Player-Asteroid Collision
            dist_to_player = self.player_pos.distance_to(asteroid['pos'])
            if dist_to_player < self.PLAYER_SIZE + asteroid['radius']:
                self._on_player_collision(asteroid)
                reward -= 10
                self.asteroids.remove(asteroid)
                continue # Skip to next asteroid

            # 2. Mining-Asteroid Interaction
            if beam_active:
                beam_end_pos = self.player_pos + pygame.Vector2(self.BEAM_LENGTH, 0).rotate(-self.player_angle)
                dist_to_beam = self._dist_point_to_segment(asteroid['pos'], self.player_pos, beam_end_pos)
                
                if dist_to_beam < asteroid['radius']:
                    # SFX: Mining impact sound
                    mined_ore = min(asteroid['ore'], 1) # Mine 1 ore per frame
                    asteroid['ore'] -= mined_ore
                    self.score += mined_ore
                    reward += mined_ore
                    ore_collected = True
                    self._create_mining_particles(asteroid)

                    if asteroid['ore'] <= 0:
                        # SFX: Asteroid destruction sound
                        self._create_explosion(asteroid['pos'], asteroid['color'], 20)
                        if asteroid['radius'] <= 15: # Small asteroid bonus
                            reward += 5
                        self.asteroids.remove(asteroid)
        
        return reward, ore_collected

    def _on_player_collision(self, asteroid):
        self.player_lives -= 1
        self.player_vel = pygame.Vector2(0, 0) # Stop the ship
        # SFX: Player explosion sound
        self._create_explosion(self.player_pos, self.COLOR_EXPLOSION, 30)

    def _check_termination(self):
        return self.player_lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])
            
        # Draw asteroids
        for asteroid in self.asteroids:
            pygame.gfxdraw.aacircle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), int(asteroid['radius']), asteroid['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), int(asteroid['radius']), asteroid['color'])

        # Draw particles
        for p in self.particles:
            if 'radius' in p: # Explosion particle
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])
            else: # Square particle
                pygame.draw.rect(self.screen, p['color'], (p['pos'].x, p['pos'].y, 2, 2))
        
        # Draw mining beam
        if self.mining_beam_timer > 0:
            beam_end_pos = self.player_pos + pygame.Vector2(self.BEAM_LENGTH, 0).rotate(-self.player_angle)
            alpha = int(255 * (self.mining_beam_timer / self.BEAM_DURATION))
            glow_color = (*self.COLOR_BEAM, alpha)
            
            # Draw a wide transparent line for glow
            pygame.draw.line(self.screen, glow_color, self.player_pos, beam_end_pos, 5)
            # Draw a thin bright line for core
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER, self.player_pos, beam_end_pos)

        # Draw player ship
        if self.player_lives > 0:
            self._draw_player()

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives display
        lives_text = self.font_ui.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "MISSION COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
        }

    # --- Helper and Effect Functions ---

    def _draw_player(self):
        # Create a rotated version of the ship polygon
        points = [
            pygame.Vector2(self.PLAYER_SIZE, 0),
            pygame.Vector2(-self.PLAYER_SIZE / 2, self.PLAYER_SIZE * 0.8),
            pygame.Vector2(-self.PLAYER_SIZE / 2, -self.PLAYER_SIZE * 0.8),
        ]
        rotated_points = [p.rotate(-self.player_angle) + self.player_pos for p in points]
        
        # Draw with antialiasing
        int_points = [(int(p.x), int(p.y)) for p in rotated_points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _generate_stars(self, num_stars):
        self.stars = []
        for _ in range(num_stars):
            size = self.np_random.choice([1, 2], p=[0.7, 0.3])
            brightness = self.np_random.integers(50, 150)
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': size,
                'color': (brightness, brightness, int(brightness*1.2)) # Bluish tint
            })

    def _spawn_asteroid(self, random_pos=False):
        ore = self.np_random.integers(5, 101)
        if ore > 50:
            color = self.COLOR_ASTEROID_HIGH
            radius = self.np_random.integers(20, 26)
        elif ore > 20:
            color = self.COLOR_ASTEROID_MED
            radius = self.np_random.integers(15, 21)
        else:
            color = self.COLOR_ASTEROID_LOW
            radius = self.np_random.integers(10, 16)
        
        if random_pos:
            pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
        else:
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), -radius)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.HEIGHT + radius)
            elif edge == 2: # Left
                pos = pygame.Vector2(-radius, self.np_random.integers(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + radius, self.np_random.integers(0, self.HEIGHT))
        
        angle = self.np_random.uniform(0, 360)
        speed = self.np_random.uniform(0.5, 1.5)
        vel = pygame.Vector2(speed, 0).rotate(angle)

        self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius, 'ore': ore, 'color': color})

    def _wrap_around_screen(self, pos_vector):
        pos_vector.x %= self.WIDTH
        pos_vector.y %= self.HEIGHT

    def _create_thruster_particles(self):
        if self.steps % 2 == 0: # Don't spawn every frame
            angle = self.player_angle + 180 + self.np_random.uniform(-15, 15)
            speed = self.np_random.uniform(2, 4)
            vel = pygame.Vector2(speed, 0).rotate(-angle) + self.player_vel * 0.5
            pos = self.player_pos - pygame.Vector2(self.PLAYER_SIZE, 0).rotate(-self.player_angle)
            self.particles.append({
                'pos': pos,
                'vel': vel,
                'lifespan': self.np_random.integers(8, 15),
                'color': self.COLOR_THRUSTER
            })

    def _create_mining_particles(self, asteroid):
        for _ in range(2):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                'pos': asteroid['pos'].copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 20),
                'color': asteroid['color']
            })

    def _create_explosion(self, position, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.integers(4, 8),
                'decay': 0.2
            })
            
    def _dist_point_to_segment(self, p, a, b):
        # p, a, and b are pygame.Vector2
        # Returns the perpendicular distance from point p to line segment ab
        if a == b:
            return p.distance_to(a)
        
        ab = b - a
        ap = p - a
        
        proj = ap.dot(ab)
        len_sq = ab.length_squared()
        
        d = proj / len_sq
        
        if d < 0:
            closest_point = a
        elif d > 1:
            closest_point = b
        else:
            closest_point = a + d * ab
            
        return p.distance_to(closest_point)

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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}. Press 'R' to restart.")
            
    env.close()