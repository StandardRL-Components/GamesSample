
# Generated: 2025-08-27T20:46:40.861030
# Source Brief: brief_02572.md
# Brief Index: 2572

        
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
        "Controls: ↑↓←→ to move. Hold space to fire your mining laser. Avoid asteroids!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, blast asteroids for ore, and avoid deadly collisions in a procedurally generated asteroid field. Collect 50 ore to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    WIN_ORE_COUNT = 50
    STARTING_LIVES = 3

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_SHIELD = (100, 255, 255, 100)
    COLOR_ASTEROID = [(160, 140, 120), (140, 120, 100), (120, 100, 80)]
    COLOR_ORE = (255, 220, 0)
    COLOR_LASER = (255, 100, 100)
    COLOR_EXPLOSION = [(255, 200, 50), (255, 100, 50), (200, 50, 50)]
    COLOR_UI_TEXT = (255, 255, 255)
    
    # Player
    PLAYER_SPEED = 6
    PLAYER_SIZE = 12
    INVULNERABILITY_FRAMES = 90 # 3 seconds at 30fps

    # Laser
    LASER_DURATION = 10
    LASER_COOLDOWN = 15
    LASER_WIDTH = 4
    LASER_RANGE = 300
    
    # Asteroids
    ASTEROID_SIZES = {'large': 40, 'medium': 25, 'small': 15}
    ASTEROID_SPEED_MAX = 2
    ASTEROID_ROTATION_SPEED_MAX = 0.03
    ASTEROID_MAX_COUNT = 20
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
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
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.player_lives = None
        self.player_invulnerable_timer = None
        self.ore_count = None
        
        self.asteroids = None
        self.ore_particles = None
        self.effects = None # For explosions and other particles
        
        self.laser_timer = None
        self.laser_cooldown_timer = None

        self.steps = None
        self.score = None
        self.game_over = None
        self.asteroid_spawn_rate = None
        self.starfield = None
        self.np_random = None

        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.player_angle = -math.pi / 2
        self.player_lives = self.STARTING_LIVES
        self.player_invulnerable_timer = self.INVULNERABILITY_FRAMES
        self.ore_count = 0
        
        self.asteroids = []
        self.ore_particles = []
        self.effects = []
        
        self.laser_timer = 0
        self.laser_cooldown_timer = 0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.asteroid_spawn_rate = 0.01

        self.starfield = [(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3)) for _ in range(100)]
        
        for _ in range(5):
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        self._handle_input(action)
        self._update_player()
        self._update_asteroids()
        self._update_ore()
        self._update_effects()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self._spawn_new_asteroids()
        self._cleanup()
        
        self.steps += 1
        self.asteroid_spawn_rate = min(0.1, 0.01 + (self.steps // 200) * 0.01)
        
        terminated = False
        if self.ore_count >= self.WIN_ORE_COUNT:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.player_lives <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Movement
        if movement == 1: # Up
            self.player_vel[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_vel[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_vel[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel[0] += self.PLAYER_SPEED
            
        # Firing laser
        if space_held and self.laser_cooldown_timer == 0:
            self.laser_timer = self.LASER_DURATION
            self.laser_cooldown_timer = self.LASER_COOLDOWN
            # sfx: laser_fire

    def _update_player(self):
        self.player_pos += self.player_vel * (1/self.FPS)
        self.player_vel *= 0.95 # friction
        
        # Clamp position
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Update timers
        if self.player_invulnerable_timer > 0:
            self.player_invulnerable_timer -= 1
        if self.laser_timer > 0:
            self.laser_timer -= 1
        if self.laser_cooldown_timer > 0:
            self.laser_cooldown_timer -= 1

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']

    def _update_ore(self):
        for ore in self.ore_particles:
            direction = self.player_pos - ore['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            ore['vel'] = ore['vel'] * 0.9 + direction * 1.5
            ore['pos'] += ore['vel']
            ore['life'] -= 1

    def _update_effects(self):
        for effect in self.effects:
            for particle in effect['particles']:
                particle['pos'] += particle['vel']
                particle['life'] -= 1
            effect['particles'] = [p for p in effect['particles'] if p['life'] > 0]
        self.effects = [e for e in self.effects if e['particles']]

    def _handle_collisions(self):
        reward = 0
        
        # Laser vs Asteroids
        if self.laser_timer > 0:
            laser_end = self.player_pos + np.array([self.LASER_RANGE, 0])
            hit_asteroids = []
            for i, asteroid in enumerate(self.asteroids):
                dist = np.linalg.norm(asteroid['pos'] - self.player_pos)
                if dist < self.LASER_RANGE + asteroid['radius']:
                    # Simple line-circle collision
                    if self._line_circle_collision(self.player_pos, laser_end, asteroid['pos'], asteroid['radius']):
                        hit_asteroids.append(i)
                        
            if hit_asteroids:
                # Hit the closest one
                closest_idx = min(hit_asteroids, key=lambda i: np.linalg.norm(self.asteroids[i]['pos'] - self.player_pos))
                asteroid = self.asteroids.pop(closest_idx)
                
                self._create_explosion(asteroid['pos'], asteroid['radius'])
                reward += 1.0 # sfx: explosion_small
                
                if asteroid['size'] == 'large':
                    self._spawn_asteroid(pos=asteroid['pos'].copy(), size='medium', count=2)
                elif asteroid['size'] == 'medium':
                    self._spawn_asteroid(pos=asteroid['pos'].copy(), size='small', count=2)
                elif asteroid['size'] == 'small':
                    for _ in range(self.np_random.integers(3, 6)):
                        self._spawn_ore(asteroid['pos'].copy())
                
                self.laser_timer = 0 # Laser is consumed on hit

        # Player vs Asteroids
        if self.player_invulnerable_timer == 0:
            for i in range(len(self.asteroids) - 1, -1, -1):
                asteroid = self.asteroids[i]
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < self.PLAYER_SIZE + asteroid['radius']:
                    self.player_lives -= 1
                    self.player_invulnerable_timer = self.INVULNERABILITY_FRAMES
                    self._create_explosion(asteroid['pos'], asteroid['radius'])
                    self.asteroids.pop(i)
                    # sfx: player_hit
                    # sfx: explosion_large
                    break # Only one collision per frame
                    
        # Player vs Ore
        for i in range(len(self.ore_particles) - 1, -1, -1):
            ore = self.ore_particles[i]
            dist = np.linalg.norm(self.player_pos - ore['pos'])
            if dist < self.PLAYER_SIZE + 5: # 5 is ore radius
                self.ore_count = min(self.WIN_ORE_COUNT, self.ore_count + 1)
                reward += 0.1
                self.ore_particles.pop(i)
                # sfx: ore_collect
                
        return reward

    def _spawn_new_asteroids(self):
        if self.np_random.random() < self.asteroid_spawn_rate and len(self.asteroids) < self.ASTEROID_MAX_COUNT:
            self._spawn_asteroid()

    def _cleanup(self):
        # Remove off-screen asteroids
        self.asteroids = [a for a in self.asteroids if -50 < a['pos'][0] < self.WIDTH + 50 and -50 < a['pos'][1] < self.HEIGHT + 50]
        # Remove expired ore
        self.ore_particles = [o for o in self.ore_particles if o['life'] > 0]

    def _spawn_asteroid(self, pos=None, size=None, count=1):
        for _ in range(count):
            if size is None:
                size = self.np_random.choice(['small', 'medium', 'large'], p=[0.5, 0.3, 0.2])
            
            radius = self.ASTEROID_SIZES[size]
            
            if pos is None:
                edge = self.np_random.integers(4)
                if edge == 0: # Top
                    spawn_pos = np.array([self.np_random.uniform(0, self.WIDTH), -radius])
                elif edge == 1: # Bottom
                    spawn_pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + radius])
                elif edge == 2: # Left
                    spawn_pos = np.array([-radius, self.np_random.uniform(0, self.HEIGHT)])
                else: # Right
                    spawn_pos = np.array([self.WIDTH + radius, self.np_random.uniform(0, self.HEIGHT)])
            else:
                spawn_pos = pos + self.np_random.uniform(-10, 10, size=2)

            angle = math.atan2(self.HEIGHT/2 - spawn_pos[1], self.WIDTH/2 - spawn_pos[0])
            speed = self.np_random.uniform(0.5, self.ASTEROID_SPEED_MAX)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            
            # Generate points for a semi-random polygon
            num_vertices = self.np_random.integers(7, 12)
            points = []
            for i in range(num_vertices):
                angle_offset = 2 * math.pi * i / num_vertices
                dist_offset = self.np_random.uniform(radius * 0.8, radius * 1.2)
                points.append((math.cos(angle_offset) * dist_offset, math.sin(angle_offset) * dist_offset))

            self.asteroids.append({
                'pos': spawn_pos,
                'vel': vel,
                'size': size,
                'radius': radius,
                'angle': 0,
                'rot_speed': self.np_random.uniform(-self.ASTEROID_ROTATION_SPEED_MAX, self.ASTEROID_ROTATION_SPEED_MAX),
                'points': points,
                'color': self.np_random.choice(self.COLOR_ASTEROID)
            })

    def _spawn_ore(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        self.ore_particles.append({
            'pos': pos.copy(),
            'vel': np.array([math.cos(angle), math.sin(angle)]) * speed,
            'life': self.np_random.integers(150, 250)
        })

    def _create_explosion(self, pos, radius):
        num_particles = int(radius * 1.5)
        particles = []
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            particles.append({
                'pos': pos.copy() + self.np_random.uniform(-5, 5, size=2),
                'vel': np.array([math.cos(angle), math.sin(angle)]) * speed,
                'life': self.np_random.integers(15, 30),
                'color': self.np_random.choice(self.COLOR_EXPLOSION)
            })
        self.effects.append({'particles': particles})

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x, y, size in self.starfield:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # --- Game Objects ---
        for ore in self.ore_particles:
            pygame.gfxdraw.filled_circle(self.screen, int(ore['pos'][0]), int(ore['pos'][1]), 4, self.COLOR_ORE)
            pygame.gfxdraw.aacircle(self.screen, int(ore['pos'][0]), int(ore['pos'][1]), 4, self.COLOR_ORE)

        for asteroid in self.asteroids:
            points_rotated = []
            for px, py in asteroid['points']:
                x_rot = px * math.cos(asteroid['angle']) - py * math.sin(asteroid['angle'])
                y_rot = px * math.sin(asteroid['angle']) + py * math.cos(asteroid['angle'])
                points_rotated.append((int(asteroid['pos'][0] + x_rot), int(asteroid['pos'][1] + y_rot)))
            if len(points_rotated) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points_rotated, asteroid['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points_rotated, asteroid['color'])
        
        # --- Player ---
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        p1 = (px + self.PLAYER_SIZE, py)
        p2 = (px - self.PLAYER_SIZE // 2, py - self.PLAYER_SIZE // 2)
        p3 = (px - self.PLAYER_SIZE // 2, py + self.PLAYER_SIZE // 2)
        player_points = [p1, p2, p3]
        
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

        if self.player_invulnerable_timer > 0 and self.steps % 10 < 5:
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_SIZE + 5, self.COLOR_PLAYER_SHIELD)

        # --- Laser ---
        if self.laser_timer > 0:
            end_pos = (self.player_pos[0] + self.LASER_RANGE, self.player_pos[1])
            pygame.draw.line(self.screen, self.COLOR_LASER, (px, py), end_pos, self.LASER_WIDTH)

        # --- Effects ---
        for effect in self.effects:
            for p in effect['particles']:
                size = int(p['life'] / 6)
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)
        
        # --- UI ---
        ore_text = self.font_small.render(f"ORE: {self.ore_count}/{self.WIN_ORE_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ore_text, (10, 10))

        lives_text = self.font_small.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.ore_count >= self.WIN_ORE_COUNT else "GAME OVER"
            color = self.COLOR_PLAYER if self.ore_count >= self.WIN_ORE_COUNT else self.COLOR_LASER
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore": self.ore_count,
            "lives": self.player_lives,
        }
        
    def _line_circle_collision(self, p1, p2, circle_center, circle_radius):
        # Simplified for horizontal laser
        cy, cr = circle_center[1], circle_radius
        if p1[1] > cy + cr or p1[1] < cy - cr:
            return False # Laser is not vertically aligned with circle
        
        cx = circle_center[0]
        if cx < p1[0] - cr or cx > p2[0] + cr:
            return False # Circle is completely to the left or right
            
        # Check if circle center is within laser's x-range
        if p1[0] <= cx <= p2[0]:
            return True
        
        # Check endpoints
        if np.linalg.norm(p1 - circle_center) < cr or np.linalg.norm(p2 - circle_center) < cr:
            return True
            
        return False

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires setting up a pygame screen to display the frames.
    # The environment itself is headless.
    
    try:
        import sys
        
        # For manual play, we need a display
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Asteroid Miner")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print("Starting manual play.")
        print(GameEnv.user_guide)

        while not done:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()

            clock.tick(GameEnv.FPS)

    except ImportError:
        print("Pygame required for manual play.")
    except Exception as e:
        print(f"An error occurred during manual play: {e}")
    finally:
        env.close()
        print("Environment closed.")