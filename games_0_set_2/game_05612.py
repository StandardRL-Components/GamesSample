import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship to mine asteroids for ore. Collect 100 units to win, but be careful: colliding with an asteroid will destroy your ship!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_ASTEROID = (120, 120, 130)
        self.COLOR_ORE = (255, 220, 0)
        self.COLOR_BEAM = (100, 255, 100)
        self.COLOR_EXPLOSION = [(255, 80, 0), (255, 180, 0), (255, 255, 100)]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BAR = (50, 60, 80)
        self.COLOR_UI_BAR_FILL = (255, 220, 0)
        self.STAR_COLORS = [(100,100,120), (150,150,180), (200,200,220)]


        # Game parameters
        self.PLAYER_SPEED = 7
        self.PLAYER_SIZE = 12
        self.ASTEROID_BASE_SPEED = 2.0
        self.ASTEROID_SPEED_INCREASE_INTERVAL = 500
        self.ASTEROID_SPEED_INCREASE_AMOUNT = 0.05
        self.ASTEROID_SPAWN_RATE = 25  # Lower is faster
        self.MAX_ASTEROIDS = 15
        self.MINING_RADIUS = 100
        self.MINING_COOLDOWN = 10 # frames
        self.WIN_ORE_TARGET = 100
        self.MAX_STEPS = 5000

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Etc...        
        self.player_pos = None
        self.asteroids = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.ore_collected = None
        self.game_over = None
        self.asteroid_speed = None
        self.asteroid_spawn_timer = None
        self.mining_cooldown_timer = None
        self.np_random = None
        
        # Initialize state variables
        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed)
        else:
             self.np_random = np.random.default_rng()

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ore_collected = 0
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32)
        self.asteroids = []
        self.particles = []
        
        self.asteroid_speed = self.ASTEROID_BASE_SPEED
        self.asteroid_spawn_timer = 0
        self.mining_cooldown_timer = 0

        self._generate_stars()
        for _ in range(5): # Start with a few asteroids
            self._spawn_asteroid(random_y=True)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if self.game_over:
            # If game is over, do nothing but allow the agent to see the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        # --- Update Timers ---
        self.steps += 1
        if self.mining_cooldown_timer > 0:
            self.mining_cooldown_timer -= 1

        # --- Handle Movement ---
        if movement == 0: # No-op
            reward -= 0.2
        elif movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # --- Handle Mining ---
        mined_asteroid_pos = None
        if space_held and self.mining_cooldown_timer == 0:
            nearest_asteroid, min_dist = self._find_nearest_asteroid()
            if nearest_asteroid is not None and min_dist < self.MINING_RADIUS:
                # SFX: Mining laser activate
                self.mining_cooldown_timer = self.MINING_COOLDOWN
                
                ore_yield, size_reward = {
                    'small': (5, 1), 'medium': (15, 2), 'large': (30, 5)
                }[nearest_asteroid['type']]

                reward += size_reward + ore_yield
                self.ore_collected = min(self.ore_collected + ore_yield, self.WIN_ORE_TARGET)

                mined_asteroid_pos = nearest_asteroid['pos'].copy()
                self._create_ore_particles(mined_asteroid_pos)
                self.asteroids.remove(nearest_asteroid)
                # SFX: Asteroid chunk collected

        # --- Update Asteroids ---
        self.asteroid_spawn_timer += 1
        if self.asteroid_spawn_timer > self.ASTEROID_SPAWN_RATE and len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = 0
        
        if self.steps > 0 and self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
            self.asteroid_speed += self.ASTEROID_SPEED_INCREASE_AMOUNT

        for asteroid in self.asteroids[:]:
            asteroid['pos'][1] += self.asteroid_speed
            if asteroid['pos'][1] > self.HEIGHT + asteroid['radius']:
                self.asteroids.remove(asteroid)

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Check Collisions ---
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['radius']:
                self.game_over = True
                terminated = True
                reward = -50  # Terminal penalty
                self._create_explosion()
                # SFX: Player explosion
                break
        
        # --- Check Win/Loss Conditions ---
        if self.ore_collected >= self.WIN_ORE_TARGET and not self.game_over:
            self.game_over = True
            terminated = True
            reward = 100 # Terminal bonus
            # SFX: Victory fanfare
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True # Game ends, so it's also terminated
            terminated = True


        self.score += reward
        obs = self._get_observation(movement, mined_asteroid_pos)
        
        # MUST return exactly this 5-tuple
        return (
            obs,
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_observation(self, movement=0, mined_asteroid_pos=None):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_stars()
        self._render_asteroids()
        if not self.game_over:
            self._render_player(movement)
            if mined_asteroid_pos is not None:
                self._render_mining_beam(mined_asteroid_pos)
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore": self.ore_collected,
        }

    # --- Helper and Rendering Methods ---

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.integers(1, 3),
                'color': self.STAR_COLORS[self.np_random.integers(len(self.STAR_COLORS))]
            })

    def _spawn_asteroid(self, random_y=False):
        size_choice = self.np_random.choice(['small', 'medium', 'large'], p=[0.6, 0.3, 0.1])
        radius, points = {
            'small': (15, 7), 'medium': (25, 9), 'large': (40, 11)
        }[size_choice]

        y_pos = -radius if not random_y else self.np_random.integers(0, self.HEIGHT // 2)
        
        asteroid = {
            'pos': np.array([self.np_random.integers(radius, self.WIDTH - radius), y_pos], dtype=np.float32),
            'radius': radius,
            'type': size_choice,
            'shape': self._generate_asteroid_shape(radius, points),
            'angle': 0,
            'rot_speed': self.np_random.uniform(-0.02, 0.02)
        }
        self.asteroids.append(asteroid)

    def _generate_asteroid_shape(self, radius, num_points):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radii = radius + self.np_random.uniform(-radius*0.3, radius*0.3, num_points)
        points = [ (r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles) ]
        return points

    def _find_nearest_asteroid(self):
        if not self.asteroids:
            return None, float('inf')
        
        min_dist = float('inf')
        nearest_asteroid = None
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest_asteroid = asteroid
        return nearest_asteroid, min_dist

    def _create_ore_particles(self, start_pos):
        # SFX: Ore particle collect
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * np.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([np.cos(angle), np.sin(angle)]) * speed
            self.particles.append({
                'pos': start_pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'color': self.COLOR_ORE,
                'type': 'ore'
            })

    def _create_explosion(self):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * np.pi)
            speed = self.np_random.uniform(2, 8)
            vel = np.array([np.cos(angle), np.sin(angle)]) * speed
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(30, 60),
                'color': self.COLOR_EXPLOSION[self.np_random.integers(len(self.COLOR_EXPLOSION))],
                'type': 'explosion'
            })

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'] // 2)

    def _render_player(self, movement):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        s = self.PLAYER_SIZE
        
        points = [(x, y - s), (x - s // 1.5, y + s // 2), (x + s // 1.5, y + s // 2)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        if movement == 1:
            tx, ty = x, y + s // 2 + 2
            ts = s // 2
            flare_points = [(tx, ty + ts), (tx - ts // 1.5, ty), (tx + ts // 1.5, ty)]
            pygame.gfxdraw.aapolygon(self.screen, flare_points, self.COLOR_THRUSTER)
            pygame.gfxdraw.filled_polygon(self.screen, flare_points, self.COLOR_THRUSTER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rot_speed']
            cos_a, sin_a = np.cos(asteroid['angle']), np.sin(asteroid['angle'])
            
            points = []
            for px, py in asteroid['shape']:
                rot_x = px * cos_a - py * sin_a
                rot_y = px * sin_a + py * cos_a
                points.append((int(rot_x + asteroid['pos'][0]), int(rot_y + asteroid['pos'][1])))
            
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                outline_color = tuple(int(min(255, c * 1.2)) for c in self.COLOR_ASTEROID[:3])
                pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = max(1, int(p['life'] / 10))
            if p['type'] == 'explosion':
                 pygame.draw.circle(self.screen, p['color'], pos, size)
            elif p['type'] == 'ore':
                 pygame.draw.circle(self.screen, p['color'], pos, 2)

    def _render_mining_beam(self, target_pos):
        start_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        end_pos = (int(target_pos[0]), int(target_pos[1]))
        width = int(self.mining_cooldown_timer / 2)
        if width > 0:
            pygame.draw.line(self.screen, self.COLOR_BEAM, start_pos, end_pos, width)

    def _render_ui(self):
        bar_width, bar_height, bar_x, bar_y = 200, 20, 10, 10
        fill_ratio = self.ore_collected / self.WIN_ORE_TARGET
        fill_width = int(bar_width * fill_ratio)

        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        ore_text = f"ORE: {self.ore_collected} / {self.WIN_ORE_TARGET}"
        text_surf = self.font_small.render(ore_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (bar_x + bar_width + 10, bar_y))

        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

        if self.game_over:
            msg = "MISSION COMPLETE" if self.ore_collected >= self.WIN_ORE_TARGET else "GAME OVER"
            end_surf = self.font_large.render(msg, True, self.COLOR_PLAYER)
            end_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    
    # To run with display, you need to instantiate GameEnv without render_mode
    # and unset the SDL_VIDEODRIVER dummy variable.
    # For this example, we'll just show the rgb_array output.
    
    pygame.display.init()
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    print(env.user_guide)
    
    while True:
        movement, space_held, shift_held = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
        
        # Key presses for interactive play
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Ore: {info['ore']}")
            pygame.time.wait(3000)
            obs, info = env.reset()