
# Generated: 2025-08-28T06:25:26.915446
# Source Brief: brief_05898.md
# Brief Index: 5898

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to apply thrust. Hold Space to mine asteroids."
    )

    game_description = (
        "Mine asteroids for ore in a top-down arcade space miner. "
        "Collect 100 ore to win, but avoid collisions or your ship will be destroyed."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500
        self.WIN_SCORE = 100
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (150, 255, 200, 50)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_ORE = (255, 220, 0)
        self.COLOR_LASER = (255, 100, 100)
        self.COLOR_UI = (220, 220, 240)
        self.ASTEROID_COLORS = {
            'low': (100, 90, 80),
            'med': (140, 120, 110),
            'high': (180, 150, 140)
        }
        
        # Player Physics
        self.PLAYER_THRUST = 0.2
        self.PLAYER_FRICTION = 0.98
        self.PLAYER_MAX_SPEED = 5
        self.PLAYER_RADIUS = 10

        # Game Mechanics
        self.INITIAL_ASTEROIDS = 5
        self.MAX_ASTEROIDS = 15
        self.DIFFICULTY_INTERVAL = 200
        self.DIFFICULTY_SCALING = 0.1
        self.MINING_RANGE = 80
        self.MINING_ANGLE = 0.4 # radians
        self.MINING_RATE = 1 # ore per frame

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 64)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans-serif", 24)
            self.font_game_over = pygame.font.SysFont("sans-serif", 64)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.ore = 0
        self.game_over = False
        self.np_random = None
        
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        
        self.asteroids = None
        self.particles = None
        self.explosions = None
        
        self.asteroid_base_speed = 0.5
        self.is_mining = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.ore = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_angle = -math.pi / 2  # Pointing up

        self.asteroids = deque()
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid()

        self.particles = deque()
        self.explosions = deque()

        self.asteroid_base_speed = 0.5
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.is_mining = False

        # --- Action Handling ---
        movement, space_held, _ = action
        
        thrust_vec = np.array([0.0, 0.0])
        if movement != 0:
            if movement == 1: # Up
                thrust_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
            elif movement == 2: # Down (brake)
                thrust_vec = -self.player_vel * 0.1
            elif movement == 3: # Left
                self.player_angle -= 0.1
            elif movement == 4: # Right
                self.player_angle += 0.1
            
            if movement in [1, 2]:
                self.player_vel += thrust_vec * self.PLAYER_THRUST
                # sound: thruster_loop.wav

        self.is_mining = space_held == 1

        # --- Game Logic ---
        if not self.game_over:
            self._update_player(movement)
            self._update_asteroids()
            
            if self.is_mining:
                reward += self._handle_mining()
                # sound: mining_laser.wav
            else:
                reward -= 0.001 # Small penalty for passivity

            reward += self._check_collisions()
            self._update_particles()
            self._update_difficulty()
            
            # Replenish asteroids
            while len(self.asteroids) < self.MAX_ASTEROIDS:
                self._spawn_asteroid()

        self._update_explosions()
        self.steps += 1
        
        # --- Termination ---
        terminated = self.game_over
        if self.ore >= self.WIN_SCORE:
            if not self.game_over: # First time reaching win score
                reward += 100
                # sound: win_jingle.wav
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement_action):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED

        # Update position
        self.player_pos += self.player_vel

        # Screen wrap
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
        
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['pos'][0] %= self.WIDTH
            asteroid['pos'][1] %= self.HEIGHT

    def _handle_mining(self):
        reward = 0
        mined_something = False
        ship_front = self.player_pos + np.array([math.cos(self.player_angle), math.sin(self.player_angle)]) * self.PLAYER_RADIUS

        for asteroid in list(self.asteroids):
            dist_vec = asteroid['pos'] - self.player_pos
            dist = np.linalg.norm(dist_vec)

            if dist < self.MINING_RANGE + asteroid['radius']:
                angle_to_asteroid = math.atan2(dist_vec[1], dist_vec[0])
                angle_diff = abs((angle_to_asteroid - self.player_angle + math.pi) % (2 * math.pi) - math.pi)
                
                if angle_diff < self.MINING_ANGLE:
                    mined_something = True
                    # Mine the asteroid
                    mined_amount = min(self.MINING_RATE, asteroid['ore'])
                    asteroid['ore'] -= mined_amount
                    self.ore += mined_amount
                    reward += 0.1 * mined_amount
                    
                    # Create ore particle
                    particle_vel = (self.player_pos - asteroid['pos']) / 30.0
                    self.particles.append({
                        'pos': asteroid['pos'].copy(),
                        'vel': particle_vel,
                        'lifetime': 30,
                        'color': self.COLOR_ORE,
                        'radius': 3
                    })

                    if asteroid['ore'] <= 0:
                        reward += 1.0 # Bonus for destroying asteroid
                        self.asteroids.remove(asteroid)
                        # sound: asteroid_destroyed.wav
        
        return reward

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                self.game_over = True
                self.explosions.append({'pos': self.player_pos.copy(), 'radius': 10, 'max_radius': 60, 'lifetime': 30})
                # sound: player_explosion.wav
                return -100 # Large negative reward for dying
        return 0

    def _update_particles(self):
        for p in list(self.particles):
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _update_explosions(self):
        for e in list(self.explosions):
            e['radius'] += e['max_radius'] / e['lifetime']
            e['lifetime'] -= 1
            if e['lifetime'] <= 0:
                self.explosions.remove(e)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.asteroid_base_speed += self.DIFFICULTY_SCALING

    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -30.0])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 30.0])
        elif edge == 2: # Left
            pos = np.array([-30.0, self.np_random.uniform(0, self.HEIGHT)])
        else: # Right
            pos = np.array([self.WIDTH + 30.0, self.np_random.uniform(0, self.HEIGHT)])

        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 1.5) * self.asteroid_base_speed
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        
        size_roll = self.np_random.random()
        if size_roll < 0.5:
            radius = self.np_random.uniform(10, 15)
            ore = self.np_random.integers(10, 20)
            color_key = 'low'
        elif size_roll < 0.85:
            radius = self.np_random.uniform(15, 25)
            ore = self.np_random.integers(20, 40)
            color_key = 'med'
        else:
            radius = self.np_random.uniform(25, 35)
            ore = self.np_random.integers(40, 70)
            color_key = 'high'

        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'radius': radius,
            'ore': ore,
            'max_ore': ore,
            'color_key': color_key
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_particles()
        if not self.game_over:
            self._render_player()
        self._render_explosions()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        # Deterministic stars based on a seed
        star_seed = 12345
        rand = random.Random(star_seed)
        for _ in range(100):
            x = rand.randint(0, self.WIDTH)
            y = rand.randint(0, self.HEIGHT)
            brightness = rand.randint(50, 150)
            self.screen.set_at((x, y), (brightness, brightness, brightness))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = asteroid['pos'].astype(int)
            radius = int(asteroid['radius'])
            
            # Interpolate color based on remaining ore
            base_color = self.ASTEROID_COLORS[asteroid['color_key']]
            ore_ratio = max(0, asteroid['ore']) / asteroid['max_ore']
            color = tuple(int(c * (0.5 + 0.5 * ore_ratio)) for c in base_color)

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_player(self):
        pos = self.player_pos
        angle = self.player_angle
        
        # Main ship body
        p1 = (pos[0] + math.cos(angle) * self.PLAYER_RADIUS, pos[1] + math.sin(angle) * self.PLAYER_RADIUS)
        p2 = (pos[0] + math.cos(angle + 2.2) * self.PLAYER_RADIUS, pos[1] + math.sin(angle + 2.2) * self.PLAYER_RADIUS)
        p3 = (pos[0] + math.cos(angle - 2.2) * self.PLAYER_RADIUS, pos[1] + math.sin(angle - 2.2) * self.PLAYER_RADIUS)
        points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
        
        # Glow effect
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        
        # Ship
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Thruster flame
        speed_ratio = np.linalg.norm(self.player_vel) / self.PLAYER_MAX_SPEED
        if speed_ratio > 0.1:
            flame_len = speed_ratio * 15
            f1 = (pos[0] - math.cos(angle) * self.PLAYER_RADIUS * 0.8, pos[1] - math.sin(angle) * self.PLAYER_RADIUS * 0.8)
            f2 = (pos[0] + math.cos(angle + 2.5) * self.PLAYER_RADIUS * 0.6, pos[1] + math.sin(angle + 2.5) * self.PLAYER_RADIUS * 0.6)
            f3 = (pos[0] + math.cos(angle - 2.5) * self.PLAYER_RADIUS * 0.6, pos[1] + math.sin(angle - 2.5) * self.PLAYER_RADIUS * 0.6)
            f_tip = (f1[0] - math.cos(angle) * flame_len, f1[1] - math.sin(angle) * flame_len)
            flame_points = [(int(p[0]), int(p[1])) for p in [f2, f3, f_tip]]
            pygame.gfxdraw.aapolygon(self.screen, flame_points, self.COLOR_THRUSTER)
            pygame.gfxdraw.filled_polygon(self.screen, flame_points, self.COLOR_THRUSTER)

        # Mining laser
        if self.is_mining:
            start_pos = (p1[0], p1[1])
            end_pos = (p1[0] + math.cos(angle) * self.MINING_RANGE, p1[1] + math.sin(angle) * self.MINING_RANGE)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)

    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    def _render_explosions(self):
        for e in self.explosions:
            pos = e['pos'].astype(int)
            radius = int(e['radius'])
            alpha = max(0, min(255, int(255 * (e['lifetime'] / 30.0))))
            color = (255, 50, 50, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        ore_text = self.font_ui.render(f"ORE: {int(self.ore)} / {self.WIN_SCORE}", True, self.COLOR_UI)
        self.screen.blit(ore_text, (10, 10))

        steps_text = self.font_ui.render(f"STEPS: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_UI)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            msg = "DEFEAT"
            color = (200, 50, 50)
            if self.ore >= self.WIN_SCORE:
                msg = "VICTORY!"
                color = (50, 200, 50)
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "ore": self.ore,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "player_vel": self.player_vel,
            "num_asteroids": len(self.asteroids),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
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
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}. Ore: {info['ore']}")
            # Wait for a moment, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Match the intended FPS

    env.close()