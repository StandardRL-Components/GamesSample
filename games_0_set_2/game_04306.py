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

    user_guide = (
        "Controls: ↑↓ to accelerate/decelerate, ←→ to turn. "
        "Hold space to activate your mining beam on nearby asteroids."
    )

    game_description = (
        "Pilot a mining ship through a hazardous asteroid field. "
        "Extract valuable ore while avoiding costly collisions. "
        "Reach 25 ore to win, but watch your ship's integrity!"
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_SHIP = (60, 180, 255)
    COLOR_SHIP_GLOW = (30, 90, 128)
    COLOR_ASTEROID = (120, 110, 100)
    COLOR_ORE = (255, 200, 0)
    COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50), (255, 255, 100)]
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_HIGH = (80, 220, 80)
    COLOR_HEALTH_MED = (220, 220, 80)
    COLOR_HEALTH_LOW = (220, 80, 80)

    # Game Parameters
    MAX_STEPS = 2000
    WIN_SCORE = 25
    INITIAL_HEALTH = 3
    INITIAL_ASTEROIDS = 8
    MAX_ASTEROIDS = 15
    SHIP_ACCELERATION = 0.15
    SHIP_TURN_SPEED = 0.1
    SHIP_FRICTION = 0.98
    SHIP_MAX_SPEED = 5
    MINING_RANGE = 80
    MINING_RATE = 0.1  # Ore per step
    ASTEROID_SPEED_INITIAL = 0.2
    ASTEROID_SPEED_MAX = 1.0
    DIFFICULTY_INTERVAL = 200
    DIFFICULTY_SPEED_INCREASE = 0.05


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ship_pos = None
        self.ship_vel = None
        self.ship_angle = None
        self.ship_health = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.asteroid_current_speed = self.ASTEROID_SPEED_INITIAL
        self.np_random = None

        self._generate_stars()
        
    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
            brightness = random.randint(50, 150)
            self.stars.append({'pos': pos, 'color': (brightness, brightness, brightness)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ship_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.ship_vel = np.array([0.0, 0.0], dtype=float)
        self.ship_angle = -math.pi / 2  # Pointing up
        self.ship_health = self.INITIAL_HEALTH
        self.asteroid_current_speed = self.ASTEROID_SPEED_INITIAL
        
        self.asteroids = []
        self.particles = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid(on_screen=True)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = -0.02  # Time penalty

        if not self.game_over:
            movement = action[0]
            space_held = action[1] == 1

            self._update_ship(movement)
            self._update_asteroids()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

            mining_reward = self._handle_mining(space_held)
            reward += mining_reward

            self._spawn_asteroids_periodically()
            self._update_difficulty()

        self._update_particles()
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100.0
            elif self.ship_health <= 0:
                reward += -100.0
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ship(self, movement):
        # Turning
        if movement == 3:  # Left
            self.ship_angle -= self.SHIP_TURN_SPEED
        if movement == 4:  # Right
            self.ship_angle += self.SHIP_TURN_SPEED

        # Acceleration
        if movement == 1:  # Up
            acceleration = np.array([math.cos(self.ship_angle), math.sin(self.ship_angle)]) * self.SHIP_ACCELERATION
            self.ship_vel += acceleration
            self._create_thruster_particles()
        elif movement == 2: # Down (Brake)
            self.ship_vel *= 0.95

        # Cap speed and apply friction
        speed = np.linalg.norm(self.ship_vel)
        if speed > self.SHIP_MAX_SPEED:
            self.ship_vel = (self.ship_vel / speed) * self.SHIP_MAX_SPEED
        self.ship_vel *= self.SHIP_FRICTION
        
        # Update position and wrap around screen
        self.ship_pos += self.ship_vel
        self.ship_pos[0] %= self.SCREEN_WIDTH
        self.ship_pos[1] %= self.SCREEN_HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            asteroid['pos'][0] %= self.SCREEN_WIDTH
            asteroid['pos'][1] %= self.SCREEN_HEIGHT

    def _handle_collisions(self):
        reward = 0
        collided_asteroid = None
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.ship_pos - asteroid['pos'])
            if dist < asteroid['radius'] + 10:  # 10 is ship radius
                collided_asteroid = asteroid
                break  # Only one collision per frame

        if collided_asteroid:
            # Filter out the collided asteroid using object identity, which is safe
            self.asteroids = [ast for ast in self.asteroids if ast is not collided_asteroid]
            self.ship_health -= 1
            self._create_explosion(self.ship_pos)
            if self.ship_health <= 0:
                self.ship_health = 0
            # Original code returned the initial reward (0) upon collision.
            return 0
        
        return reward

    def _handle_mining(self, space_held):
        reward = 0
        if not space_held:
            return reward

        mined_this_frame = False
        asteroid_to_mine = None
        
        # Find the first asteroid in range
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.ship_pos - asteroid['pos'])
            if dist < self.MINING_RANGE:
                asteroid_to_mine = asteroid
                break # Mine one asteroid at a time

        if asteroid_to_mine:
            mined_this_frame = True
            self._create_mining_particles(asteroid_to_mine['pos'])
            
            ore_mined = self.MINING_RATE
            asteroid_to_mine['ore'] -= ore_mined
            reward += ore_mined
            self.score += ore_mined

            if asteroid_to_mine['ore'] <= 0:
                # Filter out the depleted asteroid using object identity, which is safe
                self.asteroids = [ast for ast in self.asteroids if ast is not asteroid_to_mine]
    
        if mined_this_frame:
            # Add subtle ship recoil when mining
            recoil_dir = np.array([math.cos(self.ship_angle), math.sin(self.ship_angle)])
            self.ship_vel -= recoil_dir * 0.02
        return reward

    def _spawn_asteroids_periodically(self):
        if self.steps % 50 == 0 and len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()

    def _spawn_asteroid(self, on_screen=False):
        if on_screen:
            pos = self.np_random.random(2) * [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]
        else: # Spawn off-screen
            edge = self.np_random.integers(4)
            if edge == 0: pos = np.array([-30, self.np_random.random() * self.SCREEN_HEIGHT])
            elif edge == 1: pos = np.array([self.SCREEN_WIDTH + 30, self.np_random.random() * self.SCREEN_HEIGHT])
            elif edge == 2: pos = np.array([self.np_random.random() * self.SCREEN_WIDTH, -30])
            else: pos = np.array([self.np_random.random() * self.SCREEN_WIDTH, self.SCREEN_HEIGHT + 30])
        
        angle = self.np_random.random() * 2 * math.pi
        vel = np.array([math.cos(angle), math.sin(angle)]) * self.asteroid_current_speed * (self.np_random.random() * 0.5 + 0.75)
        radius = self.np_random.integers(15, 30)
        ore = self.np_random.integers(1, 4)
        
        # Create irregular shape
        num_points = self.np_random.integers(7, 12)
        shape_pts = []
        for i in range(num_points):
            angle_pt = 2 * math.pi * i / num_points
            rad_pt = radius * (self.np_random.random() * 0.4 + 0.8)
            shape_pts.append((math.cos(angle_pt) * rad_pt, math.sin(angle_pt) * rad_pt))

        self.asteroids.append({
            'pos': pos.astype(float),
            'vel': vel.astype(float),
            'radius': radius,
            'ore': ore,
            'shape_pts': shape_pts,
            'angle': self.np_random.random() * 2 * math.pi,
            'rot_speed': (self.np_random.random() - 0.5) * 0.03
        })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.asteroid_current_speed = min(self.ASTEROID_SPEED_MAX, self.asteroid_current_speed + self.DIFFICULTY_SPEED_INCREASE)

    def _check_termination(self):
        return self.ship_health <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_ship()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": int(self.score),
            "steps": self.steps,
            "health": self.ship_health,
        }

    # --- Particle Effects ---
    def _update_particles(self):
        # First, update all particle positions and lifespans
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        # Then, create a new list containing only the live particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = random.random() * 2 * math.pi
            speed = random.random() * 4 + 1
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifespan': random.randint(20, 40),
                'color': random.choice(self.COLOR_EXPLOSION), 'radius': random.randint(2, 5)
            })

    def _create_thruster_particles(self):
        angle = self.ship_angle + math.pi + (random.random() - 0.5) * 0.5
        speed = random.random() * 1.5 + 0.5
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed + self.ship_vel * 0.5
        pos = self.ship_pos - np.array([math.cos(self.ship_angle), math.sin(self.ship_angle)]) * 10
        self.particles.append({
            'pos': pos, 'vel': vel, 'lifespan': random.randint(10, 20),
            'color': (200, 220, 255), 'radius': random.randint(1, 3)
        })

    def _create_mining_particles(self, asteroid_pos):
        for _ in range(2):
            vec_to_asteroid = asteroid_pos - self.ship_pos
            dist = np.linalg.norm(vec_to_asteroid)
            if dist == 0: continue
            direction = vec_to_asteroid / dist
            
            start_pos = self.ship_pos + direction * 15
            vel = direction * (random.random() * 2 + 2)
            
            self.particles.append({
                'pos': start_pos, 'vel': vel, 'lifespan': int(dist / np.linalg.norm(vel)),
                'color': self.COLOR_ORE, 'radius': 2
            })

    # --- Rendering ---
    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], 1)

    def _render_ship(self):
        if self.ship_health <= 0: return

        p1 = (
            self.ship_pos[0] + math.cos(self.ship_angle) * 15,
            self.ship_pos[1] + math.sin(self.ship_angle) * 15
        )
        p2 = (
            self.ship_pos[0] + math.cos(self.ship_angle + 2.5) * 10,
            self.ship_pos[1] + math.sin(self.ship_angle + 2.5) * 10
        )
        p3 = (
            self.ship_pos[0] + math.cos(self.ship_angle - 2.5) * 10,
            self.ship_pos[1] + math.sin(self.ship_angle - 2.5) * 10
        )
        points = [p1, p2, p3]
        
        # Glow effect
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_SHIP_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_SHIP_GLOW)
        
        # Main body
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_SHIP)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_SHIP)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for pt in asteroid['shape_pts']:
                x = pt[0] * math.cos(asteroid['angle']) - pt[1] * math.sin(asteroid['angle']) + asteroid['pos'][0]
                y = pt[0] * math.sin(asteroid['angle']) + pt[1] * math.cos(asteroid['angle']) + asteroid['pos'][1]
                points.append((int(x), int(y)))
            
            # Use ore amount to tint the asteroid
            ore_ratio = min(1, asteroid['ore'] / 3.0)
            color = (
                int(self.COLOR_ASTEROID[0] * (1 - ore_ratio) + self.COLOR_ORE[0] * ore_ratio * 0.3),
                int(self.COLOR_ASTEROID[1] * (1 - ore_ratio) + self.COLOR_ORE[1] * ore_ratio * 0.3),
                int(self.COLOR_ASTEROID[2] * (1 - ore_ratio) + self.COLOR_ORE[2] * ore_ratio * 0.3)
            )

            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, tuple(c*0.8 for c in color))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20)) if p['lifespan'] < 20 else 255
            color = (*p['color'], alpha)
            if len(color) == 4:
                # Need a surface with per-pixel alpha to draw transparently
                surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))
            else:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        health_text = self.font_ui.render("HEALTH:", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - 180, 10))
        
        health_ratio = self.ship_health / self.INITIAL_HEALTH
        if health_ratio > 0.66: health_color = self.COLOR_HEALTH_HIGH
        elif health_ratio > 0.33: health_color = self.COLOR_HEALTH_MED
        else: health_color = self.COLOR_HEALTH_LOW

        pygame.draw.rect(self.screen, (50, 50, 50), (self.SCREEN_WIDTH - 95, 12, 85, 16))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, health_color, (self.SCREEN_WIDTH - 94, 13, int(83 * health_ratio), 14))

    def _render_game_over(self):
        if self.score >= self.WIN_SCORE:
            text = "MISSION COMPLETE"
            color = self.COLOR_HEALTH_HIGH
        else:
            text = "SHIP DESTROYED"
            color = self.COLOR_HEALTH_LOW
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((0,0,0,128))
        self.screen.blit(s, bg_rect)
        self.screen.blit(text_surf, text_rect)
        
    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- For human play ---
    # To play, you need a window. The environment is headless by default.
    # The following code sets up a window and maps keyboard keys to actions.
    
    try:
        # Unset the dummy video driver to enable display
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        pygame.display.init()
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroid Miner")
        
        obs, info = env.reset(seed=random.randint(0, 1_000_000))
        done = False
        
        print("\n" + "="*40)
        print(" " * 12 + "ASTEROID MINER")
        print("="*40)
        print(env.game_description)
        print("\n" + env.user_guide)
        print("="*40 + "\n")
        
        while not done:
            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            move = 0 # none
            if keys[pygame.K_UP]: move = 1
            elif keys[pygame.K_DOWN]: move = 2
            elif keys[pygame.K_LEFT]: move = 3
            elif keys[pygame.K_RIGHT]: move = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move, space, shift]
            # --- End Human Controls ---
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Convert observation back to a format Pygame can display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            if done:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                pygame.time.wait(3000) # Pause for 3 seconds before closing
    
    finally:
        env.close()