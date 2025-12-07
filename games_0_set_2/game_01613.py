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
        "Controls: ↑↓←→ to move. Hold space near an asteroid to mine it."
    )

    game_description = (
        "Pilot a spaceship through an asteroid field. Mine 25 asteroids to win, "
        "but watch your fuel and avoid collisions!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WIN_SCORE = 25
    MAX_STEPS = 1800  # 60 seconds at 30fps

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_ENGINE = (255, 180, 50)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_ASTEROID_GLOW = (150, 160, 170, 30)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FUEL_BG = (80, 0, 0)
    COLOR_FUEL_FG = (255, 50, 50)
    COLOR_LASER = (255, 100, 255)

    # Physics & Game Params
    PLAYER_ACCELERATION = 0.3
    PLAYER_FRICTION = 0.97
    PLAYER_SIZE = 12
    ASTEROID_COUNT = 30
    ASTEROID_SIZE_MIN = 15
    ASTEROID_SIZE_MAX = 35
    MINING_DISTANCE_FACTOR = 2.5
    MAX_FUEL = 100.0
    FUEL_DEPLETION_IDLE = 100.0 / MAX_STEPS
    FUEL_DEPLETION_THRUST = FUEL_DEPLETION_IDLE * 3

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.np_random = None

        # Game state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.asteroids = []
        self.stars = []
        self.particles = []
        self.score = 0
        self.fuel = 0.0
        self.steps = 0
        self.game_over = False
        self.termination_reason = ""
        self.is_mining = False
        self.mining_target = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            random.seed(seed)
        else:
            self.np_random = np.random.default_rng()


        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90

        self.score = 0
        self.fuel = self.MAX_FUEL
        self.steps = 0
        self.game_over = False
        self.termination_reason = ""

        self.is_mining = False
        self.mining_target = None

        self.particles = []
        self._generate_stars(200)
        self._generate_asteroids()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        # --- Handle Input and Update State ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        self._update_player_state()

        # --- Mining Logic ---
        mined_asteroid = self._handle_mining(space_held)
        if mined_asteroid:
            reward += 1.0  # +1 for mining an asteroid
            self.score += 1
            # sfx: mining success
            self._create_explosion(mined_asteroid['pos'], mined_asteroid['color'], int(mined_asteroid['radius']))

        # --- Continuous Rewards ---
        if self.fuel > self.MAX_FUEL * 0.5:
            reward += 0.01 # Reward for staying high on fuel
        elif self.fuel < self.MAX_FUEL * 0.2:
            reward -= 0.01 # Penalty for being low on fuel

        # --- Update Dynamic Elements ---
        self._update_asteroids_rotation()
        self._update_particles()
        
        # --- Check for Collisions ---
        if self._check_collisions():
            self.game_over = True
            terminated = True
            reward -= 50.0  # -50 for collision
            self.termination_reason = "CRASHED"
            # sfx: big explosion

        # --- Check Other Termination Conditions ---
        if not terminated:
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                terminated = True
                reward += 100.0  # +100 for winning
                self.termination_reason = "MISSION COMPLETE"
                # sfx: win fanfare
            elif self.fuel <= 0:
                self.game_over = True
                terminated = True
                reward -= 100.0  # -100 for running out of fuel
                self.termination_reason = "OUT OF FUEL"
                # sfx: failure sound
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.termination_reason = "TIME LIMIT"
                # sfx: timeout sound

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_movement(self, movement):
        thrusting = False
        if movement == 1:  # Up
            self.player_vel.y -= self.PLAYER_ACCELERATION
            thrusting = True
        elif movement == 2:  # Down
            self.player_vel.y += self.PLAYER_ACCELERATION
            thrusting = True
        elif movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCELERATION
            thrusting = True
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCELERATION
            thrusting = True

        self.fuel -= self.FUEL_DEPLETION_IDLE
        if thrusting:
            self.fuel -= self.FUEL_DEPLETION_THRUST
            self._create_engine_flare()
            # sfx: engine thrust loop

    def _update_player_state(self):
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel

        # Keep player on screen
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

        if self.player_vel.length() > 0.1:
            self.player_angle = self.player_vel.angle_to(pygame.Vector2(1, 0))

    def _handle_mining(self, space_held):
        self.is_mining = False
        self.mining_target = None
        if not space_held:
            return None

        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < asteroid['radius'] + self.PLAYER_SIZE * self.MINING_DISTANCE_FACTOR:
                self.is_mining = True
                self.mining_target = asteroid
                asteroid['health'] -= 1
                if asteroid['health'] <= 0:
                    self.asteroids.remove(asteroid)
                    return asteroid
                break # Mine one at a time
        return None

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < asteroid['radius'] + self.PLAYER_SIZE * 0.8: # Circle-based collision
                self._create_explosion(self.player_pos, self.COLOR_PLAYER, 30)
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_asteroids()
        if not (self.game_over and self.termination_reason == "CRASHED"):
             self._render_player()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "asteroids_left": len(self.asteroids),
        }

    # --- Generation Methods ---
    def _generate_stars(self, count):
        self.stars = []
        for _ in range(count):
            self.stars.append(
                (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.uniform(0.5, 1.5))
            )

    def _generate_asteroids(self):
        self.asteroids = []
        for _ in range(self.ASTEROID_COUNT):
            while True:
                pos = pygame.Vector2(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
                if pos.distance_to(self.player_pos) > 100:  # Don't spawn on player
                    break
            
            radius = random.uniform(self.ASTEROID_SIZE_MIN, self.ASTEROID_SIZE_MAX)
            num_points = random.randint(7, 12)
            points = []
            for i in range(num_points):
                angle = (i / num_points) * 2 * math.pi
                r = radius * random.uniform(0.8, 1.2)
                points.append((math.cos(angle) * r, math.sin(angle) * r))

            self.asteroids.append({
                'pos': pos,
                'radius': radius,
                'points': points,
                'angle': 0,
                'rot_speed': random.uniform(-1.0, 1.0),
                'color': self.COLOR_ASTEROID,
                'health': (radius / self.ASTEROID_SIZE_MIN) * 3, # Bigger asteroids are tougher
            })

    # --- Particle System ---
    def _create_engine_flare(self):
        if random.random() < 0.8: # Don't spawn every frame
            angle_rad = math.radians(self.player_angle + 180)
            spread = math.radians(20)
            
            p_angle = angle_rad + random.uniform(-spread, spread)
            p_vel = pygame.Vector2(math.cos(p_angle), -math.sin(p_angle)) * random.uniform(1, 3)
            
            offset = pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * -self.PLAYER_SIZE
            p_pos = self.player_pos + offset
            
            self.particles.append({
                'pos': p_pos, 'vel': p_vel, 'lifespan': random.randint(10, 20),
                'color': self.COLOR_ENGINE, 'radius': random.uniform(2, 4)
            })

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': random.randint(20, 50),
                'color': color, 'radius': random.uniform(1, 3)
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifespan'] -= 1
            p['radius'] -= 0.05
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _update_asteroids_rotation(self):
        for a in self.asteroids:
            a['angle'] += a['rot_speed']

    # --- Rendering Methods ---
    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

    def _render_player(self):
        # Ship body
        angle_rad = math.radians(self.player_angle)
        cos_a, sin_a = math.cos(angle_rad), -math.sin(angle_rad)
        
        p1 = (self.player_pos.x + self.PLAYER_SIZE * cos_a, self.player_pos.y + self.PLAYER_SIZE * sin_a)
        p2 = (self.player_pos.x + self.PLAYER_SIZE * 0.5 * -sin_a, self.player_pos.y + self.PLAYER_SIZE * 0.5 * cos_a)
        p3 = (self.player_pos.x - self.PLAYER_SIZE * 0.5 * -sin_a, self.player_pos.y - self.PLAYER_SIZE * 0.5 * cos_a)
        
        points = [p1, p3, p2]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Mining laser
        if self.is_mining and self.mining_target:
            start_pos = (int(p1[0]), int(p1[1]))
            end_pos = (int(self.mining_target['pos'].x), int(self.mining_target['pos'].y))
            width = int(2 + math.sin(self.steps * 0.5) * 1.5)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, width)
            # sfx: mining laser loop

    def _render_asteroids(self):
        for a in self.asteroids:
            angle_rad = math.radians(a['angle'])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            
            rotated_points = []
            for px, py in a['points']:
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                rotated_points.append((a['pos'].x + rx, a['pos'].y + ry))

            pygame.gfxdraw.aapolygon(self.screen, rotated_points, a['color'])
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, a['color'])

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Fuel
        fuel_percent = max(0, self.fuel / self.MAX_FUEL)
        bar_width = 150
        bar_height = 15
        fuel_bar_rect = pygame.Rect(self.WIDTH - bar_width - 10, 10, bar_width, bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BG, fuel_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_FUEL_FG, (fuel_bar_rect.x, fuel_bar_rect.y, bar_width * fuel_percent, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, fuel_bar_rect, 1)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_large.render(self.termination_reason, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

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
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.user_guide)

    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering to screen ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()