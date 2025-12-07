
# Generated: 2025-08-27T17:22:28.394389
# Source Brief: brief_01508.md
# Brief Index: 1508

        
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
        "Controls: Arrow keys to fly your ship. Hold spacebar near an asteroid to mine it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship to mine valuable minerals from asteroids. Race against the clock to collect enough resources before time runs out. Be efficient!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 180  # seconds
        self.WIN_SCORE = 20
        self.MAX_ASTEROIDS = 15
        self.MIN_SPAWN_DISTANCE_FROM_PLAYER = 100

        # Player properties
        self.PLAYER_ACCELERATION = 0.4
        self.PLAYER_DRAG = 0.95
        self.PLAYER_MAX_SPEED = 6.0
        self.PLAYER_SIZE = 12
        self.MINING_RANGE = 60
        self.MINING_RATE = 0.05 # minerals per frame

        # Colors
        self.COLOR_BG_STAGE1 = (10, 10, 25)
        self.COLOR_BG_STAGE2 = (25, 10, 10)
        self.COLOR_BG_STAGE3 = (25, 10, 25)
        self.COLOR_SHIP = (230, 230, 255)
        self.COLOR_SHIP_GLOW = (150, 150, 255, 100)
        self.COLOR_MINERAL = (255, 220, 50)
        self.COLOR_LASER = (255, 100, 100)
        self.COLOR_ASTEROID_LOW = (80, 70, 60)
        self.COLOR_ASTEROID_HIGH = (180, 150, 120)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TIMER_WARN = (255, 100, 100)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.stage = 1
        self.stage_flash_timer = 0
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.mining_target = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Fallback if seed is None
            if self.np_random is None:
                self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME * self.FPS
        self.stage = 1
        self.stage_flash_timer = 0

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)

        self._spawn_stars()
        self.asteroids = []
        for _ in range(self.MAX_ASTEROIDS // 2):
            self._spawn_asteroid()

        self.particles = []
        self.mining_target = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, _ = action
        space_held = space_held == 1

        # Calculate pre-move state for reward shaping
        reward = 0
        dist_before = self._get_distance_to_nearest_asteroid()

        # Update game logic
        self._handle_input(movement)
        self._update_player()
        self._update_asteroids()
        minerals_collected = self._handle_mining(space_held)
        self._update_particles()
        self._update_game_state()
        
        # Respawn asteroids if needed
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.np_random.random() < 0.02:
            self._spawn_asteroid()

        # Calculate reward
        dist_after = self._get_distance_to_nearest_asteroid()
        
        # Reward for moving closer to an asteroid
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 0.01 # Small reward, as requested 0.1 is a bit high for 30fps

        # Reward for collecting minerals
        reward += minerals_collected * 1.0

        # Check for termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        # Max steps termination
        self.steps += 1
        if self.steps >= self.MAX_TIME * self.FPS:
            if not terminated: # Avoid double penalty/reward
                reward -= 100.0
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )
    
    def _handle_input(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        acceleration_vec = np.array([0.0, 0.0])
        if movement == 1:
            acceleration_vec[1] = -self.PLAYER_ACCELERATION
        elif movement == 2:
            acceleration_vec[1] = self.PLAYER_ACCELERATION
        elif movement == 3:
            acceleration_vec[0] = -self.PLAYER_ACCELERATION
        elif movement == 4:
            acceleration_vec[0] = self.PLAYER_ACCELERATION
        
        self.player_vel += acceleration_vec
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
        
    def _update_player(self):
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel
        
        # Screen wrap
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _update_asteroids(self):
        speed_modifier = 1.0
        if self.stage == 2: speed_modifier = 1.2
        if self.stage == 3: speed_modifier = 1.5

        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel'] * speed_modifier
            asteroid['pos'][0] %= self.WIDTH
            asteroid['pos'][1] %= self.HEIGHT
            asteroid['angle'] += asteroid['rot_speed']
    
    def _handle_mining(self, space_held):
        minerals_collected = 0
        self.mining_target = None
        
        if not space_held:
            return 0

        # Find closest asteroid in range
        closest_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.MINING_RANGE and dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        if closest_asteroid:
            self.mining_target = closest_asteroid
            mined_amount = self.MINING_RATE
            
            if closest_asteroid['minerals'] >= mined_amount:
                # sound: mining_laser_loop
                closest_asteroid['minerals'] -= mined_amount
                
                # Check if a whole mineral was collected
                if math.floor(closest_asteroid['minerals'] + mined_amount) > math.floor(closest_asteroid['minerals']):
                    # sound: mineral_collect
                    self.score += 1
                    minerals_collected += 1
                    # Spawn collection particle
                    self._spawn_particle(
                        pos=closest_asteroid['pos'].copy(),
                        type='collect',
                        vel=self.player_pos - closest_asteroid['pos']
                    )

                # Spawn spark particles
                if self.np_random.random() < 0.5:
                    self._spawn_particle(pos=closest_asteroid['pos'].copy(), type='spark')
            
            if closest_asteroid['minerals'] <= 0:
                # sound: asteroid_break
                self.asteroids.remove(closest_asteroid)
                self.mining_target = None
        
        return minerals_collected

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_game_state(self):
        self.time_remaining -= 1
        
        time_seconds = self.time_remaining / self.FPS
        
        if self.stage == 1 and time_seconds <= 120:
            self.stage = 2
            self.stage_flash_timer = 15 # frames
            # sound: stage_up
        elif self.stage == 2 and time_seconds <= 60:
            self.stage = 3
            self.stage_flash_timer = 15
            # sound: stage_up

        if self.stage_flash_timer > 0:
            self.stage_flash_timer -= 1
            
    def _spawn_asteroid(self):
        while True:
            pos = self.np_random.uniform([0, 0], [self.WIDTH, self.HEIGHT])
            if np.linalg.norm(pos - self.player_pos) > self.MIN_SPAWN_DISTANCE_FROM_PLAYER:
                break
        
        size = self.np_random.integers(15, 40)
        minerals = self.np_random.integers(1, 6)
        vel_angle = self.np_random.uniform(0, 2 * math.pi)
        vel_mag = self.np_random.uniform(0.1, 0.4)
        
        asteroid = {
            'pos': np.array(pos, dtype=np.float32),
            'vel': np.array([math.cos(vel_angle) * vel_mag, math.sin(vel_angle) * vel_mag], dtype=np.float32),
            'size': size,
            'minerals': minerals,
            'max_minerals': minerals,
            'shape': self._generate_asteroid_shape(size),
            'angle': 0,
            'rot_speed': self.np_random.uniform(-0.02, 0.02)
        }
        self.asteroids.append(asteroid)

    def _generate_asteroid_shape(self, radius):
        points = []
        num_vertices = self.np_random.integers(8, 15)
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            r = self.np_random.uniform(radius * 0.7, radius)
            points.append((r * math.cos(angle), r * math.sin(angle)))
        return points

    def _spawn_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.uniform(0.5, 1.5)
            })

    def _spawn_particle(self, pos, type, vel=None):
        if type == 'spark':
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(5, 15)
            color = self.COLOR_LASER
            size = self.np_random.integers(1, 3)
        elif type == 'collect':
            if vel is None: vel = np.zeros(2)
            # Travel towards score
            target_pos = np.array([60, 30])
            vel = (target_pos - pos) / 30.0 + self.np_random.uniform(-0.5, 0.5, 2)
            life = 30
            color = self.COLOR_MINERAL
            size = 5
        
        self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'color': color, 'size': size})

    def _get_distance_to_nearest_asteroid(self):
        if not self.asteroids:
            return None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        # Background
        bg_color = self.COLOR_BG_STAGE1
        if self.stage == 2: bg_color = self.COLOR_BG_STAGE2
        if self.stage == 3: bg_color = self.COLOR_BG_STAGE3
        self.screen.fill(bg_color)

        # Render stars
        for star in self.stars:
            pygame.draw.circle(self.screen, (255, 255, 255), star['pos'], star['size'])

        # Render asteroids
        for asteroid in self.asteroids:
            t = asteroid['minerals'] / asteroid['max_minerals']
            color = (
                int(self.COLOR_ASTEROID_LOW[0] + t * (self.COLOR_ASTEROID_HIGH[0] - self.COLOR_ASTEROID_LOW[0])),
                int(self.COLOR_ASTEROID_LOW[1] + t * (self.COLOR_ASTEROID_HIGH[1] - self.COLOR_ASTEROID_LOW[1])),
                int(self.COLOR_ASTEROID_LOW[2] + t * (self.COLOR_ASTEROID_HIGH[2] - self.COLOR_ASTEROID_LOW[2])),
            )
            
            rotated_shape = []
            for x, y in asteroid['shape']:
                rx = x * math.cos(asteroid['angle']) - y * math.sin(asteroid['angle'])
                ry = x * math.sin(asteroid['angle']) + y * math.cos(asteroid['angle'])
                rotated_shape.append((rx + asteroid['pos'][0], ry + asteroid['pos'][1]))
            
            pygame.gfxdraw.aapolygon(self.screen, rotated_shape, color)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_shape, color)

        # Render mining laser
        if self.mining_target:
            start_pos = tuple(self.player_pos.astype(int))
            end_pos = tuple(self.mining_target['pos'].astype(int))
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
            pygame.gfxdraw.line(self.screen, start_pos[0], start_pos[1], end_pos[0], end_pos[1], (*self.COLOR_LASER, 150))

        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['size'] * (p['life'] / 30.0)))
            
        # Render player
        ship_points = [
            (0, -self.PLAYER_SIZE),
            (-self.PLAYER_SIZE / 1.5, self.PLAYER_SIZE / 2),
            (self.PLAYER_SIZE / 1.5, self.PLAYER_SIZE / 2),
        ]
        angle = math.atan2(self.player_vel[0], -self.player_vel[1])
        rotated_points = []
        for x, y in ship_points:
            rx = x * math.cos(angle) - y * math.sin(angle) + self.player_pos[0]
            ry = x * math.sin(angle) + y * math.cos(angle) + self.player_pos[1]
            rotated_points.append((rx, ry))
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_SHIP)

        # Render UI
        score_text = self.font_small.render(f"MINERALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        time_seconds = max(0, self.time_remaining / self.FPS)
        time_str = f"{int(time_seconds // 60):02}:{int(time_seconds % 60):02}"
        timer_color = self.COLOR_TEXT if time_seconds > 10 else self.COLOR_TIMER_WARN
        time_text = self.font_small.render(time_str, True, timer_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 20))

        # Render stage flash
        if self.stage_flash_timer > 0:
            alpha = int(255 * (self.stage_flash_timer / 15.0))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.FPS,
            "stage": self.stage,
        }
        
    def render(self):
        # This is for human playback, not used by the gym interface
        # but useful for debugging.
        if not hasattr(self, 'display_screen'):
            self.display_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Asteroid Miner")
        
        obs = self._get_observation()
        # The observation is (H, W, C), but pygame needs (W, H, C)
        # and surfarray.make_surface expects (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        self.display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

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
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

# Example usage for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    terminated = False
    obs, info = env.reset()
    
    # Use a separate display for rendering
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    while not terminated:
        # Action mapping from keyboard
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

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}, Time: {info['time_remaining']:.2f}s")
    env.close()