
# Generated: 2025-08-28T05:01:46.635177
# Source Brief: brief_05446.md
# Brief Index: 5446

        
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
    A Gymnasium environment for a top-down arcade space mining game.
    The player pilots a ship, collects ore from asteroids, and dodges enemy lasers.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, dodging enemy lasers and collecting ore from asteroids "
        "to reach a target yield before your ship is destroyed."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 100

        # Visuals & Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (255, 255, 255)
        self.COLOR_SHIP_GLOW = (200, 200, 255)
        self.COLOR_ASTEROID = (120, 120, 130)
        self.COLOR_LASER = (255, 50, 50)
        self.COLOR_LASER_GLOW = (255, 150, 150)
        self.COLOR_ORE = (255, 220, 50)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (255, 50, 0)]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_MINING_BEAM = (100, 255, 255, 150)

        # Player ship properties
        self.SHIP_SIZE = 12
        self.SHIP_SPEED = 5
        self.SHIP_ROTATION_SPEED = 15

        # Asteroid properties
        self.MIN_ASTEROIDS = 5
        self.MAX_ASTEROIDS = 8
        self.ASTEROID_MIN_ORE = 10
        self.ASTEROID_MAX_ORE = 30

        # Mining properties
        self.MINING_RANGE = 80
        self.MINING_RATE = 2 # Ore per second
        self.MINING_COOLDOWN = self.FPS / self.MINING_RATE

        # Laser properties
        self.INITIAL_LASER_SPAWN_INTERVAL = 5.0 # seconds
        self.LASER_SPAWN_ACCELERATION = 0.002 # per second
        self.LASER_SPEED = 6
        self.LASER_PROXIMITY_REWARD_DISTANCE = 50

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Internal state variables
        self.np_random = None
        self.ship_pos = None
        self.ship_angle = None
        self.target_angle = None
        self.asteroids = None
        self.lasers = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.laser_spawn_timer = None
        self.current_laser_spawn_interval = None
        self.mining_timer = None
        self.mining_target = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.ship_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.ship_angle = -90.0
        self.target_angle = -90.0

        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.stars = self._generate_stars(200)

        self.current_laser_spawn_interval = self.INITIAL_LASER_SPAWN_INTERVAL
        self.laser_spawn_timer = self.current_laser_spawn_interval * self.FPS
        
        self.mining_timer = 0
        self.mining_target = None

        while len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused per brief
        
        if not self.game_over:
            # Handle player input and movement
            self._handle_input(movement)
            
            # Update game logic
            reward += self._update_mining(space_held)
            self._update_lasers()
            self._update_particles()
            self._update_asteroids()

            # Check for collisions
            if self._check_laser_collision():
                self.game_over = True
                self.win = False
                reward -= 100
                self._create_explosion(self.ship_pos, 50)
                # sfx: player_explosion
            
            # Proximity penalty
            for laser in self.lasers:
                dist = np.linalg.norm(self.ship_pos - laser['pos'])
                if dist < self.LASER_PROXIMITY_REWARD_DISTANCE:
                    reward -= 0.1

        # Update difficulty and step count
        self.steps += 1
        self.current_laser_spawn_interval = max(0.5, self.current_laser_spawn_interval - (self.LASER_SPAWN_ACCELERATION / self.FPS))

        # Check for termination conditions
        terminated = self.game_over
        if self.score >= self.WIN_SCORE and not self.game_over:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        # Update target angle for smooth rotation
        if movement in [1, 2, 3, 4]:
            if movement == 1: self.target_angle = -90
            elif movement == 2: self.target_angle = 90
            elif movement == 3: self.target_angle = 180
            elif movement == 4: self.target_angle = 0
        
        # Smoothly rotate ship towards target angle
        angle_diff = (self.target_angle - self.ship_angle + 180) % 360 - 180
        self.ship_angle += np.clip(angle_diff, -self.SHIP_ROTATION_SPEED, self.SHIP_ROTATION_SPEED)
        self.ship_angle %= 360

        # Move ship
        if movement != 0:
            direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            move_vec = np.array(direction_map[movement], dtype=np.float32)
            self.ship_pos += move_vec * self.SHIP_SPEED

        # Clamp position to screen bounds
        self.ship_pos[0] = np.clip(self.ship_pos[0], self.SHIP_SIZE, self.WIDTH - self.SHIP_SIZE)
        self.ship_pos[1] = np.clip(self.ship_pos[1], self.SHIP_SIZE, self.HEIGHT - self.SHIP_SIZE)

    def _update_mining(self, space_held):
        reward = 0
        self.mining_target = None
        if space_held:
            closest_asteroid = None
            min_dist = float('inf')
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.ship_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
            
            if closest_asteroid and min_dist <= self.MINING_RANGE:
                self.mining_target = closest_asteroid
                self.mining_timer += 1
                if self.mining_timer >= self.MINING_COOLDOWN:
                    self.mining_timer = 0
                    mined_amount = 1
                    closest_asteroid['ore'] -= mined_amount
                    self.score += mined_amount
                    reward += mined_amount
                    
                    self._create_ore_particles(closest_asteroid['pos'], self.ship_pos, 5)
                    # sfx: ore_collect

                    if closest_asteroid['ore'] <= 0:
                        reward += 10
                        self.asteroids.remove(closest_asteroid)
                        self.mining_target = None
                        # sfx: asteroid_depleted
        else:
            self.mining_timer = 0
        return reward

    def _update_lasers(self):
        self.laser_spawn_timer -= 1
        if self.laser_spawn_timer <= 0:
            self._spawn_laser()
            self.laser_spawn_timer = self.current_laser_spawn_interval * self.FPS
        
        for laser in self.lasers[:]:
            laser['pos'] += laser['dir'] * self.LASER_SPEED
            if not (0 <= laser['pos'][0] <= self.WIDTH and 0 <= laser['pos'][1] <= self.HEIGHT):
                self.lasers.remove(laser)
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_asteroids(self):
        if len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroid()

    def _check_laser_collision(self):
        for laser in self.lasers:
            dist = np.linalg.norm(self.ship_pos - laser['pos'])
            if dist < self.SHIP_SIZE:
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_stars()
        for asteroid in self.asteroids:
            self._render_asteroid(asteroid)
        if self.mining_target:
            self._render_mining_beam(self.mining_target)
        for laser in self.lasers:
            self._render_laser(laser)
        for particle in self.particles:
            self._render_particle(particle)
        if not self.game_over:
            self._render_ship()

    def _render_ui(self):
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            message = "MISSION COMPLETE" if self.win else "SHIP DESTROYED"
            color = self.COLOR_ORE if self.win else self.COLOR_LASER
            game_over_text = self.font_game_over.render(message, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (int(star['pos'][0]), int(star['pos'][1])), star['size'])
    
    def _render_ship(self):
        angle_rad = math.radians(self.ship_angle)
        points = [
            (self.SHIP_SIZE, 0),
            (-self.SHIP_SIZE / 2, -self.SHIP_SIZE / 2),
            (-self.SHIP_SIZE / 2, self.SHIP_SIZE / 2),
        ]
        
        rotated_points = []
        for x, y in points:
            new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad) + self.ship_pos[0]
            new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad) + self.ship_pos[1]
            rotated_points.append((new_x, new_y))
        
        # Glow effect
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], self.COLOR_SHIP_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], self.COLOR_SHIP_GLOW)

        # Main ship body
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], self.COLOR_SHIP)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], self.COLOR_SHIP)

    def _render_asteroid(self, asteroid):
        scale = max(0.1, asteroid['ore'] / asteroid['max_ore'])
        points = [(v[0] * scale + asteroid['pos'][0], v[1] * scale + asteroid['pos'][1]) for v in asteroid['vertices']]
        if len(points) > 2:
            int_points = [(int(p[0]), int(p[1])) for p in points]
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ASTEROID)

    def _render_laser(self, laser):
        start_pos = laser['pos'] - laser['dir'] * 15
        end_pos = laser['pos'] + laser['dir'] * 15
        # Glow
        pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (int(start_pos[0]), int(start_pos[1])), (int(end_pos[0]), int(end_pos[1])), 5)
        # Core
        pygame.draw.line(self.screen, (255,255,255), (int(start_pos[0]), int(start_pos[1])), (int(end_pos[0]), int(end_pos[1])), 2)

    def _render_mining_beam(self, asteroid):
        pygame.draw.aaline(self.screen, self.COLOR_MINING_BEAM, (int(self.ship_pos[0]), int(self.ship_pos[1])), (int(asteroid['pos'][0]), int(asteroid['pos'][1])), 2)

    def _render_particle(self, particle):
        pygame.draw.circle(self.screen, particle['color'], (int(particle['pos'][0]), int(particle['pos'][1])), int(particle['size']))

    def _generate_stars(self, count):
        stars = []
        for _ in range(count):
            stars.append({
                'pos': [random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)],
                'size': random.choice([1, 1, 1, 2]),
                'color': random.choice([(50,50,80), (80,80,110), (120,120,150)])
            })
        return stars

    def _spawn_asteroid(self):
        ore = self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1)
        radius = ore * 1.2
        pos = np.array([
            self.np_random.uniform(radius, self.WIDTH - radius),
            self.np_random.uniform(radius, self.HEIGHT - radius)
        ], dtype=np.float32)
        
        # Avoid spawning on player
        while np.linalg.norm(pos - self.ship_pos) < radius + self.SHIP_SIZE + 50:
            pos = np.array([
                self.np_random.uniform(radius, self.WIDTH - radius),
                self.np_random.uniform(radius, self.HEIGHT - radius)
            ], dtype=np.float32)

        num_vertices = self.np_random.integers(8, 15)
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(radius * 0.7, radius)
            vertices.append((math.cos(angle) * dist, math.sin(angle) * dist))
        
        self.asteroids.append({'pos': pos, 'ore': ore, 'max_ore': ore, 'vertices': vertices})

    def _spawn_laser(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -10], dtype=np.float32)
            angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10], dtype=np.float32)
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        elif edge == 2: # Left
            pos = np.array([-10, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            angle = self.np_random.uniform(-math.pi * 0.25, math.pi * 0.25)
        else: # Right
            pos = np.array([self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)
        
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        self.lasers.append({'pos': pos, 'dir': direction})
        # sfx: laser_spawn

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(self.FPS // 2, self.FPS),
                'color': self.np_random.choice(self.COLOR_EXPLOSION),
                'size': self.np_random.uniform(2, 5)
            })

    def _create_ore_particles(self, start_pos, end_pos, num_particles):
        direction = (end_pos - start_pos)
        dist = np.linalg.norm(direction)
        if dist == 0: return
        direction /= dist
        
        for _ in range(num_particles):
            t = self.np_random.uniform(0.1, 0.9)
            speed = self.np_random.uniform(2, 5)
            vel = direction * speed
            self.particles.append({
                'pos': start_pos.copy() + direction * t * 10,
                'vel': vel,
                'life': int(dist / speed),
                'color': self.COLOR_ORE,
                'size': self.np_random.uniform(1, 3)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Action mapping for human input
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
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()