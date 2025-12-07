import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space for a speed boost (consumes extra fuel)."
    )

    # User-facing description of the game
    game_description = (
        "Pilot a spaceship through a treacherous asteroid field, collecting fuel to reach the end of the level."
    )

    # Frames auto-advance for smooth graphics and time-based gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_STAR_1 = (40, 40, 60)
    COLOR_STAR_2 = (80, 80, 100)
    COLOR_STAR_3 = (150, 150, 180)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 40)
    COLOR_THRUSTER = (255, 180, 0)
    COLOR_BOOST = (0, 180, 255)
    COLOR_ASTEROID = (210, 80, 80)
    COLOR_ASTEROID_GLOW = (210, 80, 80, 40)
    COLOR_FUEL = (255, 255, 0)
    COLOR_FUEL_GLOW = (255, 255, 0, 80)
    COLOR_TARGET = (0, 128, 255)
    COLOR_TARGET_GLOW = (0, 128, 255, 60)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (50, 50, 70, 150)
    COLOR_FUEL_BAR = (0, 255, 0)
    COLOR_FUEL_BAR_BG = (255, 0, 0)

    # Game parameters
    FPS = 30
    MAX_STEPS = 1800 # 60 seconds * 30 fps
    WORLD_WIDTH = 5000
    TARGET_ZONE_X = WORLD_WIDTH - 300
    TARGET_ZONE_WIDTH = 100

    # Player
    PLAYER_ACCEL = 0.4
    PLAYER_BOOST_ACCEL = 0.8
    PLAYER_DRAG = 0.95
    PLAYER_MAX_SPEED = 8.0
    PLAYER_SIZE = 12
    INITIAL_FUEL = 100.0
    FUEL_CONSUMPTION_IDLE = 0.03
    FUEL_CONSUMPTION_BOOST = 0.25
    FUEL_CONSUMPTION_COLLISION = 15.0

    # Asteroids
    INITIAL_ASTEROID_COUNT = 5
    ASTEROID_MIN_SPEED = 0.5
    ASTEROID_MAX_SPEED = 2.0
    ASTEROID_MIN_SIZE = 15
    ASTEROID_MAX_SIZE = 40

    # Fuel
    INITIAL_FUEL_SPAWN_RATE = 0.03
    FUEL_CANISTER_SIZE = 8
    FUEL_REPLENISH_AMOUNT = 25.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set SDL_VIDEODRIVER to dummy if not already set, for headless operation
        if 'SDL_VIDEODRIVER' not in os.environ:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.fuel = None
        self.asteroids = None
        self.fuel_canisters = None
        self.particles = None
        self.stars = None
        self.camera_offset_x = 0.0
        self.camera_offset_y = 0.0
        self.asteroid_count = 0
        self.fuel_spawn_rate = 0.0
        self.asteroid_speed_multiplier = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([150.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.fuel = self.INITIAL_FUEL
        
        self.camera_offset_x = 0.0
        self.camera_offset_y = 0.0

        # Difficulty progression variables
        self.asteroid_count = self.INITIAL_ASTEROID_COUNT
        self.fuel_spawn_rate = self.INITIAL_FUEL_SPAWN_RATE
        self.asteroid_speed_multiplier = 1.0
        
        # Create dynamic elements
        self.asteroids = [self._spawn_asteroid(initial=True) for _ in range(self.asteroid_count)]
        self.fuel_canisters = []
        self.particles = []

        # Create static background
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT * 2)]),
                'layer': self.np_random.choice([1, 2, 3])
            })
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Store pre-step state for reward calculation
        prev_dist_to_target = self._get_dist_to_target(self.player_pos[0])
        prev_fuel = self.fuel

        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_player(action)
        self._update_world()
        self._update_asteroids()
        self._update_fuel_canisters()
        self._update_particles()
        self._handle_collisions()
        self._update_difficulty()

        self.steps += 1
        
        # --- Calculate Reward ---
        reward = self._calculate_reward(prev_dist_to_target, prev_fuel)
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.player_pos[0] >= self.TARGET_ZONE_X:
                reward += 100.0 # Victory bonus
            elif self.fuel <= 0:
                reward -= 100.0 # Penalty for running out of fuel

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        accel_vec = np.array([0.0, 0.0])
        accel_magnitude = self.PLAYER_BOOST_ACCEL if space_held else self.PLAYER_ACCEL

        if movement == 1: accel_vec[1] = -accel_magnitude  # Up
        elif movement == 2: accel_vec[1] = accel_magnitude   # Down
        elif movement == 3: accel_vec[0] = -accel_magnitude  # Left
        elif movement == 4: accel_vec[0] = accel_magnitude   # Right
        
        self.player_vel += accel_vec

    def _update_player(self, action):
        movement, space_held, _ = action

        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel * (self.PLAYER_MAX_SPEED / speed)
        
        # Update position
        self.player_pos += self.player_vel
        
        # Consume fuel
        self.fuel -= self.FUEL_CONSUMPTION_IDLE
        if space_held:
            self.fuel -= self.FUEL_CONSUMPTION_BOOST
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WORLD_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # Thruster particles
        if np.linalg.norm(self.player_vel) > 0.1 or movement != 0:
            self._create_particles(
                self.player_pos, 2, 
                self.COLOR_BOOST if space_held else self.COLOR_THRUSTER,
                -self.player_vel, 0.5, 15
            )

    def _update_world(self):
        # Update camera to follow player smoothly
        target_cam_x = self.player_pos[0] - self.SCREEN_WIDTH / 2.5
        self.camera_offset_x += (target_cam_x - self.camera_offset_x) * 0.1
        
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT / 2
        self.camera_offset_y += (target_cam_y - self.camera_offset_y) * 0.1
        
        self.camera_offset_x = max(0, self.camera_offset_x)
        self.camera_offset_y = np.clip(self.camera_offset_y, -self.SCREEN_HEIGHT/2, self.SCREEN_HEIGHT/2)


    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['rot'] = (asteroid['rot'] + asteroid['rot_speed']) % 360
        
        # Remove asteroids that are far off-screen left and spawn new ones on the right
        self.asteroids = [a for a in self.asteroids if a['pos'][0] > self.camera_offset_x - 100]
        while len(self.asteroids) < self.asteroid_count:
            self.asteroids.append(self._spawn_asteroid())

    def _update_fuel_canisters(self):
        for canister in self.fuel_canisters:
            canister['rot'] = (canister['rot'] + 2) % 360
        
        # Remove canisters that are off-screen
        self.fuel_canisters = [f for f in self.fuel_canisters if f['pos'][0] > self.camera_offset_x - 50]
        
        # Spawn new canisters
        if self.np_random.random() < self.fuel_spawn_rate:
            spawn_x = self.camera_offset_x + self.SCREEN_WIDTH + 50
            spawn_y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            self.fuel_canisters.append({
                'pos': np.array([spawn_x, spawn_y]),
                'size': self.FUEL_CANISTER_SIZE,
                'rot': self.np_random.uniform(0, 360)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _handle_collisions(self):
        # Player vs Asteroids
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['size']:
                self.fuel -= self.FUEL_CONSUMPTION_COLLISION
                self._create_particles(self.player_pos, 20, self.COLOR_ASTEROID, self.player_vel, 2, 40)
                asteroid['pos'][0] = -9999 # effectively remove it
                self.score -= 2
        
        # Player vs Fuel Canisters
        for i, canister in enumerate(self.fuel_canisters):
            dist = np.linalg.norm(self.player_pos - canister['pos'])
            if dist < self.PLAYER_SIZE + canister['size']:
                self.fuel = min(self.INITIAL_FUEL, self.fuel + self.FUEL_REPLENISH_AMOUNT)
                self.score += 10
                self._create_particles(canister['pos'], 15, self.COLOR_FUEL, np.array([0,0]), 1.5, 30)
                self.fuel_canisters.pop(i)
                break

    def _update_difficulty(self):
        # Increase asteroid speed every 250 steps
        if self.steps > 0 and self.steps % 250 == 0:
            self.asteroid_speed_multiplier += 0.05
            self.fuel_spawn_rate = max(0.005, self.fuel_spawn_rate - 0.002)
        
        # Increase asteroid count every 500 steps
        if self.steps > 0 and self.steps % 500 == 0:
            self.asteroid_count += 1

    def _calculate_reward(self, prev_dist, prev_fuel):
        reward = 0.0
        
        # Reward for moving towards target
        current_dist = self._get_dist_to_target(self.player_pos[0])
        reward += (prev_dist - current_dist) * 0.1

        # Penalty for fuel loss (from any source)
        fuel_lost = prev_fuel - self.fuel
        if fuel_lost > 0:
             reward -= fuel_lost * 0.1
        
        # Reward for collecting fuel
        if self.fuel > prev_fuel and (self.fuel - prev_fuel) > 1: # check for collection event
            reward += 5.0
            
        return reward

    def _check_termination(self):
        terminated = (
            self.fuel <= 0 or
            self.steps >= self.MAX_STEPS or
            self.player_pos[0] >= self.TARGET_ZONE_X
        )
        return bool(terminated)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_game_objects()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "distance_to_target": self._get_dist_to_target(self.player_pos[0]),
        }

    def _get_dist_to_target(self, x_pos):
        return max(0, self.TARGET_ZONE_X - x_pos)

    # --- Spawning Helpers ---
    def _spawn_asteroid(self, initial=False):
        if initial:
            x = self.np_random.uniform(self.SCREEN_WIDTH / 2, self.WORLD_WIDTH)
            y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
        else:
            x = self.camera_offset_x + self.SCREEN_WIDTH + self.np_random.uniform(50, 150)
            y = self.np_random.uniform(-100, self.SCREEN_HEIGHT + 100)
        
        vel_y_component = 0
        if self.np_random.random() < 0.3: # 30% chance of vertical motion
            vel_y_component = self.np_random.uniform(-0.5, 0.5)

        return {
            'pos': np.array([x, y]),
            'vel': np.array([self.np_random.uniform(-self.ASTEROID_MAX_SPEED, -self.ASTEROID_MIN_SPEED) * self.asteroid_speed_multiplier, vel_y_component]),
            'size': self.np_random.integers(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE),
            'rot': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-2, 2),
            'shape': self._generate_asteroid_shape(self.np_random.integers(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE))
        }

    def _generate_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        # Add jitter to radius
        radii = radius + self.np_random.uniform(-radius * 0.3, radius * 0.3, num_vertices)
        points = [ (r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
        return points

    def _create_particles(self, pos, count, color, base_vel, spread, lifetime):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * spread
            vel = base_vel + np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'life': self.np_random.integers(lifetime // 2, lifetime),
                'size': self.np_random.uniform(2, 5)
            })

    # --- Rendering Helpers ---
    def _world_to_screen(self, pos):
        return int(pos[0] - self.camera_offset_x), int(pos[1] - self.camera_offset_y)

    def _render_background(self):
        for star in self.stars:
            layer = star['layer']
            x, y = star['pos']
            
            # Parallax scrolling
            screen_x = (x - self.camera_offset_x / (layer * 1.5)) % self.SCREEN_WIDTH
            screen_y = (y - self.camera_offset_y / (layer * 1.5)) % self.SCREEN_HEIGHT

            if layer == 1: color, size = self.COLOR_STAR_1, 1
            elif layer == 2: color, size = self.COLOR_STAR_2, 2
            else: color, size = self.COLOR_STAR_3, 3
            
            pygame.draw.rect(self.screen, color, (int(screen_x), int(screen_y), size, size))

    def _render_game_objects(self):
        # Target Zone
        tx, ty = self._world_to_screen((self.TARGET_ZONE_X, 0))
        if tx < self.SCREEN_WIDTH:
            s = pygame.Surface((self.TARGET_ZONE_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_TARGET_GLOW)
            self.screen.blit(s, (tx, 0))
            for i in range(0, self.SCREEN_HEIGHT, 20):
                pygame.draw.line(self.screen, self.COLOR_TARGET, (tx, i), (tx, i+10), 2)
                pygame.draw.line(self.screen, self.COLOR_TARGET, (tx+self.TARGET_ZONE_WIDTH, i), (tx+self.TARGET_ZONE_WIDTH, i+10), 2)


        # Asteroids
        for asteroid in self.asteroids:
            sx, sy = self._world_to_screen(asteroid['pos'])
            if -50 < sx < self.SCREEN_WIDTH + 50 and -50 < sy < self.SCREEN_HEIGHT + 50:
                angle_rad = math.radians(asteroid['rot'])
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                points = [(p[0] * cos_a - p[1] * sin_a + sx, p[0] * sin_a + p[1] * cos_a + sy) for p in asteroid['shape']]
                
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(asteroid['size']), self.COLOR_ASTEROID_GLOW)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Fuel Canisters
        for canister in self.fuel_canisters:
            sx, sy = self._world_to_screen(canister['pos'])
            if -20 < sx < self.SCREEN_WIDTH + 20 and -20 < sy < self.SCREEN_HEIGHT + 20:
                size = int(canister['size'])
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, size + 4, self.COLOR_FUEL_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, size, self.COLOR_FUEL)
                shine_angle = math.radians(canister['rot'] + 45)
                shine_x = sx + int(size * 0.5 * math.cos(shine_angle))
                shine_y = sy + int(size * 0.5 * math.sin(shine_angle))
                pygame.draw.circle(self.screen, (255,255,200), (shine_x, shine_y), 2)

        # Player
        sx, sy = self._world_to_screen(self.player_pos)
        size = int(self.PLAYER_SIZE)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, size + 8, self.COLOR_PLAYER_GLOW)
        
        angle = math.atan2(self.player_vel[1], self.player_vel[0] if self.player_vel[0] != 0 else 1)
        p1 = (sx + size * math.cos(angle), sy + size * math.sin(angle))
        p2 = (sx + size * math.cos(angle + 2.5), sy + size * math.sin(angle + 2.5))
        p3 = (sx + size * math.cos(angle - 2.5), sy + size * math.sin(angle - 2.5))
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            sx, sy = self._world_to_screen(p['pos'])
            size = int(p['size'])
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (sx, sy, size, size))

    def _render_ui(self):
        # Fuel bar
        fuel_ratio = max(0, self.fuel / self.INITIAL_FUEL)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BAR, (10, 10, int(bar_width * fuel_ratio), bar_height))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_large.render(f"TIME: {int(time_left):02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH/2 - time_text.get_width()/2, 10))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for human play and visualization.
    # It's not used by the evaluation system, which runs the environment headlessly.
    
    # Set the display driver for Pygame
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy', 'wayland', or 'windows'
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    terminated = False
    
    # Create a Pygame window for displaying the game
    pygame.display.set_caption("Asteroid Evader")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while not terminated:
        # Map keyboard inputs to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # 0: none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0 # Unused in this logic
        
        action = np.array([movement, space_held, shift_held])
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle the window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()