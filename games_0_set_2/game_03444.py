import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓ to steer. Press space to use a boost. Hold shift for a visual drift effect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race your retro car through a perilous track. Dodge obstacles, grab boosts, and set a new lap record!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_TRACK = (50, 50, 70)
        self.COLOR_CAR = (255, 50, 50)
        self.COLOR_OBSTACLE = (50, 200, 255)
        self.COLOR_POWERUP = (255, 255, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_FINISH_1 = (255, 255, 255)
        self.COLOR_FINISH_2 = (0, 0, 0)

        # Game constants
        self.MAX_STEPS = 1500  # Increased to allow for 3 laps
        self.LAPS_TO_WIN = 3
        self.TRACK_LENGTH = 15000
        self.CAR_X_POS = self.WIDTH // 4
        self.BASE_SPEED = 8.0
        self.BOOST_SPEED_MULTIPLIER = 2.5
        self.BOOST_DURATION = 90  # 3 seconds at 30fps
        self.CAR_ACCEL = 0.8
        self.CAR_FRICTION = 0.92
        self.MAX_VY = 8.0
        self.OBSTACLE_DENSITY_INITIAL = 0.001
        self.POWERUP_DENSITY = 0.0003
        self.NEAR_MISS_RADIUS = 50
        self.TRACK_WIDTH_BASE = 120
        self.TRACK_WAVE_AMP = 50
        self.TRACK_WAVE_FREQ = 0.001

        # Initialize state variables
        self.np_random = None
        self.car_y = 0
        self.car_vy = 0
        self.world_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laps = 0
        self.lap_start_step = 0
        self.obstacles = []
        self.powerups = []
        self.particles = []
        self.finish_line_x = 0
        self.boost_timer = 0
        self.boost_inventory = 0
        self.obstacle_density = 0

        # This call is needed to initialize the np_random generator
        # before any methods that use it are called.
        self.reset(seed=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None: # Initialize RNG only if it doesn't exist
            self.np_random = np.random.default_rng(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laps = 0
        self.lap_start_step = 0

        self.car_y = self.HEIGHT / 2
        self.car_vy = 0
        self.world_x = 0

        self.obstacles = []
        self.powerups = []
        self.particles = []

        self.boost_timer = 0
        self.boost_inventory = 0
        self.finish_line_x = self.TRACK_LENGTH
        self.obstacle_density = self.OBSTACLE_DENSITY_INITIAL

        # Pre-generate the first screen of the world
        self._generate_world_segment(0, self.WIDTH + 100)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_held = action[2] == 1

        # --- 1. Handle Input & Update Player ---
        self._handle_input(movement, space_pressed, shift_held)
        self._update_player()

        # --- 2. Update World ---
        speed = self.BASE_SPEED * (self.BOOST_SPEED_MULTIPLIER if self.boost_timer > 0 else 1.0)
        prev_world_x = self.world_x
        self.world_x += speed

        self._update_world(speed)
        self._generate_world_segment(prev_world_x + self.WIDTH, self.world_x + self.WIDTH)

        # --- 3. Collisions and Rewards ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- 4. Lap and Game State Logic ---
        # Survival reward
        reward += 0.1

        # Lap completion
        if prev_world_x < self.finish_line_x and self.world_x >= self.finish_line_x:
            self.laps += 1
            self.lap_start_step = self.steps
            self.obstacle_density *= 1.05  # Increase difficulty
            self.finish_line_x += self.TRACK_LENGTH

            if self.laps >= self.LAPS_TO_WIN:
                reward += 300  # Race completion bonus
                self.game_over = True
            else:
                reward += 100  # Lap completion bonus
            # Sound: Lap complete fanfare

        # --- 5. Finalize Step ---
        self.steps += 1
        self.score += reward

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False # Per API, truncated is for time limits, terminated for game-end states

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_held):
        # Vertical movement
        if movement == 1:  # Up
            self.car_vy -= self.CAR_ACCEL
        elif movement == 2:  # Down
            self.car_vy += self.CAR_ACCEL

        # Use boost
        if space_pressed and self.boost_inventory > 0 and self.boost_timer <= 0:
            self.boost_inventory -= 1
            self.boost_timer = self.BOOST_DURATION
            self._spawn_particles(20, (self.CAR_X_POS, self.car_y), self.COLOR_POWERUP, 8, (-15, -2), (-5, 5))
            # Sound: Boost activation

        # Visual drift effect
        if shift_held:
            self._spawn_particles(1, (self.CAR_X_POS - 10, self.car_y + 5), self.COLOR_TEXT, 2, (-3, -1), (-1, 1), 10)
            self._spawn_particles(1, (self.CAR_X_POS - 10, self.car_y - 5), self.COLOR_TEXT, 2, (-3, -1), (-1, 1), 10)

    def _update_player(self):
        # Apply friction
        self.car_vy *= self.CAR_FRICTION
        self.car_vy = np.clip(self.car_vy, -self.MAX_VY, self.MAX_VY)

        # Update position
        self.car_y += self.car_vy

        # Track boundaries
        track_center_y = self.HEIGHT / 2 + self.TRACK_WAVE_AMP * math.sin(self.world_x * self.TRACK_WAVE_FREQ)
        track_half_width = self.TRACK_WIDTH_BASE

        self.car_y = np.clip(self.car_y, track_center_y - track_half_width, track_center_y + track_half_width)

        # Update boost timer
        if self.boost_timer > 0:
            self.boost_timer -= 1
            # Engine boost particles
            self._spawn_particles(2, (self.CAR_X_POS - 15, self.car_y), self.COLOR_OBSTACLE, 4, (-10, -5), (-2, 2))

    def _update_world(self, speed):
        # Update entities
        for entity_list in [self.obstacles, self.powerups]:
            for entity in entity_list:
                entity['x'] -= speed

        # Update particles
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1

        # Remove off-screen and dead entities
        self.obstacles = [o for o in self.obstacles if o['x'] > -20]
        self.powerups = [p for p in self.powerups if p['x'] > -20]
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _generate_world_segment(self, start_x, end_x):
        num_steps = int((end_x - start_x) / self.BASE_SPEED)
        for i in range(num_steps):
            current_x = start_x + i * self.BASE_SPEED
            track_center_y = self.HEIGHT / 2 + self.TRACK_WAVE_AMP * math.sin(current_x * self.TRACK_WAVE_FREQ)
            track_half_width = self.TRACK_WIDTH_BASE - 20  # Keep entities off the very edge

            if self.np_random.random() < self.obstacle_density:
                self.obstacles.append({
                    'x': current_x + self.WIDTH,
                    'y': track_center_y + self.np_random.uniform(-track_half_width, track_half_width),
                    'r': 10
                })

            if self.np_random.random() < self.POWERUP_DENSITY:
                self.powerups.append({
                    'x': current_x + self.WIDTH,
                    'y': track_center_y + self.np_random.uniform(-track_half_width, track_half_width),
                    'r': 8
                })

    def _handle_collisions(self):
        reward = 0
        car_rect = pygame.Rect(self.CAR_X_POS - 12, self.car_y - 6, 24, 12)

        # Obstacles
        for obs in self.obstacles:
            dist = math.hypot(obs['x'] - self.CAR_X_POS, obs['y'] - self.car_y)
            if dist < obs['r'] + 10:  # Collision
                self.game_over = True
                reward = -100  # Terminal penalty
                self._spawn_particles(50, (self.CAR_X_POS, self.car_y), self.COLOR_OBSTACLE, 5)
                # Sound: Explosion
                return reward
            elif dist < self.NEAR_MISS_RADIUS:
                reward -= 0.01 # Small penalty for being near obstacles

        # Power-ups
        collected_powerups = []
        for i, p_up in enumerate(self.powerups):
            dist = math.hypot(p_up['x'] - self.CAR_X_POS, p_up['y'] - self.car_y)
            if dist < p_up['r'] + 10:
                reward += 5
                self.boost_inventory = min(3, self.boost_inventory + 1)
                collected_powerups.append(i)
                self._spawn_particles(30, (p_up['x'], p_up['y']), self.COLOR_POWERUP, 4)
                # Sound: Powerup collect

        # Remove collected powerups
        for i in sorted(collected_powerups, reverse=True):
            self.powerups.pop(i)

        return reward

    def _spawn_particles(self, count, pos, color, radius_max, vx_range=(-5, 5), vy_range=(-5, 5), lifespan=30):
        for _ in range(count):
            self.particles.append({
                'x': pos[0],
                'y': pos[1],
                'vx': self.np_random.uniform(*vx_range),
                'vy': self.np_random.uniform(*vy_range),
                'radius': self.np_random.uniform(1, radius_max),
                'color': color,
                'lifespan': self.np_random.integers(lifespan // 2, lifespan + 1)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render track
        for i in range(0, self.WIDTH, 10):
            x = self.world_x + i
            center_y = self.HEIGHT / 2 + self.TRACK_WAVE_AMP * math.sin(x * self.TRACK_WAVE_FREQ)
            track_width = self.TRACK_WIDTH_BASE

            # Draw a segment of the track path
            pygame.draw.rect(self.screen, self.COLOR_TRACK, (i, center_y - track_width, 10, track_width * 2))

        # Render finish line
        finish_screen_x = self.finish_line_x - self.world_x
        if 0 < finish_screen_x < self.WIDTH:
            for i in range(0, self.HEIGHT, 20):
                color = self.COLOR_FINISH_1 if (i // 20) % 2 == 0 else self.COLOR_FINISH_2
                pygame.draw.rect(self.screen, color, (finish_screen_x, i, 10, 20))

        # Render particles
        for p in self.particles:
            if p['lifespan'] > 0:
                alpha = int(255 * (p['lifespan'] / 30))
                try:
                    s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*p['color'], alpha), (p['radius'], p['radius']), p['radius'])
                    self.screen.blit(s, (p['x']-p['radius'], p['y']-p['radius']))
                except (ValueError, TypeError): # Catch potential errors with color alpha
                    pass

        # Render power-ups (flickering)
        flicker = 1 + 0.2 * math.sin(self.steps * 0.5)
        for p_up in self.powerups:
            r = int(p_up['r'] * flicker)
            if r > 0:
                pygame.gfxdraw.aacircle(self.screen, int(p_up['x']), int(p_up['y']), r, self.COLOR_POWERUP)
                pygame.gfxdraw.filled_circle(self.screen, int(p_up['x']), int(p_up['y']), r, self.COLOR_POWERUP)

        # Render obstacles
        for obs in self.obstacles:
            pygame.gfxdraw.aacircle(self.screen, int(obs['x']), int(obs['y']), obs['r'], self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_circle(self.screen, int(obs['x']), int(obs['y']), obs['r'], self.COLOR_OBSTACLE)

        # Render car
        if not self.game_over:
            # Glow effect
            glow_radius = 18 + 5 * (self.boost_timer / self.BOOST_DURATION)
            glow_alpha = 50 + 50 * (self.boost_timer / self.BOOST_DURATION)
            
            # Use a surface for alpha blending
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_CAR, int(glow_alpha)), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (self.CAR_X_POS-glow_radius, int(self.car_y)-glow_radius))

            # Car body (as a simple polygon)
            p1 = (self.CAR_X_POS + 12, int(self.car_y))
            p2 = (self.CAR_X_POS - 10, int(self.car_y - 6))
            p3 = (self.CAR_X_POS - 10, int(self.car_y + 6))
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_CAR)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_CAR)

    def _render_ui(self):
        # Lap counter
        lap_text = f"LAP: {min(self.laps + 1, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}"
        lap_surf = self.font_small.render(lap_text, True, self.COLOR_TEXT)
        self.screen.blit(lap_surf, (10, 10))

        # Lap time
        elapsed_steps = self.steps - self.lap_start_step
        lap_time = elapsed_steps / 30.0  # Assuming 30fps
        time_text = f"TIME: {lap_time:.2f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Boost indicator
        for i in range(self.boost_inventory):
            pygame.draw.rect(self.screen, self.COLOR_POWERUP, (10 + i * 25, 40, 20, 10))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (10 + i * 25, 40, 20, 10), 1)

        # Game Over / Win message
        if self.game_over:
            if self.laps >= self.LAPS_TO_WIN:
                msg = "RACE COMPLETE!"
            else:
                msg = "GAME OVER"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps,
            "lap_time": (self.steps - self.lap_start_step) / 30.0,
            "world_x": self.world_x,
            "boosts": self.boost_inventory,
        }

    def close(self):
        pygame.quit()


# Example usage:
if __name__ == "__main__":
    # To run with a display, set the SDL_VIDEODRIVER to a supported backend.
    # 'x11' for Linux, 'windows' for Windows, 'mac' for macOS.
    # If you have issues, you might need to install SDL2.
    os.environ['SDL_VIDEODRIVER'] = 'x11' 

    env = GameEnv(render_mode="rgb_array")

    # --- To run the game with manual controls ---
    pygame.display.set_caption("Arcade Racer")
    game_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False

    # Main game loop
    running = True
    while running:
        # Action defaults
        movement = 0  # none
        space = 0  # released
        shift = 0  # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2

        if keys[pygame.K_SPACE]:
            space = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)  # Control the frame rate

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Laps: {info['laps']}")
            pygame.time.wait(2000)  # Pause for 2 seconds
            obs, info = env.reset()

    env.close()