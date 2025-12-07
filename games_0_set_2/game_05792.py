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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓ to move, ←→ to change speed. Hold Space for a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race against time in a retro arcade world, dodging obstacles to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_TRACK = (200, 200, 200)
    COLOR_FINISH_1 = (255, 255, 255)
    COLOR_FINISH_2 = (200, 200, 200)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_BOOST_BAR = (255, 200, 0)
    COLOR_BOOST_BAR_BG = (50, 50, 50)

    # Game parameters
    FPS = 30
    MAX_STEPS = 1000
    TIME_LIMIT_SECONDS = 60

    TRACK_Y_TOP = 80
    TRACK_Y_BOTTOM = SCREEN_HEIGHT - 80
    TRACK_LENGTH = 15000  # World units

    PLAYER_X_POS = 100
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 10
    PLAYER_V_ACCEL = 1.5
    PLAYER_V_DAMPING = 0.8

    BASE_SPEED = 8.0
    MIN_SPEED = 4.0
    MAX_SPEED = 12.0
    ACCELERATION = 0.2
    BOOST_SPEED_BONUS = 10.0
    BOOST_MAX_FUEL = 100
    BOOST_CONSUMPTION_RATE = 2.5
    BOOST_REGEN_RATE = 0.5

    INITIAL_OBSTACLE_SPAWN_DIST = 500
    MIN_OBSTACLE_SPAWN_DIST = 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.player_y = 0
        self.player_vy = 0
        self.world_x = 0
        self.world_speed = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.boost_fuel = 0
        self.next_obstacle_spawn_x = 0
        self.game_over = False

        # self.reset() is not needed here as it's called by the wrapper/user

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_y = self.SCREEN_HEIGHT / 2
        self.player_vy = 0

        self.world_x = 0
        self.world_speed = self.BASE_SPEED

        self.obstacles = []
        self.particles = []
        self.next_obstacle_spawn_x = self.INITIAL_OBSTACLE_SPAWN_DIST
        self._spawn_initial_obstacles()

        self.steps = 0
        self.score = 0
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        self.boost_fuel = self.BOOST_MAX_FUEL

        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Action is expected to be a list or array of 3 integers
        movement = action[0]
        space_held = action[1] == 1
        # The third action component is unused in this logic but must be accepted

        # --- Update Game Logic ---
        self._handle_input(movement, space_held)
        self._update_player()
        self._update_world()
        self._update_particles()
        self._manage_obstacles()

        self.steps += 1
        self.timer -= 1

        # --- Collision & Termination ---
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = False # This environment does not truncate

        if terminated and not self.game_over:
            if self.world_x >= self.TRACK_LENGTH:
                # sfx: victory
                reward += 100  # Reached finish line
                self.score += 1000
            else:
                # sfx: crash
                reward -= 10  # Collision or timeout
                self._create_explosion(self.PLAYER_X_POS + self.PLAYER_WIDTH / 2, self.player_y)
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Vertical movement
        if movement == 1:  # Up
            self.player_vy -= self.PLAYER_V_ACCEL
        elif movement == 2:  # Down
            self.player_vy += self.PLAYER_V_ACCEL

        # Horizontal speed
        target_speed = self.world_speed
        if movement == 3:  # Left (Decelerate)
            target_speed = self.MIN_SPEED
        elif movement == 4:  # Right (Accelerate)
            target_speed = self.MAX_SPEED
        else:  # Drift to base speed
            target_speed = self.BASE_SPEED

        # Smoothly interpolate to target speed
        self.world_speed += np.clip(target_speed - self.world_speed, -self.ACCELERATION, self.ACCELERATION)

        # Boost
        if space_held and self.boost_fuel > 0:
            # sfx: boost_loop
            self.world_speed += self.BOOST_SPEED_BONUS
            self.boost_fuel = max(0, self.boost_fuel - self.BOOST_CONSUMPTION_RATE)
            self._create_thrust_particles(2)
        else:
            self.boost_fuel = min(self.BOOST_MAX_FUEL, self.boost_fuel + self.BOOST_REGEN_RATE)
            self._create_thrust_particles(1)

    def _update_player(self):
        self.player_vy *= self.PLAYER_V_DAMPING
        self.player_y += self.player_vy
        self.player_y = np.clip(self.player_y, self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM - self.PLAYER_HEIGHT)

    def _update_world(self):
        self.world_x += self.world_speed

    def _update_particles(self):
        for p in self.particles:
            p['x'] -= p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _manage_obstacles(self):
        # Prune off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] - self.world_x > -obs['w']]

        # Spawn new obstacles
        if self.world_x > self.next_obstacle_spawn_x:
            self._spawn_obstacle_cluster()
            difficulty_factor = 1.0 - (self.steps / self.MAX_STEPS) * 0.8
            spawn_dist = self.MIN_OBSTACLE_SPAWN_DIST + (self.INITIAL_OBSTACLE_SPAWN_DIST - self.MIN_OBSTACLE_SPAWN_DIST) * difficulty_factor
            self.next_obstacle_spawn_x += spawn_dist

    def _spawn_initial_obstacles(self):
        for i in range(5):
            self._spawn_obstacle_cluster(initial_spawn=True, cluster_idx=i)

    def _spawn_obstacle_cluster(self, initial_spawn=False, cluster_idx=0):
        if initial_spawn:
            spawn_x = cluster_idx * self.INITIAL_OBSTACLE_SPAWN_DIST
        else:
            spawn_x = self.world_x + self.SCREEN_WIDTH + 100

        gap_height = self.np_random.uniform(80, 120)
        gap_y = self.np_random.uniform(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM - gap_height)

        # Top obstacle
        h1 = gap_y - self.TRACK_Y_TOP
        if h1 > 0:
            self.obstacles.append({'x': spawn_x, 'y': self.TRACK_Y_TOP, 'w': 30, 'h': h1})

        # Bottom obstacle
        h2 = self.TRACK_Y_BOTTOM - (gap_y + gap_height)
        if h2 > 0:
            self.obstacles.append({'x': spawn_x, 'y': gap_y + gap_height, 'w': 30, 'h': h2})

    def _calculate_reward(self):
        reward = 0.1  # Base reward for survival
        if self.world_speed < self.BASE_SPEED - 1.0:
            reward -= 0.2  # Penalty for being too slow
        return reward

    def _check_termination(self):
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            return True
        if self.world_x >= self.TRACK_LENGTH:
            return True

        player_rect = pygame.Rect(self.PLAYER_X_POS, int(self.player_y), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(int(obs['x'] - self.world_x), int(obs['y']), int(obs['w']), int(obs['h']))
            if player_rect.colliderect(obs_rect):
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.timer / self.FPS,
            "distance_to_finish": max(0, self.TRACK_LENGTH - self.world_x),
        }

    def _render_game(self):
        # Track lines
        track_lines_y = [self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM]
        for y in track_lines_y:
            for i in range(-int(self.world_x % 40), self.SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, self.COLOR_TRACK, (i, y), (i + 20, y), 2)

        # Finish line
        finish_screen_x = self.TRACK_LENGTH - self.world_x
        if finish_screen_x < self.SCREEN_WIDTH:
            check_size = 20
            for y in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, check_size):
                for x_offset in range(0, 40, check_size):
                    color = self.COLOR_FINISH_1 if ((y // check_size) % 2) == (x_offset // check_size % 2) else self.COLOR_FINISH_2
                    pygame.draw.rect(self.screen, color, (int(finish_screen_x + x_offset), y, check_size, check_size))

        # Obstacles
        for obs in self.obstacles:
            screen_x = obs['x'] - self.world_x
            if screen_x < self.SCREEN_WIDTH and screen_x > -obs['w']:
                obs_rect = pygame.Rect(int(screen_x), int(obs['y']), int(obs['w']), int(obs['h']))
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * p['alpha_decay'])))
            if alpha > 0:
                self._draw_circle_alpha(self.screen, p['color'] + (alpha,), (int(p['x']), int(p['y'])), int(p['radius']))

        # Player
        if not self.game_over:
            player_rect = pygame.Rect(self.PLAYER_X_POS, int(self.player_y), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
            # Glow
            glow_rect = player_rect.inflate(10, 10)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, self.COLOR_PLAYER_GLOW + (50,), (0, 0, glow_rect.width, glow_rect.height), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)
            # Car body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Timer
        time_str = f"TIME: {max(0, self.timer // self.FPS):02d}"
        time_text = self.font_small.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 20))

        # Boost Meter
        bar_width = 150
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - 30

        fuel_ratio = self.boost_fuel / self.BOOST_MAX_FUEL

        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, (bar_x, bar_y, int(bar_width * fuel_ratio), bar_height), border_radius=3)

        # Game Over Text
        if self.game_over:
            msg = "FINISH!" if self.world_x >= self.TRACK_LENGTH else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_thrust_particles(self, num):
        px = self.PLAYER_X_POS
        py = self.player_y + self.PLAYER_HEIGHT / 2
        for _ in range(num):
            self.particles.append({
                'x': px, 'y': py,
                'vx': self.np_random.uniform(2, 5), 'vy': self.np_random.uniform(-0.5, 0.5),
                'radius': self.np_random.uniform(2, 4),
                'life': 20, 'color': (255, self.np_random.integers(150, 250), 0),
                'alpha_decay': 255 / 20
            })

    def _create_explosion(self, x, y):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            self.particles.append({
                'x': x, 'y': y,
                'vx': -math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(20, 40),
                'color': random.choice([(255, 50, 50), (255, 150, 0), (200, 200, 200)]),
                'alpha_decay': 255 / 30
            })

    def _draw_circle_alpha(self, surface, color, center, radius):
        # A workaround for pygame.draw.circle not supporting alpha
        target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius)
        surface.blit(shape_surf, target_rect)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # The main block is for human play and requires a display.
    # It will not run in a headless environment.
    # To use this, you might need to comment out the os.environ line at the top.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Create a display window only if not in dummy mode
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        pygame.display.set_caption("Arcade Racer")
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    else:
        display_screen = None
        print("Running in headless mode. No display will be shown.")

    # Game loop
    while not done:
        action = [0, 0, 0] # Default no-op action
        
        # Handle quit event and keyboard input if a display exists
        if display_screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            movement = 0  # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            # The third action is not used, so we keep it 0
            
            action = [movement, space_held, 0]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window if it exists
        if display_screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)
        
        # In headless mode, this loop will run very fast without a clock tick.
        # This is fine for testing but not for real-time simulation.

    print(f"Game Over. Final Info: {info}")
    env.close()