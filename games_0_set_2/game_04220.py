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
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to turn. The car accelerates automatically. Avoid the blue obstacles and reach the green finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade racer. Navigate a procedurally generated obstacle course against the clock. The thrill is in the near misses and precise control."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    COLOR_BG = (10, 10, 25)
    COLOR_PLAYER = (255, 0, 100)
    COLOR_PLAYER_GLOW = (255, 0, 100, 50)
    COLOR_OBSTACLE = (0, 200, 255)
    COLOR_FINISH_LINE = (0, 255, 150)
    COLOR_TRACK_LINES = (200, 200, 200)
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rng = np.random.default_rng()

        self.player_world_pos = None
        self.player_angle = None
        self.player_speed = 7.0
        self.turn_speed = 0.08

        self.obstacles = []
        self.particles = []

        self.base_obstacle_speed = 2.0
        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_rate = 20  # steps

        self.finish_line_y = -8000
        self.track_width = 1000

        # self.reset() is called by the wrapper/runner, no need to call it here.
        # self.validate_implementation() is for debugging, not needed in final class.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_world_pos = np.array([self.track_width / 2, 0.0])
        self.player_angle = -math.pi / 2  # Pointing "up"

        self.obstacles = []
        self.particles = []
        self.base_obstacle_speed = 2.0
        self.obstacle_spawn_timer = 0

        # Pre-populate some obstacles
        for _ in range(20):
            self._spawn_obstacle(initial_spawn=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]

        # --- Update Game Logic ---
        self.steps += 1
        reward = 0.1  # Survival reward

        # 1. Handle player movement
        if movement == 3:  # Left
            self.player_angle -= self.turn_speed
        elif movement == 4:  # Right
            self.player_angle += self.turn_speed

        # Normalize angle
        self.player_angle = (self.player_angle + math.pi) % (2 * math.pi) - math.pi

        move_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        self.player_world_pos += move_vec * self.player_speed

        # 2. Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_obstacle_speed += 0.05

        # 3. Update obstacles and check for collisions/near misses
        player_rect = pygame.Rect(self.WIDTH / 2 - 5, self.HEIGHT / 2 - 10, 10, 20)

        for obs in self.obstacles:
            obs['pos'] += obs['vel']

            # Create a rect for the obstacle relative to the player
            screen_pos = self._world_to_screen(obs['pos'])
            obs_rect = pygame.Rect(screen_pos[0] - obs['size'] / 2, screen_pos[1] - obs['size'] / 2, obs['size'],
                                  obs['size'])

            if not obs['dodged']:
                # Collision check
                if player_rect.colliderect(obs_rect):
                    self.game_over = True
                    reward = -100
                    self._create_explosion(self.WIDTH / 2, self.HEIGHT / 2, self.COLOR_PLAYER)
                    break

                # Near miss check
                near_miss_rect = obs_rect.inflate(60, 60)
                if near_miss_rect.colliderect(player_rect):
                    obs['dodged'] = True
                    reward += 5
                    self.score += 50
                    self._create_sparks(obs['pos'])

        # 4. Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] - p['decay'])

        # 5. Spawn/Despawn obstacles
        self.obstacle_spawn_timer += 1
        if self.obstacle_spawn_timer >= self.obstacle_spawn_rate:
            self.obstacle_spawn_timer = 0
            self._spawn_obstacle()

        # Remove obstacles far behind the player
        self.obstacles = [obs for obs in self.obstacles if obs['pos'][1] < self.player_world_pos[1] + self.HEIGHT]

        # 6. Check for termination conditions
        terminated = self.game_over
        truncated = False
        if not terminated:
            if self.player_world_pos[1] <= self.finish_line_y:
                terminated = True
                self.game_over = True
                reward = 100
                self.score += 1000
            elif self.steps >= self.MAX_STEPS:
                truncated = True # Use truncated for time limit
                self.game_over = True
                reward = -10

        self.score += 1  # Time-based score

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_obstacle(self, initial_spawn=False):
        if initial_spawn:
            # Spawn in a wide area around the player's start
            # FIX: low must be less than high.
            y_offset = self.rng.uniform(-self.HEIGHT * 15, -self.HEIGHT * 2)
        else:
            # Spawn ahead of the player, off-screen
            y_offset = -self.HEIGHT * 1.5

        x = self.rng.uniform(0, self.track_width)
        y = self.player_world_pos[1] + y_offset
        pos = np.array([x, y])

        size = self.rng.integers(20, 50)

        # Random velocity pattern
        pattern = self.rng.choice(['h', 'v', 'd'])
        speed = self.base_obstacle_speed + self.rng.uniform(-0.5, 1.5)
        if pattern == 'h':
            vel = np.array([self.rng.choice([-1, 1]) * speed / 2, 0.0])
        elif pattern == 'v':
            vel = np.array([0.0, speed])
        else:  # diagonal
            vel = np.array([self.rng.choice([-1, 1]) * speed / 2, speed * 0.75])

        self.obstacles.append({'pos': pos, 'vel': vel, 'size': size, 'dodged': False})

    def _world_to_screen(self, world_pos):
        # Center the world on the player
        screen_x = world_pos[0] - self.player_world_pos[0] + self.WIDTH / 2
        screen_y = world_pos[1] - self.player_world_pos[1] + self.HEIGHT / 2
        return np.array([screen_x, screen_y])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        # --- Render Game Elements ---
        self._render_track_lines()
        self._render_finish_line()
        self._render_obstacles()
        self._render_particles()
        if not self.game_over:
            self._render_player()

        # --- Render UI ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track_lines(self):
        # Give a sense of speed and boundaries
        for i in range(-2, self.HEIGHT // 50 + 2):
            # Left boundary
            left_p1 = self._world_to_screen(np.array([0, self.player_world_pos[1] + (i - 1) * 50]))
            left_p2 = self._world_to_screen(np.array([0, self.player_world_pos[1] + i * 50]))

            # Right boundary
            right_p1 = self._world_to_screen(np.array([self.track_width, self.player_world_pos[1] + (i - 1) * 50]))
            right_p2 = self._world_to_screen(np.array([self.track_width, self.player_world_pos[1] + i * 50]))

            pygame.draw.aaline(self.screen, self.COLOR_TRACK_LINES, left_p1, left_p2)
            pygame.draw.aaline(self.screen, self.COLOR_TRACK_LINES, right_p1, right_p2)

    def _render_finish_line(self):
        finish_pos = self._world_to_screen(np.array([self.track_width / 2, self.finish_line_y]))
        if -10 < finish_pos[1] < self.HEIGHT + 10:
            pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (0, int(finish_pos[1]), self.WIDTH, 10))

    def _render_obstacles(self):
        for obs in self.obstacles:
            screen_pos = self._world_to_screen(obs['pos'])
            if -obs['size'] < screen_pos[0] < self.WIDTH + obs['size'] and \
                    -obs['size'] < screen_pos[1] < self.HEIGHT + obs['size']:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE,
                                 (int(screen_pos[0] - obs['size'] / 2), int(screen_pos[1] - obs['size'] / 2),
                                  obs['size'], obs['size']))

    def _render_player(self):
        # Player is always at the center of the screen
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2

        # Define triangle points relative to center
        p1 = (center_x + math.cos(self.player_angle) * 15, center_y + math.sin(self.player_angle) * 15)
        p2 = (center_x + math.cos(self.player_angle + 2.5) * 10, center_y + math.sin(self.player_angle + 2.5) * 10)
        p3 = (center_x + math.cos(self.player_angle - 2.5) * 10, center_y + math.sin(self.player_angle - 2.5) * 10)

        # Glow effect
        glow_p1 = (center_x + math.cos(self.player_angle) * 25, center_y + math.sin(self.player_angle) * 25)
        glow_p2 = (center_x + math.cos(self.player_angle + 2.2) * 20, center_y + math.sin(self.player_angle + 2.2) * 20)
        glow_p3 = (center_x + math.cos(self.player_angle - 2.2) * 20, center_y + math.sin(self.player_angle - 2.2) * 20)

        pygame.gfxdraw.aapolygon(self.screen, (glow_p1, glow_p2, glow_p3), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, (glow_p1, glow_p2, glow_p3), self.COLOR_PLAYER_GLOW)

        # Main triangle
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'])
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(p['radius']),
                                            p['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Time remaining
        steps_left = max(0, self.MAX_STEPS - self.steps)
        time_left = steps_left / self.FPS
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Game Over message
        if self.game_over:
            if self.player_world_pos[1] <= self.finish_line_y:
                msg = "FINISH!"
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
            else:
                msg = "CRASHED"

            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_finish": max(0, self.player_world_pos[1] - self.finish_line_y),
        }

    def _create_sparks(self, world_pos):
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': world_pos.copy(),
                'vel': vel,
                'lifespan': self.rng.integers(10, 20),
                'radius': self.rng.uniform(1, 3),
                'decay': 0.1,
                'color': (255, 255, 100, 150)
            })

    def _create_explosion(self, screen_x, screen_y, color):
        world_pos = self.player_world_pos.copy()
        for _ in range(50):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 8)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': world_pos.copy(),
                'vel': vel,
                'lifespan': self.rng.integers(20, 40),
                'radius': self.rng.uniform(2, 5),
                'decay': 0.15,
                'color': color + (self.rng.integers(100, 200),)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set render_mode to "human" to see the game window
    env = GameEnv()
    obs, info = env.reset(seed=42)

    # Use a separate screen for human rendering
    human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)

    terminated = False
    truncated = False
    total_reward = 0

    print(GameEnv.user_guide)

    while not (terminated or truncated):
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0  # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()