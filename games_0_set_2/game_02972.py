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


class Particle:
    """A simple class for managing particles for visual effects."""

    def __init__(self, x, y, color, size, lifetime, velocity_range, rng):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.initial_lifetime = lifetime
        self.lifetime = lifetime
        self.vx = rng.uniform(*velocity_range[0])
        self.vy = rng.uniform(*velocity_range[1])
        self.gravity = 0.1

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.lifetime -= 1
        self.size = max(0, self.size * (self.lifetime / self.initial_lifetime))

    def draw(self, surface):
        if self.lifetime > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ↑ for a high jump, ↓ for a short hop. Avoid the obstacles!"
    )
    game_description = (
        "Guide a neon line through a procedurally generated obstacle course to reach the finish line as quickly as possible."
    )

    # Frame advance behavior
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 50

    # Colors
    COLOR_BG = (10, 10, 26)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_OBSTACLE = (255, 140, 0)
    COLOR_OBSTACLE_GLOW = (128, 70, 0)
    COLOR_START = (0, 255, 0)
    COLOR_FINISH = (255, 0, 0)
    COLOR_GROUND = (100, 100, 120)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE_CRASH = (255, 80, 80)

    # Player physics
    PLAYER_X_POS = 120
    PLAYER_WIDTH = 4
    PLAYER_HEIGHT = 20
    GRAVITY = 0.5
    HIGH_JUMP_STRENGTH = -10.5
    SHORT_HOP_STRENGTH = -7.5
    GROUND_Y = SCREEN_HEIGHT - 50

    # Game parameters
    COURSE_LENGTH = 8000
    PLAYER_SPEED = COURSE_LENGTH / (20 * FPS)
    MAX_STEPS = 2500  # 50 seconds, allows for slower runs
    TIME_LIMIT_SECONDS = 20.0

    # Obstacle parameters
    INITIAL_OBSTACLE_SPEED = PLAYER_SPEED
    INITIAL_SPAWN_RATE = 80
    MIN_SPAWN_RATE = 35

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = [0, 0]
        self.player_vel_y = 0.0
        self.on_ground = True
        self.world_scroll = 0.0
        self.obstacles = []
        self.particles = []
        self.obstacle_speed = 0.0
        self.obstacle_spawn_rate = 0.0
        self.spawn_timer = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.background_stars = []

        # self.reset() is called by the wrapper, but we can call it to initialize state
        # In this case, it is called by the original code, so we keep it.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.PLAYER_X_POS, self.GROUND_Y - self.PLAYER_HEIGHT]
        self.player_vel_y = 0.0
        self.on_ground = True

        self.world_scroll = 0.0
        self.obstacles = []
        self.particles = []

        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_spawn_rate = self.INITIAL_SPAWN_RATE
        self.spawn_timer = self.INITIAL_SPAWN_RATE

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False

        # Pre-populate some stars for the parallax background
        self.background_stars = [
            [[self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)], 0.2 + self.np_random.random() * 0.6]
            for _ in range(150)
        ]

        # Pre-populate initial obstacles
        for i in range(5):
            self._spawn_obstacle(initial_offset=(i + 1) * 300)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0.0

        if not self.game_over:
            self._handle_input(movement)
            self._update_player()
            self._update_world()
            self._update_difficulty()
            passed_obstacle = self._update_obstacles()
            self._handle_collisions()

            # Calculate reward
            reward += 0.01  # Small reward for surviving
            if passed_obstacle:
                reward += 5.0

        self._update_particles()

        self.steps += 1
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS
        truncated = False # This environment does not truncate based on time limit

        if terminated:
            if self.game_over:
                reward = -100.0
            elif self.game_won:
                elapsed_time = self.steps / self.FPS
                if elapsed_time < self.TIME_LIMIT_SECONDS:
                    time_bonus = 100 * (1 - (elapsed_time / self.TIME_LIMIT_SECONDS))
                    reward += time_bonus
                else:  # Won, but over time limit
                    reward += 10  # Small consolation prize

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if self.on_ground:
            if movement == 1:  # Up for high jump
                self.player_vel_y = self.HIGH_JUMP_STRENGTH
                self.on_ground = False
                # sfx: jump_high.wav
            elif movement == 2:  # Down for short hop
                self.player_vel_y = self.SHORT_HOP_STRENGTH
                self.on_ground = False
                # sfx: jump_short.wav

    def _update_player(self):
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        # Check for ground collision
        if self.player_pos[1] >= self.GROUND_Y - self.PLAYER_HEIGHT:
            self.player_pos[1] = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vel_y = 0
            if not self.on_ground:
                # sfx: land.wav
                self.on_ground = True

    def _update_world(self):
        self.world_scroll += self.PLAYER_SPEED

        # Check for win condition
        if self.world_scroll >= self.COURSE_LENGTH and not self.game_won:
            self.game_won = True
            # sfx: win_jingle.wav

    def _update_difficulty(self):
        # Increase obstacle spawn frequency
        if self.steps > 0 and self.steps % 50 == 0:
            self.obstacle_spawn_rate = max(self.MIN_SPAWN_RATE, self.obstacle_spawn_rate * 0.99)
        # Increase obstacle speed
        if self.steps > 0 and self.steps % 250 == 0:
            self.obstacle_speed += 0.05

    def _update_obstacles(self):
        passed_obstacle = False

        # Spawn new obstacles
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_obstacle()
            self.spawn_timer = int(self.obstacle_spawn_rate)

        # Update and check for passed obstacles
        for obs in self.obstacles:
            if not obs['passed'] and obs['rect'].right < self.world_scroll + self.PLAYER_X_POS:
                obs['passed'] = True
                passed_obstacle = True

        # Remove off-screen obstacles to the left
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > self.world_scroll]

        return passed_obstacle

    def _spawn_obstacle(self, initial_offset=0):
        obs_w = self.np_random.integers(40, 100)
        obs_h = self.np_random.integers(30, 120)

        is_ground_obstacle = self.np_random.choice([True, False])

        if is_ground_obstacle:
            obs_y = self.GROUND_Y - obs_h
        else:
            min_clearance = self.PLAYER_HEIGHT + abs(self.HIGH_JUMP_STRENGTH) * 15
            obs_y = self.np_random.integers(50, self.GROUND_Y - obs_h - min_clearance)

        obs_x = self.world_scroll + self.SCREEN_WIDTH + initial_offset
        rect = pygame.Rect(obs_x, obs_y, obs_w, obs_h)
        self.obstacles.append({'rect': rect, 'passed': False})

    def _handle_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].move(-self.world_scroll, 0)
            if player_rect.colliderect(obs_screen_rect):
                self.game_over = True
                self._create_crash_particles(player_rect.center)
                # sfx: crash.wav
                break

    def _create_crash_particles(self, pos):
        for _ in range(30):
            self.particles.append(Particle(
                pos[0], pos[1], self.COLOR_PARTICLE_CRASH,
                self.np_random.random() * 5 + 2,
                self.np_random.integers(20, 40),
                ((-5, 5), (-8, 2)),
                self.np_random
            ))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_world()
        self._render_player()
        self._render_particles()
        self._render_ui()

        if self.game_over or self.game_won:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_effects(self):
        for star in self.background_stars:
            pos, speed_multiplier = star
            pos[0] = (pos[0] - self.PLAYER_SPEED * speed_multiplier) % self.SCREEN_WIDTH
            color_val = int(100 * speed_multiplier)
            pygame.gfxdraw.pixel(self.screen, int(pos[0]), int(pos[1]), (color_val, color_val, color_val + 20))

    def _render_world(self):
        # Ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)

        # Start/Finish Lines
        start_x = int(0 - self.world_scroll)
        if 0 < start_x < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_START, (start_x, 0), (start_x, self.GROUND_Y), 3)

        finish_x = int(self.COURSE_LENGTH - self.world_scroll)
        if 0 < finish_x < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.GROUND_Y), 3)

        # Obstacles
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].move(-self.world_scroll, 0)
            if obs_screen_rect.right > 0 and obs_screen_rect.left < self.SCREEN_WIDTH:
                # Glow effect
                glow_rect = obs_screen_rect.inflate(6, 6)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=3)
                # Main shape
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_screen_rect, border_radius=3)

    def _render_player(self):
        p_x, p_y = int(self.player_pos[0]), int(self.player_pos[1])
        p_h = self.PLAYER_HEIGHT

        # Glow effect
        glow_thickness = 10
        pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, (p_x, p_y), (p_x, p_y + p_h), glow_thickness)

        # Main line
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (p_x, p_y), (p_x, p_y + p_h), self.PLAYER_WIDTH)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Timer
        elapsed_time = self.steps / self.FPS
        timer_text = f"Time: {elapsed_time:.2f}s"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Progress
        progress = min(100, (self.world_scroll / self.COURSE_LENGTH) * 100)
        progress_text = f"Progress: {progress:.1f}%"
        progress_surf = self.font_ui.render(progress_text, True, self.COLOR_TEXT)
        self.screen.blit(progress_surf, (self.SCREEN_WIDTH - progress_surf.get_width() - 10, 10))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        if self.game_won:
            text = "FINISH!"
            color = self.COLOR_START
        else:
            text = "GAME OVER"
            color = self.COLOR_FINISH

        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": (self.world_scroll / self.COURSE_LENGTH),
            "game_won": self.game_won,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    # To use, you might need to unset the dummy video driver
    # and install pygame dependencies:
    # unset SDL_VIDEODRIVER
    # pip install pygame
    
    # For headed mode, comment out the os.environ line at the top
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Line Jumper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    running = True
    total_reward = 0.0

    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1  # High jump
        elif keys[pygame.K_DOWN]:
            action[0] = 2  # Short hop

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Info: {info}")
            # The game will show its end screen, wait for 'R' to reset.

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(GameEnv.FPS)

    env.close()