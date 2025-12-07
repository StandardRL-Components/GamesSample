from gymnasium.spaces import MultiDiscrete
import os
import pygame


import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random

# Pygame must run headless.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Maneuver a trio of platforms using thrusters to land them all in a designated zone. "
        "Land the first platform to start a sync timer, then land the rest before it runs out."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move horizontally and ↑↓ to apply vertical thrust. "
        "Hold space for an extra boost. Land all three platforms in the target zone to win."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    LEVELS_TO_WIN = 5

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_WALL = (50, 50, 80)
    COLOR_WALL_OUTLINE = (100, 100, 200)
    COLOR_LANDING_ZONE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 255)
    PLATFORM_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Cyan, Magenta, Yellow

    # Physics
    GRAVITY = 0.4
    HORIZONTAL_FORCE = 0.8
    VERTICAL_FORCE = 0.5
    SPACE_BOOST = 0.6
    FRICTION = 0.92
    MAX_VX = 8
    MAX_VY = 10

    # Game Mechanics
    PLATFORM_WIDTH = 40
    PLATFORM_HEIGHT = 10
    PLATFORM_SPACING = 80
    SYNC_WINDOW_FRAMES = 15  # Frames to land all platforms after the first one

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 16)
        self.font_small = pygame.font.SysFont("monospace", 12)

        # Initialize state variables that are not reset every episode
        self.platforms = []
        self.walls = []
        self.landing_zone = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.level = 1
        self.game_over_message = ""
        self.level_timer = 0
        self.landed_mask = [False, False, False]
        self.sync_timer = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.level = 1
        self.game_over_message = ""

        self._setup_level()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the state for the current level."""
        self.level_timer = 60 * self.FPS
        self.landed_mask = [False, False, False]
        self.sync_timer = -1
        self.particles = []

        # Reset platforms
        center_x = self.SCREEN_WIDTH / 2
        self.platforms = [
            {
                "x": center_x - self.PLATFORM_SPACING, "y": 50,
                "vx": 0, "vy": 0, "color": self.PLATFORM_COLORS[0]
            },
            {
                "x": center_x, "y": 50,
                "vx": 0, "vy": 0, "color": self.PLATFORM_COLORS[1]
            },
            {
                "x": center_x + self.PLATFORM_SPACING, "y": 50,
                "vx": 0, "vy": 0, "color": self.PLATFORM_COLORS[2]
            },
        ]

        # Generate procedural gap and landing zone
        gap_width = max(200 - (self.level - 1) * 20, 140)
        gap_center = self.np_random.integers(low=gap_width / 2 + 20, high=self.SCREEN_WIDTH - gap_width / 2 - 20)
        gap_y = self.SCREEN_HEIGHT * 0.6

        self.walls = [
            pygame.Rect(0, gap_y, gap_center - gap_width / 2, 20),
            pygame.Rect(gap_center + gap_width / 2, gap_y, self.SCREEN_WIDTH - (gap_center + gap_width / 2), 20)
        ]
        self.landing_zone = pygame.Rect(gap_center - gap_width / 2, gap_y + 20, gap_width, 15)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.win, not self.win, self._get_info()

        self.steps += 1
        self.level_timer -= 1

        self._handle_input(action)
        self._update_physics()

        reward, terminated = self._update_game_state()
        truncated = self.steps >= self.MAX_STEPS

        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        center_platform = self.platforms[1]

        if movement == 3:  # Left
            center_platform["vx"] -= self.HORIZONTAL_FORCE
        elif movement == 4:  # Right
            center_platform["vx"] += self.HORIZONTAL_FORCE

        vertical_mod = 0
        if movement == 1:  # Up
            vertical_mod = -self.VERTICAL_FORCE
        elif movement == 2:  # Down
            vertical_mod = self.VERTICAL_FORCE

        if space_held:
            vertical_mod -= self.SPACE_BOOST

        for p in self.platforms:
            p["vy"] += vertical_mod
            if space_held:
                self._spawn_particles(p['x'] + self.PLATFORM_WIDTH / 2, p['y'] + self.PLATFORM_HEIGHT, 1, p['color'], (0, 1), 0.5)

    def _update_physics(self):
        center_vx = self.platforms[1]["vx"]
        self.platforms[0]["vx"] = center_vx
        self.platforms[2]["vx"] = center_vx

        for p in self.platforms:
            p["vy"] += self.GRAVITY
            p["vx"] *= self.FRICTION
            p["vx"] = np.clip(p["vx"], -self.MAX_VX, self.MAX_VX)
            p["vy"] = np.clip(p["vy"], -self.MAX_VY, self.MAX_VY)
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["x"] = np.clip(p["x"], 0, self.SCREEN_WIDTH - self.PLATFORM_WIDTH)

        self.particles = [particle for particle in self.particles if particle["life"] > 1]
        for particle in self.particles:
            particle["x"] += particle["vx"]
            particle["y"] += particle["vy"]
            particle["life"] -= 1

    def _update_game_state(self):
        reward = 0.01  # Survival reward
        terminated = False

        for p in self.platforms:
            platform_rect = pygame.Rect(p["x"], p["y"], self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
            for wall in self.walls:
                if platform_rect.colliderect(wall):
                    p["vy"] *= -0.5
                    p["y"] = wall.y - self.PLATFORM_HEIGHT if p["y"] < wall.y else wall.y + wall.height

        is_sync_window_open = self.sync_timer > 0
        if not is_sync_window_open:
            self.sync_timer = -1

        for i, p in enumerate(self.platforms):
            if not self.landed_mask[i]:
                platform_rect = pygame.Rect(p["x"], p["y"], self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
                if platform_rect.colliderect(self.landing_zone):
                    if abs(p['vy']) < 3 and abs(p['vx']) < 3:
                        self.landed_mask[i] = True
                        p["vy"], p["vx"] = 0, 0
                        p["y"] = self.landing_zone.y
                        if self.sync_timer == -1:
                            self.sync_timer = self.SYNC_WINDOW_FRAMES
                    else:
                        p["vy"] *= -0.5

        if self.sync_timer > 0:
            self.sync_timer -= 1

        if all(self.landed_mask):
            reward += 10
            self.score += 10
            self._spawn_particles(self.landing_zone.centerx, self.landing_zone.centery, 100, self.COLOR_LANDING_ZONE)
            self.level += 1
            if self.level > self.LEVELS_TO_WIN:
                reward += 500
                self.score += 500
                terminated = True
                self.win = True
                self.game_over_message = "YOU WIN!"
            else:
                reward += 100
                self.score += 100
                self._setup_level()
            return reward, terminated

        for p in self.platforms:
            if p["y"] > self.SCREEN_HEIGHT:
                reward -= 100
                terminated = True
                self.game_over_message = "PLATFORM LOST"
                return reward, terminated

        if self.sync_timer == 0 and not all(self.landed_mask):
            reward -= 50
            terminated = True
            self.game_over_message = "SYNC FAILED"
            return reward, terminated

        if self.level_timer <= 0:
            reward -= 50
            terminated = True
            self.game_over_message = "TIME'S UP"
            return reward, terminated

        return reward, terminated

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def _render_game(self):
        for y in range(self.SCREEN_HEIGHT):
            color = [self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * (y / self.SCREEN_HEIGHT) for i in range(3)]
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        self._render_particles()
        self._render_level_elements()
        self._render_platforms()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

    def _render_level_elements(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            pygame.draw.rect(self.screen, self.COLOR_WALL_OUTLINE, wall, 2)
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = 100 + int(pulse * 155)
        glow_rect = self.landing_zone.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        glow_color = self.COLOR_LANDING_ZONE + (int(alpha / 3),)
        pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_LANDING_ZONE, self.landing_zone, border_radius=3)
        if self.sync_timer > 0:
            ratio = self.sync_timer / self.SYNC_WINDOW_FRAMES
            bar_width = self.landing_zone.width * ratio
            bar_rect = pygame.Rect(self.landing_zone.x, self.landing_zone.y - 10, bar_width, 5)
            pygame.draw.rect(self.screen, self.COLOR_LANDING_ZONE, bar_rect)

    def _render_platforms(self):
        for i, p in enumerate(self.platforms):
            if self.landed_mask[i]: continue
            platform_rect = pygame.Rect(int(p["x"]), int(p["y"]), self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
            color = p["color"]
            glow_rect = platform_rect.inflate(10, 10)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            glow_color = color + (64,)
            pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=8)
            self.screen.blit(s, (int(p["x"] - 5), int(p["y"] - 5)))
            pygame.draw.rect(self.screen, color, platform_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 255, 255), platform_rect, 1, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(p["size"]), color)

    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        level_text = self.font_large.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(level_text, level_rect)
        time_left = max(0, self.level_timer / self.FPS)
        timer_text = self.font_medium.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(right=self.SCREEN_WIDTH - 10, y=10)
        self.screen.blit(timer_text, timer_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        color = (0, 255, 100) if self.win else (255, 50, 50)
        end_text = self.font_large.render(self.game_over_message, True, color)
        end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(end_text, end_rect)
        score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
        self.screen.blit(score_text, score_rect)

    def _spawn_particles(self, x, y, count, color, direction=(0, 0), speed=3):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_x = math.cos(angle) * random.uniform(0.5, 1) * speed + direction[0]
            vel_y = math.sin(angle) * random.uniform(0.5, 1) * speed + direction[1]
            self.particles.append({
                "x": x, "y": y, "vx": vel_x, "vy": vel_y,
                "life": random.randint(20, 40), "max_life": 40,
                "color": color, "size": random.randint(2, 5)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()

        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Platform Synchronizer")
        clock = pygame.time.Clock()

        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            action = [movement, space_held, shift_held]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated, truncated = False, False

            clock.tick(GameEnv.FPS)

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        env.close()
    except pygame.error as e:
        print(f"Could not run in interactive mode: {e}")
        print("This is expected in a headless environment.")