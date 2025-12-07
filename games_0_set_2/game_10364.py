import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set Pygame to run in headless mode for the environment
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Guide a small ant to grow by consuming aphids while avoiding dangerous spiders. "
        "Reach the queen to achieve victory before time runs out or you shrink to nothing."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the ant. "
        "Press the spacebar when near an aphid to consume it and grow."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 25, 15)
    COLOR_ANT = (100, 255, 100)
    COLOR_ANT_OUTLINE = (50, 150, 50)
    COLOR_SPIDER = (20, 20, 20)
    COLOR_SPIDER_EYES = (255, 0, 0)
    COLOR_QUEEN = (255, 255, 255)
    COLOR_QUEEN_GLOW = (255, 255, 200)
    APHID_COLORS = {
        "red": (255, 50, 50),
        "blue": (50, 100, 255),
        "yellow": (255, 255, 50),
    }
    COLOR_UI_TEXT = (220, 220, 220)

    # Game Parameters
    ANT_START_SIZE = 10
    ANT_MAX_SIZE = 50
    ANT_SPEED = 4.0
    ANT_MATCH_RADIUS = 25

    SPIDER_START_COUNT = 3
    SPIDER_SIZE = 15
    SPIDER_START_SPEED = 1.0
    SPIDER_MAX_SPEED = 2.5
    SPIDER_SPEED_INCREASE_INTERVAL = 200
    SPIDER_SPEED_INCREASE_AMOUNT = 0.05

    APHID_MIN_COUNT = 10
    APHID_MAX_COUNT = 20
    APHID_SIZE = 5

    PARTICLE_LIFESPAN = 20
    PARTICLE_COUNT = 15
    PARTICLE_SPEED = 2.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.ant_pos = [0.0, 0.0]
        self.ant_size = self.ANT_START_SIZE
        self.aphids = []
        self.spiders = []
        self.queen_pos = [0.0, 0.0]
        self.particles = []
        self.spider_current_speed = self.SPIDER_START_SPEED
        self.damage_flash_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.ant_pos = [self.WIDTH / 4, self.HEIGHT / 2]
        self.ant_size = self.ANT_START_SIZE
        self.spider_current_speed = self.SPIDER_START_SPEED
        self.damage_flash_timer = 0

        self.particles.clear()
        self.aphids.clear()
        self.spiders.clear()

        self.queen_pos = [
            self.WIDTH * 0.9,
            self.np_random.uniform(self.HEIGHT * 0.1, self.HEIGHT * 0.9),
        ]

        for _ in range(self.SPIDER_START_COUNT):
            self._spawn_spider()

        while len(self.aphids) < self.APHID_MAX_COUNT:
            self._spawn_aphid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)

        # --- Update Game State ---
        self._update_spiders()
        self._update_particles()

        # --- Handle Interactions & Rewards ---
        if space_held:
            reward += self._handle_matching()

        damage_reward = self._handle_spider_collisions()
        reward += damage_reward
        if damage_reward < 0:
            self.damage_flash_timer = 5  # Flash for 5 frames

        # --- Check Termination Conditions ---
        terminated = False
        if self._check_victory():
            reward += 5.0  # Event reward
            reward += 100.0  # Terminal reward
            terminated = True

        if self.ant_size <= 0:
            reward += -100.0  # Terminal reward
            terminated = True

        # --- Check Truncation Conditions ---
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True

        # --- Maintain Game World ---
        self._maintain_aphids()
        self._update_difficulty()

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_movement(self, movement_action):
        dx, dy = 0, 0
        if movement_action == 1:
            dy = -1  # Up
        elif movement_action == 2:
            dy = 1  # Down
        elif movement_action == 3:
            dx = -1  # Left
        elif movement_action == 4:
            dx = 1  # Right

        self.ant_pos[0] += dx * self.ANT_SPEED
        self.ant_pos[1] += dy * self.ANT_SPEED

        # World wrapping
        self.ant_pos[0] %= self.WIDTH
        self.ant_pos[1] %= self.HEIGHT

    def _handle_matching(self):
        closest_aphid = None
        min_dist_sq = self.ANT_MATCH_RADIUS**2

        for aphid in self.aphids:
            dist_sq = (self.ant_pos[0] - aphid["pos"][0]) ** 2 + (
                self.ant_pos[1] - aphid["pos"][1]
            ) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_aphid = aphid

        if closest_aphid:
            self.aphids.remove(closest_aphid)
            self.ant_size = min(self.ANT_MAX_SIZE, self.ant_size + 1)
            self._spawn_particles(
                closest_aphid["pos"], self.APHID_COLORS[closest_aphid["color"]]
            )
            return 0.1
        return 0.0

    def _handle_spider_collisions(self):
        reward = 0.0
        ant_rect = pygame.Rect(
            self.ant_pos[0] - self.ant_size / 2,
            self.ant_pos[1] - self.ant_size / 2,
            self.ant_size,
            self.ant_size,
        )

        for spider in self.spiders:
            spider_rect = pygame.Rect(
                spider["pos"][0] - self.SPIDER_SIZE / 2,
                spider["pos"][1] - self.SPIDER_SIZE / 2,
                self.SPIDER_SIZE,
                self.SPIDER_SIZE,
            )
            if ant_rect.colliderect(spider_rect):
                damage = max(1, self.SPIDER_SIZE - self.ant_size)
                self.ant_size -= damage
                reward -= 1.0
        return reward

    def _check_victory(self):
        dist_sq = (self.ant_pos[0] - self.queen_pos[0]) ** 2 + (
            self.ant_pos[1] - self.queen_pos[1]
        ) ** 2
        return dist_sq < (self.ant_size / 2 + 10) ** 2

    def _update_spiders(self):
        for spider in self.spiders:
            target = spider["target"]
            pos = spider["pos"]

            dist_sq = (pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2
            if dist_sq < (self.spider_current_speed**2):
                spider["target"] = [
                    self.np_random.uniform(0, self.WIDTH),
                    self.np_random.uniform(0, self.HEIGHT),
                ]
            else:
                angle = math.atan2(target[1] - pos[1], target[0] - pos[0])
                pos[0] += math.cos(angle) * self.spider_current_speed
                pos[1] += math.sin(angle) * self.spider_current_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

    def _maintain_aphids(self):
        while len(self.aphids) < self.APHID_MIN_COUNT and not self.game_over:
            self._spawn_aphid()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.SPIDER_SPEED_INCREASE_INTERVAL == 0:
            self.spider_current_speed = min(
                self.SPIDER_MAX_SPEED,
                self.spider_current_speed + self.SPIDER_SPEED_INCREASE_AMOUNT,
            )

    def _spawn_aphid(self):
        self.aphids.append(
            {
                "pos": [
                    self.np_random.uniform(0, self.WIDTH),
                    self.np_random.uniform(0, self.HEIGHT),
                ],
                "color": self.np_random.choice(list(self.APHID_COLORS.keys())),
            }
        )

    def _spawn_spider(self):
        self.spiders.append(
            {
                "pos": [
                    self.np_random.uniform(0, self.WIDTH),
                    self.np_random.uniform(0, self.HEIGHT),
                ],
                "target": [
                    self.np_random.uniform(0, self.WIDTH),
                    self.np_random.uniform(0, self.HEIGHT),
                ],
            }
        )

    def _spawn_particles(self, pos, color):
        for _ in range(self.PARTICLE_COUNT):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.0) * self.PARTICLE_SPEED
            self.particles.append(
                {
                    "pos": list(pos),
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "lifespan": self.np_random.integers(10, self.PARTICLE_LIFESPAN),
                    "color": color,
                }
            )

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
            "ant_size": self.ant_size,
        }

    def _render_game(self):
        self._render_queen()

        for aphid in self.aphids:
            pygame.draw.circle(
                self.screen,
                self.APHID_COLORS[aphid["color"]],
                (int(aphid["pos"][0]), int(aphid["pos"][1])),
                self.APHID_SIZE,
            )

        for spider in self.spiders:
            pos = (int(spider["pos"][0]), int(spider["pos"][1]))
            size = self.SPIDER_SIZE
            body_rect = pygame.Rect(pos[0] - size / 2, pos[1] - size / 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_SPIDER, body_rect, border_radius=3)
            eye_y = pos[1] - size * 0.1
            pygame.draw.circle(
                self.screen,
                self.COLOR_SPIDER_EYES,
                (int(pos[0] - size * 0.2), int(eye_y)),
                2,
            )
            pygame.draw.circle(
                self.screen,
                self.COLOR_SPIDER_EYES,
                (int(pos[0] + size * 0.2), int(eye_y)),
                2,
            )

        self._render_ant()

        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN))
            color = p["color"]
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, (*color, alpha)
            )

        if self.game_over:
            if self.ant_size <= 0:
                text = "GAME OVER"
            elif self._check_victory():
                text = "VICTORY!"
            else:  # Max steps
                text = "TIME UP"

            rendered_text = self.font_large.render(text, True, self.COLOR_UI_TEXT)
            text_rect = rendered_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(rendered_text, text_rect)

    def _render_ant(self):
        pos = (int(self.ant_pos[0]), int(self.ant_pos[1]))
        size = int(max(1, self.ant_size))

        color = (255, 50, 50) if self.damage_flash_timer > 0 else self.COLOR_ANT
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

        pygame.draw.circle(self.screen, self.COLOR_ANT_OUTLINE, pos, size // 2 + 2)
        pygame.draw.circle(self.screen, color, pos, size // 2)

    def _render_queen(self):
        pos = (int(self.queen_pos[0]), int(self.queen_pos[1]))
        pulse = (math.sin(self.steps * 0.1) + 1) / 2  # 0 to 1

        for i in range(4, 0, -1):
            alpha = int(50 * (1 - (i / 5)) * pulse)
            radius = 10 + i * 4
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], radius, (*self.COLOR_QUEEN_GLOW, alpha)
            )

        pygame.draw.circle(self.screen, self.COLOR_QUEEN, pos, 10)

    def _render_ui(self):
        size_text = self.font.render(f"Size: {int(self.ant_size)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(size_text, (10, 10))

        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 35))

        steps_text = self.font.render(
            f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT
        )
        text_rect = steps_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(steps_text, text_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing.
    # It is not part of the Gymnasium environment.
    # It requires pygame to be installed with display drivers.
    # To run, unset the headless environment variable:
    # `unset SDL_VIDEODRIVER` or `os.environ.pop("SDL_VIDEODRIVER", None)`
    
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Ant's Quest")

    running = True
    terminated, truncated = False, False

    while running:
        movement = 0  # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)

        # Blit the observation from the environment to the display screen
        # Need to transpose it from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated, truncated = False, False

    env.close()