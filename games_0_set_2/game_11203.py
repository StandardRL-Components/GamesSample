import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:48:00.329366
# Source Brief: brief_01203.md
# Brief Index: 1203
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a growing vine.
    The goal is to collect 100 units of sunlight while avoiding thorny bushes.
    Ladybugs act as static obstacles that block the vine's growth path.
    """

    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Grow a vine to collect sunlight and reach the target score. Avoid thorny bushes and stationary ladybugs that block your path."
    user_guide = "Use the arrow keys (↑↓←→) to guide the growing vine."
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.VINE_GROWTH_STEP = 10
        self.VINE_WIDTH = 6
        self.ENTITY_RADIUS = 10
        self.BUSH_RADIUS = 12
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.MAX_BUSH_HITS = 3

        # Colors
        self.COLOR_BG = (20, 40, 30)
        self.COLOR_VINE = (139, 69, 19)
        self.COLOR_VINE_HEAD = (160, 82, 45)
        self.COLOR_SUNLIGHT = (255, 255, 0)
        self.COLOR_SUNLIGHT_GLOW = (255, 255, 150)
        self.COLOR_BUSH = (200, 0, 0)
        self.COLOR_LADYBUG = (220, 20, 60)
        self.COLOR_LADYBUG_DOT = (0, 0, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_FLASH = (255, 0, 0, 100)

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 64, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_game_over = pygame.font.SysFont(None, 70)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.vine_segments = []
        self.sunlight_pos = None
        self.bushes = []
        self.ladybugs = []
        self.bush_collisions = 0
        self.game_over = False
        self.particles = []
        self.screen_flash_timer = 0
        self.last_growth_direction = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.bush_collisions = 0
        self.game_over = False
        self.particles = []
        self.screen_flash_timer = 0
        self.last_growth_direction = 0

        self.vine_segments = [(self.WIDTH // 2, self.HEIGHT // 2)]

        # Clear entities before spawning new ones
        self.bushes = []
        self.ladybugs = []
        self.sunlight_pos = None

        initial_bushes = 2
        self.bushes = [
            self._spawn_entity(self.BUSH_RADIUS) for _ in range(initial_bushes)
        ]

        initial_ladybugs = 5
        self.ladybugs = [
            self._spawn_entity(self.ENTITY_RADIUS) for _ in range(initial_ladybugs)
        ]

        self.sunlight_pos = self._spawn_entity(self.ENTITY_RADIUS)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0.0

        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1

        if movement != 0:
            self._grow_vine(movement)

        reward += self._check_sunlight_collection()
        reward += self._update_bush_collisions()

        self._update_particles()

        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
        elif self.bush_collisions >= self.MAX_BUSH_HITS:
            reward -= 100.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Gymnasium standard is that truncated implies terminated
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _grow_vine(self, movement):
        head = self.vine_segments[-1]
        new_head = list(head)

        prevent_reverse = {1: 2, 2: 1, 3: 4, 4: 3}
        if (
            len(self.vine_segments) > 1
            and movement == prevent_reverse.get(self.last_growth_direction)
        ):
            return

        if movement == 1:
            new_head[1] -= self.VINE_GROWTH_STEP
        elif movement == 2:
            new_head[1] += self.VINE_GROWTH_STEP
        elif movement == 3:
            new_head[0] -= self.VINE_GROWTH_STEP
        elif movement == 4:
            new_head[0] += self.VINE_GROWTH_STEP

        new_head[0] %= self.WIDTH
        new_head[1] %= self.HEIGHT
        new_head = tuple(new_head)

        can_grow = True
        for bush_pos in self.bushes:
            if self._check_collision(
                new_head, bush_pos, self.BUSH_RADIUS + self.VINE_WIDTH / 2
            ):
                can_grow = False
                break

        if can_grow:
            for i in range(len(self.vine_segments) - 2):
                if self._check_collision(
                    new_head, self.vine_segments[i], self.VINE_WIDTH
                ):
                    can_grow = False
                    break

        if can_grow:
            for ladybug_pos in self.ladybugs:
                if self._check_collision(
                    new_head, ladybug_pos, self.ENTITY_RADIUS + self.VINE_WIDTH / 2
                ):
                    can_grow = False
                    break

        if can_grow:
            self.vine_segments.append(new_head)
            self.last_growth_direction = movement
            # sfx_vine_grow.wav

    def _check_sunlight_collection(self):
        head = self.vine_segments[-1]
        if self._check_collision(
            head, self.sunlight_pos, self.ENTITY_RADIUS + self.VINE_WIDTH / 2
        ):
            self.score += 1
            # sfx_sunlight_collect.wav

            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                self.particles.append(
                    {
                        "pos": list(self.sunlight_pos),
                        "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                        "lifespan": random.randint(15, 30),
                        "color": self.COLOR_SUNLIGHT,
                    }
                )

            self.sunlight_pos = self._spawn_entity(self.ENTITY_RADIUS)

            if self.score > 0 and self.score % 25 == 0:
                new_bush_count = 2 + (self.score // 25)
                if len(self.bushes) < new_bush_count:
                    self.bushes.append(self._spawn_entity(self.BUSH_RADIUS))
                    # sfx_bush_spawn.wav
            return 1.0
        return 0.0

    def _update_bush_collisions(self):
        head = self.vine_segments[-1]
        reward = 0.0
        for bush_pos in self.bushes:
            if self._check_collision(
                head, bush_pos, self.BUSH_RADIUS + self.VINE_WIDTH / 2
            ):
                if self.bush_collisions < self.MAX_BUSH_HITS:
                    self.bush_collisions += 1
                    reward -= 5.0
                    self.screen_flash_timer = 5
                    # sfx_bush_hit.wav
                break
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

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
            "bush_collisions": self.bush_collisions,
        }

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30.0))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, (*p["color"], alpha)
            )

        glow_radius = self.ENTITY_RADIUS + 5 + 3 * math.sin(self.steps * 0.1)
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(self.sunlight_pos[0]),
            int(self.sunlight_pos[1]),
            int(glow_radius),
            (*self.COLOR_SUNLIGHT_GLOW, 50),
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            int(self.sunlight_pos[0]),
            int(self.sunlight_pos[1]),
            int(glow_radius),
            (*self.COLOR_SUNLIGHT_GLOW, 50),
        )
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(self.sunlight_pos[0]),
            int(self.sunlight_pos[1]),
            self.ENTITY_RADIUS,
            self.COLOR_SUNLIGHT,
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            int(self.sunlight_pos[0]),
            int(self.sunlight_pos[1]),
            self.ENTITY_RADIUS,
            self.COLOR_SUNLIGHT,
        )

        for pos in self.bushes:
            points = [
                (pos[0], pos[1] - self.BUSH_RADIUS),
                (pos[0] - self.BUSH_RADIUS, pos[1] + self.BUSH_RADIUS * 0.7),
                (pos[0] + self.BUSH_RADIUS, pos[1] + self.BUSH_RADIUS * 0.7),
            ]
            pygame.gfxdraw.filled_trigon(
                self.screen,
                int(points[0][0]),
                int(points[0][1]),
                int(points[1][0]),
                int(points[1][1]),
                int(points[2][0]),
                int(points[2][1]),
                self.COLOR_BUSH,
            )
            pygame.gfxdraw.aatrigon(
                self.screen,
                int(points[0][0]),
                int(points[0][1]),
                int(points[1][0]),
                int(points[1][1]),
                int(points[2][0]),
                int(points[2][1]),
                self.COLOR_BUSH,
            )

        for pos in self.ladybugs:
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos[0]), int(pos[1]), self.ENTITY_RADIUS, self.COLOR_LADYBUG
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(pos[0]), int(pos[1]), self.ENTITY_RADIUS, self.COLOR_LADYBUG
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos[0] - 4), int(pos[1] - 4), 2, self.COLOR_LADYBUG_DOT
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos[0] + 4), int(pos[1] - 4), 2, self.COLOR_LADYBUG_DOT
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos[0]), int(pos[1] + 4), 2, self.COLOR_LADYBUG_DOT
            )

        if len(self.vine_segments) > 1:
            pygame.draw.lines(
                self.screen, self.COLOR_VINE, False, self.vine_segments, self.VINE_WIDTH * 2
            )
        for segment in self.vine_segments:
            pygame.gfxdraw.filled_circle(
                self.screen, int(segment[0]), int(segment[1]), self.VINE_WIDTH, self.COLOR_VINE
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(segment[0]), int(segment[1]), self.VINE_WIDTH, self.COLOR_VINE
            )

        head = self.vine_segments[-1]
        head_pulse = 2 * math.sin(self.steps * 0.2)
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(head[0]),
            int(head[1]),
            int(self.VINE_WIDTH + head_pulse),
            self.COLOR_VINE_HEAD,
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            int(head[0]),
            int(head[1]),
            int(self.VINE_WIDTH + head_pulse),
            self.COLOR_VINE_HEAD,
        )

        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(self.screen_flash_timer / 5.0 * 100)
            flash_surface.fill((*self.COLOR_BUSH, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        score_text = self.font_ui.render(
            f"Sunlight: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(score_text, (10, 10))

        hits_text = self.font_ui.render(f"Hits: ", True, self.COLOR_UI_TEXT)
        self.screen.blit(hits_text, (self.WIDTH - 150, 10))
        for i in range(self.MAX_BUSH_HITS):
            color = self.COLOR_BUSH if i < self.bush_collisions else (100, 100, 100)
            pos = (self.WIDTH - 70 + i * 20, 22)
            points = [
                (pos[0], pos[1] - 8),
                (pos[0] - 8, pos[1] + 5),
                (pos[0] + 8, pos[1] + 5),
            ]
            pygame.gfxdraw.filled_trigon(
                self.screen,
                int(points[0][0]),
                int(points[0][1]),
                int(points[1][0]),
                int(points[1][1]),
                int(points[2][0]),
                int(points[2][1]),
                color,
            )

        if self.game_over:
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_SUNLIGHT if self.score >= self.WIN_SCORE else self.COLOR_BUSH
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_entity(self, radius):
        max_attempts = 100
        for _ in range(max_attempts):
            pos = (
                self.np_random.integers(radius, self.WIDTH - radius),
                self.np_random.integers(radius, self.HEIGHT - radius),
            )
            is_overlapping = False

            all_entities = (
                [(seg, self.VINE_WIDTH) for seg in self.vine_segments]
                + [(b, self.BUSH_RADIUS) for b in self.bushes]
                + [(l, self.ENTITY_RADIUS) for l in self.ladybugs]
                + (
                    [(self.sunlight_pos, self.ENTITY_RADIUS)]
                    if self.sunlight_pos
                    else []
                )
            )

            for entity_pos, entity_radius in all_entities:
                if self._check_collision(
                    pos, entity_pos, radius + entity_radius + 10
                ):  # Extra buffer
                    is_overlapping = True
                    break

            if not is_overlapping:
                return pos
        return (
            self.np_random.integers(radius, self.WIDTH - radius),
            self.np_random.integers(radius, self.HEIGHT - radius),
        )

    def _check_collision(self, pos1, pos2, distance_threshold):
        dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        return dist < distance_threshold

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()

    # The following is for human play and is not part of the environment logic
    # It requires a display to be available.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Vine Grower")
        clock = pygame.time.Clock()

        done = False
        total_reward = 0

        # Mapping keys to actions for human play
        key_to_action = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        while not done:
            movement_action = 0  # No-op by default
            space_held = 0
            shift_held = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            for key, action_val in key_to_action.items():
                if keys[key]:
                    movement_action = action_val
                    break

            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1

            action = [movement_action, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(
                    f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}"
                )
                done = True

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(15)  # Control game speed for human play

    except pygame.error as e:
        print(f"Pygame display could not be initialized: {e}")
        print("Running a short step-loop without rendering for validation.")
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Info: {info}")
                obs, info = env.reset()

    env.close()