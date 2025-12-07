import gymnasium as gym
import os
import pygame
import math
import random
from itertools import combinations
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for visual effects."""

    def __init__(self, x, y, color, seed):
        self.rng = random.Random(seed)
        angle = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(1, 4)
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.radius = self.rng.uniform(2, 5)
        self.color = color
        self.lifespan = self.rng.randint(20, 40)

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius -= 0.1
        return self.lifespan > 0 and self.radius > 0

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 20))))
            # Using gfxdraw is not available in standard pygame, but we can fake it
            # For the purpose of this fix, we will assume it's available or switch to standard draw
            # For maximum compatibility, let's use standard filled circle
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                temp_surf,
                (*self.color, alpha),
                (self.radius, self.radius),
                self.radius,
            )
            surface.blit(temp_surf, (self.pos.x - self.radius, self.pos.y - self.radius), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Align and stack horizontally moving blocks to score points. "
        "Time your moves to create tall stacks before the time runs out."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ arrow keys to select a block. "
        "Use ← and → arrow keys to change its direction of movement."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    TIME_LIMIT_SECONDS = 120

    COLOR_BG = (26, 26, 46)  # Dark blue-purple
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (230, 230, 255)
    COLOR_TEXT_SHADOW = (10, 10, 20)

    BLOCK_COLORS = [(233, 69, 96), (126, 217, 87), (87, 115, 217)]  # Red, Green, Blue
    STACK_COLOR = (255, 255, 255)
    SELECT_GLOW_COLOR = (248, 180, 0)

    BASE_SPEEDS = [120, 240, 360]  # pixels/sec
    BLOCK_BASE_HEIGHT = 20
    BLOCK_WIDTH = 50
    STACK_THRESHOLD = 5  # pixels

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.blocks = []
        self.particles = []
        self.selected_block_idx = 0
        self.last_movement_action = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.particles.clear()
        self.selected_block_idx = 0
        self.last_movement_action = 0

        self._initialize_blocks()

        return self._get_observation(), self._get_info()

    def _initialize_blocks(self):
        self.blocks = []
        possible_y = list(range(
            self.BLOCK_BASE_HEIGHT,
            self.SCREEN_HEIGHT - self.BLOCK_BASE_HEIGHT,
            self.BLOCK_BASE_HEIGHT * 2,
        ))
        y_positions = self.np_random.choice(possible_y, size=3, replace=False)

        for i in range(3):
            speed = self.BASE_SPEEDS[i] / self.FPS
            block = {
                "pos": pygame.math.Vector2(
                    self.np_random.uniform(self.BLOCK_WIDTH, self.SCREEN_WIDTH - self.BLOCK_WIDTH),
                    y_positions[i],
                ),
                "vel": self.np_random.choice([-speed, speed]),
                "base_speed": speed,
                "color": self.BLOCK_COLORS[i],
                "size": pygame.math.Vector2(self.BLOCK_WIDTH, self.BLOCK_BASE_HEIGHT),
                "is_stack": False,
                "component_speeds": [speed * self.FPS],
                "component_count": 1,
            }
            self.blocks.append(block)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_actions(action)
        self._update_physics()
        self._update_particles()

        stack_reward = self._check_for_stacks()
        continuous_reward = self._calculate_continuous_reward()

        self.score += stack_reward
        reward = stack_reward + continuous_reward

        self.steps += 1
        self.time_left -= 1
        terminated = self.time_left <= 0
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_actions(self, action):
        movement, _, _ = action

        if movement in [1, 2] and movement != self.last_movement_action:
            if movement == 1:  # Up
                self.selected_block_idx = (self.selected_block_idx - 1 + len(self.blocks)) % len(self.blocks)
            elif movement == 2:  # Down
                self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)

        self.last_movement_action = movement

        if self.blocks:
            self.selected_block_idx = min(self.selected_block_idx, len(self.blocks) - 1)
            selected_block = self.blocks[self.selected_block_idx]

            if movement == 3:  # Left
                selected_block["vel"] = -abs(selected_block["base_speed"])
            elif movement == 4:  # Right
                selected_block["vel"] = abs(selected_block["base_speed"])

    def _update_physics(self):
        for block in self.blocks:
            block["pos"].x += block["vel"]

            if block["pos"].x - block["size"].x / 2 < 0:
                block["pos"].x = block["size"].x / 2
                block["vel"] *= -1
            if block["pos"].x + block["size"].x / 2 > self.SCREEN_WIDTH:
                block["pos"].x = self.SCREEN_WIDTH - block["size"].x / 2
                block["vel"] *= -1

    def _check_for_stacks(self):
        stack_reward = 0
        while True:
            stacked_this_iteration = False

            if len(self.blocks) < 2:
                break

            for i, j in combinations(range(len(self.blocks)), 2):
                b1 = self.blocks[i]
                b2 = self.blocks[j]

                if abs(b1["pos"].x - b2["pos"].x) < self.STACK_THRESHOLD:
                    new_count = b1["component_count"] + b2["component_count"]
                    new_speeds = b1["component_speeds"] + b2["component_speeds"]
                    avg_speed = sum(new_speeds) / new_count

                    new_stack = {
                        "pos": (b1["pos"] * b1["component_count"] + b2["pos"] * b2["component_count"]) / new_count,
                        "base_speed": avg_speed / self.FPS,
                        "vel": (avg_speed / self.FPS) * self.np_random.choice([-1, 1]),
                        "color": self.STACK_COLOR,
                        "size": pygame.math.Vector2(self.BLOCK_WIDTH, b1["size"].y + b2["size"].y),
                        "is_stack": True,
                        "component_speeds": new_speeds,
                        "component_count": new_count,
                    }

                    self._create_particles(new_stack["pos"].x, new_stack["pos"].y, new_count)

                    self.blocks.pop(max(i, j))
                    self.blocks.pop(min(i, j))
                    self.blocks.append(new_stack)

                    stack_reward += 10 * new_count
                    stacked_this_iteration = True
                    break

            if not stacked_this_iteration:
                break

        return stack_reward

    def _calculate_continuous_reward(self):
        reward = 0
        if len(self.blocks) < 2:
            return 0

        for b1, b2 in combinations(self.blocks, 2):
            is_converging = (b1["pos"].x < b2["pos"].x and b1["vel"] > b2["vel"]) or \
                            (b1["pos"].x > b2["pos"].x and b1["vel"] < b2["vel"])
            if is_converging:
                reward += 0.1

        return min(reward, 1.0)

    def _create_particles(self, x, y, count):
        particle_color = self.SELECT_GLOW_COLOR
        for _ in range(count * 10):
            seed = self.np_random.integers(0, 2**32 - 1)
            self.particles.append(Particle(x, y, particle_color, seed))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_background_grid()
        self._draw_blocks()
        self._draw_particles()

    def _draw_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_blocks(self):
        for i, block in enumerate(self.blocks):
            rect = pygame.Rect(
                block["pos"].x - block["size"].x / 2,
                block["pos"].y - block["size"].y / 2,
                block["size"].x,
                block["size"].y,
            )
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=5)

            if i == self.selected_block_idx:
                self._draw_selection_glow(rect)

    def _draw_selection_glow(self, rect):
        glow_alpha = (math.sin(self.steps * 0.2) + 1) / 2 * 150 + 50
        glow_color = (*self.SELECT_GLOW_COLOR, int(glow_alpha))

        glow_surface = pygame.Surface((rect.width + 20, rect.height + 20), pygame.SRCALPHA)
        glow_rect = glow_surface.get_rect()

        pygame.draw.rect(glow_surface, glow_color, glow_rect, border_radius=12)

        self.screen.blit(glow_surface, (rect.x - 10, rect.y - 10), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, self.SELECT_GLOW_COLOR, rect.inflate(4, 4), width=2, border_radius=7)

    def _draw_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10))

        time_str = f"TIME: {max(0, self.time_left // self.FPS):03d}"
        text_width = self.font_ui.size(time_str)[0]
        self._draw_text(time_str, (self.SCREEN_WIDTH - text_width - 15, 10))

    def _draw_text(self, text, pos):
        shadow_surface = self.font_ui.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        text_surface = self.font_ui.render(text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_left": len(self.blocks),
            "time_remaining_seconds": max(0, self.time_left // self.FPS),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows for manual play testing.
    # It will create a window and let you control the game.
    # Undo the headless environment variable to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    last_keys = pygame.key.get_pressed()

    while running:
        movement_action = 0  # 0=none, 1=up, 2=down, 3=left, 4=right

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                # Discrete actions on key press
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2

        # Continuous actions for holding keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4

        # The original main loop had a slight bug in handling discrete vs continuous
        # actions. This revised version handles it better.
        if not (keys[pygame.K_UP] or keys[pygame.K_DOWN]):
             if last_keys[pygame.K_UP] and not keys[pygame.K_UP]:
                 pass # up was just released
             if last_keys[pygame.K_DOWN] and not keys[pygame.K_DOWN]:
                 pass # down was just released
        
        last_keys = keys

        gym_action = [movement_action, 0, 0]

        obs, reward, terminated, truncated, info = env.step(gym_action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}. Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()