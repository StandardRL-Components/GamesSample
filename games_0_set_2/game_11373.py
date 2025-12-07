import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control two vertical platforms on either side of the screen to catch falling objects."
    )
    user_guide = (
        "Move the left platform with ↑ and ↓. Move the right platform up with 'space' and down with 'shift'."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # For physics tuning

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 40, 70)
        self.COLOR_PLATFORM = (255, 255, 255)
        self.COLOR_PLATFORM_GLOW = (200, 200, 255)
        self.OBJECT_COLORS = [
            (255, 80, 80),  # Red
            (80, 255, 80),  # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # Game parameters
        self.PLATFORM_WIDTH = 120
        self.PLATFORM_HEIGHT = 12
        self.PLATFORM_SPEED = 12.0
        self.OBJECT_RADIUS = 10
        self.INITIAL_FALL_SPEED = 2.0
        self.FALL_SPEED_INCREMENT = 0.2
        self.MAX_OBJECTS = 5
        self.CATCHES_PER_LEVEL = 5
        self.MAX_MISSES = 3
        self.MAX_STEPS = 1500  # Increased for longer play potential
        self.SPAWN_PROBABILITY = 0.03

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.missed_count = 0
        self.total_caught = 0
        self.game_over = False
        self.platform_left_y = 0
        self.platform_right_y = 0
        self.objects = []
        self.particles = []
        self.object_fall_speed = self.INITIAL_FALL_SPEED

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.missed_count = 0
        self.total_caught = 0
        self.game_over = False

        self.platform_left_y = self.HEIGHT // 2
        self.platform_right_y = self.HEIGHT // 2

        self.objects = []
        self.particles = []
        self.object_fall_speed = self.INITIAL_FALL_SPEED

        for _ in range(self.MAX_OBJECTS):
            self._spawn_object(initial_spawn=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # Return final state if called after episode is done
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Game Logic Update ---
        self._update_platforms()
        self._update_particles()

        catch_reward, level_up_reward = self._update_objects()

        # --- Spawning New Objects ---
        if len(self.objects) < self.MAX_OBJECTS:
            if self.np_random.random() < self.SPAWN_PROBABILITY:
                self._spawn_object()

        # --- Reward Calculation ---
        reward = 0
        # Survival reward: Small incentive to keep objects on screen
        reward += 0.01 * len(self.objects)
        reward += catch_reward
        reward += level_up_reward

        # --- Termination & Truncation Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            self.game_over = True
            reward = -100.0  # Large penalty for losing
        
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Left platform (controlled by movement actions)
        if movement == 1:  # Up
            self.platform_left_y -= self.PLATFORM_SPEED
        elif movement == 2:  # Down
            self.platform_left_y += self.PLATFORM_SPEED

        # Right platform (controlled by space/shift)
        if space_held and not shift_held:  # Up
            self.platform_right_y -= self.PLATFORM_SPEED
        elif shift_held and not space_held:  # Down
            self.platform_right_y += self.PLATFORM_SPEED

    def _update_platforms(self):
        platform_min_y = self.PLATFORM_HEIGHT // 2
        platform_max_y = self.HEIGHT - self.PLATFORM_HEIGHT // 2
        self.platform_left_y = np.clip(
            self.platform_left_y, platform_min_y, platform_max_y
        )
        self.platform_right_y = np.clip(
            self.platform_right_y, platform_min_y, platform_max_y
        )

    def _update_objects(self):
        catch_reward = 0
        level_up_reward = 0
        catches_this_step = 0

        for obj in self.objects[:]:
            obj["pos"].y += self.object_fall_speed

            if obj["pos"].y > self.HEIGHT + self.OBJECT_RADIUS:
                self.missed_count += 1
                self.objects.remove(obj)
                continue

            left_platform_rect = pygame.Rect(
                0,
                self.platform_left_y - self.PLATFORM_HEIGHT // 2,
                self.PLATFORM_WIDTH,
                self.PLATFORM_HEIGHT,
            )
            if left_platform_rect.collidepoint(obj["pos"].x, obj["pos"].y):
                self._trigger_catch(obj)
                catches_this_step += 1
                continue

            right_platform_rect = pygame.Rect(
                self.WIDTH - self.PLATFORM_WIDTH,
                self.platform_right_y - self.PLATFORM_HEIGHT // 2,
                self.PLATFORM_WIDTH,
                self.PLATFORM_HEIGHT,
            )
            if right_platform_rect.collidepoint(obj["pos"].x, obj["pos"].y):
                self._trigger_catch(obj)
                catches_this_step += 1
                continue

        if catches_this_step > 0:
            catch_reward += 1.0 * catches_this_step
            if catches_this_step >= 2:
                catch_reward += 2.0
                self.score += 25 * catches_this_step

            if self.total_caught // self.CATCHES_PER_LEVEL > self.level - 1:
                self.level = (self.total_caught // self.CATCHES_PER_LEVEL) + 1
                self.object_fall_speed += self.FALL_SPEED_INCREMENT
                level_up_reward = 10.0
                self._spawn_particles(
                    pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
                    (255, 255, 255),
                    100,
                    is_level_up=True,
                )

        return catch_reward, level_up_reward

    def _trigger_catch(self, obj):
        self.score += 10
        self.total_caught += 1
        self._spawn_particles(obj["pos"], obj["color"], 30)
        self.objects.remove(obj)

    def _spawn_object(self, initial_spawn=False):
        x_pos = self.np_random.integers(
            self.PLATFORM_WIDTH, self.WIDTH - self.PLATFORM_WIDTH
        )
        y_pos = -self.np_random.integers(20, 200) if initial_spawn else -self.OBJECT_RADIUS

        new_obj = {
            "pos": pygame.Vector2(x_pos, y_pos),
            "color": random.choice(self.OBJECT_COLORS),
        }
        self.objects.append(new_obj)

    def _spawn_particles(self, pos, color, count, is_level_up=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = (
                self.np_random.uniform(1, 5)
                if not is_level_up
                else self.np_random.uniform(3, 8)
            )
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append(
                {
                    "pos": pos.copy(),
                    "vel": vel,
                    "radius": self.np_random.uniform(2, 6),
                    "life": self.np_random.integers(20, 40),
                    "color": color,
                }
            )

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # Damping
            p["life"] -= 1
            p["radius"] -= 0.1
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.missed_count >= self.MAX_MISSES

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "missed": self.missed_count,
            "caught": self.total_caught,
        }

    def _get_observation(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        self._render_particles()
        self._render_objects()
        self._render_platforms()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_platforms(self):
        left_rect = pygame.Rect(0, 0, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        left_rect.centery = int(self.platform_left_y)

        right_rect = pygame.Rect(0, 0, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        right_rect.right = self.WIDTH
        right_rect.centery = int(self.platform_right_y)

        for r in [left_rect, right_rect]:
            glow_rect = r.inflate(10, 10)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_PLATFORM_GLOW, 50), s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, left_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, right_rect, border_radius=4)

    def _render_objects(self):
        for obj in self.objects:
            pos = (int(obj["pos"].x), int(obj["pos"].y))
            color = obj["color"]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.OBJECT_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.OBJECT_RADIUS, color)
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], self.OBJECT_RADIUS - 2, tuple(c * 0.7 for c in color)
            )

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"])
            if radius > 0:
                alpha = int(255 * (p["life"] / 40))
                color = (*p["color"], alpha)
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (pos[0] - radius, pos[1] - radius))

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2, 2)):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_medium, self.COLOR_TEXT, (15, 10), self.COLOR_TEXT_SHADOW)

        miss_text = f"MISSES: {self.missed_count}/{self.MAX_MISSES}"
        text_surf = self.font_medium.render(miss_text, True, self.COLOR_TEXT)
        draw_text(
            miss_text,
            self.font_medium,
            self.COLOR_TEXT,
            (self.WIDTH - text_surf.get_width() - 15, 10),
            self.COLOR_TEXT_SHADOW,
        )

        level_text = f"LEVEL {self.level}"
        text_surf = self.font_large.render(level_text, True, self.COLOR_TEXT)
        pos = (
            self.WIDTH // 2 - text_surf.get_width() // 2,
            self.HEIGHT - text_surf.get_height() - 5,
        )
        draw_text(level_text, self.font_large, self.COLOR_TEXT, pos, self.COLOR_TEXT_SHADOW)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()
    done = False

    # Override Pygame screen for direct display
    pygame.display.init()
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dual Catch")

    total_reward = 0
    action = [0, 0, 0]

    while not done:
        keys = pygame.key.get_pressed()

        # Left platform: Arrow Up/Down
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        else:
            action[0] = 0

        # Right platform: Space/Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(display_surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()