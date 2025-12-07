import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend a celestial body from waves of incoming invaders. Charge and launch energetic fragments to destroy them."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to aim your launcher and ←→ to adjust power. "
        "Hold Shift to charge your shot and press Space to fire."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.MAX_LEVEL = 10

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_FRAGMENT = (255, 255, 100)
        self.COLOR_ENEMY_A = (255, 50, 50)
        self.COLOR_ENEMY_B = (255, 120, 50)
        self.COLOR_CELESTIAL_BODY = (80, 120, 255)
        self.COLOR_ATMOSPHERE = (150, 180, 255)
        self.COLOR_NOVA = (255, 200, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.celestial_health = 0
        self.max_celestial_health = 100
        self.current_level = 1
        self.current_wave = 1
        self.waves_per_level = 3

        # Launcher state
        self.launch_angle = 0.0
        self.launch_power = 0.0
        self.fragment_size = 0.0
        self.min_fragment_size = 5
        self.max_fragment_size = 25
        self.fragment_charge_rate = 0.5

        # Entity lists
        self.invaders = []
        self.fragments = []
        self.particles = []
        self.nova_impacts = []

        # Control state
        self.prev_space_held = False

        # Background
        self._starfield = self._generate_starfield()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.celestial_health = self.max_celestial_health
        self.current_level = 1
        self.current_wave = 1

        self.launch_angle = -math.pi / 2
        self.launch_power = 50
        self.fragment_size = self.min_fragment_size

        self.invaders.clear()
        self.fragments.clear()
        self.particles.clear()
        self.nova_impacts.clear()

        self.prev_space_held = False

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        # --- Handle Actions ---
        self._handle_actions(action)

        # --- Update Game State ---
        self._update_fragments()
        reward += self._update_invaders()
        self._update_particles()
        self._cleanup_nova_impacts()

        # --- Check Wave/Level Completion ---
        if not self.invaders and self.wave_spawn_complete:
            self.current_wave += 1
            reward += 10  # Wave clear reward
            if self.current_wave > self.waves_per_level:
                self.current_wave = 1
                self.current_level += 1
                reward += 20  # Level clear reward

            if self.current_level <= self.MAX_LEVEL:
                self._spawn_wave()

        self.steps += 1

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.celestial_health <= 0:
            reward -= 100  # Loss penalty
            terminated = True
            self.game_over = True
        elif self.current_level > self.MAX_LEVEL:
            reward += 100  # Win bonus
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_actions(self, action):
        movement, space_held_raw, shift_held = (
            action[0],
            action[1] == 1,
            action[2] == 1,
        )
        space_pressed = space_held_raw and not self.prev_space_held
        self.prev_space_held = space_held_raw

        # Action 0: Movement (aiming)
        if movement == 1:  # Up
            self.launch_angle -= 0.05
        elif movement == 2:  # Down
            self.launch_angle += 0.05
        elif movement == 3:  # Left
            self.launch_power -= 2
        elif movement == 4:  # Right
            self.launch_power += 2

        self.launch_angle = max(-math.pi, min(0, self.launch_angle))
        self.launch_power = max(20, min(100, self.launch_power))

        # Action 1: Space (Launch)
        if space_pressed:
            launch_x = self.WIDTH / 2
            launch_y = self.HEIGHT - 50
            vel_x = math.cos(self.launch_angle) * (self.launch_power / 10)
            vel_y = math.sin(self.launch_angle) * (self.launch_power / 10)
            self.fragments.append(
                {
                    "pos": pygame.Vector2(launch_x, launch_y),
                    "vel": pygame.Vector2(vel_x, vel_y),
                    "size": self.fragment_size,
                    "trail": [],
                }
            )
            self.fragment_size = self.min_fragment_size

        # Action 2: Shift (Charge Fragment)
        if shift_held:
            self.fragment_size = min(
                self.max_fragment_size, self.fragment_size + self.fragment_charge_rate
            )

    def _update_fragments(self):
        for f in self.fragments[:]:
            f["trail"].append(pygame.Vector2(f["pos"]))
            if len(f["trail"]) > 15:
                f["trail"].pop(0)

            f["pos"] += f["vel"]
            f["size"] -= 0.05  # Shrink over time

            if f["size"] <= 1 or not (
                0 < f["pos"].x < self.WIDTH and 0 < f["pos"].y < self.HEIGHT
            ):
                self.fragments.remove(f)

    def _update_invaders(self):
        reward = 0
        target_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)

        for inv in self.invaders[:]:
            direction = (target_pos - inv["pos"]).normalize()
            inv["pos"] += direction * inv["speed"]
            inv["angle"] = math.atan2(direction.y, direction.x)

            # Collision with celestial body
            if inv["pos"].distance_to(target_pos) < inv["size"] + 40:
                damage = int(inv["health"])
                self.celestial_health -= damage
                reward -= 0.1 * damage
                self.invaders.remove(inv)
                self._create_explosion(inv["pos"], self.COLOR_ENEMY_A, 30, 2)
                continue

            # Collision with fragments
            for f in self.fragments[:]:
                if inv["pos"].distance_to(f["pos"]) < inv["size"] + f["size"]:
                    damage = f["size"]
                    inv["health"] -= damage
                    reward += 0.1  # Hit reward
                    self.fragments.remove(f)
                    self._create_explosion(f["pos"], self.COLOR_FRAGMENT, 20, 1)

                    if inv["health"] <= 0:
                        self.score += 10
                        if inv in self.invaders:
                            self.invaders.remove(inv)
                        self._create_explosion(inv["pos"], self.COLOR_ENEMY_A, 50, 3)

                        # Check for Nova Burst
                        self.nova_impacts.append(
                            {"pos": inv["pos"], "time": self.steps}
                        )
                        nearby_impacts = [
                            imp
                            for imp in self.nova_impacts
                            if inv["pos"].distance_to(imp["pos"]) < 50
                        ]
                        if len(nearby_impacts) >= 2:
                            reward += 5  # Nova Burst reward
                            self.score += 50
                            self._trigger_nova_burst(inv["pos"])
                        break
        return reward

    def _trigger_nova_burst(self, pos):
        nova_radius = 100
        nova_damage = 50
        self._create_explosion(pos, self.COLOR_NOVA, 200, 5, is_nova=True)

        for inv in self.invaders[:]:
            if pos.distance_to(inv["pos"]) < nova_radius:
                inv["health"] -= nova_damage
                if inv["health"] <= 0:
                    self.score += 10
                    if inv in self.invaders:
                        self.invaders.remove(inv)
                    self._create_explosion(inv["pos"], self.COLOR_ENEMY_A, 50, 3)

    def _cleanup_nova_impacts(self):
        self.nova_impacts = [
            imp for imp in self.nova_impacts if self.steps - imp["time"] < 30
        ]  # 1 second window

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        self.wave_spawn_complete = False
        num_invaders = 1 + (self.current_level - 1) + (self.current_wave - 1)
        num_invaders = min(num_invaders, 15)  # Max invaders

        base_speed = 0.5 + self.current_level * 0.1

        for i in range(num_invaders):
            angle = random.uniform(0, 2 * math.pi)
            distance = self.WIDTH / 2 + 50
            pos = pygame.Vector2(
                self.WIDTH / 2 + math.cos(angle) * distance,
                self.HEIGHT / 2 + math.sin(angle) * distance,
            )
            self.invaders.append(
                {
                    "pos": pos,
                    "speed": base_speed + random.uniform(-0.1, 0.1),
                    "health": 20 + self.current_level * 5,
                    "size": 10,
                    "angle": 0,
                }
            )
        self.wave_spawn_complete = True

    def _create_explosion(self, pos, color, count, speed_scale, is_nova=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            if is_nova:
                life = random.randint(30, 60)
                size = random.uniform(3, 7)
            else:
                life = random.randint(15, 30)
                size = random.uniform(1, 4)
            self.particles.append(
                {
                    "pos": pygame.Vector2(pos),
                    "vel": vel,
                    "life": life,
                    "max_life": life,
                    "color": color,
                    "size": size,
                }
            )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "wave": self.current_wave,
            "health": self.celestial_health,
        }

    def _generate_starfield(self):
        stars = []
        for _ in range(200):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.choice([1, 1, 1, 2])
            brightness = random.randint(50, 150)
            stars.append(((x, y), size, (brightness, brightness, brightness)))
        return stars

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for pos, size, color in self._starfield:
            pygame.draw.circle(self.screen, color, pos, size)

    def _render_game(self):
        center = (self.WIDTH // 2, self.HEIGHT // 2)
        pygame.gfxdraw.filled_circle(
            self.screen, center[0], center[1], 45, self.COLOR_ATMOSPHERE + (50,)
        )
        pygame.gfxdraw.filled_circle(
            self.screen, center[0], center[1], 40, self.COLOR_CELESTIAL_BODY
        )
        pygame.gfxdraw.aacircle(
            self.screen, center[0], center[1], 40, self.COLOR_CELESTIAL_BODY
        )

        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = p["color"]
            size_int = int(p["size"])
            if size_int <= 0: continue
            s = pygame.Surface((size_int * 2, size_int * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color + (alpha,), (size_int, size_int), size_int)
            self.screen.blit(
                s,
                (int(p["pos"].x - size_int), int(p["pos"].y - size_int)),
                special_flags=pygame.BLEND_RGBA_ADD,
            )

        for f in self.fragments:
            for i, pos in enumerate(f["trail"]):
                alpha = int(100 * (i / len(f["trail"])))
                radius = int(f["size"] * 0.5 * (i / len(f["trail"])))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(
                        self.screen,
                        int(pos.x),
                        int(pos.y),
                        radius,
                        self.COLOR_FRAGMENT + (alpha,),
                    )
            pos_int = (int(f["pos"].x), int(f["pos"].y))
            size_int = max(1, int(f["size"]))
            pygame.gfxdraw.filled_circle(
                self.screen, pos_int[0], pos_int[1], size_int, self.COLOR_FRAGMENT
            )
            pygame.gfxdraw.aacircle(
                self.screen, pos_int[0], pos_int[1], size_int, self.COLOR_FRAGMENT
            )

        for inv in self.invaders:
            size_int = int(inv["size"])
            points = [
                pygame.Vector2(size_int, 0).rotate_rad(inv["angle"]),
                pygame.Vector2(-size_int, -size_int * 0.7).rotate_rad(inv["angle"]),
                pygame.Vector2(-size_int, size_int * 0.7).rotate_rad(inv["angle"]),
            ]
            points_abs = [(p + inv["pos"]) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, points_abs, self.COLOR_ENEMY_A)
            pygame.gfxdraw.filled_polygon(self.screen, points_abs, self.COLOR_ENEMY_A)

        launch_start = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        launch_end = launch_start + pygame.Vector2(
            math.cos(self.launch_angle), math.sin(self.launch_angle)
        ) * self.launch_power
        pygame.draw.line(self.screen, self.COLOR_PLAYER, launch_start, launch_end, 2)

        charge_ratio = (self.fragment_size - self.min_fragment_size) / (
            self.max_fragment_size - self.min_fragment_size
        )
        indicator_color = (
            self.COLOR_PLAYER if charge_ratio < 1.0 else self.COLOR_FRAGMENT
        )
        radius = int(self.fragment_size)
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(launch_start.x),
            int(launch_start.y),
            radius,
            indicator_color + (100,),
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(launch_start.x), int(launch_start.y), radius, indicator_color
        )

    def _render_ui(self):
        score_text = self.font_main.render(
            f"SCORE: {self.score}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(score_text, (10, 10))
        level_text = self.font_main.render(
            f"LEVEL: {self.current_level}-{self.current_wave}",
            True,
            self.COLOR_UI_TEXT,
        )
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        health_ratio = max(0, self.celestial_health / self.max_celestial_health)
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 20
        pygame.draw.rect(
            self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height)
        )
        pygame.draw.rect(
            self.screen,
            self.COLOR_HEALTH_BAR,
            (bar_x, bar_y, bar_width * health_ratio, bar_height),
        )

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            win_or_lose_text = (
                "LEVELS CLEARED"
                if self.current_level > self.MAX_LEVEL
                else "DEFENSES FAILED"
            )
            title_surf = self.font_title.render(
                win_or_lose_text, True, self.COLOR_UI_TEXT
            )
            title_rect = title_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(title_surf, title_rect)

            final_score_surf = self.font_main.render(
                f"FINAL SCORE: {self.score}", True, self.COLOR_UI_TEXT
            )
            final_score_rect = final_score_surf.get_rect(
                center=(self.WIDTH / 2, self.HEIGHT / 2 + 20)
            )
            self.screen.blit(final_score_surf, final_score_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()
    done = False

    pygame.display.set_caption("Astral Defender")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = [0, 0, 0]  # No-op, space released, shift released

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(
                f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Health: {info['health']}"
            )

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    print("Game Over!")
    print(f"Final Info: {info}")

    pygame.time.wait(3000)

    env.close()