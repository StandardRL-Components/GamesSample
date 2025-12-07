import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates time to solve rune-based puzzles,
    opening portals to gather resources and craft a Viking longship to escape a cursed fjord.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Manipulate time to align rotating runes, gather resources from portals, "
        "and repair your longship to escape a cursed fjord."
    )
    user_guide = (
        "Controls: ↑↓ to change speed, ←→ to reverse direction of the selected rune. "
        "Press shift to cycle runes and space to repair the ship."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400

        # Colors
        self.COLOR_BG = (26, 42, 58)
        self.COLOR_WATER = (42, 74, 106)
        self.COLOR_SHIP = (96, 96, 112)
        self.COLOR_SHIP_DAMAGE = (192, 57, 43)
        self.COLOR_RUNE = (74, 35, 90)
        self.COLOR_RUNE_TARGET = (174, 214, 241, 100)
        self.COLOR_RUNE_SELECTED_GLOW = (241, 196, 15)
        self.COLOR_PORTAL = (216, 27, 96)
        self.COLOR_RESOURCE = (241, 196, 15)
        self.COLOR_UI_TEXT = (236, 240, 241)
        self.COLOR_UI_BAR = (39, 174, 96)
        self.COLOR_UI_BAR_BG = (44, 62, 80)

        # Game Parameters
        self.MAX_STEPS = 2000
        self.WIN_HEALTH = 100.0
        self.SHIP_DECAY_RATE = 0.05
        self.REPAIR_AMOUNT_PER_RESOURCE = 5.0
        self.RUNE_SPEED_INCREMENT = 0.005
        self.MAX_RUNE_SPEED = 0.1
        self.RUNE_ALIGNMENT_TOLERANCE = 0.1  # Radians
        self.TARGET_ANGLE = -math.pi / 2  # Pointing up
        self.MAX_RUNES = 5

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)

        # --- Internal State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.longship_health = 0.0
        self.resources = 0
        self.num_runes = 0
        self.runes = []
        self.portal_activations = 0
        self.selected_rune_idx = 0
        self.portal_particles = []
        self.resource_particles = []
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.longship_health = self.WIN_HEALTH - 1.0  # Start slightly damaged
        self.resources = 10
        self.num_runes = 2
        self.portal_activations = 0
        self.selected_rune_idx = 0
        self.prev_shift_held = False

        self.portal_particles = []
        self.resource_particles = []

        self.runes = []
        for i in range(self.MAX_RUNES):
            self.runes.append({
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.np_random.uniform(-0.02, 0.02),
                "pos": (100 + i * 110, self.HEIGHT / 2 - 20),
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        # --- Action Processing ---
        if shift_held and not self.prev_shift_held:
            self.selected_rune_idx = (self.selected_rune_idx + 1) % self.num_runes
        self.prev_shift_held = shift_held

        if movement == 0:  # Pause all
            for i in range(self.num_runes):
                self.runes[i]["speed"] = 0
        else:
            sel_rune = self.runes[self.selected_rune_idx]
            if movement == 1:  # Speed up
                sel_rune["speed"] += self.RUNE_SPEED_INCREMENT
            elif movement == 2:  # Speed down
                sel_rune["speed"] -= self.RUNE_SPEED_INCREMENT
            elif movement == 3 or movement == 4:  # Reverse direction
                sel_rune["speed"] *= -1
            sel_rune["speed"] = np.clip(sel_rune["speed"], -self.MAX_RUNE_SPEED, self.MAX_RUNE_SPEED)

        # --- Game Logic Update ---
        for i in range(self.num_runes):
            self.runes[i]["angle"] += self.runes[i]["speed"]
            self.runes[i]["angle"] %= (2 * math.pi)

        initial_health = self.longship_health
        if self.longship_health > 0:
            self.longship_health -= self.SHIP_DECAY_RATE

        if space_held and self.resources > 0 and self.longship_health < self.WIN_HEALTH:
            self.resources -= 1
            self.longship_health += self.REPAIR_AMOUNT_PER_RESOURCE
            self.longship_health = min(self.longship_health, self.WIN_HEALTH)
            reward += 0.5  # Reward for repairing

        if self.longship_health < initial_health:
            reward -= 0.01

        all_aligned = True
        for i in range(self.num_runes):
            angle_diff = abs(self.runes[i]["angle"] - (self.TARGET_ANGLE + 2 * math.pi) % (2 * math.pi))
            if min(angle_diff, 2 * math.pi - angle_diff) > self.RUNE_ALIGNMENT_TOLERANCE:
                all_aligned = False
                break

        if all_aligned:
            reward += 1.0
            self.portal_activations += 1
            resources_spawned = self.np_random.integers(2, 5)
            for _ in range(resources_spawned):
                self._spawn_resource_particle()
            self._spawn_portal_effect()
            if self.num_runes < self.MAX_RUNES and self.portal_activations > 0 and self.portal_activations % 20 == 0:
                self.num_runes += 1
                reward += 5.0

        reward += self._update_all_particles()

        # --- Termination Check ---
        terminated = False
        if self.longship_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.longship_health >= self.WIN_HEALTH:
            reward += 100
            terminated = True
            self.game_over = True
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "longship_health": self.longship_health,
            "resources": self.resources,
            "unlocked_runes": self.num_runes,
        }

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_WATER, (0, self.HEIGHT - 80, self.WIDTH, 80))
        for i in range(5):
            x = i * self.WIDTH / 4
            h = 100 + (i % 2) * 40
            pygame.draw.polygon(self.screen, self.COLOR_WATER,
                                [(x - 100, self.HEIGHT - 80), (x + 100, self.HEIGHT - 80),
                                 (x, self.HEIGHT - 80 - h)])

        self._render_longship()
        self._render_runes()
        self._render_particles()
        self._render_ui()

    def _render_longship(self):
        ship_base_y = self.HEIGHT - 60
        pygame.draw.polygon(self.screen, self.COLOR_SHIP, [
            (self.WIDTH / 2 - 200, ship_base_y), (self.WIDTH / 2 + 200, ship_base_y),
            (self.WIDTH / 2 + 160, ship_base_y + 30), (self.WIDTH / 2 - 160, ship_base_y + 30)
        ])
        pygame.draw.rect(self.screen, self.COLOR_SHIP, (self.WIDTH / 2 - 5, ship_base_y - 80, 10, 80))

        damage_level = 1 - (self.longship_health / self.WIN_HEALTH)
        num_cracks = int(damage_level * 15)
        if num_cracks > 0:
            crack_rng = np.random.default_rng(seed=int(self.longship_health))
            for _ in range(num_cracks):
                start_x = crack_rng.integers(self.WIDTH / 2 - 150, self.WIDTH / 2 + 151)
                start_y = crack_rng.integers(ship_base_y, ship_base_y + 26)
                end_x = start_x + crack_rng.integers(-10, 11)
                end_y = start_y + crack_rng.integers(5, 11)
                pygame.draw.line(self.screen, self.COLOR_SHIP_DAMAGE, (start_x, start_y), (end_x, end_y), 2)

    def _render_runes(self):
        for i in range(self.num_runes):
            rune = self.runes[i]
            pos = (int(rune["pos"][0]), int(rune["pos"][1]))
            is_selected = (i == self.selected_rune_idx)

            if is_selected:
                self._draw_glowing_circle(pos, 50, self.COLOR_RUNE_SELECTED_GLOW)

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 40, self.COLOR_RUNE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 40, self.COLOR_RUNE)
            self._draw_glowing_circle(pos, 42, self.COLOR_RUNE_TARGET, max_alpha=50, radius_scale=1.05, steps=5)

            points = [(-20, -25), (20, -25), (0, -25), (0, 25), (0, -5), (15, -5)]
            rotated_points = []
            for p in points:
                x = p[0] * math.cos(rune["angle"]) - p[1] * math.sin(rune["angle"])
                y = p[0] * math.sin(rune["angle"]) + p[1] * math.cos(rune["angle"])
                rotated_points.append((pos[0] + x, pos[1] + y))

            pygame.draw.lines(self.screen, self.COLOR_UI_TEXT, False, rotated_points, 3)

    def _render_particles(self):
        for p in self.portal_particles:
            if p["size"] > 0:
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = (*self.COLOR_PORTAL, alpha)
                self._draw_glowing_circle(p["pos"], int(p["size"]), color, steps=3, max_alpha=alpha)

        for p in self.resource_particles:
            self._draw_glowing_circle(p["pos"], int(p["size"]), self.COLOR_RESOURCE)

    def _render_ui(self):
        bar_width, bar_height = 300, 20
        bar_x, bar_y = (self.WIDTH - bar_width) / 2, self.HEIGHT - 35
        health_ratio = max(0, self.longship_health / self.WIN_HEALTH)

        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_width * health_ratio, bar_height),
                         border_radius=5)

        health_text = self.font_small.render("LONGSHIP INTEGRITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (bar_x + (bar_width - health_text.get_width()) / 2, bar_y - 18))

        resource_text = self.font_large.render(f"RESOURCES: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(resource_text, (self.WIDTH - resource_text.get_width() - 20, 20))

        step_text = self.font_small.render(f"DAY: {self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(step_text, (20, 20))
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 40))

    def _draw_glowing_circle(self, pos, radius, color, steps=5, max_alpha=150, radius_scale=1.2):
        x, y = int(pos[0]), int(pos[1])
        base_color = color[:3]

        for i in range(steps):
            frac = 1 - (i / steps)
            alpha = int(max_alpha * frac ** 2)
            current_radius = int(radius * (1 + (radius_scale - 1) * (i / steps)))

            if len(color) == 4:
                alpha = min(alpha, color[3])

            if alpha <= 0 or current_radius <= 0: continue

            s = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*base_color, alpha), (current_radius, current_radius), current_radius)
            self.screen.blit(s, (x - current_radius, y - current_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, x, y, radius, base_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, base_color)

    def _spawn_portal_effect(self):
        center_x = sum(r["pos"][0] for r in self.runes[:self.num_runes]) / self.num_runes
        center_y = sum(r["pos"][1] for r in self.runes[:self.num_runes]) / self.num_runes

        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.portal_particles.append({
                "pos": pygame.Vector2(center_x, center_y),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "size": self.np_random.uniform(5, 10),
                "life": self.np_random.integers(20, 40),
                "max_life": 40
            })

    def _spawn_resource_particle(self):
        center_x = sum(r["pos"][0] for r in self.runes[:self.num_runes]) / self.num_runes
        center_y = sum(r["pos"][1] for r in self.runes[:self.num_runes]) / self.num_runes

        self.resource_particles.append({
            "pos": pygame.Vector2(center_x, center_y),
            "speed": self.np_random.uniform(2, 4),
            "size": self.np_random.uniform(5, 8),
            "life": 150
        })

    def _update_all_particles(self):
        """Processes all particles, returns reward from collected resources."""
        # Update portal particles
        self.portal_particles = [p for p in self.portal_particles if p["life"] > 1]
        for p in self.portal_particles:
            p["life"] -= 1
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["size"] -= 0.2

        # Update resource particles
        collected_reward = 0
        new_resource_particles = []
        for p in self.resource_particles:
            p["life"] -= 1
            if p["life"] <= 0:
                continue

            target_vec = pygame.Vector2(self.WIDTH - 80, 25) - p["pos"]
            if target_vec.length() < 10:
                self.resources += 1
                collected_reward += 0.1
            else:
                target_vec.scale_to_length(min(target_vec.length(), p["speed"]))
                p["pos"] += target_vec
                p["speed"] += 0.2
                new_resource_particles.append(p)
        self.resource_particles = new_resource_particles
        return collected_reward

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)