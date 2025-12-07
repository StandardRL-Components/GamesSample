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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the crosshair. Press space to fire."
    )

    game_description = (
        "A side-view target practice game. Hit 10 moving targets with your "
        "15 shots to win. Aim carefully to maximize your score!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Pygame setup
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.color_bg_top = (10, 10, 40)
        self.color_bg_bottom = (40, 10, 60)
        self.color_target = (220, 50, 50)
        self.color_projectile = (100, 255, 255)
        self.color_crosshair = (255, 255, 0)
        self.color_text = (255, 255, 255)
        self.color_cannon = (100, 100, 110)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # Game constants
        self.initial_ammo = 15
        self.win_condition_hits = 10
        self.max_steps = 1000
        self.crosshair_speed = 8.0
        self.projectile_speed = 15.0

        # Initialize state variables
        self.crosshair_pos = None
        self.ammo = 0
        self.targets_hit = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.space_was_pressed = False
        self.target_base_speed = 1.0
        self.reward_this_step = 0.0

        # The following call to reset() is to initialize self.np_random
        # which is needed for some of the initialization steps.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ammo = self.initial_ammo
        self.targets_hit = 0
        self.crosshair_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=float)
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.space_was_pressed = False
        self.target_base_speed = 1.0

        # Spawn initial targets
        for _ in range(5):
            self._spawn_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0.0
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            self._update_projectiles()
            self._update_targets()
            self._update_particles()
            self._handle_collisions()

            if self.np_random.random() < 0.025 and len(self.targets) < 15:
                self._spawn_target()

        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.crosshair_pos[1] -= self.crosshair_speed
        elif movement == 2: self.crosshair_pos[1] += self.crosshair_speed
        elif movement == 3: self.crosshair_pos[0] -= self.crosshair_speed
        elif movement == 4: self.crosshair_pos[0] += self.crosshair_speed

        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.screen_width)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.screen_height)

        if space_held and not self.space_was_pressed:
            self._fire_projectile()
        self.space_was_pressed = space_held

    def _fire_projectile(self):
        if self.ammo > 0:
            self.ammo -= 1
            self.reward_this_step -= 0.1  # Cost of firing (miss penalty)

            origin = np.array([30.0, 370.0])
            direction = self.crosshair_pos - origin
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist

            velocity = direction * self.projectile_speed
            self.projectiles.append({"pos": origin.copy(), "vel": velocity, "radius": 4})

    def _update_projectiles(self):
        updated_projectiles = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            if 0 <= p["pos"][0] <= self.screen_width and 0 <= p["pos"][1] <= self.screen_height:
                updated_projectiles.append(p)
        self.projectiles = updated_projectiles

    def _update_targets(self):
        updated_targets = []
        for t in self.targets:
            if t.get("type") == "sine":
                t["pos"][0] += t["vel"][0]
                t["pos"][1] = t["sine_y_base"] + t["sine_amplitude"] * math.sin(t["sine_freq"] * self.steps + t["sine_phase"])
            else:
                t["pos"] += t["vel"]
                if "gravity" in t:
                    t["vel"][1] += t["gravity"]
            
            is_in_bounds = (
                -t["radius"] < t["pos"][0] < self.screen_width + t["radius"] and
                -t["radius"] < t["pos"][1] < self.screen_height + t["radius"] + 100
            )
            if is_in_bounds:
                updated_targets.append(t)
        self.targets = updated_targets

    def _update_particles(self):
        updated_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] > 0:
                updated_particles.append(p)
        self.particles = updated_particles

    def _handle_collisions(self):
        kept_projectiles = []
        hit_target_indices = set()

        for p in self.projectiles:
            hit_a_target = False
            for i, t in enumerate(self.targets):
                if i in hit_target_indices:
                    continue

                dist = np.linalg.norm(p["pos"] - t["pos"])
                if dist < p["radius"] + t["radius"]:
                    self.targets_hit += 1
                    self.score += 100
                    self.reward_this_step += 1.1  # +1 for hit, cancelling out -0.1 fire cost
                    
                    self._create_explosion(t["pos"], self.color_target)

                    if self.targets_hit > 0 and self.targets_hit % 5 == 0:
                        self.target_base_speed += 0.2

                    hit_target_indices.add(i)
                    hit_a_target = True
                    break  # Projectile hits only one target
            
            if not hit_a_target:
                kept_projectiles.append(p)

        self.targets = [t for i, t in enumerate(self.targets) if i not in hit_target_indices]
        self.projectiles = kept_projectiles

    def _check_termination(self):
        terminated = False
        if self.targets_hit >= self.win_condition_hits:
            terminated = True
            self.reward_this_step += 10.0
            if not self.game_over: self.score += 1000
        elif self.ammo <= 0 and not self.projectiles:
            terminated = True
            self.reward_this_step -= 10.0
        elif self.steps >= self.max_steps:
            terminated = True
        
        if terminated:
            self.game_over = True
        return terminated

    def _spawn_target(self):
        side = self.np_random.choice(["left", "right"])
        x = -20 if side == "left" else self.screen_width + 20
        vx_base = self.np_random.uniform(0.8, 1.5) * self.target_base_speed
        vx = vx_base if side == "left" else -vx_base

        y = self.np_random.uniform(50, self.screen_height - 100)
        radius = self.np_random.integers(10, 25)
        
        target = {"pos": np.array([x, y], dtype=float), "radius": radius, "color": self.color_target}

        trajectory_type = self.np_random.choice(["linear", "parabolic", "sine"], p=[0.4, 0.3, 0.3])
        if trajectory_type == "linear":
            vy = self.np_random.uniform(-0.5, 0.5) * self.target_base_speed
            target["vel"] = np.array([vx, vy])
        elif trajectory_type == "parabolic":
            vy = self.np_random.uniform(-2.5, -1)
            target["vel"] = np.array([vx * 0.8, vy])
            target["gravity"] = 0.05
        elif trajectory_type == "sine":
            target["vel"] = np.array([vx, 0])
            target["sine_y_base"] = y
            target["sine_amplitude"] = self.np_random.uniform(30, 80)
            target["sine_freq"] = self.np_random.uniform(0.02, 0.05)
            target["sine_phase"] = self.np_random.uniform(0, 2 * math.pi)
            target["type"] = "sine"
            
        self.targets.append(target)

    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(1, 4), "color": color
            })

    def _get_observation(self):
        # Background gradient
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(self.color_bg_top, self.color_bg_bottom))
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
        
        # Render cannon
        pygame.draw.polygon(self.screen, self.color_cannon, [(10, 400), (10, 360), (50, 360), (70, 400)])
        
        # Render targets
        for t in self.targets:
            pos = t["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(t["radius"]), t["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(t["radius"]), (255, 255, 255))

        # Render projectiles
        for p in self.projectiles:
            pos = p["pos"].astype(int)
            end_pos = (p["pos"] - p["vel"] * 0.7).astype(int)
            pygame.draw.line(self.screen, self.color_projectile, (pos[0], pos[1]), (end_pos[0], end_pos[1]), 4)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), (200, 255, 255))
            
        # Render particles
        for p in self.particles:
            pos = p["pos"].astype(int)
            life_ratio = p["life"] / 30.0
            fade_color = tuple(int(c * life_ratio) for c in p["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"] * life_ratio), fade_color)

        # Render crosshair
        cx, cy = self.crosshair_pos.astype(int)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 15, (80, 80, 0)) # Glow
        pygame.draw.line(self.screen, self.color_crosshair, (cx - 12, cy), (cx + 12, cy), 2)
        pygame.draw.line(self.screen, self.color_crosshair, (cx, cy - 12), (cx, cy + 12), 2)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, 8, self.color_crosshair)

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text_with_outline(self, text, font, color, pos, outline_color=(0,0,0)):
        text_surface = font.render(text, True, color)
        outline_surface = font.render(text, True, outline_color)
        text_rect = text_surface.get_rect(topleft=pos)
        offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for dx, dy in offsets:
            self.screen.blit(outline_surface, (text_rect.x + dx, text_rect.y + dy))
        self.screen.blit(text_surface, text_rect)

    def _render_ui(self):
        self._render_text_with_outline(f"SCORE: {self.score}", self.font_ui, self.color_text, (10, 10))
        self._render_text_with_outline(f"HITS: {self.targets_hit}/{self.win_condition_hits}", self.font_ui, self.color_text, (self.screen_width - 200, 10))
        
        ammo_color = (255, 255, 0) if self.ammo > 5 else ((255, 100, 0) if self.ammo > 0 else (255, 0, 0))
        self._render_text_with_outline(f"AMMO: {self.ammo}", self.font_ui, ammo_color, (10, 35))

        if self.game_over:
            msg, msg_color = ("YOU WIN!", (0, 255, 0)) if self.targets_hit >= self.win_condition_hits else ("GAME OVER", (255, 0, 0))
            text_surf = self.font_msg.render(msg, True, msg_color)
            text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            outline_surf = self.font_msg.render(msg, True, (0,0,0))
            offsets = [(-2, -2), (2, -2), (-2, 2), (2, 2)]
            for dx, dy in offsets:
                self.screen.blit(outline_surf, (text_rect.x + dx, text_rect.y + dy))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_hit": self.targets_hit,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()