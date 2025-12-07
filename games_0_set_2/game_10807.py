import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:06:18.748835
# Source Brief: brief_00807.md
# Brief Index: 807
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    game_description = (
        "Aim and fire projectiles to hit targets. Chain successful hits to build a score "
        "multiplier and get the high score before time runs out."
    )
    user_guide = "Controls: Use ← and → arrow keys to aim the launcher. Press space to fire."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_TIME_SECONDS = 60
    MAX_STEPS = MAX_TIME_SECONDS * metadata["render_fps"]
    WIN_SCORE = 500

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALLS = (60, 70, 90)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PROJECTILE = (255, 220, 0)
    COLOR_PROJECTILE_GLOW = (255, 180, 0, 64)  # RGBA
    COLOR_TARGET = (220, 50, 50)
    COLOR_TARGET_GLOW = (220, 50, 50, 64)  # RGBA
    COLOR_PARTICLE = (255, 120, 0)
    COLOR_UI_TEXT = (240, 240, 240)

    # Physics & Gameplay
    GRAVITY = 0.15
    LAUNCHER_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)
    LAUNCH_SPEED = 10.0
    ANGLE_SPEED = 0.05  # radians per step
    MIN_ANGLE = -math.pi * 0.95
    MAX_ANGLE = -math.pi * 0.05
    TARGET_RADIUS = 15

    # Rewards
    REWARD_HIT_TARGET_BASE = 0.1
    REWARD_HIT_TARGET_SCORE = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_multiplier = pygame.font.SysFont("Impact", 28, bold=True)

        # For human rendering mode
        self.window = None

        # Initialize state variables
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.launcher_angle = -math.pi / 2  # Straight up
        self.projectile = None
        self.targets = []
        self.particles = []
        self.multiplier = 1
        self.last_space_press = False

        self._spawn_target()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        if not self.game_over:
            movement = action[0]
            space_held = action[1] == 1

            self._handle_input(movement, space_held)
            self._update_projectile()
            self._update_particles()
            
            hit_reward = self._check_collisions()
            reward += hit_reward

            self.steps += 1
            self.time_remaining -= 1

        terminated = False
        if self.score >= self.WIN_SCORE:
            if not self.game_over: reward += self.REWARD_WIN
            terminated = True
        elif self.time_remaining <= 0:
            if not self.game_over: reward += self.REWARD_LOSE
            terminated = True

        self.game_over = terminated
        
        truncated = False # This environment does not have a truncation condition separate from termination

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # movement=1 aims left, movement=2 aims right
        if movement == 1: self.launcher_angle -= self.ANGLE_SPEED
        elif movement == 2: self.launcher_angle += self.ANGLE_SPEED
        self.launcher_angle = np.clip(self.launcher_angle, self.MIN_ANGLE, self.MAX_ANGLE)

        if space_held and not self.last_space_press and self.projectile is None:
            # SFX: Launch sound
            vel_x = math.cos(self.launcher_angle) * self.LAUNCH_SPEED
            vel_y = math.sin(self.launcher_angle) * self.LAUNCH_SPEED
            self.projectile = {"pos": list(self.LAUNCHER_POS), "vel": [vel_x, vel_y], "trail": []}
        self.last_space_press = space_held

    def _update_projectile(self):
        if self.projectile is None: return

        self.projectile["trail"].append(tuple(self.projectile["pos"]))
        if len(self.projectile["trail"]) > 15: self.projectile["trail"].pop(0)

        self.projectile["vel"][1] += self.GRAVITY
        self.projectile["pos"][0] += self.projectile["vel"][0]
        self.projectile["pos"][1] += self.projectile["vel"][1]

        px, py = self.projectile["pos"]
        if not (0 < px < self.SCREEN_WIDTH and py < self.SCREEN_HEIGHT):
            # SFX: Fizzle sound
            self.projectile = None
            if self.multiplier > 1: self.multiplier = 1 # SFX: Multiplier reset sound

    def _check_collisions(self):
        if self.projectile is None: return 0.0
        
        proj_pos = np.array(self.projectile["pos"])
        for target in self.targets[:]:
            target_pos = np.array(target["pos"])
            if np.linalg.norm(proj_pos - target_pos) < self.TARGET_RADIUS:
                # SFX: Explosion and Score sounds
                reward = self.REWARD_HIT_TARGET_BASE + (self.REWARD_HIT_TARGET_SCORE * self.multiplier)
                self.score += 10 * self.multiplier
                self.multiplier += 1
                
                self._create_particles(target["pos"], 30, self.COLOR_PARTICLE)
                self.targets.remove(target)
                self._spawn_target()
                
                self.projectile = None
                return reward
        return 0.0

    def _spawn_target(self):
        x = self.np_random.integers(self.TARGET_RADIUS + 20, self.SCREEN_WIDTH - self.TARGET_RADIUS - 20)
        y = self.np_random.integers(self.TARGET_RADIUS + 20, int(self.SCREEN_HEIGHT * 0.6))
        self.targets.append({"pos": (x, y)})

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({"pos": list(pos), "vel": vel, "lifetime": lifetime, "max_lifetime": lifetime, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.98; p["vel"][1] *= 0.98
            p["lifetime"] -= 1
            if p["lifetime"] <= 0: self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Chain Reaction")
        
        if self.clock is None: self.clock = pygame.time.Clock()

        observation = self._get_observation()
        surface = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))
        self.window.blit(surface, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_game(self):
        pygame.draw.line(self.screen, self.COLOR_WALLS, (0, self.LAUNCHER_POS[1]+10), (self.SCREEN_WIDTH, self.LAUNCHER_POS[1]+10), 3)

        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            radius = int(3 * (p["lifetime"] / p["max_lifetime"]))
            if radius > 0: self._draw_aa_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, p["color"])

        for target in self.targets:
            pos = (int(target["pos"][0]), int(target["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TARGET_RADIUS + 4, self.COLOR_TARGET_GLOW)
            self._draw_aa_circle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET)

        if self.projectile:
            if len(self.projectile["trail"]) > 1:
                points = self.projectile["trail"]
                for i in range(len(points) - 1):
                    alpha = int(255 * (i / len(points)))
                    color = (*self.COLOR_PROJECTILE, alpha)
                    start_pos = (int(points[i][0]), int(points[i][1]))
                    end_pos = (int(points[i+1][0]), int(points[i+1][1]))
                    pygame.draw.line(self.screen, color, start_pos, end_pos, max(1, int(4 * (i / len(points)))))

            pos = (int(self.projectile["pos"][0]), int(self.projectile["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_PROJECTILE_GLOW)
            self._draw_aa_circle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)

        launcher_len = 30
        end_x = self.LAUNCHER_POS[0] + math.cos(self.launcher_angle) * launcher_len
        end_y = self.LAUNCHER_POS[1] + math.sin(self.launcher_angle) * launcher_len
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.LAUNCHER_POS, (int(end_x), int(end_y)), 5)
        self._draw_aa_circle(self.screen, self.LAUNCHER_POS[0], self.LAUNCHER_POS[1], 7, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_sec = self.time_remaining // self.metadata["render_fps"]
        time_text = self.font_ui.render(f"TIME: {time_sec}", True, self.COLOR_UI_TEXT)
        text_rect = time_text.get_rect(center=(self.SCREEN_WIDTH // 2, 22))
        self.screen.blit(time_text, text_rect)

        if self.multiplier > 1:
            size_bonus = min(20, (self.multiplier - 1) * 2)
            dynamic_font = pygame.font.SysFont("Impact", 28 + size_bonus, bold=True)
            
            lerp = min(1.0, (self.multiplier - 1) / 10.0)
            color = tuple(int(c1 * (1-lerp) + c2 * lerp) for c1, c2 in zip((255,255,255), self.COLOR_PROJECTILE))

            mult_text = dynamic_font.render(f"x{self.multiplier}", True, color)
            text_rect = mult_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 5))
            self.screen.blit(mult_text, text_rect)

    def _draw_aa_circle(self, surface, x, y, radius, color):
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                # This outer loop break is important for a clean exit.
                continue

        if terminated:
            break

        keys = pygame.key.get_pressed()
        # Consistent with user_guide: ←/→ to aim
        if keys[pygame.K_LEFT]: movement = 1
        elif keys[pygame.K_RIGHT]: movement = 2
        
        if keys[pygame.K_SPACE]: space = 1
        # Shift is in the action space but unused in this game logic
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, term, trunc, info = env.step(action)

        if term or trunc:
            print(f"Game Over! Final Score: {info['score']}")
            # Auto-restart for continuous play in human mode
            obs, info = env.reset()

    env.close()