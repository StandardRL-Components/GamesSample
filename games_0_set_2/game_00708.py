# Generated: 2025-08-27T14:32:37.868599
# Source Brief: brief_00708.md
# Brief Index: 708

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ for steep up, ↓ for steep down, ← for gentle up, → for gentle down. No action is a slight decline."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track in real-time to guide your sled to the finish. Manage speed and angle to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.FPS = 30

        # Colors
        self.COLOR_BG_TOP = (173, 216, 230)
        self.COLOR_BG_BOTTOM = (135, 206, 250)
        self.COLOR_TRACK = (20, 20, 20)
        self.COLOR_SLED = (255, 69, 0)
        self.COLOR_SLED_BORDER = (139, 0, 0)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_START = (60, 179, 113, 150)
        self.COLOR_FINISH = (220, 20, 60, 150)
        self.COLOR_CHECKPOINT = (255, 215, 0, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # Physics & Game
        self.GRAVITY = 0.35
        self.FRICTION = 0.998
        self.DRAW_STEP_LENGTH = 15
        self.SLED_SIZE = pygame.Vector2(20, 10)
        self.PARTICLE_LIFESPAN = 25

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_ui_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_ui_small = pygame.font.SysFont(None, 24)

        # Pre-render background
        self.bg_surface = self._create_gradient_background()

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sled_pos = pygame.Vector2(0, 0)
        self.sled_vel = pygame.Vector2(0, 0)
        self.track_points = []
        self.checkpoints = []
        self.finish_line_x = 0
        self.particles = []
        
        # Initialize state
        # self.reset() is called by the wrapper, no need to call it here.

    def _create_gradient_background(self):
        """Creates a surface with a vertical gradient."""
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        r1, g1, b1 = self.COLOR_BG_TOP
        r2, g2, b2 = self.COLOR_BG_BOTTOM
        for y in range(self.HEIGHT):
            r = r1 + (r2 - r1) * (y / self.HEIGHT)
            g = g1 + (g2 - g1) * (y / self.HEIGHT)
            b = b1 + (b2 - b1) * (y / self.HEIGHT)
            pygame.draw.line(bg, (int(r), int(g), int(b)), (0, y), (self.WIDTH, y))
        return bg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.sled_pos = pygame.Vector2(50, 100)
        self.sled_vel = pygame.Vector2(2, 0)
        
        self.track_points = [pygame.Vector2(20, 120), pygame.Vector2(150, 125)]
        
        self.finish_line_x = self.WIDTH - 40
        self.checkpoints = [
            {"x": 250, "passed": False},
            {"x": 450, "passed": False},
        ]
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._extend_track(action)
        self._update_physics()
        self._update_particles()
        
        reward = 0
        terminated = False

        # Reward for forward movement
        if self.sled_vel.x > 0.1:
            reward += 0.1
        else:
            reward -= 0.01

        # Checkpoints
        for cp in self.checkpoints:
            if not cp["passed"] and self.sled_pos.x >= cp["x"]:
                cp["passed"] = True
                reward += 1.0
                self.score += 100
                # Sound: checkpoint.wav

        # Termination conditions
        if self.sled_pos.x >= self.finish_line_x:
            time_bonus = min(10.0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS * 10.0)
            reward += 10.0 + time_bonus
            self.score += 1000 + int(time_bonus * 100)
            terminated = True
            # Sound: victory.wav
        elif not (0 < self.sled_pos.y < self.HEIGHT):
            reward -= 1.0
            terminated = True
            # Sound: crash.wav
        elif self.steps >= self.MAX_STEPS - 1:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _extend_track(self, action):
        movement = action[0]
        
        angle_deg = {
            0: 5,   # None: Gentle down
            1: -45, # Up: Steep up
            2: 45,  # Down: Steep down
            3: -15, # Left: Gentle up
            4: 15,  # Right: Gentle down
        }[movement]
        
        angle_rad = math.radians(angle_deg)
        
        last_point = self.track_points[-1]
        new_point = last_point + pygame.Vector2(
            math.cos(angle_rad) * self.DRAW_STEP_LENGTH,
            math.sin(angle_rad) * self.DRAW_STEP_LENGTH,
        )

        # Clamp to screen bounds to prevent drawing off-screen
        new_point.x = max(0, min(self.WIDTH, new_point.x))
        new_point.y = max(0, min(self.HEIGHT, new_point.y))

        # Only add point if it extends the track forward
        if new_point.x > last_point.x:
            self.track_points.append(new_point)

    def _update_physics(self):
        self.sled_vel.y += self.GRAVITY
        
        old_pos = self.sled_pos.copy()
        self.sled_pos += self.sled_vel

        on_track = False
        if len(self.track_points) >= 2:
            for i in range(len(self.track_points) - 1):
                p1 = self.track_points[i]
                p2 = self.track_points[i+1]
                
                # Check if sled is horizontally within this segment
                if (p1.x <= self.sled_pos.x < p2.x) or (p2.x <= self.sled_pos.x < p1.x):
                    # Linearly interpolate track height at sled's x
                    dx = p2.x - p1.x
                    if abs(dx) < 1e-6: continue # Avoid division by zero for vertical lines
                    
                    t = (self.sled_pos.x - p1.x) / dx
                    track_y = p1.y + t * (p2.y - p1.y)

                    # Collision detected if sled was above and is now on or below track
                    if self.sled_pos.y >= track_y and old_pos.y < track_y:
                        self.sled_pos.y = track_y
                        
                        track_vec = p2 - p1
                        if track_vec.length() > 0:
                            proj = self.sled_vel.dot(track_vec.normalize())
                            self.sled_vel = track_vec.normalize() * proj
                        
                        self.sled_vel *= self.FRICTION
                        
                        on_track = True
                        # Sound: sled_grind_soft.wav
                        break
        
        # Add particles if on track and moving fast
        if on_track and self.sled_vel.length() > 1:
            for _ in range(int(self.sled_vel.length() / 4) + 1):
                self.particles.append({
                    "pos": self.sled_pos.copy() + pygame.Vector2(random.uniform(-3, 3), random.uniform(0, 5)),
                    "vel": self.sled_vel.rotate(random.uniform(-15, 15)) * -0.1,
                    "life": self.PARTICLE_LIFESPAN,
                    "size": random.uniform(2, 4)
                })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] -= 0.1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["size"] > 0]
    
    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw transparent vertical lines for start, finish, checkpoints
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        start_x = self.track_points[0].x
        pygame.draw.line(s, self.COLOR_START, (int(start_x), 0), (int(start_x), self.HEIGHT), 5)
        pygame.draw.line(s, self.COLOR_FINISH, (int(self.finish_line_x), 0), (int(self.finish_line_x), self.HEIGHT), 5)
        for cp in self.checkpoints:
            color = self.COLOR_CHECKPOINT if not cp["passed"] else (*self.COLOR_CHECKPOINT[:3], 50)
            pygame.draw.line(s, color, (int(cp["x"]), 0), (int(cp["x"]), self.HEIGHT), 3)
        self.screen.blit(s, (0, 0))

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / self.PARTICLE_LIFESPAN))
            color = (*self.COLOR_PARTICLE, int(alpha))
            pos = (int(p["pos"].x), int(p["pos"].y))
            size = int(p["size"])
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)

        # Draw track
        if len(self.track_points) >= 2:
            point_list = [(int(p.x), int(p.y)) for p in self.track_points]
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, point_list, 4)

        # Draw sled
        sled_rect = pygame.Rect(0, 0, self.SLED_SIZE.x, self.SLED_SIZE.y)
        sled_rect.center = (int(self.sled_pos.x), int(self.sled_pos.y))
        
        angle = self.sled_vel.angle_to(pygame.Vector2(1, 0))
        
        # Create a rotated surface for the sled
        scaled_size = (int(sled_rect.width * 1.2), int(sled_rect.height * 1.2))
        rotated_surf = pygame.Surface(scaled_size, pygame.SRCALPHA)
        inner_rect = pygame.Rect(sled_rect.width * 0.1, sled_rect.height * 0.1, sled_rect.width, sled_rect.height)
        pygame.draw.rect(rotated_surf, self.COLOR_SLED, inner_rect, border_radius=2)
        pygame.draw.rect(rotated_surf, self.COLOR_SLED_BORDER, inner_rect, 1, border_radius=2)
        
        rotated_surf = pygame.transform.rotate(rotated_surf, angle)
        rotated_rect = rotated_surf.get_rect(center=sled_rect.center)
        self.screen.blit(rotated_surf, rotated_rect)

    def _draw_text(self, text, font, color, pos, shadow_color, shadow_offset=(1, 1)):
        text_shadow = font.render(text, True, shadow_color)
        self.screen.blit(text_shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _render_ui(self):
        time_text = f"Time: {self.steps * (1/self.FPS):.1f}s"
        speed_text = f"Speed: {self.sled_vel.length():.1f}"
        
        self._draw_text(time_text, self.font_ui, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        speed_pos_x = self.WIDTH - self.font_ui.size(speed_text)[0] - 10
        self._draw_text(speed_text, self.font_ui, self.COLOR_TEXT, (speed_pos_x, 10), self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            if self.sled_pos.x >= self.finish_line_x:
                msg = "FINISH!"
            else:
                msg = "CRASHED!"
            
            msg_size = self.font_ui.size(msg)
            msg_pos = (self.WIDTH / 2 - msg_size[0] / 2, self.HEIGHT / 2 - msg_size[1] / 2)
            self._draw_text(msg, self.font_ui, self.COLOR_TEXT, msg_pos, self.COLOR_TEXT_SHADOW, (2, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": (self.sled_pos.x, self.sled_pos.y),
            "sled_vel": (self.sled_vel.x, self.sled_vel.y),
        }

    def render(self):
        return self._get_observation()