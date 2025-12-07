import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:08:31.687062
# Source Brief: brief_00830.md
# Brief Index: 830
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A steampunk-themed puzzle-platformer Gymnasium environment.
    The agent must place steam-powered platforms and flip gravity to guide a robot
    to a goal portal before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a robot to the goal by placing steam-powered platforms and flipping gravity "
        "to navigate the level before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to place a platform "
        "and shift to flip gravity."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1800  # 60 seconds at 30 FPS

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GEAR_1 = (30, 45, 60)
        self.COLOR_GEAR_2 = (40, 60, 80)
        self.COLOR_PLAYER = (255, 215, 0)  # Gold
        self.COLOR_PLAYER_GLOW = (255, 215, 0, 50)
        self.COLOR_PLATFORM = (0, 200, 255)  # Bright Cyan
        self.COLOR_PLATFORM_GLOW = (0, 200, 255, 50)
        self.COLOR_STATIC_PLATFORM = (100, 110, 120)
        self.COLOR_GOAL = (255, 50, 50)  # Red
        self.COLOR_GOAL_GLOW = (255, 50, 50, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_STEAM = (200, 200, 210)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 0
        self.time_remaining = 0
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_size = (16, 24)
        self.on_ground = False
        self.gravity_val = 0.5
        self.gravity_direction = 1  # 1 for down, -1 for up
        self.cursor_pos = np.array([0.0, 0.0])
        self.cursor_speed = 10
        self.platforms = []
        self.static_platforms = []
        self.platform_size = (80, 12)
        self.goal_pos = np.array([0.0, 0.0])
        self.goal_radius = 20
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.background_gears = []
        
        # self.reset() # reset is called by the environment runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.game_over:
             self.level += 1

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_level()

        self.player_vel = np.array([0.0, 0.0])
        self.gravity_direction = 1
        self.cursor_pos = np.array([self.player_pos[0], self.player_pos[1] - 50])
        self.platforms = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        if not self.background_gears:
            for _ in range(15):
                self.background_gears.append({
                    "pos": (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                    "radius": self.np_random.integers(20, 100),
                    "teeth": self.np_random.integers(8, 20),
                    "speed": self.np_random.uniform(-0.5, 0.5),
                    "angle": self.np_random.uniform(0, 360),
                    "color": self.COLOR_GEAR_1 if self.np_random.random() > 0.5 else self.COLOR_GEAR_2
                })

        difficulty = self.level // 5
        self.time_remaining = max(30 * 20, (30 * 60) - difficulty * (30 * 5)) # 20s to 60s

        self.static_platforms = []
        start_y = self.np_random.integers(self.HEIGHT - 80, self.HEIGHT - 40)
        start_x = self.np_random.integers(40, 100)
        start_platform = pygame.Rect(start_x, start_y, 100, 20)
        self.static_platforms.append(start_platform)
        self.player_pos = np.array([start_platform.centerx, start_platform.top - self.player_size[1]], dtype=float)

        goal_y = self.np_random.integers(40, 120)
        goal_x = self.np_random.integers(self.WIDTH - 140, self.WIDTH - 60)
        
        min_dist = 200 + difficulty * 40
        if abs(goal_x - start_x) < min_dist:
            goal_x = start_x + self.np_random.integers(min_dist, min_dist + 100)
            goal_x = min(goal_x, self.WIDTH - 60)

        self.goal_pos = np.array([goal_x, goal_y], dtype=float)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        old_dist_to_goal = math.dist(self.player_pos, self.goal_pos)
        
        self._handle_input(action)
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        self.time_remaining -= 1

        new_dist_to_goal = math.dist(self.player_pos, self.goal_pos)
        reward += (old_dist_to_goal - new_dist_to_goal) * 0.1
        self.score += (old_dist_to_goal - new_dist_to_goal) * 0.1

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated:
            if math.dist(self.player_pos, self.goal_pos) < self.goal_radius + self.player_size[0] / 2:
                reward = 100.0
                self.score += 100.0
            else:
                reward = -100.0
                self.score -= 100.0
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        
        if movement == 1: self.cursor_pos[1] -= self.cursor_speed
        elif movement == 2: self.cursor_pos[1] += self.cursor_speed
        elif movement == 3: self.cursor_pos[0] -= self.cursor_speed
        elif movement == 4: self.cursor_pos[0] += self.cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
        space_held = space_action == 1
        if space_held and not self.last_space_held:
            self._place_platform()
            # sfx: platform_create.wav
        self.last_space_held = space_held

        shift_held = shift_action == 1
        if shift_held and not self.last_shift_held:
            self.gravity_direction *= -1
            # sfx: gravity_shift.wav
        self.last_shift_held = shift_held

    def _place_platform(self):
        self.score -= 1.0  # Cost for placing a platform

        new_plat_rect = pygame.Rect(
            self.cursor_pos[0] - self.platform_size[0] / 2,
            self.cursor_pos[1] - self.platform_size[1] / 2,
            *self.platform_size
        )
        
        if new_plat_rect.left < 0 or new_plat_rect.right > self.WIDTH or \
           new_plat_rect.top < 0 or new_plat_rect.bottom > self.HEIGHT:
            return

        player_rect = self._get_player_rect()
        if new_plat_rect.colliderect(player_rect):
            return

        if math.dist(new_plat_rect.center, self.goal_pos) < self.goal_radius + new_plat_rect.width / 2:
            return

        self.platforms.append(new_plat_rect)

        for _ in range(20):
            self.particles.append({
                'pos': np.array(new_plat_rect.center, dtype=float) + self.np_random.uniform(-10, 10, 2),
                'vel': self.np_random.uniform(-0.5, 0.5, 2) + np.array([0, -self.gravity_direction * 0.5]),
                'life': self.np_random.integers(20, 40),
                'size': self.np_random.uniform(2, 5)
            })

    def _update_physics(self):
        self.player_vel[1] += self.gravity_val * self.gravity_direction
        self.player_vel[1] = np.clip(self.player_vel[1], -15, 15)
        self.player_pos += self.player_vel

        self.on_ground = False
        player_rect = self._get_player_rect()
        
        for plat in self.platforms + self.static_platforms:
            if player_rect.colliderect(plat):
                if self.gravity_direction == 1:
                    if self.player_pos[1] + self.player_size[1] - self.player_vel[1] <= plat.top + 1:
                        self.player_pos[1] = plat.top - self.player_size[1]
                        self.player_vel[1] = 0
                        self.on_ground = True
                else:
                    if self.player_pos[1] - self.player_vel[1] >= plat.bottom - 1:
                        self.player_pos[1] = plat.bottom
                        self.player_vel[1] = 0
                        self.on_ground = True
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_size[0] / 2, self.WIDTH - self.player_size[0] / 2)

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos[0] - self.player_size[0] / 2, self.player_pos[1], *self.player_size)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.98)

    def _check_termination(self):
        if math.dist(self.player_pos, self.goal_pos) < self.goal_radius + self.player_size[0] / 2:
            return True # sfx: win.wav
        if self.time_remaining <= 0:
            return True # sfx: timeout.wav
        if not (-self.player_size[1] < self.player_pos[1] < self.HEIGHT):
            return True # sfx: fall.wav
            
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining, "level": self.level}

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for gear in self.background_gears:
            gear['angle'] += gear['speed']
            self._draw_gear(self.screen, gear['pos'], gear['radius'], gear['teeth'], gear['angle'], gear['color'])

    def _render_game_elements(self):
        for plat in self.static_platforms:
            pygame.draw.rect(self.screen, self.COLOR_STATIC_PLATFORM, plat, border_radius=3)
        for plat in self.platforms:
            self._draw_glowing_rect(self.screen, plat, self.COLOR_PLATFORM, self.COLOR_PLATFORM_GLOW)
        for p in self.particles:
            if p['size'] > 1:
                pygame.draw.circle(self.screen, self.COLOR_STEAM, p['pos'], p['size'])
        self._draw_glowing_circle(self.screen, self.goal_pos, self.goal_radius, self.COLOR_GOAL, self.COLOR_GOAL_GLOW)
        self._draw_player()
        if not self.game_over:
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (self.cursor_pos[0] - 2, self.cursor_pos[1] - 10, 4, 20), 1)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (self.cursor_pos[0] - 10, self.cursor_pos[1] - 2, 20, 4), 1)

    def _draw_player(self):
        player_rect = self._get_player_rect()
        self._draw_glowing_rect(self.screen, player_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 10, 4)
        
        eye_y = player_rect.centery - 5 if self.gravity_direction == 1 else player_rect.centery + 5
        pygame.draw.circle(self.screen, self.COLOR_BG, (player_rect.centerx, eye_y), 3)

        if not self.on_ground and abs(self.player_vel[1]) > 0.1:
            flame_length = min(20, abs(self.player_vel[1]) * 2) * (1 if self.player_vel[1] * self.gravity_direction > 0 else -1)
            flame_color = (255, 150 + self.np_random.integers(0, 105), 0)
            if self.gravity_direction == 1:
                flame_rect = pygame.Rect(player_rect.left + 4, player_rect.top - flame_length, 8, flame_length)
            else:
                flame_rect = pygame.Rect(player_rect.left + 4, player_rect.bottom, 8, flame_length)
            pygame.draw.rect(self.screen, flame_color, flame_rect, border_radius=4)

    def _render_ui(self):
        time_text = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 20, 10))
        
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        level_text = f"LEVEL: {self.level + 1}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (20, 35))

    def _draw_gear(self, surface, pos, radius, num_teeth, angle_deg, color):
        angle_rad = math.radians(angle_deg)
        points = []
        for i in range(num_teeth * 2):
            r = radius if i % 2 == 0 else radius - radius / 10
            a = angle_rad + (i / (num_teeth * 2)) * 2 * math.pi
            points.append((pos[0] + r * math.cos(a), pos[1] + r * math.sin(a)))
        if len(points) > 2: pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.7), color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.3), self.COLOR_BG)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(4):
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius + i * 3, glow_color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)

    def _draw_glowing_rect(self, surface, rect, color, glow_color, glow_radius=8, border_radius=3):
        temp_surf = pygame.Surface((rect.width + glow_radius * 2, rect.height + glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, glow_color, temp_surf.get_rect(), border_radius=border_radius + glow_radius)
        surface.blit(temp_surf, (rect.left - glow_radius, rect.top - glow_radius))
        pygame.draw.rect(surface, color, rect, border_radius=border_radius)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for local testing and visualization.
    # It will not be executed in the headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Steampunk Gravity Platformer")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Map keyboard inputs to actions
        movement, space_action, shift_action = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        # Use event queue for single-press actions
        space_pressed_this_frame = False
        shift_pressed_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_SPACE:
                    space_pressed_this_frame = True
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift_pressed_this_frame = True

        if not done:
            # Continuous actions (cursor movement)
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            # The environment handles the "press" logic (edge detection)
            # So we send the current state of the key (held or not)
            if keys[pygame.K_SPACE]: space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

            action = [movement, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if done:
            font = pygame.font.SysFont("Consolas", 50, bold=True)
            win_condition = math.dist(env.player_pos, env.goal_pos) < env.goal_radius + env.player_size[0] / 2
            msg = "GOAL REACHED!" if win_condition else "FAILED"
            text_surf = font.render(msg, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(env.WIDTH / 2, env.HEIGHT / 2 - 30))
            screen.blit(text_surf, text_rect)
            
            font_small = pygame.font.SysFont("Consolas", 20)
            reset_surf = font_small.render("Press 'R' to restart", True, (255, 255, 255))
            reset_rect = reset_surf.get_rect(center=(env.WIDTH / 2, env.HEIGHT / 2 + 20))
            screen.blit(reset_surf, reset_rect)

        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()