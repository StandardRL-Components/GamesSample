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
    user_guide = "Controls: ←→ to move in air. Press Space to jump. Press Shift to cycle color."

    # Must be a short, user-facing description of the game:
    game_description = "Climb a procedurally generated tower of color-coded platforms, matching your avatar's color to ascend."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    WIN_HEIGHT = 100

    # Colors
    COLORS = [
        (255, 87, 87),    # Red
        (87, 255, 87),    # Green
        (87, 87, 255),    # Blue
        (255, 255, 87),   # Yellow
        (170, 87, 255),   # Purple
    ]
    COLOR_BG_TOP = (13, 17, 23)
    COLOR_BG_BOTTOM = (25, 31, 40)
    COLOR_UI = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)

    # Physics
    GRAVITY = 0.5
    JUMP_STRENGTH = -10
    AIR_CONTROL_SPEED = 5
    PLAYER_SIZE = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.large_font = pygame.font.SysFont("Consolas", 60, bold=True)

        # Initialize state variables
        # self.reset() is called by the validation function
        
        # A seed is required for the first reset, let's use a default one.
        self.reset(seed=0)
        
        # self.validate_implementation() # This is a helper for development, not part of the standard env.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # If seed is None, we continue using the existing RNG

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_color_idx = 0
        self.on_ground = True

        # Action state
        self.last_space_held = False
        self.last_shift_held = False

        # Game state
        self.platforms = []
        self.particles = []
        self.camera_y = 0
        self.max_height_reached = 0
        self._generate_initial_platforms()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        reward = 0
        self.steps += 1

        # --- Handle Input ---
        if shift_held and not self.last_shift_held:
            self.player_color_idx = (self.player_color_idx + 1) % len(self.COLORS)
            # sfx: color_change_sound
            self._create_particles(self.player_pos, self.COLORS[self.player_color_idx], 15, 'burst')

        if space_held and not self.last_space_held and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound
            self._create_particles(self.player_pos + pygame.Vector2(0, self.PLAYER_SIZE/2), self.COLOR_WHITE, 10, 'down_burst')

        if not self.on_ground:
            if movement == 3: # Left
                self.player_vel.x = -self.AIR_CONTROL_SPEED
            elif movement == 4: # Right
                self.player_vel.x = self.AIR_CONTROL_SPEED
            else:
                self.player_vel.x *= 0.9 # Air friction
        else:
            self.player_vel.x = 0

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Physics Update ---
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel

        if self.player_pos.x < self.PLAYER_SIZE / 2:
            self.player_pos.x = self.PLAYER_SIZE / 2
        if self.player_pos.x > self.WIDTH - self.PLAYER_SIZE / 2:
            self.player_pos.x = self.WIDTH - self.PLAYER_SIZE / 2

        # --- Collision and Game Logic ---
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        if self.player_vel.y >= 0:
            for plat in self.platforms:
                plat_rect = pygame.Rect(plat['pos'].x, plat['pos'].y, plat['size'].x, plat['size'].y)
                if player_rect.colliderect(plat_rect) and self.player_pos.y < plat['pos'].y + self.player_vel.y:
                    if self.player_color_idx == plat['color_idx']:
                        self.player_pos.y = plat['pos'].y - self.PLAYER_SIZE / 2
                        self.player_vel.y = 0
                        self.on_ground = True
                        reward += 1.0
                        # sfx: land_sound
                        self._create_particles(pygame.Vector2(self.player_pos.x, plat['pos'].y), self.COLORS[plat['color_idx']], 20, 'land_burst')
                        break
                    else:
                        self.game_over = True
                        reward -= 1.0
                        # sfx: fail_sound
                        self._create_particles(self.player_pos, self.COLORS[self.player_color_idx], 50, 'burst')
                        break

        # --- Update State & Rewards ---
        current_height = (self.HEIGHT - 30 - self.player_pos.y) / 10
        if current_height > self.max_height_reached:
            height_gain = current_height - self.max_height_reached
            reward += height_gain * 0.1
            self.max_height_reached = current_height

        self.score += reward

        self._update_particles()
        self._manage_platforms()

        # --- Termination Conditions ---
        terminated = self.game_over
        if self.max_height_reached >= self.WIN_HEIGHT:
            reward += 100
            self.score += 100
            terminated = True
        if self.player_pos.y > self.camera_y + self.HEIGHT + self.PLAYER_SIZE:
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated, terminated should also be true.
            
        self.game_over = terminated

        # --- Camera Update ---
        target_camera_y = self.player_pos.y - self.HEIGHT * 0.7
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_initial_platforms(self):
        self.platforms.append({
            'pos': pygame.Vector2(self.WIDTH / 2 - 50, self.HEIGHT - 30),
            'size': pygame.Vector2(100, 20),
            'color_idx': 0
        })
        while self.platforms[-1]['pos'].y > -20:
             self._generate_next_platform()

    def _generate_next_platform(self):
        last_plat = self.platforms[-1]
        difficulty = min(1.0, self.max_height_reached / self.WIN_HEIGHT)
        
        plat_dist_factor = 0.1 * (self.max_height_reached // 10)
        min_dy = 60 + 20 * difficulty
        max_dy = 100 + 40 * difficulty + plat_dist_factor
        dy = self.rng.uniform(min_dy, max_dy)
        new_y = last_plat['pos'].y - dy

        max_dx = 150 + 100 * difficulty
        min_x = max(20, last_plat['pos'].x - max_dx)
        max_x = min(self.WIDTH - 120, last_plat['pos'].x + max_dx)
        new_x = self.rng.uniform(min_x, max_x)

        new_width = self.rng.uniform(max(50, 80 - 30 * difficulty), max(80, 150 - 70 * difficulty))

        self.platforms.append({
            'pos': pygame.Vector2(new_x, new_y),
            'size': pygame.Vector2(new_width, 20),
            'color_idx': self.rng.integers(0, len(self.COLORS))
        })

    def _manage_platforms(self):
        if self.platforms[-1]['pos'].y > self.camera_y - 20:
            self._generate_next_platform()
        self.platforms = [p for p in self.platforms if p['pos'].y < self.camera_y + self.HEIGHT + 50]

    def _create_particles(self, pos, color, count, p_type):
        for _ in range(count):
            if p_type == 'burst':
                angle = self.rng.uniform(0, 2 * math.pi)
                speed = self.rng.uniform(1, 5)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            elif p_type == 'down_burst':
                angle = self.rng.uniform(0.25 * math.pi, 0.75 * math.pi)
                speed = self.rng.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            elif p_type == 'land_burst':
                angle = self.rng.uniform(math.pi, 2 * math.pi)
                speed = self.rng.uniform(0.5, 2.5)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            else:
                vel = pygame.Vector2(0,0)

            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': self.rng.uniform(15, 30),
                'color': color, 'size': self.rng.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        # Clear screen with background
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(
                self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            x, y = int(plat['pos'].x), int(plat['pos'].y - self.camera_y)
            w, h = int(plat['size'].x), int(plat['size'].y)
            color = self.COLORS[plat['color_idx']]
            shadow_color = tuple(c * 0.5 for c in color)
            
            if y + h > 0 and y < self.HEIGHT:
                pygame.draw.rect(self.screen, shadow_color, (x, y + 5, w, h), border_radius=5)
                pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=5)
                pygame.draw.line(self.screen, self.COLOR_WHITE, (x + 2, y), (x + w - 3, y), 2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, p['lifespan'] / 30)
            color = (*p['color'], int(255 * alpha))
            pos = (int(p['pos'].x), int(p['pos'].y - self.camera_y))
            size = int(p['size'] * alpha)
            if size > 0:
                try:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
                except OverflowError: # Can happen if particles fly way off screen
                    pass


        # Draw player
        if not self.game_over:
            player_color = self.COLORS[self.player_color_idx]
            player_screen_pos = (int(self.player_pos.x), int(self.player_pos.y - self.camera_y))
            player_rect = pygame.Rect(player_screen_pos[0] - self.PLAYER_SIZE / 2, player_screen_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
            
            glow_size = int(self.PLAYER_SIZE * 1.5)
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size // 2, glow_size // 2, glow_size // 2, (*player_color, 60))
            self.screen.blit(glow_surf, (player_rect.centerx - glow_size // 2, player_rect.centery - glow_size // 2))

            pygame.draw.rect(self.screen, player_color, player_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_WHITE, player_rect, width=2, border_radius=3)

    def _render_ui(self):
        height_text = f"Height: {max(0, self.max_height_reached):.1f}m / {self.WIN_HEIGHT}m"
        text_surface = self.font.render(height_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (10, 10))

        color_text = "Color:"
        text_surface = self.font.render(color_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (10, 40))
        pygame.draw.rect(self.screen, self.COLORS[self.player_color_idx], (100, 42, 25, 20), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (100, 42, 25, 20), width=2, border_radius=3)

        if self.game_over:
            status_text = "GAME OVER"
            if self.max_height_reached >= self.WIN_HEIGHT:
                status_text = "YOU WIN!"
            
            text_surface = self.large_font.render(status_text, True, self.COLOR_WHITE)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            shadow_surface = self.large_font.render(status_text, True, (0, 0, 0))
            self.screen.blit(shadow_surface, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.max_height_reached}

    def close(self):
        pygame.quit()