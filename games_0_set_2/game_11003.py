import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:28:17.201662
# Source Brief: brief_01003.md
# Brief Index: 1003
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate through fractal dimensions, collecting energy shards to craft and place anchors. "
        "Stabilize all dimensions to win before your energy is depleted."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to craft/place an anchor. "
        "Press shift to switch between dimensions."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    NUM_DIMENSIONS = 5
    PLAYER_RADIUS = 10
    PLAYER_SPEED = 6
    MAX_ENERGY = 100.0
    ENERGY_COST_MOVE = 0.02
    ENERGY_COST_SHIFT = 15.0
    SHARD_RADIUS = 4
    ANCHOR_COST = 20
    SHARDS_PER_DIMENSION = 25
    SHARD_RESPAWN_STEPS = 300
    
    # --- Colors ---
    COLOR_BG = (10, 5, 15)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_SHARD = (0, 200, 255)
    COLOR_SHARD_GLOW = (0, 200, 255, 60)
    COLOR_ANCHOR = (255, 200, 0)
    COLOR_ANCHOR_GLOW = (255, 200, 0, 80)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)
    COLOR_ENERGY_HIGH = (0, 255, 128)
    COLOR_ENERGY_LOW = (255, 50, 50)
    
    # Dimension-specific color palettes for fractals
    DIM_PALETTES = [
        [(66, 30, 15), (25, 7, 26), (9, 1, 47), (4, 4, 73), (0, 7, 100), (12, 44, 138), (24, 82, 177), (57, 125, 209), (134, 181, 229), (211, 236, 248), (241, 233, 191), (248, 201, 95), (255, 170, 0), (204, 128, 0), (153, 87, 0), (106, 52, 3)],
        [(20, 0, 0), (40, 10, 10), (80, 20, 20), (160, 40, 40), (255, 80, 80), (255, 120, 120), (255, 180, 180), (255, 220, 220), (255, 255, 255)],
        [(0, 20, 0), (10, 40, 10), (20, 80, 20), (40, 160, 40), (80, 255, 80), (120, 255, 120), (180, 255, 180), (220, 255, 220), (255, 255, 255)],
        [(0, 0, 20), (10, 10, 40), (20, 20, 80), (40, 40, 160), (80, 80, 255), (120, 120, 255), (180, 180, 255), (220, 220, 255), (255, 255, 255)],
        [(20, 0, 20), (40, 10, 40), (80, 20, 80), (160, 40, 160), (255, 80, 255), (255, 120, 255), (255, 180, 255), (255, 220, 255), (255, 255, 255)]
    ]

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        self.fractal_backgrounds = [
            self._generate_fractal_surface(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, p, 30 + i * 5)
            for i, p in enumerate(self.DIM_PALETTES)
        ]

        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # validation is for dev, not production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.energy = self.MAX_ENERGY
        self.shards = 0
        self.current_dimension = 0
        self.stabilized_dimensions = [False] * self.NUM_DIMENSIONS
        self.has_anchor = False
        
        self.dimension_anchor_pos = [None] * self.NUM_DIMENSIONS
        self.dimension_resources = []
        self.dimension_shard_respawn_timers = [0] * self.NUM_DIMENSIONS
        
        for i in range(self.NUM_DIMENSIONS):
            num_shards = int(self.SHARDS_PER_DIMENSION * (1 - 0.05 * i))
            self.dimension_resources.append(self._spawn_shards(num_shards))

        self.last_space_held = False
        self.last_shift_held = False

        self.dim_shift_fade = 0 # for visual effect

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Handle Actions ---
        # Movement
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement != 0:
            if movement == 1: move_vector[1] -= 1 # Up
            elif movement == 2: move_vector[1] += 1 # Down
            elif movement == 3: move_vector[0] -= 1 # Left
            elif movement == 4: move_vector[0] += 1 # Right
            
            self.player_pos += move_vector * self.PLAYER_SPEED
            self.energy -= self.ENERGY_COST_MOVE
            reward -= 0.01 # Penalty for inefficient movement
            self._clamp_player_position()
        
        # Craft/Place Anchor (Space)
        if space_press:
            if self.has_anchor and not self.stabilized_dimensions[self.current_dimension]:
                # Place anchor
                self.dimension_anchor_pos[self.current_dimension] = self.player_pos.copy()
                self.stabilized_dimensions[self.current_dimension] = True
                self.has_anchor = False
                reward += 10.0 # Stabilize dimension reward
                # SFX: Anchor placed sound
            elif not self.has_anchor and self.shards >= self.ANCHOR_COST:
                # Craft anchor
                self.shards -= self.ANCHOR_COST
                self.has_anchor = True
                reward += 5.0 # Craft anchor reward
                # SFX: Crafting success sound

        # Switch Dimension (Shift)
        if shift_press and self.energy >= self.ENERGY_COST_SHIFT:
            max_unlocked_dim = 0
            for i, stabilized in enumerate(self.stabilized_dimensions):
                if i < self.NUM_DIMENSIONS - 1 and stabilized:
                    max_unlocked_dim = i + 1
            
            if max_unlocked_dim > 0: # Ensure there's a dimension to switch to
                self.current_dimension = (self.current_dimension + 1) % (max_unlocked_dim + 1)
                self.energy -= self.ENERGY_COST_SHIFT
                self.dim_shift_fade = 255 # Start fade effect
                # SFX: Dimension shift whoosh
        
        # --- Update Game State ---
        self._update_shards()
        reward += self._check_shard_collection()
        self.dim_shift_fade = max(0, self.dim_shift_fade - 15)

        # --- Check Termination ---
        terminated = False
        if self.energy <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        if all(self.stabilized_dimensions):
            reward += 100.0
            terminated = True
            self.game_over = True
            self.victory = True

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
        # Draw background fractal
        self.screen.blit(self.fractal_backgrounds[self.current_dimension], (0, 0))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "shards": self.shards,
            "current_dimension": self.current_dimension,
            "stabilized_dimensions": sum(self.stabilized_dimensions),
            "has_anchor": self.has_anchor,
        }

    # --- Helper Methods ---

    def _clamp_player_position(self):
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _spawn_shards(self, num_shards):
        shards = []
        for _ in range(num_shards):
            shards.append(np.array([
                self.np_random.uniform(self.SHARD_RADIUS, self.SCREEN_WIDTH - self.SHARD_RADIUS),
                self.np_random.uniform(self.SHARD_RADIUS, self.SCREEN_HEIGHT - self.SHARD_RADIUS)
            ]))
        return shards

    def _update_shards(self):
        # Respawn logic
        self.dimension_shard_respawn_timers[self.current_dimension] += 1
        if self.dimension_shard_respawn_timers[self.current_dimension] >= self.SHARD_RESPAWN_STEPS:
            self.dimension_shard_respawn_timers[self.current_dimension] = 0
            if len(self.dimension_resources[self.current_dimension]) < self.SHARDS_PER_DIMENSION // 2:
                new_shards = self._spawn_shards(self.SHARDS_PER_DIMENSION // 4)
                self.dimension_resources[self.current_dimension].extend(new_shards)

    def _check_shard_collection(self):
        collected_reward = 0
        shards_in_dim = self.dimension_resources[self.current_dimension]
        for i in range(len(shards_in_dim) - 1, -1, -1):
            shard_pos = shards_in_dim[i]
            dist = np.linalg.norm(self.player_pos - shard_pos)
            if dist < self.PLAYER_RADIUS + self.SHARD_RADIUS:
                shards_in_dim.pop(i)
                self.shards += 1
                collected_reward += 0.1
                # SFX: Shard collection ping
        return collected_reward

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw placed anchor
        if self.dimension_anchor_pos[self.current_dimension] is not None:
            self._draw_tetrahedron(self.dimension_anchor_pos[self.current_dimension], 20, self.COLOR_ANCHOR, self.COLOR_ANCHOR_GLOW)

        # Draw shards
        pulse = math.sin(self.steps * 0.1) * 2
        for shard_pos in self.dimension_resources[self.current_dimension]:
            x, y = int(shard_pos[0]), int(shard_pos[1])
            glow_rad = int(self.SHARD_RADIUS * 2.5 + pulse)
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_rad, self.COLOR_SHARD_GLOW)
            pygame.gfxdraw.aacircle(self.screen, x, y, glow_rad, self.COLOR_SHARD_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.SHARD_RADIUS, self.COLOR_SHARD)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.SHARD_RADIUS, self.COLOR_SHARD)

        # Draw player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        glow_rad = int(self.PLAYER_RADIUS * 2.0 + math.sin(self.steps * 0.2) * 3)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_rad, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, glow_rad, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Energy Bar
        energy_ratio = np.clip(self.energy / self.MAX_ENERGY, 0, 1)
        bar_width = 200
        bar_height = 20
        energy_color = (
            int(self.COLOR_ENERGY_LOW[0] + (self.COLOR_ENERGY_HIGH[0] - self.COLOR_ENERGY_LOW[0]) * energy_ratio),
            int(self.COLOR_ENERGY_LOW[1] + (self.COLOR_ENERGY_HIGH[1] - self.COLOR_ENERGY_LOW[1]) * energy_ratio),
            int(self.COLOR_ENERGY_LOW[2] + (self.COLOR_ENERGY_HIGH[2] - self.COLOR_ENERGY_LOW[2]) * energy_ratio)
        )
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, energy_color, (10, 10, int(bar_width * energy_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)
        
        # Resource Count
        shard_text = f"Shards: {self.shards}"
        if self.has_anchor:
            shard_text = "ANCHOR READY"
        self._render_text(shard_text, (15, 40), self.COLOR_UI_TEXT, 24)

        # Stabilized Dimensions
        self._render_text("Dimensions Stabilized:", (self.SCREEN_WIDTH - 220, 15), self.COLOR_UI_TEXT, 24)
        for i in range(self.NUM_DIMENSIONS):
            color = self.COLOR_ANCHOR if self.stabilized_dimensions[i] else (80, 80, 80)
            center = (self.SCREEN_WIDTH - 120 + i * 25, 45)
            pygame.draw.circle(self.screen, color, center, 8)
            pygame.draw.circle(self.screen, self.COLOR_UI_TEXT, center, 8, 1)

        # Dimension Shift Fade
        if self.dim_shift_fade > 0:
            fade_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            fade_surface.fill((0, 0, 0, self.dim_shift_fade))
            self.screen.blit(fade_surface, (0, 0))

        # Game Over / Victory Text
        if self.game_over:
            text = "VICTORY" if self.victory else "ENERGY DEPLETED"
            color = self.COLOR_ENERGY_HIGH if self.victory else self.COLOR_ENERGY_LOW
            self._render_text(text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), color, 64, center=True)

    def _draw_tetrahedron(self, pos, size, color, glow_color):
        angle = self.steps * 0.02
        points = []
        for i in range(3):
            theta = 2 * math.pi * i / 3 + angle
            points.append((pos[0] + size * math.cos(theta), pos[1] + size * math.sin(theta) * 0.7))
        top_point = (pos[0], pos[1] - size * 0.8)

        # Glow
        for i in range(3):
            pygame.draw.aaline(self.screen, glow_color, top_point, points[i], 3)
            pygame.draw.aaline(self.screen, glow_color, points[i], points[(i + 1) % 3], 3)
        # Lines
        for i in range(3):
            pygame.draw.aaline(self.screen, color, top_point, points[i], 1)
            pygame.draw.aaline(self.screen, color, points[i], points[(i + 1) % 3], 1)

    def _render_text(self, text, pos, color, size, shadow=True, center=False):
        font = self.font_large if size > 32 else self.font_small
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        if shadow:
            shadow_surface = font.render(text, True, self.COLOR_UI_SHADOW)
            self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        
        self.screen.blit(text_surface, text_rect)

    @staticmethod
    def _generate_fractal_surface(width, height, palette, max_iter=30):
        surface = pygame.Surface((width, height))
        pixels = pygame.surfarray.pixels3d(surface)
        
        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.0, 1.0

        for x in range(width):
            for y in range(height):
                zx, zy = x * (x_max - x_min) / (width - 1) + x_min, y * (y_max - y_min) / (height - 1) + y_min
                c = complex(zx, zy)
                z = 0j
                for i in range(max_iter):
                    if abs(z) > 2.0:
                        break
                    z = z * z + c
                
                color_index = int(i / max_iter * (len(palette) - 1))
                pixels[x, y] = palette[color_index]
        
        del pixels
        return surface

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium environment API.
    # It will not be executed by the test suite.
    
    # --- For human play, we need a real display. ---
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv(render_mode="rgb_array")
    
    # Setup for manual play
    pygame.display.set_caption("Fractal Dimensions")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    print(env.user_guide)
    
    # Game loop for human player
    while not terminated and not truncated:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()