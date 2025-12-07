
# Generated: 2025-08-27T20:05:34.532525
# Source Brief: brief_02342.md
# Brief Index: 2342

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for particle effects
class Particle:
    """A simple particle for visual effects like sparks."""
    def __init__(self, pos, vel, life, color, size_range):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        """Update particle position and lifetime."""
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1

    def draw(self, surface):
        """Draw the particle with alpha blending for a glow effect."""
        if self.life > 0:
            # Fade out over life
            alpha = int(255 * (self.life / self.max_life))
            color_with_alpha = self.color + (alpha,)
            # Use a temporary surface for additive blending
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (self.size, self.size), self.size)
            surface.blit(temp_surf, (self.pos[0] - self.size, self.pos[1] - self.size), special_flags=pygame.BLEND_RGBA_ADD)

# Helper class for pulse animations
class PulseAnimation:
    """An animation for a radial pulse effect."""
    def __init__(self, pos, life, start_radius, end_radius, color):
        self.pos = pos
        self.life = life
        self.max_life = life
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.color = color

    def update(self):
        """Update animation lifetime."""
        self.life -= 1

    def draw(self, surface):
        """Draw the expanding, fading circle."""
        if self.life > 0:
            progress = (self.max_life - self.life) / self.max_life
            current_radius = self.start_radius + (self.end_radius - self.start_radius) * progress
            # Fade out as it expands
            alpha = int(255 * (1 - progress))
            
            # Draw anti-aliased circles for a smooth look
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(current_radius), self.color + (alpha,))
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(current_radius)-1, self.color + (alpha,))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to place a crystal. Press shift to restart."
    )

    game_description = (
        "An isometric puzzle game. Place crystals to trigger chain reactions and fill the entire cavern grid."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 5, 4
        self.NUM_SPACES = self.GRID_COLS * self.GRID_ROWS
        self.INITIAL_CRYSTALS = 12
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_EMPTY = (40, 50, 70)
        self.COLOR_FILLED = (20, 100, 120)
        self.COLOR_FILLED_GLOW = (40, 180, 200)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 80, 80)
        self.COLOR_GRID_LINES = (30, 40, 60)
        self.COLOR_PULSE = (100, 220, 255)
        self.COLOR_PARTICLE = (150, 230, 255)
        self.COLOR_UI_TEXT = (200, 220, 255)

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
        self.font_ui = pygame.font.Font(None, 28)

        # --- Grid Layout ---
        self.tile_width = 64
        self.tile_height = 32
        self.grid_origin_x = self.WIDTH // 2
        self.grid_origin_y = 140
        self.grid_screen_coords = [self._iso_to_screen(c, r) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]
        
        # --- Game State (initialized in reset) ---
        self.grid_state = None
        self.crystals_remaining = None
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = []
        self.animations = []
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def _iso_to_screen(self, c, r):
        """Converts grid coordinates (col, row) to screen coordinates (x, y)."""
        x = self.grid_origin_x + (c - r) * self.tile_width / 2
        y = self.grid_origin_y + (c + r) * self.tile_height / 2
        return x, y

    def _get_tile_points(self, x, y):
        """Calculates the four corner points of an isometric tile."""
        return [
            (x, y - self.tile_height / 2),
            (x + self.tile_width / 2, y),
            (x, y + self.tile_height / 2),
            (x - self.tile_width / 2, y),
        ]

    def _get_neighbors(self, index):
        """Gets valid grid neighbors for a given linear index."""
        neighbors = []
        c, r = index % self.GRID_COLS, index // self.GRID_COLS
        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < self.GRID_COLS and 0 <= nr < self.GRID_ROWS:
                neighbors.append(nr * self.GRID_COLS + nc)
        return neighbors

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.grid_state = np.zeros(self.NUM_SPACES, dtype=int)
        self.crystals_remaining = self.INITIAL_CRYSTALS
        self.cursor_pos = self.GRID_COLS * (self.GRID_ROWS // 2) + (self.GRID_COLS // 2)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.animations = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        if shift_held:
            self.game_over = True
            reward = -100.0 # Penalty for giving up
        else:
            self._handle_movement(movement)
            if space_held:
                reward = self._handle_place_crystal()

        self._check_termination()
        if self.game_over:
            if all(s == 1 for s in self.grid_state): # Win condition
                reward += 100.0
            elif not shift_held: # Loss condition (out of crystals)
                reward -= 100.0
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 0: return # No-op
        
        c, r = self.cursor_pos % self.GRID_COLS, self.cursor_pos // self.GRID_COLS
        if movement == 1: r = (r - 1 + self.GRID_ROWS) % self.GRID_ROWS # Up
        elif movement == 2: r = (r + 1) % self.GRID_ROWS # Down
        elif movement == 3: c = (c - 1 + self.GRID_COLS) % self.GRID_COLS # Left
        elif movement == 4: c = (c + 1) % self.GRID_COLS # Right
        self.cursor_pos = r * self.GRID_COLS + c

    def _handle_place_crystal(self):
        if self.crystals_remaining > 0 and self.grid_state[self.cursor_pos] == 0:
            # // SFX: Crystal_Place_Success
            self.crystals_remaining -= 1
            filled_count = self._trigger_chain_reaction(self.cursor_pos)
            self.score += filled_count
            return float(filled_count)
        else:
            # // SFX: Action_Fail
            self._add_invalid_action_effect()
            return -0.1

    def _trigger_chain_reaction(self, start_index):
        if self.grid_state[start_index] != 0: return 0

        filled_count = 0
        q = collections.deque([start_index])
        visited = {start_index}

        while q:
            current_index = q.popleft()
            self.grid_state[current_index] = 1 # Mark as filled
            filled_count += 1
            self._add_fill_animation(current_index)
            # // SFX: Chain_Reaction_Tick

            for neighbor_index in self._get_neighbors(current_index):
                if self.grid_state[neighbor_index] == 0 and neighbor_index not in visited:
                    visited.add(neighbor_index)
                    q.append(neighbor_index)
        
        for _ in range(filled_count * 10):
            self._add_spark_particle(self.grid_screen_coords[start_index])
        return filled_count

    def _check_termination(self):
        is_win = all(s == 1 for s in self.grid_state)
        is_loss = self.crystals_remaining == 0 and any(s == 0 for s in self.grid_state)
        if is_win or is_loss:
            self.game_over = True
            if is_win:
                # // SFX: Game_Win
                for i in range(self.NUM_SPACES): self._add_fill_animation(i)
            # else: // SFX: Game_Loss

    def _get_observation(self):
        self._update_visuals()
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid_lines()
        self._render_grid_cells()
        self._render_cursor()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _update_visuals(self):
        self.animations = [a for a in self.animations if a.life > 0]
        self.particles = [p for p in self.particles if p.life > 0]
        for a in self.animations: a.update()
        for p in self.particles: p.update()
        
    def _render_grid_lines(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                x, y = self._iso_to_screen(c, r)
                if c < self.GRID_COLS - 1:
                    nx, ny = self._iso_to_screen(c + 1, r)
                    pygame.draw.aaline(self.screen, self.COLOR_GRID_LINES, (x, y), (nx, ny))
                if r < self.GRID_ROWS - 1:
                    nx, ny = self._iso_to_screen(c, r + 1)
                    pygame.draw.aaline(self.screen, self.COLOR_GRID_LINES, (x, y), (nx, ny))

    def _render_grid_cells(self):
        for i in range(self.NUM_SPACES):
            x, y = self.grid_screen_coords[i]
            points = self._get_tile_points(x, y)
            
            color = self.COLOR_EMPTY if self.grid_state[i] == 0 else self.COLOR_FILLED
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

            if self.grid_state[i] == 1:
                glow_points = self._get_tile_points(x, y - 2)
                pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_FILLED_GLOW + (50,))

    def _render_cursor(self):
        x, y = self.grid_screen_coords[self.cursor_pos]
        points = self._get_tile_points(x, y)
        
        is_valid_move = self.crystals_remaining > 0 and self.grid_state[self.cursor_pos] == 0
        color = self.COLOR_CURSOR if is_valid_move else self.COLOR_CURSOR_INVALID

        for i in range(4):
            start = points[i]
            end = points[(i + 1) % 4]
            pygame.draw.line(self.screen, color, start, end, 2)

    def _render_effects(self):
        effect_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for p in self.particles: p.draw(effect_surface)
        self.screen.blit(effect_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        for a in self.animations: a.draw(self.screen)

    def _render_ui(self):
        crystal_text = self.font_ui.render(f"CRYSTALS: {self.crystals_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (20, 20))
        
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))

        if self.game_over:
            is_win = all(s == 1 for s in self.grid_state)
            end_text_str = "GRID COMPLETE" if is_win else "GAME OVER"
            end_text = self.font_ui.render(end_text_str, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT-40))
            self.screen.blit(end_text, text_rect)
            
    def _add_fill_animation(self, index):
        pos = self.grid_screen_coords[index]
        anim = PulseAnimation(pos, 30, self.tile_width * 0.1, self.tile_width * 0.6, self.COLOR_PULSE)
        self.animations.append(anim)

    def _add_spark_particle(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        life = self.np_random.integers(20, 41)
        p = Particle(pos, vel, life, self.COLOR_PARTICLE, size_range=(1, 3))
        self.particles.append(p)

    def _add_invalid_action_effect(self):
        pos = self.grid_screen_coords[self.cursor_pos]
        anim = PulseAnimation(pos, 15, self.tile_width * 0.4, self.tile_width * 0.6, self.COLOR_CURSOR_INVALID)
        self.animations.append(anim)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_remaining": self.crystals_remaining,
            "spaces_filled": int(np.sum(self.grid_state)),
        }

    def validate_implementation(self):
        """CRITICAL: Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")