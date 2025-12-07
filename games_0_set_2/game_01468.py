
# Generated: 2025-08-27T17:14:36.564521
# Source Brief: brief_01468.md
# Brief Index: 1468

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to place a rock. Shift to cycle through rock types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A serene zen garden simulator. Place rocks and rake the sand to achieve a high aesthetic score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GARDEN_RECT = pygame.Rect(50, 50, 540, 300)
        self.MAX_EPISODE_STEPS = 180
        self.PHASE_LENGTH = 60
        self.MAX_ROCKS_PER_PHASE = 10

        # Colors
        self.COLOR_BG = (215, 225, 210)
        self.COLOR_SAND_LIGHT = (235, 225, 200)
        self.COLOR_SAND_DARK = (210, 200, 175)
        self.COLOR_ROCK = (80, 80, 85)
        self.COLOR_BORDER = (140, 140, 140)
        self.COLOR_TEXT = (50, 50, 50)
        self.COLOR_CURSOR = (100, 150, 255)
        self.COLOR_SHADOW = (0, 0, 0, 50)

        # Fonts
        try:
            self.font_main = pygame.font.SysFont("georgia", 22)
            self.font_small = pygame.font.SysFont("georgia", 16)
        except pygame.error:
            self.font_main = pygame.font.SysFont("sans-serif", 22)
            self.font_small = pygame.font.SysFont("sans-serif", 16)

        # State variables (initialized in reset)
        self.total_score = 0
        self.episode_steps = 0
        self.current_phase = 0
        self.phase_steps = 0
        self.aesthetic_score = 0
        self.rocks = []
        self.cursor_pos = np.array([0.0, 0.0])
        self.selected_rock_type = 0
        self.target_score = 0
        self.sand_grid = None
        self.sand_surface = None
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_score = 0
        self.episode_steps = 0
        self.current_phase = 0
        self._start_new_phase()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _start_new_phase(self):
        self.current_phase += 1
        self.phase_steps = 0
        self.rocks = []
        self.cursor_pos = np.array(self.GARDEN_RECT.center, dtype=float)
        self.selected_rock_type = 0
        self.target_score = 80 + (self.current_phase - 1) * 5
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        self._initialize_sand()
        self.aesthetic_score = self._calculate_aesthetic_score()

    def _initialize_sand(self):
        w, h = self.GARDEN_RECT.size
        x = np.linspace(0, self.np_random.uniform(8, 12), w)
        y = np.linspace(0, self.np_random.uniform(4, 7), h)
        xx, yy = np.meshgrid(x, y)
        
        freq1 = self.np_random.uniform(0.8, 1.2)
        freq2 = self.np_random.uniform(1.8, 2.2)
        amp1 = self.np_random.uniform(0.4, 0.6)
        amp2 = self.np_random.uniform(0.2, 0.3)
        
        self.sand_grid = amp1 * np.sin(freq1 * xx + self.np_random.uniform(0, math.pi)) + \
                         amp2 * np.cos(freq2 * yy + self.np_random.uniform(0, math.pi))
        
        self.sand_grid = (self.sand_grid - self.sand_grid.min()) / (self.sand_grid.max() - self.sand_grid.min())
        self._update_sand_surface()

    def step(self, action):
        movement, space_binary, shift_binary = action
        space_held = space_binary == 1
        shift_held = shift_binary == 1

        self.episode_steps += 1
        self.phase_steps += 1
        reward = 0
        
        prev_aesthetic_score = self.aesthetic_score

        self._handle_movement(movement)
        
        rock_placed = self._handle_rock_placement(space_held)
        if rock_placed:
            reward += 1.0
            # sfx: rock_place.wav

        self._handle_rock_selection(shift_held)
        
        self.aesthetic_score = self._calculate_aesthetic_score()
        
        score_diff = self.aesthetic_score - prev_aesthetic_score
        reward += score_diff * 0.01 # Small continuous reward

        if self.phase_steps >= self.PHASE_LENGTH:
            self.total_score += self.aesthetic_score
            if self.aesthetic_score >= self.target_score:
                reward += 100
                # sfx: phase_win.wav
            else:
                reward -= 10
                # sfx: phase_lose.wav
            
            if self.current_phase < 3:
                self._start_new_phase()
        
        terminated = self.episode_steps >= self.MAX_EPISODE_STEPS
        
        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        old_pos = self.cursor_pos.copy()
        speed = 5
        if movement == 1: self.cursor_pos[1] -= speed
        elif movement == 2: self.cursor_pos[1] += speed
        elif movement == 3: self.cursor_pos[0] -= speed
        elif movement == 4: self.cursor_pos[0] += speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], self.GARDEN_RECT.left, self.GARDEN_RECT.right - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], self.GARDEN_RECT.top, self.GARDEN_RECT.bottom - 1)

        if movement != 0:
            self._rake_sand(old_pos, self.cursor_pos)
            self._update_sand_surface()
            # sfx: sand_rake.wav

    def _rake_sand(self, p1, p2):
        g_p1 = (int(p1[0] - self.GARDEN_RECT.left), int(p1[1] - self.GARDEN_RECT.top))
        g_p2 = (int(p2[0] - self.GARDEN_RECT.left), int(p2[1] - self.GARDEN_RECT.top))
        
        x1, y1 = g_p1; x2, y2 = g_p2
        dx, dy = abs(x2 - x1), -abs(y2 - y1)
        sx, sy = 1 if x1 < x2 else -1, 1 if y1 < y2 else -1
        err = dx + dy
        rake_radius = 15
        
        h, w = self.sand_grid.shape
        Y, X = np.ogrid[:h, :w]
        
        while True:
            dist_from_rake = np.sqrt((X - x1)**2 + (Y - y1)**2)
            mask = dist_from_rake <= rake_radius
            
            if np.any(mask):
                avg_val = np.mean(self.sand_grid[mask])
                self.sand_grid[mask] = self.sand_grid[mask] * 0.8 + avg_val * 0.2

            if x1 == x2 and y1 == y2: break
            e2 = 2 * err
            if e2 >= dy: err += dy; x1 += sx
            if e2 <= dx: err += dx; y1 += sy

    def _handle_rock_placement(self, space_held):
        if space_held and not self.space_pressed_last_frame and len(self.rocks) < self.MAX_ROCKS_PER_PHASE:
            can_place = True
            new_rock_props = self._get_rock_properties(self.selected_rock_type)
            new_rock_size = new_rock_props['size']
            for rock in self.rocks:
                dist = np.linalg.norm(self.cursor_pos - rock['pos'])
                if dist < (rock['size'] + new_rock_size) * 0.8:
                    can_place = False
                    break
            
            if can_place:
                self.rocks.append({'pos': self.cursor_pos.copy(), **new_rock_props})
                self._disturb_sand_at(self.cursor_pos, new_rock_size)
                self._update_sand_surface()
                return True
        return False
        
    def _disturb_sand_at(self, pos, size):
        gx, gy = int(pos[0] - self.GARDEN_RECT.left), int(pos[1] - self.GARDEN_RECT.top)
        h, w = self.sand_grid.shape
        Y, X = np.ogrid[:h, :w]
        dist_from_rock = np.sqrt((X - gx)**2 + (Y - gy)**2)
        
        ripple_mask = (dist_from_rock > 0) & (dist_from_rock < size * 1.5)
        if np.any(ripple_mask):
            ripple_effect = np.sin(dist_from_rock[ripple_mask] * 0.5) * 0.1
            self.sand_grid[ripple_mask] = np.clip(self.sand_grid[ripple_mask] + ripple_effect, 0, 1)

    def _handle_rock_selection(self, shift_held):
        if shift_held and not self.shift_pressed_last_frame:
            self.selected_rock_type = (self.selected_rock_type + 1) % 3
            # sfx: ui_tick.wav

    def _get_rock_properties(self, rock_type):
        if rock_type == 0: return {'type': 0, 'size': 25, 'shape': 'circle'}
        elif rock_type == 1: return {'type': 1, 'size': 20, 'shape': 'oval'}
        return {'type': 2, 'size': 15, 'shape': 'cluster'}

    def _calculate_aesthetic_score(self):
        if not self.rocks: return 0
        num_rocks = len(self.rocks)
        positions = np.array([r['pos'] for r in self.rocks])
        
        center_of_mass = np.mean(positions, axis=0)
        garden_center = np.array(self.GARDEN_RECT.center)
        dist_from_center = np.linalg.norm(center_of_mass - garden_center)
        max_dist = np.linalg.norm(np.array([self.GARDEN_RECT.width/2, self.GARDEN_RECT.height/2]))
        balance_score = 30 * max(0, 1 - dist_from_center / max_dist)

        spacing_score = 30
        min_dist = 50
        for i in range(num_rocks):
            for j in range(i + 1, num_rocks):
                spacing_score -= 10 * max(0, 1 - np.linalg.norm(positions[i] - positions[j]) / min_dist)
            if not self.GARDEN_RECT.inflate(-min_dist, -min_dist).collidepoint(positions[i]):
                spacing_score -= 5
        
        variety_score = 20 * (len({r['type'] for r in self.rocks}) / 3.0)

        composition_score = 0
        for rock in self.rocks:
            rx, ry = int(rock['pos'][0] - self.GARDEN_RECT.left), int(rock['pos'][1] - self.GARDEN_RECT.top)
            radius = int(rock['size'] * 1.5)
            h, w = self.sand_grid.shape
            sub_grid = self.sand_grid[max(0, ry-radius):min(h, ry+radius), max(0, rx-radius):min(w, rx+radius)]
            if sub_grid.size > 0:
                composition_score += max(0, 2 - np.std(sub_grid) * 10)
        composition_score = min(20, composition_score)

        total_score = balance_score + spacing_score + variety_score + composition_score
        return max(0, min(100, int(total_score)))
        
    def _update_sand_surface(self):
        c_light = np.array(self.COLOR_SAND_LIGHT)
        c_dark = np.array(self.COLOR_SAND_DARK)
        sand_color_values = self.sand_grid[..., np.newaxis] * c_light + (1 - self.sand_grid[..., np.newaxis]) * c_dark
        self.sand_surface = pygame.surfarray.make_surface(sand_color_values.astype(np.uint8))
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, self.GARDEN_RECT.inflate(6,6), 0, 5)
        pygame.draw.rect(self.screen, tuple(x*0.9 for x in self.COLOR_BG), self.GARDEN_RECT, 0, 3)

        self.screen.blit(self.sand_surface, self.GARDEN_RECT.topleft)
        
        for rock in self.rocks: self._render_rock(rock)
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_rock(self, rock):
        pos = rock['pos'].astype(int)
        size = rock['size']
        shadow_pos = (pos[0] + 3, pos[1] + 3)
        
        if rock['shape'] == 'circle':
            pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], size, self.COLOR_SHADOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ROCK)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_ROCK)
        elif rock['shape'] == 'oval':
            rect = pygame.Rect(0, 0, int(size * 1.5), int(size * 1.2))
            rect.center = shadow_pos
            pygame.draw.ellipse(self.screen, self.COLOR_SHADOW, rect)
            rect.center = pos
            pygame.draw.ellipse(self.screen, self.COLOR_ROCK, rect)
        elif rock['shape'] == 'cluster':
            offsets = [(-size*0.6, -size*0.4), (size*0.5, -size*0.3), (0, size*0.6)]
            for offset in offsets:
                s_pos = (int(shadow_pos[0] + offset[0]), int(shadow_pos[1] + offset[1]))
                r_pos = (int(pos[0] + offset[0]), int(pos[1] + offset[1]))
                s = int(size * 0.7)
                pygame.gfxdraw.filled_circle(self.screen, s_pos[0], s_pos[1], s, self.COLOR_SHADOW)
                pygame.gfxdraw.filled_circle(self.screen, r_pos[0], r_pos[1], s, self.COLOR_ROCK)
                pygame.gfxdraw.aacircle(self.screen, r_pos[0], r_pos[1], s, self.COLOR_ROCK)

    def _render_cursor(self):
        pos = self.cursor_pos.astype(int)
        ghost_props = self._get_rock_properties(self.selected_rock_type)
        ghost_rock = {'pos': self.cursor_pos, **ghost_props}
        
        temp_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self._render_rock_ghost(temp_surface, ghost_rock)
        self.screen.blit(temp_surface, (0, 0))
        
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 30, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 31, self.COLOR_CURSOR)

    def _render_rock_ghost(self, surface, rock):
        pos = rock['pos'].astype(int)
        size = rock['size']
        color = (*self.COLOR_ROCK, 100)
        
        if rock['shape'] == 'circle':
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], size, color)
        elif rock['shape'] == 'oval':
            rect = pygame.Rect(0, 0, int(size * 1.5), int(size * 1.2))
            rect.center = pos
            pygame.draw.ellipse(surface, color, rect)
        elif rock['shape'] == 'cluster':
            offsets = [(-size*0.6, -size*0.4), (size*0.5, -size*0.3), (0, size*0.6)]
            for offset in offsets:
                r_pos = (int(pos[0] + offset[0]), int(pos[1] + offset[1]))
                s = int(size * 0.7)
                pygame.gfxdraw.filled_circle(surface, r_pos[0], r_pos[1], s, color)

    def _render_ui(self):
        score_text = self.font_main.render(f"Total Score: {self.total_score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        phase_text = self.font_small.render(f"Day: {self.current_phase}/3", True, self.COLOR_TEXT)
        self.screen.blit(phase_text, (self.WIDTH - 100, 10))
        
        aesthetic_text = self.font_main.render(f"Aesthetic: {self.aesthetic_score} / {self.target_score}", True, self.COLOR_TEXT)
        self.screen.blit(aesthetic_text, (self.GARDEN_RECT.left, self.GARDEN_RECT.bottom + 10))

        steps_left = self.PHASE_LENGTH - self.phase_steps
        steps_text = self.font_small.render(f"Actions Left: {steps_left}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.GARDEN_RECT.right - 120, self.GARDEN_RECT.bottom + 15))

    def _get_info(self):
        return {
            "total_score": self.total_score,
            "aesthetic_score": self.aesthetic_score,
            "phase": self.current_phase,
            "steps": self.episode_steps,
            "rocks_placed": len(self.rocks),
        }
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test assertions
        assert 0 <= self._calculate_aesthetic_score() <= 100
        assert len(self.rocks) <= self.MAX_ROCKS_PER_PHASE
        assert self.episode_steps > 0
        assert self.current_phase >= 1

        print("✓ Implementation validated successfully")