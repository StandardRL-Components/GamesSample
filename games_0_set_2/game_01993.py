
# Generated: 2025-08-28T03:20:32.124514
# Source Brief: brief_01993.md
# Brief Index: 1993

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "move to an adjacent gem, and press Space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match gems in an isometric grid to score points. Create matches of 3 or more "
        "to clear them from the board. Reach 500 points before you run out of 50 moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # Game constants
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 5
    MOVES_LIMIT = 50
    SCORE_TARGET = 500
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (20, 30, 40)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_WHITE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 230)
    CURSOR_COLOR_IDLE = (255, 255, 255, 100)
    CURSOR_COLOR_SELECTED = (255, 255, 0, 150)

    # Isometric projection
    TILE_WIDTH_HALF = 32
    TILE_HEIGHT_HALF = 16
    ORIGIN_X = 640 // 2
    ORIGIN_Y = 400 // 2 - (GRID_HEIGHT * TILE_HEIGHT_HALF) // 2 + 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.np_random = None

        # State machine and animation
        self.game_state = "AWAIT_INPUT"
        self.animation_timer = 0
        self.swap_info = {}
        self.matched_gems = set()
        self.falling_gems = []
        self.particles = []

        # Game variables
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.cursor_pos = (0, 0)
        self.selected_gem_pos = None
        self.score = 0
        self.moves_remaining = 0
        self.steps = 0
        self.game_over = False
        
        self.prev_space_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback if seed is None
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MOVES_LIMIT
        self.game_over = False
        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.selected_gem_pos = None
        self.prev_space_held = False
        self.game_state = "AWAIT_INPUT"
        self.animation_timer = 0
        self.particles = []

        self._initialize_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        if self.game_state == "AWAIT_INPUT":
            reward += self._handle_action(action)
        
        frame_reward = self._update_game_state()
        reward += frame_reward

        self.steps += 1
        
        win_condition = self.score >= self.SCORE_TARGET
        loss_condition = self.moves_remaining <= 0
        timeout_condition = self.steps >= self.MAX_STEPS
        
        terminated = win_condition or loss_condition or timeout_condition

        if terminated and not self.game_over:
            if win_condition:
                reward += 100
            elif loss_condition:
                reward -= 100
            elif timeout_condition:
                reward -= 50 # Smaller penalty for timeout
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, _ = action
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        reward = 0

        if movement != 0:
            r, c = self.cursor_pos
            if movement == 1: r -= 1
            elif movement == 2: r += 1
            elif movement == 3: c -= 1
            elif movement == 4: c += 1
            self.cursor_pos = (max(0, min(r, self.GRID_HEIGHT - 1)), max(0, min(c, self.GRID_WIDTH - 1)))
        
        if space_pressed:
            # sound: 'select.wav'
            if self.selected_gem_pos is None:
                self.selected_gem_pos = self.cursor_pos
            else:
                r1, c1 = self.selected_gem_pos
                r2, c2 = self.cursor_pos
                
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    self.moves_remaining -= 1
                    self.game_state = "SWAP_ANIM"
                    self.animation_timer = 10
                    self.swap_info = {
                        "pos1": (r1, c1), "pos2": (r2, c2),
                        "gem1_type": self.grid[r1, c1], "gem2_type": self.grid[r2, c2],
                        "is_revert": False
                    }
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                else:
                    # sound: 'error.wav'
                    self.selected_gem_pos = self.cursor_pos # Select new gem instead
                    reward = -0.01
        
        return reward

    def _update_game_state(self):
        reward = 0
        
        if self.animation_timer > 0:
            self.animation_timer -= 1
        
        if self.game_state == "SWAP_ANIM" and self.animation_timer == 0:
            matches = self._find_matches_on_grid(self.grid)
            if matches:
                # sound: 'match.wav'
                self.matched_gems = matches
                self.game_state = "MATCH_ANIM"
                self.animation_timer = 15
                self.selected_gem_pos = None
                if len(matches) == 3: reward += 1
                elif len(matches) == 4: reward += 2
                else: reward += 3
                for r, c in matches: self._create_particles(r, c, self.grid[r, c])
            else:
                if not self.swap_info.get("is_revert", False):
                    # sound: 'no_match.wav'
                    r1, c1 = self.swap_info["pos1"]
                    r2, c2 = self.swap_info["pos2"]
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    self.game_state = "SWAP_ANIM"
                    self.animation_timer = 10
                    self.swap_info["is_revert"] = True
                    reward = -0.1
                else:
                    self.game_state = "AWAIT_INPUT"
                    self.selected_gem_pos = None
        
        elif self.game_state == "MATCH_ANIM" and self.animation_timer == 0:
            for r, c in self.matched_gems:
                self.score += 10
                self.grid[r, c] = -1
            self.matched_gems = set()
            self._prepare_falling_gems()
            if self.falling_gems:
                self.game_state = "DROP_ANIM"
                self.animation_timer = 10
                # sound: 'fall.wav'
            else: # No gems fell, just go to refill
                self._apply_fall_and_refill()
                self._check_for_cascades_or_end_turn()

        elif self.game_state == "DROP_ANIM" and self.animation_timer == 0:
            self._apply_fall_and_refill()
            reward += self._check_for_cascades_or_end_turn()

        self._update_particles()
        return reward
    
    def _check_for_cascades_or_end_turn(self):
        matches = self._find_matches_on_grid(self.grid)
        if matches:
            # sound: 'match_cascade.wav'
            self.matched_gems = matches
            self.game_state = "MATCH_ANIM"
            self.animation_timer = 15
            reward = 5 # Cascade bonus
            if len(matches) == 3: reward += 1
            elif len(matches) == 4: reward += 2
            else: reward += 3
            for r, c in matches: self._create_particles(r, c, self.grid[r, c])
            return reward
        else:
            if not self._find_possible_moves():
                # sound: 'reshuffle.wav'
                self._initialize_board()
            self.game_state = "AWAIT_INPUT"
            return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": self.cursor_pos,
            "game_state": self.game_state,
        }

    # --- Board Logic ---
    def _initialize_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_and_remove_initial_matches():
                pass
            if self._find_possible_moves():
                break

    def _find_matches_on_grid(self, grid):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_and_remove_initial_matches(self):
        matches = self._find_matches_on_grid(self.grid)
        if not matches: return False
        for r, c in matches: self.grid[r, c] = -1
        self._prepare_falling_gems()
        self._apply_fall_and_refill()
        return True

    def _find_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                temp_grid = self.grid.copy()
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_matches_on_grid(temp_grid): return True
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_matches_on_grid(temp_grid): return True
        return False

    def _prepare_falling_gems(self):
        self.falling_gems = []
        for c in range(self.GRID_WIDTH):
            fall_dist = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == -1:
                    fall_dist += 1
                elif fall_dist > 0:
                    gem_type = self.grid[r, c]
                    self.falling_gems.append({ "from_pos": (r, c), "to_pos": (r + fall_dist, c), "gem_type": gem_type })
                    self.grid[r + fall_dist, c] = gem_type
                    self.grid[r, c] = -1

    def _apply_fall_and_refill(self):
        self.falling_gems = []
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    # --- Rendering ---
    def _grid_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH_HALF
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT_HALF
        return int(x), int(y)

    def _render_game(self):
        rendered_gems = set()
        anim_progress = 1.0 - (self.animation_timer / 10.0) if self.animation_timer > 0 else 1.0

        if self.game_state == "SWAP_ANIM":
            p1, p2 = self.swap_info["pos1"], self.swap_info["pos2"]
            x1, y1 = self._grid_to_screen(*p1); x2, y2 = self._grid_to_screen(*p2)
            ix1 = x1 + (x2 - x1) * anim_progress; iy1 = y1 + (y2 - y1) * anim_progress
            ix2 = x2 + (x1 - x2) * anim_progress; iy2 = y2 + (y1 - y2) * anim_progress
            self._draw_gem(ix1, iy1, self.swap_info["gem1_type"]); self._draw_gem(ix2, iy2, self.swap_info["gem2_type"])
            rendered_gems.add(p1); rendered_gems.add(p2)

        if self.game_state == "DROP_ANIM":
            for gem in self.falling_gems:
                x1, y1 = self._grid_to_screen(*gem["from_pos"]); x2, y2 = self._grid_to_screen(*gem["to_pos"])
                iy = y1 + (y2 - y1) * anim_progress
                self._draw_gem(x2, iy, gem["gem_type"])
                rendered_gems.add(gem["to_pos"])

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) in rendered_gems or self.grid[r, c] == -1: continue
                x, y = self._grid_to_screen(r, c)
                if self.game_state == "MATCH_ANIM" and (r, c) in self.matched_gems:
                    match_progress = self.animation_timer / 15.0
                    self._draw_gem(x, y, self.grid[r,c], match_progress)
                    if self.animation_timer > 7:
                        self._draw_gem(x, y, -1, 1.1, (1.0 - (self.animation_timer - 7) / 8.0) * 255)
                else: self._draw_gem(x, y, self.grid[r,c])
        
        if self.game_state == "AWAIT_INPUT":
            r, c = self.cursor_pos
            x, y = self._grid_to_screen(r, c)
            color = self.CURSOR_COLOR_SELECTED if self.selected_gem_pos == self.cursor_pos else self.CURSOR_COLOR_IDLE
            self._draw_cursor(x, y, color)
        if self.selected_gem_pos:
            r, c = self.selected_gem_pos
            x, y = self._grid_to_screen(r, c)
            self._draw_cursor(x, y, self.CURSOR_COLOR_SELECTED)

        for p in self.particles: pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.screen.get_width() - moves_text.get_width() - 20, 10))
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
            msg = "You Win!" if self.score >= self.SCORE_TARGET else "Game Over"
            end_text = self.font_large.render(msg, True, self.COLOR_WHITE)
            self.screen.blit(end_text, end_text.get_rect(center=(320, 200)))

    def _draw_gem(self, x, y, gem_type, size_mult=1.0, alpha=255):
        color = self.COLOR_WHITE if gem_type == -1 else self.GEM_COLORS[gem_type]
        w, h = self.TILE_WIDTH_HALF * size_mult, self.TILE_HEIGHT_HALF * size_mult
        points = [(x, y - h), (x + w, y), (x, y + h), (x - w, y)]
        if alpha < 255:
            temp_surf = pygame.Surface((w*2, h*2), pygame.SRCALPHA)
            points_local = [(p[0] - (x-w), p[1] - (y-h)) for p in points]
            pygame.gfxdraw.filled_polygon(temp_surf, points_local, (*color, int(alpha)))
            pygame.gfxdraw.aapolygon(temp_surf, points_local, (*color, int(alpha)))
            self.screen.blit(temp_surf, (x - w, y - h))
        else:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_cursor(self, x, y, color):
        w, h = self.TILE_WIDTH_HALF * 1.1, self.TILE_HEIGHT_HALF * 1.1
        points = [(x, y - h), (x + w, y), (x, y + h), (x - w, y)]
        temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(temp_surf, points, color)
        pygame.gfxdraw.aapolygon(temp_surf, points, color)
        self.screen.blit(temp_surf, (0,0))
    
    def _create_particles(self, r, c, gem_type):
        x, y = self._grid_to_screen(r, c)
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed - 1, 'size': self.np_random.uniform(2, 5), 'life': 20, 'color': self.GEM_COLORS[gem_type]})

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']; p['y'] += p['vy']; p['vy'] += 0.1; p['life'] -= 1; p['size'] *= 0.97
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0.5]

    def validate_implementation(self):
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