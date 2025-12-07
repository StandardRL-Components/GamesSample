
# Generated: 2025-08-28T05:45:25.734551
# Source Brief: brief_05680.md
# Brief Index: 5680

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "then move to an adjacent tile and press Space again to swap."
    )

    game_description = (
        "A vibrant match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Create combos and chain reactions to maximize your score before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    NUM_COLORS = 6
    MOVES_LIMIT = 20
    SCORE_TARGET = 1000
    MAX_STEPS = 1000

    # Visuals
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 36
    TILE_SPACING = 4
    GRID_AREA_WIDTH = GRID_WIDTH * (TILE_SIZE + TILE_SPACING)
    GRID_AREA_HEIGHT = GRID_HEIGHT * (TILE_SIZE + TILE_SPACING)
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) + 10

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SCORE = (255, 220, 100)
    COLOR_MOVES = (100, 200, 255)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTION = (255, 255, 0)
    
    TILE_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 150, 50),   # Orange
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid = None
        self.cursor_pos = None
        self.first_selection = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.win_status = ""
        self.step_reward = 0
        
        self.animations = []
        self.particles = []

        self.reset()
        
        # self.validate_implementation() # Optional: Call to check compliance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MOVES_LIMIT
        self.game_over = False
        self.win_status = ""
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.first_selection = None
        self.animations.clear()
        self.particles.clear()
        
        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        # Ensure no initial matches
        while self._find_all_matches():
            matches = self._find_all_matches()
            for x, y in matches:
                # Avoid matching colors from neighbors
                forbidden_colors = set()
                if x > 0: forbidden_colors.add(self.grid[x-1, y])
                if y > 0: forbidden_colors.add(self.grid[x, y-1])
                
                possible_colors = [c for c in range(self.NUM_COLORS) if c not in forbidden_colors]
                if not possible_colors: possible_colors = list(range(self.NUM_COLORS))
                self.grid[x, y] = self.np_random.choice(possible_colors)

    def step(self, action):
        self.steps += 1
        self.step_reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Animations take precedence over player actions
        if self.animations:
            self._update_animations_and_particles()
        else:
            self._handle_input(action)

        terminated = self._check_termination()
        
        # Apply terminal rewards only once
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.SCORE_TARGET:
                self.step_reward += 100
                self.win_status = "YOU WIN!"
            else:
                self.step_reward -= 100
                self.win_status = "GAME OVER"
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over:
                 self.step_reward -= 100
                 self.win_status = "TIME UP!"
            self.game_over = True
            
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_press, _ = action
        space_pressed = (space_press == 1)

        # Move cursor
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

        if space_pressed:
            if self.first_selection is None:
                # First selection
                self.first_selection = list(self.cursor_pos)
                # SFX: select_gem.wav
            else:
                # Second selection - attempt swap
                if self._is_adjacent(self.first_selection, self.cursor_pos):
                    self._attempt_swap(self.first_selection, self.cursor_pos)
                else:
                    # Invalid (not adjacent) second selection, reset
                    self.first_selection = list(self.cursor_pos) # Treat as new first selection
                    # SFX: error.wav
    
    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        x1, y1 = pos1
        x2, y2 = pos2

        # Animate the swap
        self.animations.append(SwapAnimation(pos1, pos2))

        # Perform swap in grid data
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
        
        matches1 = self._find_matches_at(x1, y1)
        matches2 = self._find_matches_at(x2, y2)
        all_matches = set(matches1 + matches2)
        
        if not all_matches:
            # Invalid swap, animate back
            # SFX: invalid_swap.wav
            self.step_reward = -0.1
            self.animations.append(SwapAnimation(pos1, pos2, is_revert=True))
            # Swap back in grid data
            self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
        else:
            # Valid swap, start resolving board
            # SFX: match_found.wav
            self._resolve_matches(all_matches)

        self.first_selection = None

    def _resolve_matches(self, matches):
        if not matches:
            return

        # Score and rewards
        num_cleared = len(matches)
        self.score += num_cleared * 10
        self.step_reward += num_cleared
        if num_cleared == 4: self.step_reward += 5
        if num_cleared >= 5: self.step_reward += 10
        
        # Create clear animations and particles
        for x, y in matches:
            self.animations.append(ClearAnimation((x, y)))
            self._create_particles(x, y, self.grid[x,y])
        
        # Update grid data: set matched tiles to -1 (empty)
        for x, y in matches:
            self.grid[x, y] = -1

        # Drop and refill
        self._apply_gravity_and_refill()
        
        # Check for new chain-reaction matches
        new_matches = self._find_all_matches()
        if new_matches:
            # SFX: combo.wav
            self.animations.append(ChainReactionDelay(lambda: self._resolve_matches(new_matches)))

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            fall_map = {} # old_y -> new_y
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    fall_map[y] = y + empty_count
                    self.grid[x, y + empty_count] = self.grid[x, y]
                    self.grid[x, y] = -1
            
            # Animate falling tiles
            for old_y, new_y in fall_map.items():
                self.animations.append(FallAnimation((x, old_y), (x, new_y)))

            # Refill top with new tiles
            for y in range(empty_count):
                self.grid[x, y] = self.np_random.integers(0, self.NUM_COLORS)
                self.animations.append(FallAnimation((x, y - empty_count), (x, y)))


    def _find_matches_at(self, x, y):
        color = self.grid[x, y]
        if color == -1: return []

        # Horizontal
        h_matches = [(x, y)]
        # Left
        for i in range(x - 1, -1, -1):
            if self.grid[i, y] == color: h_matches.append((i, y))
            else: break
        # Right
        for i in range(x + 1, self.GRID_WIDTH):
            if self.grid[i, y] == color: h_matches.append((i, y))
            else: break

        # Vertical
        v_matches = [(x, y)]
        # Up
        for j in range(y - 1, -1, -1):
            if self.grid[x, j] == color: v_matches.append((x, j))
            else: break
        # Down
        for j in range(y + 1, self.GRID_HEIGHT):
            if self.grid[x, j] == color: v_matches.append((x, j))
            else: break
            
        final_matches = []
        if len(h_matches) >= 3: final_matches.extend(h_matches)
        if len(v_matches) >= 3: final_matches.extend(v_matches)
        
        return list(set(final_matches))

    def _find_all_matches(self):
        all_matches = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                matches = self._find_matches_at(x, y)
                if matches:
                    all_matches.update(matches)
        return list(all_matches)

    def _check_termination(self):
        return self.score >= self.SCORE_TARGET or self.moves_left <= 0

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "is_resolving": bool(self.animations)
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid_background()
        self._draw_tiles()
        self._draw_cursor_and_selection()
        self._update_animations_and_particles(draw_only=True)

    def _draw_grid_background(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = self._get_tile_rect(x, y)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, border_radius=4)
    
    def _draw_tiles(self):
        # Draw static tiles first
        animated_tiles = {anim.get_involved_tile() for anim in self.animations if hasattr(anim, 'get_involved_tile')}
        
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in animated_tiles:
                    color_idx = self.grid[x, y]
                    if color_idx != -1:
                        self._draw_tile(x, y, color_idx)

    def _draw_tile(self, x, y, color_idx, size_mod=0, alpha=255):
        rect = self._get_tile_rect(x, y)
        if size_mod != 0:
            rect = rect.inflate(size_mod, size_mod)
        
        color = self.TILE_COLORS[color_idx]
        
        # Use gfxdraw for antialiasing
        pygame.gfxdraw.box(self.screen, rect, (*color, alpha))
        
        # Add a subtle highlight for 3D effect
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.line(self.screen, (*highlight_color, alpha), rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, (*highlight_color, alpha), rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, (*shadow_color, alpha), rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, (*shadow_color, alpha), rect.topright, rect.bottomright, 2)

    def _draw_cursor_and_selection(self):
        # Draw first selection highlight
        if self.first_selection and not self.animations:
            x, y = self.first_selection
            rect = self._get_tile_rect(x, y).inflate(6, 6)
            # Pulsing effect
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 100 + 155
            pygame.draw.rect(self.screen, (*self.COLOR_SELECTION, int(pulse)), rect, 3, border_radius=6)

        # Draw cursor
        if not self.animations:
            cx, cy = self.cursor_pos
            rect = self._get_tile_rect(cx, cy).inflate(6, 6)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=6)
    
    def _update_animations_and_particles(self, draw_only=False):
        # Update and draw animations
        if not draw_only:
            for anim in self.animations:
                anim.update()
            self.animations = [anim for anim in self.animations if not anim.is_done()]

        for anim in self.animations:
            anim.draw(self)

        # Update and draw particles
        if not draw_only:
            for p in self.particles:
                p.update()
            self.particles = [p for p in self.particles if p.is_alive()]

        for p in self.particles:
            p.draw(self.screen)

    def _create_particles(self, grid_x, grid_y, color_idx):
        px, py = self._get_tile_rect(grid_x, grid_y).center
        color = self.TILE_COLORS[color_idx]
        for _ in range(15):
            self.particles.append(Particle(px, py, color, self.np_random))
            
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 15))

        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_MOVES)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(moves_text, moves_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_status, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)
            
            score_text = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_SCORE)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40))
            self.screen.blit(score_text, score_rect)

    def _get_tile_rect(self, grid_x, grid_y):
        px = self.GRID_OFFSET_X + grid_x * (self.TILE_SIZE + self.TILE_SPACING)
        py = self.GRID_OFFSET_Y + grid_y * (self.TILE_SIZE + self.TILE_SPACING)
        return pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
    
    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")


# --- Animation and Particle Classes ---

class Animation:
    def __init__(self, duration):
        self.duration = duration
        self.progress = 0
    
    def update(self):
        self.progress += 1
    
    def is_done(self):
        return self.progress >= self.duration

    def draw(self, env):
        raise NotImplementedError

class SwapAnimation(Animation):
    def __init__(self, pos1, pos2, is_revert=False):
        super().__init__(duration=8)
        self.pos1 = pos1
        self.pos2 = pos2
        self.is_revert = is_revert

    def draw(self, env):
        p = self.progress / self.duration
        if self.is_revert: p = 1.0 - p # Animate backwards for revert
        
        x1, y1 = self.pos1
        x2, y2 = self.pos2
        
        # Interpolate positions
        draw_x1 = x1 + (x2 - x1) * p
        draw_y1 = y1 + (y2 - y1) * p
        draw_x2 = x2 + (x1 - x2) * p
        draw_y2 = y2 + (y1 - y2) * p

        # Get colors from the grid state *at the time of drawing*
        # For a normal swap, the grid is already swapped. For revert, it's swapped back.
        # This logic ensures the correct tile moves along the path.
        if self.is_revert:
            color1_idx = env.grid[x1, y1]
            color2_idx = env.grid[x2, y2]
        else:
            color1_idx = env.grid[x2, y2]
            color2_idx = env.grid[x1, y1]

        if color1_idx != -1: env._draw_tile(draw_x1, draw_y1, color1_idx)
        if color2_idx != -1: env._draw_tile(draw_x2, draw_y2, color2_idx)

    def get_involved_tile(self): return None # Involves two, handled specially

class FallAnimation(Animation):
    def __init__(self, start_pos, end_pos):
        super().__init__(duration=10)
        self.start_pos = start_pos
        self.end_pos = end_pos

    def draw(self, env):
        p = self.progress / self.duration
        p = 1 - (1 - p) * (1 - p) # Ease-out quadratic
        
        x = self.end_pos[0]
        draw_y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * p
        
        color_idx = env.grid[self.end_pos[0], self.end_pos[1]]
        if color_idx != -1:
            env._draw_tile(x, draw_y, color_idx)
    
    def get_involved_tile(self):
        return self.end_pos

class ClearAnimation(Animation):
    def __init__(self, pos):
        super().__init__(duration=12)
        self.pos = pos
        
    def draw(self, env):
        p = self.progress / self.duration
        size_mod = -env.TILE_SIZE * p
        alpha = 255 * (1 - p)
        
        # Color is gone from grid, so we can't look it up.
        # This animation is purely visual, relying on particles.
        # We'll just draw a shrinking white flash.
        rect = env._get_tile_rect(self.pos[0], self.pos[1]).inflate(size_mod, size_mod)
        pygame.gfxdraw.box(env.screen, rect, (255, 255, 255, alpha))
        
    def get_involved_tile(self):
        return self.pos

class ChainReactionDelay(Animation):
    def __init__(self, callback):
        super().__init__(duration=15)
        self.callback = callback
        self.called = False

    def update(self):
        super().update()
        if self.is_done() and not self.called:
            self.callback()
            self.called = True
    
    def draw(self, env):
        pass # No visual component

class Particle:
    def __init__(self, x, y, color, rng):
        self.x = x
        self.y = y
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = 20
        self.color = color
        self.radius = rng.integers(3, 6)
        
    def is_alive(self):
        return self.lifespan > 0
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.lifespan -= 1
        
    def draw(self, screen):
        alpha = max(0, 255 * (self.lifespan / 20))
        color_with_alpha = (*self.color, int(alpha))
        pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), int(self.radius), color_with_alpha)