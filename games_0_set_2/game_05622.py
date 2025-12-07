
# Generated: 2025-08-28T05:34:47.290624
# Source Brief: brief_05622.md
# Brief Index: 5622

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to clear highlighted group. Shift to reshuffle (costs 1 move)."
    )

    game_description = (
        "Match-3 puzzle game. Clear groups of 2 or more same-colored tiles. Clear the board or get the highest score in 20 moves."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 12, 8
    TILE_SIZE = 40
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    GRID_X = (WIDTH - GRID_WIDTH) // 2
    GRID_Y = (HEIGHT - GRID_HEIGHT) // 2 + 20

    COLOR_BG = (25, 30, 45)
    COLOR_GRID = (45, 55, 75)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_TEXT_SHADOW = (10, 10, 20)
    COLOR_CURSOR = (255, 255, 255)
    
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    INITIAL_MOVES = 20
    MAX_STEPS = 1000
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)
        
        # Game state variables
        self.grid = None
        self.cursor_x = 0
        self.cursor_y = 0
        self.highlighted_group = []
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.step_reward = 0.0

        # Input handling
        self.prev_space_held = False
        self.prev_shift_held = False

        # Animation state
        self.animations = []

        self.reset()
        
        # This check is for development and can be removed in production
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_over = False
        self.game_won = False
        
        self._create_and_validate_grid()
        
        self.cursor_x = self.GRID_COLS // 2
        self.cursor_y = self.GRID_ROWS // 2
        self._update_highlighted_group()

        self.animations = []
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.step_reward = 0.0
        self.steps += 1

        if not self.game_over:
            # Process input only if no major animations are running
            if not any(isinstance(anim, (TileFallAnimation, TilePopAnimation)) for anim in self.animations):
                self._handle_input(action)
        
        self._update_animations()

        terminated = self._check_termination()
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        moved = False
        if movement == 1 and self.cursor_y > 0: # Up
            self.cursor_y -= 1
            moved = True
        elif movement == 2 and self.cursor_y < self.GRID_ROWS - 1: # Down
            self.cursor_y += 1
            moved = True
        elif movement == 3 and self.cursor_x > 0: # Left
            self.cursor_x -= 1
            moved = True
        elif movement == 4 and self.cursor_x < self.GRID_COLS - 1: # Right
            self.cursor_x += 1
            moved = True
        
        if moved:
            self._update_highlighted_group()

        if space_pressed and self.highlighted_group:
            self._execute_match()
        
        if shift_pressed:
            self._execute_reshuffle()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
    def _execute_match(self):
        # Sound: tile_clear.wav
        group_size = len(self.highlighted_group)
        
        # Calculate reward
        self.step_reward += group_size  # +1 per tile
        if group_size == 2:
            self.step_reward -= 0.1
        if group_size > 5:
            self.step_reward += 5.0

        self.score += group_size * group_size # Exponential scoring
        self.moves_left -= 1
        
        # Animate and remove tiles
        for x, y in self.highlighted_group:
            color = self.TILE_COLORS[self.grid[y][x]]
            self.animations.append(TilePopAnimation((x, y), color))
            self.grid[y][x] = None
            # Spawn particles
            for _ in range(3):
                self.animations.append(Particle((x, y), color))
        
        self.highlighted_group = []
        self._start_fall_animation()

    def _execute_reshuffle(self):
        if self.moves_left > 0:
            # Sound: reshuffle.wav
            self.moves_left -= 1
            self.step_reward -= 5.0 # Penalty for reshuffling
            self._create_and_validate_grid()
            self._update_highlighted_group()
            # Add a visual effect for reshuffling
            for y in range(self.GRID_ROWS):
                for x in range(self.GRID_COLS):
                    color = self.TILE_COLORS[self.grid[y][x]]
                    self.animations.append(TilePopAnimation((x,y), color, pop_in=True))


    def _start_fall_animation(self):
        # Determine which tiles need to fall
        for x in range(self.GRID_COLS):
            fall_dist = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[y][x] is None:
                    fall_dist += 1
                elif fall_dist > 0:
                    tile_color_idx = self.grid[y][x]
                    self.animations.append(TileFallAnimation(
                        (x, y), (x, y + fall_dist), tile_color_idx, self
                    ))
                    self.grid[y + fall_dist][x] = self.grid[y][x]
                    self.grid[y][x] = None
    
    def _post_animation_refill(self):
        # This is called by TileFallAnimation when the last one finishes
        # Refill empty top spots with new tiles
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[y][x] is None:
                    self.grid[y][x] = self.np_random.integers(0, len(self.TILE_COLORS))
        
        # Check for board clear
        if not any(None in row for row in self.grid):
            # The board is full, now check for valid moves
            if not self._find_any_valid_move():
                # Anti-softlock: No moves available, reshuffle for free
                self._create_and_validate_grid()
                # Add a visual effect for the automatic reshuffle
                for y in range(self.GRID_ROWS):
                    for x in range(self.GRID_COLS):
                        color = self.TILE_COLORS[self.grid[y][x]]
                        self.animations.append(TilePopAnimation((x,y), color, pop_in=True))

        self._update_highlighted_group()

    def _update_animations(self):
        # Update and remove finished animations
        was_animating_fall = any(isinstance(anim, TileFallAnimation) for anim in self.animations)
        self.animations = [anim for anim in self.animations if not anim.update()]
        is_animating_fall = any(isinstance(anim, TileFallAnimation) for anim in self.animations)

        # If fall animations just finished, trigger refill
        if was_animating_fall and not is_animating_fall:
            self._post_animation_refill()

    def _check_termination(self):
        if self.game_over:
            return True

        # Win condition: board is cleared
        is_board_empty = all(all(tile is None for tile in row) for row in self.grid)
        if is_board_empty:
            self.game_over = True
            self.game_won = True
            self.step_reward += 100.0 # Win bonus
            return True

        # Lose condition: out of moves
        if self.moves_left <= 0:
            self.game_over = True
            self.game_won = False
            self.step_reward -= 50.0 # Lose penalty
            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, 
            (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT), 
            border_radius=8)

        # Draw static tiles
        for y, row in enumerate(self.grid):
            for x, tile_color_idx in enumerate(row):
                if tile_color_idx is not None:
                    # Check if this tile is part of an animation
                    is_animating = any(
                        hasattr(anim, 'grid_pos') and anim.grid_pos == (x, y) 
                        for anim in self.animations
                    )
                    if not is_animating:
                        self._draw_tile(x, y, self.TILE_COLORS[tile_color_idx])

        # Draw highlighted group
        if self.highlighted_group:
            for x, y in self.highlighted_group:
                rect = pygame.Rect(
                    self.GRID_X + x * self.TILE_SIZE,
                    self.GRID_Y + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                highlight_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(highlight_surf, (255, 255, 255, 90), highlight_surf.get_rect(), border_radius=8)
                self.screen.blit(highlight_surf, rect.topleft)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_X + self.cursor_x * self.TILE_SIZE,
            self.GRID_Y + self.cursor_y * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)

        # Draw animations
        for anim in self.animations:
            anim.draw(self.screen)

    def _draw_tile(self, grid_x, grid_y, color, size_factor=1.0, y_offset=0):
        size = int(self.TILE_SIZE * 0.85 * size_factor)
        padding = (self.TILE_SIZE - size) // 2
        rect = pygame.Rect(
            self.GRID_X + grid_x * self.TILE_SIZE + padding,
            self.GRID_Y + grid_y * self.TILE_SIZE + padding + y_offset,
            size, size
        )
        light_color = tuple(min(255, c + 40) for c in color)
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        pygame.gfxdraw.aacircle(self.screen, rect.left + 5, rect.top + 5, 2, light_color)


    def _render_ui(self):
        # Render Score
        self._render_text(f"Score: {self.score}", (20, 20), self.font_small, self.COLOR_UI_TEXT)
        # Render Moves
        self._render_text(f"Moves: {self.moves_left}", (self.WIDTH - 120, 20), self.font_small, self.COLOR_UI_TEXT)
        
        if self.game_over:
            msg = "BOARD CLEARED!" if self.game_won else "OUT OF MOVES"
            color = (150, 255, 150) if self.game_won else (255, 150, 150)
            self._render_text(msg, (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, color, center=True)

    def _render_text(self, text, pos, font, color, center=False):
        shadow = font.render(text, True, self.COLOR_UI_TEXT_SHADOW)
        main = font.render(text, True, color)
        
        rect = main.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos

        self.screen.blit(shadow, (rect.x + 2, rect.y + 2))
        self.screen.blit(main, rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "game_won": self.game_won,
        }

    # --- Helper Functions ---

    def _create_and_validate_grid(self):
        while True:
            self.grid = [
                [self.np_random.integers(0, len(self.TILE_COLORS)) for _ in range(self.GRID_COLS)]
                for _ in range(self.GRID_ROWS)
            ]
            if self._find_any_valid_move():
                break

    def _find_any_valid_move(self):
        visited = set()
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                if (x, y) not in visited:
                    group = self._find_connected_group(x, y)
                    if len(group) >= 2:
                        return True
                    visited.update(group)
        return False

    def _update_highlighted_group(self):
        if self.grid[self.cursor_y][self.cursor_x] is not None:
            group = self._find_connected_group(self.cursor_x, self.cursor_y)
            if len(group) >= 2:
                self.highlighted_group = group
                return
        self.highlighted_group = []

    def _find_connected_group(self, start_x, start_y):
        if self.grid[start_y][start_x] is None:
            return []

        target_color = self.grid[start_y][start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        group = []

        while q:
            x, y = q.popleft()
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    if (nx, ny) not in visited and self.grid[ny][nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# --- Animation Classes ---

class Animation:
    def update(self):
        raise NotImplementedError
    def draw(self, screen):
        raise NotImplementedError

class TilePopAnimation(Animation):
    def __init__(self, grid_pos, color, pop_in=False):
        self.grid_pos = grid_pos
        self.color = color
        self.duration = 10
        self.timer = 0 if pop_in else self.duration
        self.pop_in = pop_in

    def update(self):
        if self.pop_in:
            self.timer += 1
            return self.timer >= self.duration
        else:
            self.timer -= 1
            return self.timer <= 0

    def draw(self, screen):
        progress = self.timer / self.duration
        size_factor = progress if self.pop_in else 1.0 - progress
        
        size = int(GameEnv.TILE_SIZE * 0.85 * size_factor)
        padding = (GameEnv.TILE_SIZE - size) // 2
        rect = pygame.Rect(
            GameEnv.GRID_X + self.grid_pos[0] * GameEnv.TILE_SIZE + padding,
            GameEnv.GRID_Y + self.grid_pos[1] * GameEnv.TILE_SIZE + padding,
            size, size
        )
        
        alpha = int(255 * progress)
        temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, (*self.color, alpha), (0, 0, *rect.size), border_radius=6)
        screen.blit(temp_surf, rect.topleft)

class TileFallAnimation(Animation):
    def __init__(self, start_pos, end_pos, color_idx, env):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color_idx = color_idx
        self.color = GameEnv.TILE_COLORS[color_idx]
        self.env = env # To call draw_tile
        self.grid_pos = start_pos # For filtering in main loop
        
        self.duration = 15 # frames
        self.timer = 0

    def update(self):
        self.timer = min(self.duration, self.timer + 1)
        return self.timer >= self.duration

    def draw(self, screen):
        progress = self.timer / self.duration
        # Ease out quad
        eased_progress = 1 - (1 - progress) ** 2

        start_pixel_y = self.start_pos[1] * GameEnv.TILE_SIZE
        end_pixel_y = self.end_pos[1] * GameEnv.TILE_SIZE
        
        current_y_offset = (end_pixel_y - start_pixel_y) * eased_progress
        
        self.env._draw_tile(self.start_pos[0], self.start_pos[1], self.color, y_offset=current_y_offset)

class Particle(Animation):
    def __init__(self, grid_pos, color):
        self.x = GameEnv.GRID_X + (grid_pos[0] + 0.5) * GameEnv.TILE_SIZE
        self.y = GameEnv.GRID_Y + (grid_pos[1] + 0.5) * GameEnv.TILE_SIZE
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.max_lifetime = 20
        self.lifetime = self.max_lifetime
        self.size = random.randint(3, 6)
        self.grid_pos = None # Doesn't represent a tile

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # Gravity
        self.lifetime -= 1
        return self.lifetime <= 0

    def draw(self, screen):
        progress = self.lifetime / self.max_lifetime
        size = int(self.size * progress)
        if size > 0:
            alpha = int(255 * progress)
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (size, size), size)
            screen.blit(temp_surf, (int(self.x) - size, int(self.y) - size))