import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import math
import os
import pygame


# Set the environment to run headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "then move to an adjacent tile and press Space again to swap."
    )
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Clear the entire board before you run out of moves to win!"
    )
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_TILE_TYPES = 6
    TILE_SIZE = 45
    GRID_LINE_WIDTH = 2
    ANIMATION_SPEED = 0.2  # Progress per step
    MAX_MOVES = 50
    MAX_EPISODE_STEPS = 2000

    # --- Colors ---
    COLOR_BG = (44, 62, 80)
    COLOR_GRID_LINES = (52, 73, 94)
    COLOR_CURSOR = (236, 240, 241)
    COLOR_SELECTED = (46, 204, 113)
    COLOR_TEXT = (236, 240, 241)

    TILE_COLORS = [
        (0, 0, 0),  # Empty
        (231, 76, 60),   # Red
        (26, 188, 156),  # Teal
        (52, 152, 219),  # Blue
        (155, 89, 182),  # Purple
        (241, 196, 15),  # Yellow
        (230, 126, 34),  # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400
        
        self.grid_pixel_width = self.GRID_WIDTH * self.TILE_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.TILE_SIZE
        self.grid_offset_x = (self.screen_width - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_pixel_height) // 2

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.game_state = "IDLE"
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = (0, 0)
        self.selected_pos = None
        self.animation_progress = 0.0
        self.animation_data = {}
        self.prev_space_held = False

        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        
        # self.np_random is initialized by super().reset()
        self.np_random = None

        # This validation was causing a timeout due to a bug in board generation.
        # The bug is now fixed.
        self.validate_implementation()

    def _get_lerp_pos(self, pos1, pos2, t):
        return pos1[0] * (1 - t) + pos2[0] * t, pos1[1] * (1 - t) + pos2[1] * t

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for x, y in matches:
                self.grid[x, y] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
        
        if not self._find_possible_moves():
            self._shuffle_board()

    def _shuffle_board(self):
        flat_grid = self.grid.flatten()
        self.np_random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Ensure shuffled board is valid
        while self._find_all_matches() or not self._find_possible_moves():
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))

    def _find_all_matches_on_grid(self, grid):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if grid[x, y] == 0: continue
                # Horizontal
                if x < self.GRID_WIDTH - 2 and grid[x, y] == grid[x+1, y] == grid[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical
                if y < self.GRID_HEIGHT - 2 and grid[x, y] == grid[x, y+1] == grid[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return list(matches)
    
    def _find_all_matches(self):
        return self._find_all_matches_on_grid(self.grid)

    def _find_possible_moves(self):
        moves = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Swap right
                if x < self.GRID_WIDTH - 1:
                    temp_grid = self.grid.copy()
                    temp_grid[x, y], temp_grid[x+1, y] = temp_grid[x+1, y], temp_grid[x, y]
                    if self._find_all_matches_on_grid(temp_grid):
                        moves.append(((x, y), (x+1, y)))
                # Swap down
                if y < self.GRID_HEIGHT - 1:
                    temp_grid = self.grid.copy()
                    temp_grid[x, y], temp_grid[x, y+1] = temp_grid[x, y+1], temp_grid[x, y]
                    if self._find_all_matches_on_grid(temp_grid):
                        moves.append(((x, y), (x, y+1)))
        return moves

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.np_random is now initialized by super().reset()

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        
        self._generate_board()
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_pos = None
        self.game_state = "IDLE"
        self.animation_progress = 0.0
        self.animation_data = {}
        self.prev_space_held = True # prevent action on first frame

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        movement, space_held, _ = action
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        # --- State Machine ---
        if self.game_state == "IDLE":
            if not self.game_over:
                # Handle cursor movement
                cx, cy = self.cursor_pos
                if movement == 1: cy = max(0, cy - 1)
                elif movement == 2: cy = min(self.GRID_HEIGHT - 1, cy + 1)
                elif movement == 3: cx = max(0, cx - 1)
                elif movement == 4: cx = min(self.GRID_WIDTH - 1, cx + 1)
                self.cursor_pos = (cx, cy)

                # Handle selection/swap
                if space_press:
                    if self.selected_pos is None:
                        self.selected_pos = self.cursor_pos
                    else:
                        dist = abs(self.cursor_pos[0] - self.selected_pos[0]) + abs(self.cursor_pos[1] - self.selected_pos[1])
                        if dist == 1: # Adjacent
                            self.animation_data = {'from': self.selected_pos, 'to': self.cursor_pos, 'reverse': False}
                            self.game_state = "SWAP_ANIM"
                            self.animation_progress = 0.0
                            self.moves_left -= 1
                        self.selected_pos = None
        
        elif self.game_state == "SWAP_ANIM":
            self.animation_progress += self.ANIMATION_SPEED
            if self.animation_progress >= 1.0:
                p1 = self.animation_data['from']
                p2 = self.animation_data['to']
                self.grid[p1], self.grid[p2] = self.grid[p2], self.grid[p1]

                matches = self._find_all_matches()
                if matches and not self.animation_data.get('reverse', False):
                    self.animation_data = {'matches': matches}
                    self.game_state = "CLEAR_ANIM"
                    self.animation_progress = 0.0
                    
                    match_count = len(matches)
                    if match_count == 3: reward += 1
                    elif match_count == 4: reward += 2
                    else: reward += 3
                    self.score += match_count * 10
                elif not self.animation_data.get('reverse', False):
                    # Invalid move, swap back
                    self.animation_data = {'from': p2, 'to': p1, 'reverse': True}
                    self.animation_progress = 0.0
                    reward += -0.1
                else: # Finished swapping back
                    self.game_state = "IDLE"
                    self.animation_progress = 0.0
        
        elif self.game_state == "CLEAR_ANIM":
            self.animation_progress += self.ANIMATION_SPEED * 1.5
            if self.animation_progress >= 1.0:
                for x, y in self.animation_data['matches']:
                    self.grid[x, y] = 0
                
                self.animation_data = {'fall_map': self._calculate_fall_map()}
                self.game_state = "FALL_ANIM"
                self.animation_progress = 0.0
        
        elif self.game_state == "FALL_ANIM":
            self.animation_progress += self.ANIMATION_SPEED
            if self.animation_progress >= 1.0:
                self._apply_fall_map(self.animation_data['fall_map'])
                
                cascade_matches = self._find_all_matches()
                if cascade_matches:
                    self.animation_data = {'matches': cascade_matches}
                    self.game_state = "CLEAR_ANIM"
                    self.animation_progress = 0.0
                    
                    match_count = len(cascade_matches)
                    if match_count == 3: reward += 2 # Bonus for cascade
                    elif match_count == 4: reward += 4
                    else: reward += 6
                    self.score += match_count * 20
                else:
                    self.game_state = "IDLE"
                    self.animation_progress = 0.0
                    if not self._find_possible_moves() and np.any(self.grid != 0):
                        self._shuffle_board()

        # --- Check Termination Conditions ---
        if not self.game_over:
            is_board_clear = np.all(self.grid == 0)
            if is_board_clear:
                reward += 100
                self.score += 1000
                terminated = True
                self.game_over = True
                self.game_state = "WIN"
            elif self.moves_left <= 0 and self.game_state == "IDLE":
                terminated = True
                self.game_over = True
                self.game_state = "LOSE"
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _calculate_fall_map(self):
        fall_map = {}
        for x in range(self.GRID_WIDTH):
            fall_dist = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    fall_dist += 1
                elif fall_dist > 0:
                    fall_map[(x, y)] = (x, y + fall_dist)
        return fall_map
    
    def _apply_fall_map(self, fall_map):
        new_grid = np.zeros_like(self.grid)
        occupied = set(fall_map.values())
        for (fx, fy), (tx, ty) in fall_map.items():
            new_grid[tx, ty] = self.grid[fx, fy]
        
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in fall_map and (x, y) not in occupied and self.grid[x, y] != 0:
                    new_grid[x, y] = self.grid[x, y]
        self.grid = new_grid

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_lines()
        self._render_tiles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_lines(self):
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_pixel_height), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_pixel_width, y), self.GRID_LINE_WIDTH)
            
    def _draw_tile(self, surface, pos, tile_type, size_mult=1.0, alpha=255):
        if tile_type == 0: return
        color = self.TILE_COLORS[tile_type]
        
        size = int(self.TILE_SIZE * 0.8 * size_mult)
        radius = int(size * 0.3)
        rect = pygame.Rect(0, 0, size, size)
        rect.center = pos
        
        alpha_color = (*color, alpha)
        
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, alpha_color, (0, 0, *rect.size), border_radius=radius)
        
        highlight_color = (255, 255, 255, 50)
        pygame.draw.rect(shape_surf, highlight_color, (0,0,size, size/2), border_top_left_radius=radius, border_top_right_radius=radius)

        surface.blit(shape_surf, rect)

    def _render_tiles(self):
        animating_tiles = set()
        
        if self.game_state == "SWAP_ANIM":
            p1_grid, p2_grid = self.animation_data['from'], self.animation_data['to']
            p1_type, p2_type = self.grid[p1_grid], self.grid[p2_grid]
            
            p1_screen = (self.grid_offset_x + p1_grid[0] * self.TILE_SIZE + self.TILE_SIZE // 2, self.grid_offset_y + p1_grid[1] * self.TILE_SIZE + self.TILE_SIZE // 2)
            p2_screen = (self.grid_offset_x + p2_grid[0] * self.TILE_SIZE + self.TILE_SIZE // 2, self.grid_offset_y + p2_grid[1] * self.TILE_SIZE + self.TILE_SIZE // 2)

            t = self.animation_progress
            pos1_anim = self._get_lerp_pos(p1_screen, p2_screen, t)
            pos2_anim = self._get_lerp_pos(p2_screen, p1_screen, t)
            
            self._draw_tile(self.screen, pos1_anim, p1_type)
            self._draw_tile(self.screen, pos2_anim, p2_type)
            animating_tiles.add(p1_grid)
            animating_tiles.add(p2_grid)

        elif self.game_state == "CLEAR_ANIM":
            t = self.animation_progress
            size = 1.0 - t
            alpha = 255 * (1.0 - t)
            for x, y in self.animation_data['matches']:
                pos = (self.grid_offset_x + x * self.TILE_SIZE + self.TILE_SIZE // 2, self.grid_offset_y + y * self.TILE_SIZE + self.TILE_SIZE // 2)
                self._draw_tile(self.screen, pos, self.grid[x, y], size_mult=max(0,size), alpha=max(0,int(alpha)))
                animating_tiles.add((x, y))

        elif self.game_state == "FALL_ANIM":
            t = self.animation_progress
            for (fx, fy), (tx, ty) in self.animation_data['fall_map'].items():
                start_pos = (self.grid_offset_x + fx * self.TILE_SIZE + self.TILE_SIZE // 2, self.grid_offset_y + fy * self.TILE_SIZE + self.TILE_SIZE // 2)
                end_pos = (self.grid_offset_x + tx * self.TILE_SIZE + self.TILE_SIZE // 2, self.grid_offset_y + ty * self.TILE_SIZE + self.TILE_SIZE // 2)
                anim_pos = self._get_lerp_pos(start_pos, end_pos, t)
                self._draw_tile(self.screen, anim_pos, self.grid[fx, fy])
                animating_tiles.add((fx, fy))

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in animating_tiles:
                    pos = (self.grid_offset_x + x * self.TILE_SIZE + self.TILE_SIZE // 2, self.grid_offset_y + y * self.TILE_SIZE + self.TILE_SIZE // 2)
                    self._draw_tile(self.screen, pos, self.grid[x, y])
        
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.TILE_SIZE,
            self.grid_offset_y + self.cursor_pos[1] * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)
        
        if self.selected_pos:
            selected_rect = pygame.Rect(
                self.grid_offset_x + self.selected_pos[0] * self.TILE_SIZE,
                self.grid_offset_y + self.selected_pos[1] * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, selected_rect, 4, border_radius=6)
    
    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 20, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.game_state == "WIN":
                msg = "Board Cleared!"
                color = (46, 204, 113)
            else: # LOSE or truncated
                msg = "Game Over!"
                color = (231, 76, 60)
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "game_state": self.game_state,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Beginning implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    while running:
        action = [0, 0, 0] # Default: no-op
        
        # Event handling for keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
        
        # For continuous actions like holding space, use get_pressed
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Step the environment only if an action is taken or game is animating
        if any(action) or env.game_state != "IDLE":
             obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame rendering for human play ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(30)

    env.close()