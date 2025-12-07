
# Generated: 2025-08-28T02:27:52.164810
# Source Brief: brief_04452.md
# Brief Index: 4452

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Space to drop the block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Place falling blocks to create and clear horizontal lines. Aim for chain reactions to maximize your score and clear 100 blocks to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 10 # Brief specified 10x10
    CELL_SIZE = 30
    MAX_STEPS = 1000
    WIN_CONDITION_BLOCKS = 100

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_GHOST = (255, 255, 255, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        
        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.grid_top_left_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_top_left_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        self._define_pieces()
        
        # State variables are initialized in reset()
        self.grid = None
        self.current_piece = None
        self.steps = None
        self.score = None
        self.blocks_cleared = None
        self.game_over = None
        self.particles = None

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def _define_pieces(self):
        self.piece_colors = [
            (230, 60, 60),    # Red
            (60, 230, 60),    # Green
            (60, 60, 230),    # Blue
            (230, 230, 60),   # Yellow
            (230, 60, 230),   # Magenta
            (60, 230, 230),   # Cyan
            (240, 140, 40)    # Orange
        ]
        # Shapes are defined by 4x4 grids of coordinates relative to a pivot
        self.piece_shapes = {
            'I': [[(0, 1), (1, 1), (2, 1), (3, 1)], [(1, 0), (1, 1), (1, 2), (1, 3)]],
            'J': [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 2)], [(1, 0), (1, 1), (0, 2), (1, 2)]],
            'L': [[(2, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
            'O': [[(0, 0), (1, 0), (0, 1), (1, 1)]],
            'S': [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
            'T': [[(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (2, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (0, 1), (1, 1), (1, 2)]],
            'Z': [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (0, 2)]]
        }
        self.piece_keys = list(self.piece_shapes.keys())
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0.0
        self.blocks_cleared = 0
        self.game_over = False
        self.particles = []
        
        self._spawn_new_piece()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        reward = 0.0
        terminated = False
        
        if not space_held: # Action: Adjust piece
            new_x = self.current_piece['x']
            new_rot = self.current_piece['rotation']
            
            if movement == 1: # up -> Rotate CW
                new_rot = (self.current_piece['rotation'] + 1) % len(self.current_piece['shapes'])
            elif movement == 2: # down -> Rotate CCW
                new_rot = (self.current_piece['rotation'] - 1 + len(self.current_piece['shapes'])) % len(self.current_piece['shapes'])
            elif movement == 3: # left -> Move Left
                new_x -= 1
            elif movement == 4: # right -> Move Right
                new_x += 1
            
            if self._is_valid_position(x_offset=new_x, rotation=new_rot):
                self.current_piece['x'] = new_x
                self.current_piece['rotation'] = new_rot
            
            # No reward or state change for just moving/rotating the preview
            reward = 0.0

        else: # Action: Place piece
            self.steps += 1
            
            # Find hard drop location
            drop_y = self._get_hard_drop_y()
            
            # Place piece on grid
            self._place_piece(self.current_piece['x'], drop_y, self.current_piece['rotation'])
            
            # Clear lines and calculate reward from it
            line_reward, cleared_count = self._clear_lines_and_get_reward()
            reward += line_reward
            self.score += reward
            self.blocks_cleared += cleared_count

            # Check for win condition
            if self.blocks_cleared >= self.WIN_CONDITION_BLOCKS:
                reward += 100
                self.score += 100
                terminated = True
                self.game_over = True
            
            # Spawn new piece and check for loss condition
            if not terminated:
                if not self._spawn_new_piece():
                    reward = -100 # Game over penalty
                    self.score -= 100 # Overwrite any score change from this step
                    terminated = True
                    self.game_over = True
        
        # Check max steps termination
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_cleared": self.blocks_cleared,
        }

    def _render_game(self):
        self._draw_grid()
        self._draw_placed_blocks()
        if not self.game_over and self.current_piece:
            self._draw_ghost_piece()
        self._update_and_draw_particles()

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.grid_top_left_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.grid_top_left_y), (px, self.grid_top_left_y + self.grid_pixel_height))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.grid_top_left_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_top_left_x, py), (self.grid_top_left_x + self.grid_pixel_width, py))

    def _draw_placed_blocks(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] > 0:
                    color_index = self.grid[r][c] - 1
                    self._draw_block(c, r, self.piece_colors[color_index])

    def _draw_ghost_piece(self):
        drop_y = self._get_hard_drop_y()
        shape_coords = self.current_piece['shapes'][self.current_piece['rotation']]
        
        for x, y in shape_coords:
            grid_x, grid_y = self.current_piece['x'] + x, drop_y + y
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                px = self.grid_top_left_x + grid_x * self.CELL_SIZE
                py = self.grid_top_left_y + grid_y * self.CELL_SIZE
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill(self.COLOR_GHOST)
                self.screen.blit(s, (px, py))

    def _draw_block(self, grid_x, grid_y, color):
        px = self.grid_top_left_x + grid_x * self.CELL_SIZE
        py = self.grid_top_left_y + grid_y * self.CELL_SIZE
        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        pygame.gfxdraw.box(self.screen, rect, color)
        border_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.rect(self.screen, border_color, rect, 1)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p['life'] * 12)))
                radius = max(1, int(p['life'] / p['max_life'] * (self.CELL_SIZE / 4)))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], alpha)
                )

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        blocks_text = self.font_large.render(f"CLEARED: {self.blocks_cleared}/{self.WIN_CONDITION_BLOCKS}", True, self.COLOR_UI_TEXT)
        text_rect = blocks_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(blocks_text, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            status = "YOU WIN!" if self.blocks_cleared >= self.WIN_CONDITION_BLOCKS else "GAME OVER"
            end_text = self.font_large.render(status, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _spawn_new_piece(self):
        piece_key = self.np_random.choice(self.piece_keys)
        color_idx = self.np_random.integers(0, len(self.piece_colors))
        shapes = self.piece_shapes[piece_key]
        
        self.current_piece = {
            'key': piece_key,
            'color_idx': color_idx,
            'shapes': shapes,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0
        }
        if not self._is_valid_position():
            self.current_piece = None
            return False # Game over
        return True

    def _is_valid_position(self, piece=None, x_offset=None, y_offset=None, rotation=None):
        if piece is None: piece = self.current_piece
        if x_offset is None: x_offset = piece['x']
        if y_offset is None: y_offset = piece['y']
        if rotation is None: rotation = piece['rotation']
        
        shape_coords = piece['shapes'][rotation]
        
        for x, y in shape_coords:
            grid_x, grid_y = x_offset + x, y_offset + y
            if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                return False
            if self.grid[grid_y][grid_x] > 0:
                return False
        return True

    def _get_hard_drop_y(self):
        y = self.current_piece['y']
        while self._is_valid_position(y_offset=y + 1):
            y += 1
        return y

    def _place_piece(self, x_offset, y_offset, rotation):
        shape_coords = self.current_piece['shapes'][rotation]
        color_val = self.current_piece['color_idx'] + 1
        for x, y in shape_coords:
            grid_x, grid_y = x_offset + x, y_offset + y
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_y][grid_x] = color_val

    def _clear_lines_and_get_reward(self):
        chain_multiplier = 0
        total_lines_cleared = 0
        reward = 0.0
        
        while True:
            full_rows = [r for r in range(self.GRID_HEIGHT) if np.all(self.grid[r, :] > 0)]
            if not full_rows: break
            
            lines_this_pass = len(full_rows)
            total_lines_cleared += lines_this_pass
            chain_multiplier += 1
            
            grid_colors_cache = self.grid.copy()
            for r in full_rows:
                for c in range(self.GRID_WIDTH):
                    color_idx = grid_colors_cache[r, c] - 1
                    color = self.piece_colors[color_idx]
                    self._spawn_particles(c, r, color)
            
            self.grid = np.delete(self.grid, full_rows, axis=0)
            new_rows = np.zeros((lines_this_pass, self.GRID_WIDTH), dtype=int)
            self.grid = np.vstack((new_rows, self.grid))
        
        if total_lines_cleared == 0:
            reward = -0.02
        else:
            reward += total_lines_cleared * 0.1
            if chain_multiplier > 1:
                reward += (chain_multiplier - 1) * 1.0
        
        blocks_cleared_count = total_lines_cleared * self.GRID_WIDTH
        return reward, blocks_cleared_count

    def _spawn_particles(self, grid_x, grid_y, color):
        # sfx_clear_line.play()
        px = self.grid_top_left_x + (grid_x + 0.5) * self.CELL_SIZE
        py = self.grid_top_left_y + (grid_y + 0.5) * self.CELL_SIZE
        
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life, 'max_life': life, 'color': color
            })

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Manual Control - " + env.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # No-op
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # --- Key Down Events ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
            
            # --- Key Up Events ---
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                    action[0] = 0
                if event.key == pygame.K_SPACE:
                    action[1] = 0
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action[2] = 0

        # Step the environment with the current action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # After a "place" action, reset the spacebar part of the action
        if action[1] == 1:
            action[1] = 0

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    print(f"Game Over! Final Info: {info}")
    
    # Keep the window open for a few seconds to see the final state
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        pygame.display.flip()
        env.clock.tick(30)
        
    pygame.quit()