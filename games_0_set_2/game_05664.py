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
    user_guide = (
        "Controls: ←→ to move, ↑/Space to rotate. ↓ for soft drop, Shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Rotate and drop falling stars to clear lines and reach a target score before the screen fills up in this fast-paced, top-down arcade puzzler."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    
    # Playfield position
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_GRID = (40, 40, 80)
    COLOR_TEXT = (220, 220, 255)
    COLOR_UI_BG = (30, 30, 60)
    COLOR_UI_BORDER = (60, 60, 120)

    # Tetromino shapes (stars) and their colors
    SHAPES = [
        # I
        [[[0, 0], [-1, 0], [1, 0], [2, 0]], [[0, 0], [0, -1], [0, 1], [0, 2]]],
        # O
        [[[0, 0], [1, 0], [0, 1], [1, 1]]],
        # T
        [[[0, 0], [-1, 0], [1, 0], [0, -1]], [[0, 0], [0, -1], [0, 1], [1, 0]], [[0, 0], [-1, 0], [1, 0], [0, 1]], [[0, 0], [0, -1], [0, 1], [-1, 0]]],
        # L
        [[[0, 0], [-1, 0], [1, 0], [1, -1]], [[0, 0], [0, -1], [0, 1], [1, 1]], [[0, 0], [-1, 0], [1, 0], [-1, 1]], [[0, 0], [0, -1], [0, 1], [-1, -1]]],
        # J
        [[[0, 0], [-1, 0], [1, 0], [-1, -1]], [[0, 0], [0, -1], [0, 1], [1, -1]], [[0, 0], [-1, 0], [1, 0], [1, 1]], [[0, 0], [0, -1], [0, 1], [-1, 1]]],
        # S
        [[[0, 0], [-1, 0], [0, -1], [1, -1]], [[0, 0], [0, -1], [1, 0], [1, 1]]],
        # Z
        [[[0, 0], [1, 0], [0, -1], [-1, -1]], [[0, 0], [0, 1], [1, 0], [1, -1]]],
    ]
    SHAPE_COLORS = [
        (0, 255, 255),  # I - Cyan
        (255, 255, 0),  # O - Yellow
        (160, 0, 255),  # T - Purple
        (255, 165, 0),  # L - Orange
        (0, 0, 255),    # J - Blue
        (0, 255, 0),    # S - Green
        (255, 0, 0),    # Z - Red
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
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 36)
        
        self.grid = []
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        
        self.fall_timer = 0
        self.fall_speed = 15 # frames per grid cell
        
        self.particles = []
        self.last_reward = 0

        # This is here because reset() is called in __init__
        # and it needs self.np_random to be initialized.
        # It will be seeded again in the public reset call.
        self.np_random, _ = gym.utils.seeding.np_random()

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.last_reward = 0
        
        self.fall_timer = 0
        self.particles = []

        # Ensure first few pieces are simple for a random agent
        self.piece_bag = list(range(len(self.SHAPES)))
        self.np_random.shuffle(self.piece_bag)
        # Force first piece to be I or O
        first_piece_idx = self.np_random.choice([0, 1])
        if first_piece_idx in self.piece_bag:
            self.piece_bag.remove(first_piece_idx)
            self.piece_bag.insert(0, first_piece_idx)
        
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.last_reward = 0
        
        self._handle_input(action)
        
        if not self.game_over:
            self._update_game_state(action)

        terminated = self.game_over or self.lines_cleared >= 50 or self.steps >= 5000
        if terminated and not self.game_over and self.lines_cleared >= 50:
            self.last_reward += 100 # Win bonus
        
        return (
            self._get_observation(),
            self.last_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held:
            # Hard drop
            while self._is_valid_position(self.current_piece, dy=1):
                self.current_piece['y'] += 1
            self._lock_piece()
            # sfx: hard_drop_sound
            return # Hard drop action overrides all other actions for this step

        # Movement
        if movement == 3: # Left
            if self._is_valid_position(self.current_piece, dx=-1):
                self.current_piece['x'] -= 1
                # sfx: move_sound
        elif movement == 4: # Right
            if self._is_valid_position(self.current_piece, dx=1):
                self.current_piece['x'] += 1
                # sfx: move_sound

        # Rotation
        if movement == 1: # Rotate Right (Up arrow)
            self._rotate_piece(1)
        if space_held: # Rotate Left (Space bar)
            self._rotate_piece(-1)
            
    def _update_game_state(self, action):
        self.fall_timer += 1
        
        soft_drop = action[0] == 2
        current_fall_speed = self.fall_speed // 4 if soft_drop else self.fall_speed
        
        if self.fall_timer >= current_fall_speed:
            self.fall_timer = 0
            if self._is_valid_position(self.current_piece, dy=1):
                self.current_piece['y'] += 1
            else:
                self._lock_piece()

    def _lock_piece(self):
        # sfx: lock_piece_sound
        self.last_reward += 0.1 # Base reward for placing a piece
        
        # Penalize based on height
        height_penalty = (self.current_piece['y'] / self.GRID_HEIGHT) * 0.5
        self.last_reward -= height_penalty

        for pos in self.current_piece['shape'][self.current_piece['rotation']]:
            px, py = self.current_piece['x'] + pos[0], self.current_piece['y'] + pos[1]
            if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                self.grid[py][px] = self.current_piece['color_idx']
        
        cleared_count = self._clear_lines()
        if cleared_count > 0:
            rewards = {1: 1, 2: 2, 3: 4, 4: 8}
            self.last_reward += rewards.get(cleared_count, 0)
            self.score += [0, 100, 300, 500, 800][cleared_count] * (self.lines_cleared // 10 + 1)
            self.lines_cleared += cleared_count
        
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()

        if not self._is_valid_position(self.current_piece):
            self.game_over = True
            self.last_reward -= 100 # Loss penalty
            # sfx: game_over_sound

    def _clear_lines(self):
        lines_to_clear = []
        for i, row in enumerate(self.grid):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(i)

        if not lines_to_clear:
            return 0
            
        # sfx: line_clear_sound
        for row_idx in lines_to_clear:
            self._create_line_clear_particles(row_idx)
        
        # This is a more efficient way to remove multiple lines
        lines_to_clear.sort(reverse=True)
        for row_idx in lines_to_clear:
            self.grid.pop(row_idx)
            self.grid.insert(0, [0 for _ in range(self.GRID_WIDTH)])

        return len(lines_to_clear)
        
    def _new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.SHAPES)))
            self.np_random.shuffle(self.piece_bag)
        
        shape_idx = self.piece_bag.pop(0)
        shape = self.SHAPES[shape_idx]
        
        return {
            'x': self.GRID_WIDTH // 2,
            'y': 1, # Start just below the top
            'shape': shape,
            'rotation': 0,
            'color_idx': shape_idx + 1,
            'color': self.SHAPE_COLORS[shape_idx]
        }
        
    def _rotate_piece(self, direction):
        original_rotation = self.current_piece['rotation']
        num_rotations = len(self.current_piece['shape'])
        self.current_piece['rotation'] = (original_rotation + direction) % num_rotations
        
        if not self._is_valid_position(self.current_piece):
            # Wall kick logic
            if self._is_valid_position(self.current_piece, dx=1):
                self.current_piece['x'] += 1
            elif self._is_valid_position(self.current_piece, dx=-1):
                self.current_piece['x'] -= 1
            elif self._is_valid_position(self.current_piece, dx=2): # For I-piece
                self.current_piece['x'] += 2
            elif self._is_valid_position(self.current_piece, dx=-2):
                self.current_piece['x'] -= 2
            else:
                self.current_piece['rotation'] = original_rotation # Revert if all kicks fail
                return
        # sfx: rotate_sound
        
    def _is_valid_position(self, piece, dx=0, dy=0):
        for pos in piece['shape'][piece['rotation']]:
            px, py = piece['x'] + pos[0] + dx, piece['y'] + pos[1] + dy
            
            if not (0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT):
                return False # Out of bounds
            if self.grid[py][px] != 0:
                return False # Collision with locked piece
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "lines": self.lines_cleared,
            "steps": self.steps,
        }

    def _render_background_effects(self):
        # Simple moving stars
        for i in range(50):
            x = (hash(i * 10) + self.steps) % self.SCREEN_WIDTH
            y = (hash(i * 20) + self.steps // 3) % self.SCREEN_HEIGHT
            size = (hash(i*30) % 2) + 1
            color = (hash(i*40)%50 + 20, hash(i*50)%50 + 20, hash(i*60)%50 + 50)
            pygame.draw.rect(self.screen, color, (x, y, size, size))

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw locked pieces
        for y, row in enumerate(self.grid):
            for x, cell_color_idx in enumerate(row):
                if cell_color_idx != 0:
                    self._draw_block(x, y, self.SHAPE_COLORS[cell_color_idx - 1])
        
        if not self.game_over and self.current_piece:
            # Draw ghost piece
            ghost = self.current_piece.copy()
            while self._is_valid_position(ghost, dy=1):
                ghost['y'] += 1
            self._draw_piece(ghost, ghost=True)

            # Draw current piece
            self._draw_piece(self.current_piece)
            
        # Draw particles
        self._update_and_draw_particles()

    def _draw_piece(self, piece, ghost=False):
        for pos in piece['shape'][piece['rotation']]:
            px, py = piece['x'] + pos[0], piece['y'] + pos[1]
            self._draw_block(px, py, piece['color'], ghost)

    def _draw_block(self, grid_x, grid_y, color, ghost=False):
        screen_x = self.GRID_X + grid_x * self.CELL_SIZE
        screen_y = self.GRID_Y + grid_y * self.CELL_SIZE
        
        if ghost:
            rect = (screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, tuple(c//2 for c in color), rect, 2)
        else:
            # Main block
            main_rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
            # Darker border
            border_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, border_color, main_rect)
            # Brighter inner part
            inner_color = tuple(min(255, c + 50) for c in color)
            inner_rect = pygame.Rect(screen_x + 2, screen_y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, inner_color, inner_rect)
            # Highlight
            pygame.gfxdraw.line(self.screen, screen_x + 2, screen_y + 2, screen_x + self.CELL_SIZE - 3, screen_y + 2, (255, 255, 255, 100))
            pygame.gfxdraw.line(self.screen, screen_x + 2, screen_y + 2, screen_x + 2, screen_y + self.CELL_SIZE - 3, (255, 255, 255, 100))

    def _render_ui(self):
        # UI Panels
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 30, 180, 120), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (10, 30, 180, 120), 2, border_radius=8)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - 190, 30, 180, 150), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (self.SCREEN_WIDTH - 190, 30, 180, 150), 2, border_radius=8)

        # Score and Lines text
        score_text = self.font_title.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (190 // 2 - score_text.get_width() // 2 + 10, 40))
        score_val = self.font_main.render(f"{self.score:08d}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (190 // 2 - score_val.get_width() // 2 + 10, 70))
        
        lines_text = self.font_title.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (190 // 2 - lines_text.get_width() // 2 + 10, 100))
        lines_val = self.font_main.render(f"{self.lines_cleared} / 50", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (190 // 2 - lines_val.get_width() // 2 + 10, 130))

        # Next Piece display
        next_text = self.font_title.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 100 - next_text.get_width() // 2, 40))
        if self.next_piece:
            # Center the piece in the box
            piece_coords = self.next_piece['shape'][0]
            min_x = min(c[0] for c in piece_coords)
            max_x = max(c[0] for c in piece_coords)
            min_y = min(c[1] for c in piece_coords)
            max_y = max(c[1] for c in piece_coords)
            piece_width = (max_x - min_x + 1) * self.CELL_SIZE
            piece_height = (max_y - min_y + 1) * self.CELL_SIZE
            
            offset_x = self.SCREEN_WIDTH - 100 - piece_width / 2
            offset_y = 110 - piece_height / 2
            
            for pos in self.next_piece['shape'][0]:
                screen_x = offset_x - min_x * self.CELL_SIZE + pos[0] * self.CELL_SIZE
                screen_y = offset_y - min_y * self.CELL_SIZE + pos[1] * self.CELL_SIZE
                
                # Draw block without using grid coordinates
                main_rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
                border_color = tuple(max(0, c - 50) for c in self.next_piece['color'])
                pygame.draw.rect(self.screen, border_color, main_rect)
                inner_color = tuple(min(255, c + 50) for c in self.next_piece['color'])
                inner_rect = pygame.Rect(screen_x + 2, screen_y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                pygame.draw.rect(self.screen, inner_color, inner_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text = self.font_title.render("GAME OVER", True, (255, 50, 50))
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - 50))

    def _create_line_clear_particles(self, row_idx):
        y_pos = self.GRID_Y + (row_idx + 0.5) * self.CELL_SIZE
        for _ in range(50):
            x_pos = self.np_random.uniform(self.GRID_X, self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE)
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            color = random.choice([(255, 255, 220), (255, 255, 255), (200, 200, 255)])
            lifetime = self.np_random.integers(20, 41)
            self.particles.append({'pos': [x_pos, y_pos], 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p['lifetime'] * 6)))
                color = (*p['color'], alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

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
    # This block allows you to play the game directly
    # To render, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to your action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # To render, we need a display
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Falling Stars")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not done:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        # This input handling is simple; it doesn't handle auto-repeat delays
        # but is fine for testing.
        move_key_pressed = False
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                move_key_pressed = True
                break # Prioritize first key in map
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        # --- End Human Input ---

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

    pygame.quit()