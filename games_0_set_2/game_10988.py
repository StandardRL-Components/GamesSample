import gymnasium as gym
import os
import pygame
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "A competitive Tetris-like puzzle game where you clear lines to score points and outlast your opponent."
    )
    user_guide = (
        "Controls: ←→ to move, ↓ for soft drop, ↑ to rotate clockwise, and shift to rotate counter-clockwise. Press space to hard drop."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.MAX_STEPS = 10000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 15)
        self.COLOR_GAMEOVER = (255, 80, 80)
        self.TETROMINO_COLORS = [
            (0, 0, 0),          # 0: Empty
            (40, 220, 220),     # 1: I (Cyan)
            (220, 220, 40),     # 2: O (Yellow)
            (180, 40, 220),     # 3: T (Purple)
            (40, 40, 220),      # 4: J (Blue)
            (220, 120, 40),     # 5: L (Orange)
            (40, 220, 40),      # 6: S (Green)
            (220, 40, 40),      # 7: Z (Red)
        ]

        # --- Tetromino Shapes ---
        # Indexed by color index
        self.TETROMINOES = {
            1: [[1, 1, 1, 1]], # I
            2: [[1, 1], [1, 1]], # O
            3: [[0, 1, 0], [1, 1, 1]], # T
            4: [[1, 0, 0], [1, 1, 1]], # J
            5: [[0, 0, 1], [1, 1, 1]], # L
            6: [[0, 1, 1], [1, 1, 0]], # S
            7: [[1, 1, 0], [0, 1, 1]]  # Z
        }
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- State Variables ---
        self.steps = None
        self.game_over = None
        self.player_grid = None
        self.player_piece = None
        self.player_next_piece_id = None
        self.player_bag = None
        self.player_score = None
        self.player_just_held_space = None
        self.player_just_held_shift = None
        self.player_just_held_up = None
        self.opponent_grid = None
        self.opponent_piece = None
        self.opponent_next_piece_id = None
        self.opponent_bag = None
        self.opponent_score = None
        self.opponent_ai_target = None
        self.fall_timer = None
        self.fall_speed_delay = None
        self.line_clear_animation = None
        self.particles = None
        
    def _get_new_piece(self, bag, next_piece_id):
        if not bag:
            bag.extend(self.np_random.permutation(list(self.TETROMINOES.keys())).tolist())
        
        piece_id = next_piece_id
        next_piece_id = bag.pop(0)
        
        shape = self.TETROMINOES[piece_id]
        piece = {
            'id': piece_id,
            'shape': shape,
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'rotation': 0
        }
        return piece, bag, next_piece_id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.fall_speed_delay = 0.5  # Seconds per fall step
        self.fall_timer = self.fall_speed_delay * self.metadata['render_fps']
        self.line_clear_animation = []
        self.particles = []

        # Player state
        self.player_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.player_bag = []
        self.player_score = 0
        self.player_just_held_space = False
        self.player_just_held_shift = False
        self.player_just_held_up = False
        # Initialize first two pieces for player
        self.player_bag.extend(self.np_random.permutation(list(self.TETROMINOES.keys())).tolist())
        _, self.player_bag, self.player_next_piece_id = self._get_new_piece(self.player_bag, self.player_bag[0])
        self.player_piece, self.player_bag, self.player_next_piece_id = self._get_new_piece(self.player_bag, self.player_next_piece_id)

        # Opponent state
        self.opponent_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.opponent_bag = []
        self.opponent_score = 0
        self.opponent_ai_target = None
        # Initialize first two pieces for opponent
        self.opponent_bag.extend(self.np_random.permutation(list(self.TETROMINOES.keys())).tolist())
        _, self.opponent_bag, self.opponent_next_piece_id = self._get_new_piece(self.opponent_bag, self.opponent_bag[0])
        self.opponent_piece, self.opponent_bag, self.opponent_next_piece_id = self._get_new_piece(self.opponent_bag, self.opponent_next_piece_id)

        return self._get_observation(), self._get_info()
    
    def _check_collision(self, piece, grid, dx=0, dy=0):
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    new_x, new_y = piece['x'] + c + dx, piece['y'] + r + dy
                    if not (0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT):
                        return True  # Collision with wall/floor
                    if grid[new_y, new_x] != 0:
                        return True  # Collision with another block
        return False

    def _rotate_piece(self, piece, grid, clockwise=True):
        if piece['id'] == 2: return # Cannot rotate 'O' piece
        
        original_shape = piece['shape']
        if clockwise:
            new_shape = [list(row) for row in zip(*original_shape[::-1])]
        else: # counter-clockwise
            new_shape = [list(row) for row in zip(*original_shape)][::-1]

        test_piece = piece.copy()
        test_piece['shape'] = new_shape
        
        # Wall kick logic
        offsets = [(0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (0, -1)]
        for ox, oy in offsets:
            if not self._check_collision(test_piece, grid, dx=ox, dy=oy):
                piece['shape'] = new_shape
                piece['x'] += ox
                piece['y'] += oy
                return True
        return False

    def _get_ghost_y(self, piece, grid):
        y = piece['y']
        while not self._check_collision(piece, grid, dy=1):
            piece['y'] += 1
        ghost_y = piece['y']
        piece['y'] = y
        return ghost_y

    def _lock_piece(self, piece, grid):
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    x, y = piece['x'] + c, piece['y'] + r
                    if 0 <= y < self.GRID_HEIGHT:
                        grid[y, x] = piece['id']
        # Check for game over
        if piece['y'] < 0:
            return True # Game over
        return False

    def _clear_lines(self, grid, player_id):
        lines_cleared = 0
        full_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(grid[r, :] != 0):
                lines_cleared += 1
                full_rows.append(r)
        
        if lines_cleared > 0:
            self.line_clear_animation.append({'rows': full_rows, 'timer': 10, 'player_id': player_id})
            for r in full_rows:
                for x in range(self.GRID_WIDTH):
                    self._spawn_particles(x, r, self.TETROMINO_COLORS[grid[r,x]], player_id)
            
            new_grid = np.zeros_like(grid)
            new_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if r not in full_rows:
                    new_grid[new_row, :] = grid[r, :]
                    new_row -= 1
            return new_grid, lines_cleared
        return grid, 0

    def _spawn_particles(self, grid_x, grid_y, color, player_id):
        offset_x = 50 if player_id == 'player' else 410
        px = offset_x + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = 20 + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'color': color, 'life': 20
            })

    def _get_ai_action(self):
        if self.opponent_ai_target is None or self.opponent_piece['x'] == self.opponent_ai_target['x']:
            best_score = -float('inf')
            best_target = None
            
            for rot in range(4):
                test_piece = self.opponent_piece.copy()
                test_piece['shape'] = self.opponent_piece['shape']
                for _ in range(rot):
                    if test_piece['id'] != 2:
                        test_piece['shape'] = [list(row) for row in zip(*test_piece['shape'][::-1])]
                
                for x in range(-2, self.GRID_WIDTH):
                    test_piece['x'] = x
                    if not self._check_collision(test_piece, self.opponent_grid):
                        y = self._get_ghost_y(test_piece, self.opponent_grid)
                        
                        height = self.GRID_HEIGHT - (y + len(test_piece['shape']))
                        score = -height * 10
                        if self._check_collision(test_piece, self.opponent_grid, dy=1):
                            score += 5
                        
                        if score > best_score:
                            best_score = score
                            best_target = {'x': x, 'rot': rot}
            self.opponent_ai_target = best_target if best_target else {'x': 3, 'rot': 0}

        current_rot = self.opponent_piece.get('rotation', 0)
        if current_rot != self.opponent_ai_target['rot']:
            self.opponent_piece['rotation'] = (self.opponent_piece.get('rotation', 0) + 1) % 4
            self._rotate_piece(self.opponent_piece, self.opponent_grid)
        elif self.opponent_piece['x'] < self.opponent_ai_target['x']:
            if not self._check_collision(self.opponent_piece, self.opponent_grid, dx=1):
                self.opponent_piece['x'] += 1
        elif self.opponent_piece['x'] > self.opponent_ai_target['x']:
            if not self._check_collision(self.opponent_piece, self.opponent_grid, dx=-1):
                self.opponent_piece['x'] -= 1
        
        if self.np_random.random() < 0.1:
            return True
        return False
        
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_button, shift_button = action
        space_pressed = space_button == 1 and not self.player_just_held_space
        shift_pressed = shift_button == 1 and not self.player_just_held_shift
        up_pressed = movement == 1 and not self.player_just_held_up
        self.player_just_held_space = space_button == 1
        self.player_just_held_shift = shift_button == 1
        self.player_just_held_up = movement == 1

        if movement == 3:
            if not self._check_collision(self.player_piece, self.player_grid, dx=-1): self.player_piece['x'] -= 1
        elif movement == 4:
            if not self._check_collision(self.player_piece, self.player_grid, dx=1): self.player_piece['x'] += 1
        
        if up_pressed:
            self._rotate_piece(self.player_piece, self.player_grid, clockwise=True)
        if shift_pressed:
            self._rotate_piece(self.player_piece, self.player_grid, clockwise=False)
        
        if movement == 2:
            self.fall_timer -= self.metadata['render_fps'] * 0.1

        ai_hard_drop = self._get_ai_action()

        self.fall_timer -= 1
        
        player_hard_drop = space_pressed
        player_is_on_floor = self._check_collision(self.player_piece, self.player_grid, dy=1)
        opponent_is_on_floor = self._check_collision(self.opponent_piece, self.opponent_grid, dy=1)

        if self.fall_timer <= 0:
            if not player_is_on_floor: self.player_piece['y'] += 1
            if not opponent_is_on_floor: self.opponent_piece['y'] += 1
            self.fall_timer = self.fall_speed_delay * self.metadata['render_fps']

        player_locks = player_hard_drop or (self.fall_timer <= 0 and player_is_on_floor)
        opponent_locks = ai_hard_drop or (self.fall_timer <= 0 and opponent_is_on_floor)

        if player_locks or opponent_locks:
            if player_hard_drop: self.player_piece['y'] = self._get_ghost_y(self.player_piece, self.player_grid)
            if ai_hard_drop: self.opponent_piece['y'] = self._get_ghost_y(self.opponent_piece, self.opponent_grid)
            
            player_topped_out = self._lock_piece(self.player_piece, self.player_grid)
            opponent_topped_out = self._lock_piece(self.opponent_piece, self.opponent_grid)
            
            self.player_grid, p_cleared = self._clear_lines(self.player_grid, 'player')
            self.opponent_grid, o_cleared = self._clear_lines(self.opponent_grid, 'opponent')

            if p_cleared > 0:
                self.player_score += p_cleared
                reward += p_cleared * 1.0
            if o_cleared > 0:
                self.opponent_score += o_cleared

            player_max_h = 0
            if np.any(self.player_grid):
                player_max_h = self.GRID_HEIGHT - np.min(np.where(np.any(self.player_grid, axis=1)))
            if player_max_h >= self.GRID_HEIGHT - 2:
                reward += 1.0

            if self.player_score // 20 > (self.player_score - p_cleared) // 20:
                self.fall_speed_delay = max(0.1, self.fall_speed_delay - 0.05)
            
            self.player_piece, self.player_bag, self.player_next_piece_id = self._get_new_piece(self.player_bag, self.player_next_piece_id)
            self.opponent_piece, self.opponent_bag, self.opponent_next_piece_id = self._get_new_piece(self.opponent_bag, self.opponent_next_piece_id)
            self.opponent_ai_target = None

            if self._check_collision(self.player_piece, self.player_grid): player_topped_out = True
            if self._check_collision(self.opponent_piece, self.opponent_grid): opponent_topped_out = True

            if player_topped_out and opponent_topped_out:
                self.game_over = True
                reward += 0
            elif player_topped_out:
                self.game_over = True
                reward -= 100
            elif opponent_topped_out:
                self.game_over = True
                reward += 100
            
            self.fall_timer = self.fall_speed_delay * self.metadata['render_fps']

        for anim in self.line_clear_animation[:]:
            anim['timer'] -= 1
            if anim['timer'] <= 0:
                self.line_clear_animation.remove(anim)
        
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
             reward += 100 if self.player_score > self.opponent_score else -100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_playfield(self, grid, offset_x, offset_y, player_id):
        pygame.draw.rect(self.screen, (10, 10, 15), (offset_x, offset_y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        
        for x in range(self.GRID_WIDTH + 1):
            px = offset_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, offset_y), (px, offset_y + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = offset_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x, py), (offset_x + self.GRID_WIDTH * self.CELL_SIZE, py))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if grid[r, c] != 0:
                    color_idx = grid[r, c]
                    self._draw_block(c, r, self.TETROMINO_COLORS[color_idx], offset_x, offset_y)
        
        for anim in self.line_clear_animation:
            if anim['player_id'] == player_id:
                alpha = 255 * (anim['timer'] / 10)
                flash_color = (255, 255, 255, alpha)
                s = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill(flash_color)
                self.screen.blit(s, (offset_x, offset_y + r * self.CELL_SIZE))

    def _draw_block(self, x, y, color, offset_x, offset_y, alpha=255):
        px, py = offset_x + x * self.CELL_SIZE, offset_y + y * self.CELL_SIZE
        
        if alpha < 255:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], alpha))
            self.screen.blit(s, (px, py))
            return

        main_color = color
        light_color = tuple(min(255, c + 50) for c in color)
        dark_color = tuple(max(0, c - 50) for c in color)
        
        pygame.draw.rect(self.screen, main_color, (px, py, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.polygon(self.screen, light_color, [(px, py), (px + self.CELL_SIZE, py), (px + self.CELL_SIZE - 2, py + 2), (px + 2, py + 2)])
        pygame.draw.polygon(self.screen, dark_color, [(px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), (px + self.CELL_SIZE - 2, py + self.CELL_SIZE - 2), (px + 2, py + self.CELL_SIZE - 2)])
        pygame.draw.line(self.screen, (0,0,0,50), (px, py), (px + self.CELL_SIZE, py + self.CELL_SIZE))


    def _render_piece(self, piece, grid, offset_x, offset_y):
        if piece is None: return
        
        ghost_y = self._get_ghost_y(piece, grid)
        shape = piece['shape']
        color = self.TETROMINO_COLORS[piece['id']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(piece['x'] + c, ghost_y + r, color, offset_x, offset_y, alpha=50)

        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(piece['x'] + c, piece['y'] + r, color, offset_x, offset_y)

    def _render_ui(self):
        self._render_text("PLAYER", self.font_medium, self.COLOR_TEXT, (50, 5))
        self._render_text(f"SCORE: {self.player_score}", self.font_small, self.COLOR_TEXT, (52, 35))
        
        self._render_text("OPPONENT", self.font_medium, self.COLOR_TEXT, (410, 5))
        self._render_text(f"SCORE: {self.opponent_score}", self.font_small, self.COLOR_TEXT, (412, 35))
        
        mid_x = self.WIDTH // 2
        self._render_text("NEXT", self.font_medium, self.COLOR_TEXT, (mid_x - 30, 80))
        if self.player_next_piece_id:
            shape = self.TETROMINOES[self.player_next_piece_id]
            color = self.TETROMINO_COLORS[self.player_next_piece_id]
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(c, r, color, mid_x - 40, 120)

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            outcome_text = ""
            if self.player_score > self.opponent_score: outcome_text = "YOU WIN!"
            elif self.opponent_score > self.player_score: outcome_text = "YOU LOSE"
            else: outcome_text = "DRAW"

            self._render_text("GAME OVER", self.font_large, self.COLOR_GAMEOVER, (self.WIDTH/2 - 160, self.HEIGHT/2 - 60))
            self._render_text(outcome_text, self.font_medium, self.COLOR_TEXT, (self.WIDTH/2 - 60, self.HEIGHT/2 + 10))

    def _render_game(self):
        self._render_playfield(self.player_grid, 50, 20, 'player')
        self._render_piece(self.player_piece, self.player_grid, 50, 20)
        
        self._render_playfield(self.opponent_grid, 410, 20, 'opponent')
        self._render_piece(self.opponent_piece, self.opponent_grid, 410, 20)

        for p in self.particles:
            size = max(0, p['life'] / 4)
            pygame.draw.rect(self.screen, p['color'], (p['x'] - size/2, p['y'] - size/2, size, size))
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # The environment itself runs headlessly.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dual Tetris Gym Environment")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Player Score: {info['player_score']}, Opponent Score: {info['opponent_score']}")
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])
        
    print("Game Over!")
    pygame.time.wait(2000)
    env.close()