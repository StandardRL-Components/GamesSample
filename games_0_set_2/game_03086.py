
# Generated: 2025-08-27T22:19:41.587003
# Source Brief: brief_03086.md
# Brief Index: 3086

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Hold Shift for soft drop, press Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated falling block puzzle where strategic rotations and placements are key to clearing lines and achieving a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_DRAW_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_DRAW_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_DRAW_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_DRAW_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID_BG = (30, 30, 50)
    COLOR_GRID_LINES = (50, 50, 70)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_GHOST = (255, 255, 255, 50)

    # Game settings
    WIN_SCORE = 1000
    MAX_STEPS = 10000
    INITIAL_FALL_SPEED = 1.0  # cells per second
    FPS = 30

    TETROMINOES = {
        'I': [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)]],
        'O': [[(1, 0), (2, 0), (1, 1), (2, 1)]],
        'T': [[(0, 1), (1, 1), (2, 1), (1, 0)], [(1, 0), (1, 1), (1, 2), (2, 1)],
              [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (1, 1), (1, 2), (0, 1)]],
        'S': [[(1, 0), (2, 0), (0, 1), (1, 1)], [(1, 0), (1, 1), (2, 1), (2, 2)]],
        'Z': [[(0, 0), (1, 0), (1, 1), (2, 1)], [(2, 0), (2, 1), (1, 1), (1, 2)]],
        'J': [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (1, 1), (1, 2)],
              [(0, 1), (1, 1), (2, 1), (2, 2)], [(1, 0), (1, 1), (1, 2), (0, 2)]],
        'L': [[(2, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 2)],
              [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]]
    }

    PIECE_COLORS = {
        'I': (0, 240, 240), 'O': (240, 240, 0), 'T': (160, 0, 240),
        'S': (0, 240, 0), 'Z': (240, 0, 0), 'J': (0, 0, 240), 'L': (240, 160, 0)
    }

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid = []
        self.current_piece = {}
        self.next_piece_type = ''
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.fall_speed_cells_per_sec = 0
        self.particles = []
        self.lines_to_clear_anim = []
        self.line_clear_anim_timer = 0
        self.space_was_held = False
        self.last_move_timer = 0
        self.move_cooldown = self.FPS // 10 # 10 moves per second max

        self.reset()
        
        # This can be commented out for performance in production
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.fall_speed_cells_per_sec = self.INITIAL_FALL_SPEED
        self.particles = []
        self.lines_to_clear_anim = []
        self.line_clear_anim_timer = 0
        self.space_was_held = False
        self.last_move_timer = 0

        self.next_piece_type = self.np_random.choice(list(self.TETROMINOES.keys()))
        self._spawn_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty per step to encourage speed
        self.steps += 1
        self.last_move_timer += 1
        
        if self.line_clear_anim_timer > 0:
            self.line_clear_anim_timer -=1
            if self.line_clear_anim_timer == 0:
                self._finish_line_clear()
        else:
            self._handle_input(action)
            reward += self._update_gravity()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if self.game_over:
            reward += -50.0 # Loss penalty
        elif self.score >= self.WIN_SCORE:
            reward += 100.0 # Win bonus
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement and rotation with cooldown
        if self.last_move_timer >= self.move_cooldown:
            moved = False
            # 3=left, 4=right
            if movement == 3:
                self._move(-1)
                moved = True
            elif movement == 4:
                self._move(1)
                moved = True
            # 1=up (rotate right), 2=down (rotate left)
            elif movement == 1:
                self._rotate(1)
                moved = True
            elif movement == 2:
                self._rotate(-1)
                moved = True
            
            if moved:
                self.last_move_timer = 0
        
        # Hard drop on space *press*
        if space_held and not self.space_was_held:
            self._hard_drop()
        self.space_was_held = space_held
        
        # Soft drop
        if shift_held:
            self.fall_timer += 2 # Extra gravity pull

    def _update_gravity(self):
        self.fall_timer += 1
        fall_threshold = self.FPS / self.fall_speed_cells_per_sec
        
        if self.fall_timer >= fall_threshold:
            self.fall_timer = 0
            
            # Attempt to move down
            new_y = self.current_piece['y'] + 1
            if not self._check_collision(self.current_piece['shape'], self.current_piece['x'], new_y):
                self.current_piece['y'] = new_y
            else:
                # Lock piece
                return self._lock_piece()
        return 0.0 # No reward for just falling

    def _spawn_piece(self):
        self.current_piece = {
            'type': self.next_piece_type,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0
        }
        self.current_piece['shape'] = self.TETROMINOES[self.current_piece['type']][0]
        self.next_piece_type = self.np_random.choice(list(self.TETROMINOES.keys()))
        
        if self._check_collision(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
            self.game_over = True

    def _lock_piece(self):
        piece_reward = 0
        for x, y in self.current_piece['shape']:
            grid_x, grid_y = self.current_piece['x'] + x, self.current_piece['y'] + y
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_y][grid_x] = self.current_piece['type']
        
        lines_cleared = self._check_line_clears()
        
        if lines_cleared == 0:
            piece_reward = -0.2 # Penalty for placing a block without clearing a line
        else:
            # Reward for clearing lines
            rewards = {1: 1.0, 2: 3.0, 3: 5.0, 4: 10.0}
            piece_reward = rewards.get(lines_cleared, 0)
            self.score += lines_cleared * 10
            # Update fall speed based on score
            self.fall_speed_cells_per_sec = self.INITIAL_FALL_SPEED + (self.score // 200) * 0.5
        
        if not self.game_over:
            self._spawn_piece()
            
        return piece_reward

    def _check_line_clears(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if all(self.grid[y]):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            self.lines_to_clear_anim = lines_to_clear
            self.line_clear_anim_timer = self.FPS // 6 # 1/6th of a second animation
            # Create particles
            for y in lines_to_clear:
                for x in range(self.GRID_WIDTH):
                    px = self.GRID_X + x * self.CELL_SIZE + self.CELL_SIZE / 2
                    py = self.GRID_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2
                    color = self.PIECE_COLORS[self.grid[y][x]]
                    for _ in range(3): # 3 particles per cell
                        self.particles.append(Particle(px, py, color, self.np_random))
            # sfx: line clear
            
        return len(lines_to_clear)

    def _finish_line_clear(self):
        for y_clear in sorted(self.lines_to_clear_anim, reverse=True):
            self.grid.pop(y_clear)
            self.grid.insert(0, [None for _ in range(self.GRID_WIDTH)])
        self.lines_to_clear_anim = []

    def _move(self, dx):
        new_x = self.current_piece['x'] + dx
        if not self._check_collision(self.current_piece['shape'], new_x, self.current_piece['y']):
            self.current_piece['x'] = new_x
            # sfx: move

    def _rotate(self, dr):
        rotations = self.TETROMINOES[self.current_piece['type']]
        new_rotation_idx = (self.current_piece['rotation'] + dr) % len(rotations)
        new_shape = rotations[new_rotation_idx]
        
        # Wall kick logic
        for dx in [0, 1, -1, 2, -2]: # Basic wall kick checks
            if not self._check_collision(new_shape, self.current_piece['x'] + dx, self.current_piece['y']):
                self.current_piece['x'] += dx
                self.current_piece['rotation'] = new_rotation_idx
                self.current_piece['shape'] = new_shape
                # sfx: rotate
                break

    def _hard_drop(self):
        # sfx: hard drop
        y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape'], self.current_piece['x'], y + 1):
            y += 1
        self.current_piece['y'] = y
        self.fall_timer = self.FPS # Force lock on next frame
    
    def _check_collision(self, shape, x, y):
        for sx, sy in shape:
            grid_x, grid_y = x + sx, y + sy
            if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                return True # Out of bounds
            if self.grid[grid_y][grid_x] is not None:
                return True # Collides with existing block
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.GRID_X, self.GRID_Y, self.GRID_DRAW_WIDTH, self.GRID_DRAW_HEIGHT))
        
        self._render_grid_and_locked_pieces()
        if not self.game_over:
            self._render_ghost_piece()
            self._render_piece(self.current_piece, self.GRID_X, self.GRID_Y)
        
        self._render_particles()

        # Draw grid border
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, 
                         (self.GRID_X, self.GRID_Y, self.GRID_DRAW_WIDTH, self.GRID_DRAW_HEIGHT), 2)

    def _render_grid_and_locked_pieces(self):
        for y in range(self.GRID_HEIGHT):
            # Line clear animation
            if y in self.lines_to_clear_anim:
                flash_alpha = 255 * (self.line_clear_anim_timer / (self.FPS // 6))
                flash_surface = pygame.Surface((self.GRID_DRAW_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill((255, 255, 255, flash_alpha))
                self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE))
                continue

            for x in range(self.GRID_WIDTH):
                cell_type = self.grid[y][x]
                if cell_type:
                    self._draw_block(x, y, self.PIECE_COLORS[cell_type], self.GRID_X, self.GRID_Y)

    def _render_piece(self, piece, grid_x, grid_y):
        color = self.PIECE_COLORS[piece['type']]
        for x, y in piece['shape']:
            self._draw_block(piece['x'] + x, piece['y'] + y, color, grid_x, grid_y)

    def _render_ghost_piece(self):
        ghost_y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape'], self.current_piece['x'], ghost_y + 1):
            ghost_y += 1
        
        color = self.PIECE_COLORS[self.current_piece['type']]
        for x, y in self.current_piece['shape']:
            self._draw_block(self.current_piece['x'] + x, ghost_y + y, color, self.GRID_X, self.GRID_Y, is_ghost=True)

    def _draw_block(self, x, y, color, grid_x, grid_y, is_ghost=False):
        rect = pygame.Rect(
            grid_x + x * self.CELL_SIZE,
            grid_y + y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        if is_ghost:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((*color, 60))
            self.screen.blit(s, rect.topleft)
            pygame.draw.rect(self.screen, (*color, 120), rect, 1)
        else:
            # 3D effect
            light_color = tuple(min(255, c + 40) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, dark_color, rect)
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))
            pygame.draw.line(self.screen, light_color, rect.topleft, rect.topright)
            pygame.draw.line(self.screen, light_color, rect.topleft, rect.bottomleft)

    def _render_particles(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()
            p.draw(self.screen)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.GRID_X + self.GRID_DRAW_WIDTH + 20, self.GRID_Y + 30))

        # Next piece preview
        next_text = self.font_small.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.GRID_X + self.GRID_DRAW_WIDTH + 20, self.GRID_Y + 100))
        
        preview_bg_rect = pygame.Rect(self.GRID_X + self.GRID_DRAW_WIDTH + 20, self.GRID_Y + 130, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, preview_bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, preview_bg_rect, 1)

        preview_piece = {
            'type': self.next_piece_type,
            'shape': self.TETROMINOES[self.next_piece_type][0],
            'x': 0, 'y': 0
        }
        self._render_piece(preview_piece, preview_bg_rect.x, preview_bg_rect.y)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Particle:
    def __init__(self, x, y, color, rng):
        self.x = x
        self.y = y
        self.color = color
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 3)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = 20
        self.radius = rng.integers(2, 5)

    def is_alive(self):
        return self.lifespan > 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # gravity
        self.lifespan -= 1
        self.radius = max(0, self.radius - 0.1)

    def draw(self, surface):
        if self.is_alive():
            alpha = int(255 * (self.lifespan / 20))
            color_with_alpha = (*self.color, alpha)
            
            # Use a temporary surface for alpha blending
            s = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (self.radius, self.radius), self.radius)
            surface.blit(s, (int(self.x - self.radius), int(self.y - self.radius)))

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Puzzle Game")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Pygame uses a different coordinate system for surfaces vs arrays
        # So we need to transpose back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()