
# Generated: 2025-08-28T01:38:37.398536
# Source Brief: brief_04178.md
# Brief Index: 4178

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, SHIFT to rotate counter-clockwise. ↓ for soft drop, SPACE for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic falling block puzzle game. Clear lines to score points and prevent the stack from reaching the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 20 # Lines to clear

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_VALUE = (255, 255, 255)
        self.COLOR_GHOST = (255, 255, 255, 60)
        self.PIECE_COLORS = [
            (0, 240, 240),  # I piece (cyan)
            (240, 240, 0),  # O piece (yellow)
            (160, 0, 240),  # T piece (purple)
            (0, 240, 0),    # S piece (green)
            (240, 0, 0),    # Z piece (red)
            (0, 0, 240),    # J piece (blue)
            (240, 160, 0),  # L piece (orange)
        ]
        
        # Tetromino shapes
        self.PIECES = [
            [[1, 5, 9, 13], [4, 5, 6, 7]],  # I
            [[4, 5, 8, 9]],                 # O
            [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],  # T
            [[4, 5, 9, 10], [2, 6, 5, 9]], # S
            [[6, 5, 9, 8], [1, 5, 6, 10]], # Z
            [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]], # J
            [[0, 1, 5, 9], [4, 5, 6, 2], [1, 5, 8, 9], [4, 8, 9, 10]] # L
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 64)

        # Grid positioning
        self.GRID_PIXEL_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GRID_PIXEL_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_PIXEL_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_PIXEL_HEIGHT) // 2

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_counter = 0.0
        self.fall_speed = 1.0 # cells per second
        self.previous_space_held = False
        self.reward_this_step = 0
        self.line_clear_animation = []
        self.action_cooldowns = {'move': 0, 'rotate': 0}

        self.reset()
        self.validate_implementation()
    
    def _create_piece(self, piece_id=None):
        if piece_id is None:
            piece_id = self.np_random.integers(0, len(self.PIECES))
        shape_rotations = self.PIECES[piece_id]
        
        return {
            'id': piece_id,
            'shapes': shape_rotations,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0,
            'color': self.PIECE_COLORS[piece_id]
        }

    def _spawn_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self._create_piece()
        self.fall_counter = 0.0
        if self._check_collision(self.current_piece):
            self.game_over = True
            self.reward_this_step -= 50 # Losing penalty

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.next_piece = self._create_piece()
        self._spawn_piece()
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.fall_speed = 1.0
        self.fall_counter = 0.0
        self.previous_space_held = False
        self.reward_this_step = 0
        self.line_clear_animation = []
        self.action_cooldowns = {'move': 0, 'rotate': 0}

        return self._get_observation(), self._get_info()
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cooldowns to prevent overly sensitive controls
        cooldown_time = 0.1 # seconds
        for k in self.action_cooldowns:
            self.action_cooldowns[k] = max(0, self.action_cooldowns[k] - self.clock.get_time() / 1000.0)

        # Horizontal Movement
        if movement in [3, 4] and self.action_cooldowns['move'] == 0:
            dx = -1 if movement == 3 else 1
            if not self._check_collision(self.current_piece, dx=dx):
                self.current_piece['x'] += dx
                self.reward_this_step -= 0.02
                self.action_cooldowns['move'] = cooldown_time

        # Rotation
        if (movement == 1 or shift_held) and self.action_cooldowns['rotate'] == 0:
            d_rot = 1 if movement == 1 else -1 # Up for CW, Shift for CCW
            if not self._check_collision(self.current_piece, d_rot=d_rot):
                self.current_piece['rotation'] = (self.current_piece['rotation'] + d_rot) % len(self.current_piece['shapes'])
            # Wall kick
            else:
                for kick in [-1, 1, -2, 2]: # Try shifting left/right
                    if not self._check_collision(self.current_piece, dx=kick, d_rot=d_rot):
                        self.current_piece['x'] += kick
                        self.current_piece['rotation'] = (self.current_piece['rotation'] + d_rot) % len(self.current_piece['shapes'])
                        break
            self.reward_this_step -= 0.02
            self.action_cooldowns['rotate'] = cooldown_time * 1.5

        # Soft Drop
        if movement == 2:
            self.fall_counter += 5.0 / self.fall_speed * (self.clock.get_time() / 1000.0)

        # Hard Drop
        if space_held and not self.previous_space_held:
            # sfx: hard_drop
            while not self._check_collision(self.current_piece, dy=1):
                self.current_piece['y'] += 1
            self._place_piece()
        self.previous_space_held = space_held

    def _get_piece_coords(selfself, piece, dx=0, dy=0, d_rot=0):
        shape_idx = (piece['rotation'] + d_rot) % len(piece['shapes'])
        shape = piece['shapes'][shape_idx]
        coords = []
        for i in range(4):
            x = piece['x'] + dx + shape[i] % 4
            y = piece['y'] + dy + shape[i] // 4
            coords.append((x, y))
        return coords

    def _check_collision(self, piece, dx=0, dy=0, d_rot=0):
        coords = self._get_piece_coords(piece, dx, dy, d_rot)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True # Wall collision
            if y >= 0 and self.grid[x, y] > 0:
                return True # Grid collision
        return False

    def _place_piece(self):
        # sfx: place_block
        coords = self._get_piece_coords(self.current_piece)
        for x, y in coords:
            if y >= 0:
                self.grid[x, y] = self.current_piece['id'] + 1
        
        self.reward_this_step += 0.1 # Placement reward
        self._check_lines()
        self._spawn_piece()

    def _check_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y] > 0):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            # sfx: line_clear
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            self.score += [0, 100, 300, 500, 800][num_cleared]
            self.reward_this_step += [0, 1, 3, 5, 10][num_cleared]

            for y in lines_to_clear:
                self.grid[:, y] = 0 # Clear line visually
                self.line_clear_animation.append({'y': y, 'timer': 0.2})
            
            # Shift blocks down
            lines_to_clear.sort(reverse=True)
            for y_clear in lines_to_clear:
                for y in range(y_clear, 0, -1):
                    self.grid[:, y] = self.grid[:, y-1]
                self.grid[:, 0] = 0
            
            # Increase difficulty
            self.fall_speed = 1.0 + (self.lines_cleared // 5) * 0.5


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1
        
        self._handle_input(action)

        # Update fall timer
        dt = self.clock.get_time() / 1000.0
        self.fall_counter += self.fall_speed * dt

        if self.fall_counter >= 1.0:
            self.fall_counter = 0.0
            if not self._check_collision(self.current_piece, dy=1):
                self.current_piece['y'] += 1
            else:
                self._place_piece()
        
        # Update animations
        for anim in self.line_clear_animation[:]:
            anim['timer'] -= dt
            if anim['timer'] <= 0:
                self.line_clear_animation.remove(anim)

        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.lines_cleared >= self.WIN_SCORE
        if terminated and self.lines_cleared >= self.WIN_SCORE and not self.game_over:
            self.reward_this_step += 100 # Win bonus
        
        reward = self.reward_this_step

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _draw_block(self, surface, x, y, color, is_ghost=False):
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        if is_ghost:
            pygame.draw.rect(surface, color, rect, 2, border_radius=3)
        else:
            light_color = tuple(min(255, c + 50) for c in color)
            dark_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(surface, light_color, rect.move(-1, -1), border_radius=4)
            pygame.draw.rect(surface, dark_color, rect.move(1, 1), border_radius=4)
            pygame.draw.rect(surface, color, rect, border_radius=3)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw placed blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] > 0:
                    color_id = int(self.grid[x, y] - 1)
                    self._draw_block(
                        self.screen,
                        self.GRID_X + x * self.CELL_SIZE,
                        self.GRID_Y + y * self.CELL_SIZE,
                        self.PIECE_COLORS[color_id]
                    )
        
        # Draw ghost piece
        if not self.game_over and self.current_piece:
            ghost_piece = self.current_piece.copy()
            while not self._check_collision(ghost_piece, dy=1):
                ghost_piece['y'] += 1
            ghost_coords = self._get_piece_coords(ghost_piece)
            for x, y in ghost_coords:
                 if y >= 0:
                    self._draw_block(
                        self.screen,
                        self.GRID_X + x * self.CELL_SIZE,
                        self.GRID_Y + y * self.CELL_SIZE,
                        self.COLOR_GHOST,
                        is_ghost=True
                    )

        # Draw current piece
        if not self.game_over and self.current_piece:
            piece_coords = self._get_piece_coords(self.current_piece)
            for x, y in piece_coords:
                if y >= 0:
                    self._draw_block(
                        self.screen,
                        self.GRID_X + x * self.CELL_SIZE,
                        self.GRID_Y + y * self.CELL_SIZE,
                        self.current_piece['color']
                    )

        # Draw line clear animation
        for anim in self.line_clear_animation:
            alpha = 255 * (anim['timer'] / 0.2)
            flash_surface = pygame.Surface((self.GRID_PIXEL_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + anim['y'] * self.CELL_SIZE))

    def _render_ui(self):
        # Score
        score_text = self.font_m.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_l.render(f"{self.score}", True, self.COLOR_UI_VALUE)
        self.screen.blit(score_text, (self.GRID_X - score_text.get_width() - 20, self.GRID_Y + 20))
        self.screen.blit(score_val, (self.GRID_X - score_val.get_width() - 20, self.GRID_Y + 45))

        # Lines
        lines_text = self.font_m.render("LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_l.render(f"{self.lines_cleared}", True, self.COLOR_UI_VALUE)
        self.screen.blit(lines_text, (self.GRID_X - lines_text.get_width() - 20, self.GRID_Y + 120))
        self.screen.blit(lines_val, (self.GRID_X - lines_val.get_width() - 20, self.GRID_Y + 145))

        # Next Piece
        next_text = self.font_m.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.GRID_X + self.GRID_PIXEL_WIDTH + 20, self.GRID_Y + 20))
        if self.next_piece:
            next_coords = self._get_piece_coords(self.next_piece)
            min_x = min(c[0] for c in next_coords)
            min_y = min(c[1] for c in next_coords)
            for x, y in next_coords:
                self._draw_block(
                    self.screen,
                    self.GRID_X + self.GRID_PIXEL_WIDTH + 40 + (x - min_x) * self.CELL_SIZE,
                    self.GRID_Y + 60 + (y - min_y) * self.CELL_SIZE,
                    self.next_piece['color']
                )
        
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
        elif self.lines_cleared >= self.WIN_SCORE:
            msg = "YOU WIN!"
        else:
            return

        msg_surf = self.font_l.render(msg, True, self.COLOR_UI_VALUE)
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Update clock for auto-advance timing
        if self.auto_advance:
            self.clock.tick(30)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gymnasium Block Puzzle")
    
    terminated = False
    running = True
    
    # Game loop for human play
    while running:
        # Action defaults
        movement = 0 # none
        space = 0    # released
        shift = 0    # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Print info for debugging
        if info['steps'] % 30 == 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Lines: {info['lines_cleared']}, Reward: {reward:.2f}")

    env.close()
    pygame.quit()