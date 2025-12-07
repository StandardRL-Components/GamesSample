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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    user_guide = (
        "Controls: ←→ to move, ↓ for soft drop, ↑ to rotate clockwise, "
        "Shift to rotate counter-clockwise. Press Space for hard drop."
    )

    game_description = (
        "Fast-paced falling block puzzle game. Clear lines to score points "
        "and advance through stages before the timer runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless operation
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 20
        self.BLOCK_SIZE = 20
        self.GRID_X = (self.screen_width - (self.GRID_WIDTH * self.BLOCK_SIZE)) // 2
        self.GRID_Y = 0
        self.MAX_STAGES = 3
        self.LINES_PER_STAGE = 10
        self.FPS = 30
        self.STAGE_TIME = 60 * self.FPS

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)
        self.COLORS = [
            (0, 0, 0),         # 0: Empty
            (0, 240, 240),     # 1: I (Cyan)
            (0, 0, 240),       # 2: J (Blue)
            (240, 160, 0),     # 3: L (Orange)
            (240, 240, 0),     # 4: O (Yellow)
            (0, 240, 0),       # 5: S (Green)
            (160, 0, 240),     # 6: T (Purple)
            (240, 0, 0),       # 7: Z (Red)
        ]
        
        # Tetromino shapes and their rotations
        self.SHAPES = {
            'I': [[(0, -1), (0, 0), (0, 1), (0, 2)], [( -1, 0), (0, 0), (1, 0), (2, 0)]],
            'J': [[(-1, -1), (0, -1), (0, 0), (0, 1)], [(-1, 0), (0, 0), (1, 0), (1, -1)], [ (1, 1), (0, 1), (0, 0), (0, -1)], [(1, 0), (0, 0), (-1, 0), (-1, 1)]],
            'L': [[(1, -1), (0, -1), (0, 0), (0, 1)], [(-1, 0), (0, 0), (1, 0), (1, 1)], [(-1, 1), (0, 1), (0, 0), (0, -1)], [(1, 0), (0, 0), (-1, 0), (-1, -1)]],
            'O': [[(0, 0), (0, 1), (1, 0), (1, 1)]],
            'S': [[(0, 0), (1, 0), (0, 1), (-1, 1)], [(-1, 0), (-1, -1), (0, 0), (0, 1)]],
            'T': [[(0, 0), (-1, 0), (1, 0), (0, -1)], [ (0, 0), (0, -1), (0, 1), (-1, 0)], [ (0, 0), (-1, 0), (1, 0), (0, 1)], [(0, 0), (0, -1), (0, 1), (1, 0)]],
            'Z': [[(0, 0), (-1, 0), (0, 1), (1, 1)], [ (1, 0), (1, -1), (0, 0), (0, 1)]],
        }
        self.SHAPE_KEYS = list(self.SHAPES.keys())
        
        # Initialize state variables (will be properly set in reset)
        self.grid = None
        self.current_piece = None
        self.score = 0
        self.stage = 1
        self.lines_cleared_in_stage = 0
        self.timer = 0
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.fall_timer = 0
        self.fall_speed = 0
        self.input_cooldowns = {'move': 0, 'rotate': 0}
        self.line_clear_animation = []
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.stage = 1
        self.lines_cleared_in_stage = 0
        self.timer = self.STAGE_TIME
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.input_cooldowns = {'move': 0, 'rotate': 0}
        self.line_clear_animation = []
        self._set_fall_speed()
        self._spawn_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.001  # Small penalty for time passing
        terminated = False
        truncated = False

        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Update cooldowns
        for key in self.input_cooldowns:
            if self.input_cooldowns[key] > 0:
                self.input_cooldowns[key] -= 1

        # Handle hard drop
        if space_held:
            reward += self._hard_drop()
        else:
            # Handle horizontal movement
            if movement in [3, 4] and self.input_cooldowns['move'] == 0:
                dx = -1 if movement == 3 else 1
                self._move(dx)
                self.input_cooldowns['move'] = 4 # 4 frame cooldown
            
            # Handle rotation
            if (movement == 1 or shift_held) and self.input_cooldowns['rotate'] == 0:
                direction = 1 if movement == 1 else -1 # Up for CW, Shift for CCW
                self._rotate(direction)
                self.input_cooldowns['rotate'] = 6 # 6 frame cooldown
            
            # Handle soft drop and auto-fall
            soft_drop = movement == 2
            self.fall_timer += 2 if soft_drop else 1
            if self.fall_timer >= self.fall_speed:
                self.fall_timer = 0
                if not self._move(0, 1): # If move down fails, lock piece
                    reward += self._lock_piece()

        # Update timer
        self.timer -= 1
        if self.timer <= 0:
            self.game_over = True
            reward -= 50

        # Check for stage clear
        if self.lines_cleared_in_stage >= self.LINES_PER_STAGE:
            reward += self._advance_stage()

        terminated = self.game_over or self.game_won
        if self.game_won:
            reward += 300
            
        self.steps += 1
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    def _set_fall_speed(self):
        base_speed = 15 # frames per grid cell
        speed_increase_per_stage = 2
        self.fall_speed = max(3, base_speed - (self.stage - 1) * speed_increase_per_stage)
        self.fall_timer = 0

    def _spawn_piece(self):
        shape_key = self.np_random.choice(self.SHAPE_KEYS)
        self.current_piece = {
            'shape': shape_key,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 1,
            'y': 0,
            'color_index': self.SHAPE_KEYS.index(shape_key) + 1
        }
        if not self._is_valid_position():
            self.game_over = True

    def _get_piece_coords(self, piece=None):
        if piece is None:
            piece = self.current_piece
        shape_template = self.SHAPES[piece['shape']][piece['rotation']]
        return [(piece['x'] + dx, piece['y'] + dy) for dx, dy in shape_template]

    def _is_valid_position(self, piece=None):
        coords = self._get_piece_coords(piece)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False
            if y >= 0 and self.grid[y, x] != 0:
                return False
        return True

    def _move(self, dx, dy=0):
        if self.current_piece is None: return False
        
        test_piece = self.current_piece.copy()
        test_piece['x'] += dx
        test_piece['y'] += dy

        if self._is_valid_position(test_piece):
            self.current_piece = test_piece
            return True
        return False

    def _rotate(self, direction):
        if self.current_piece is None: return

        test_piece = self.current_piece.copy()
        num_rotations = len(self.SHAPES[test_piece['shape']])
        test_piece['rotation'] = (test_piece['rotation'] + direction) % num_rotations
        
        # Wall kick logic
        for kick_dx in [0, 1, -1, 2, -2]:
            kick_piece = test_piece.copy()
            kick_piece['x'] += kick_dx
            if self._is_valid_position(kick_piece):
                self.current_piece = kick_piece
                return

    def _hard_drop(self):
        if self.current_piece is None: return 0
        
        dy = 0
        while self._is_valid_position({'x': self.current_piece['x'], 'y': self.current_piece['y'] + dy + 1, 'shape': self.current_piece['shape'], 'rotation': self.current_piece['rotation']}):
            dy += 1
        
        self.current_piece['y'] += dy
        return self._lock_piece() + 0.1 # Small bonus for hard dropping

    def _lock_piece(self):
        if self.current_piece is None: return 0
        
        coords = self._get_piece_coords()
        for x, y in coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = self.current_piece['color_index']
        
        self.current_piece = None
        reward = self._clear_lines()
        self._spawn_piece()
        return reward

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] != 0):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0
        
        # Animation
        self.line_clear_animation = [(y, self.FPS // 4) for y in lines_to_clear]
        
        # Clear lines and shift down
        for y in sorted(lines_to_clear, reverse=True):
            self.grid[1:y+1, :] = self.grid[0:y, :]
            self.grid[0, :] = 0
        
        num_cleared = len(lines_to_clear)
        self.lines_cleared_in_stage += num_cleared
        self.score += [0, 100, 300, 500, 800][num_cleared]
        
        # Reward based on brief (1, 2, 4, 8)
        return [0, 1, 2, 4, 8][num_cleared]

    def _advance_stage(self):
        self.stage += 1
        if self.stage > self.MAX_STAGES:
            self.game_won = True
            return 100 # Stage complete reward
        
        self.lines_cleared_in_stage = 0
        self.timer = self.STAGE_TIME
        self.grid.fill(0) # Clear grid for new stage
        self._set_fall_speed()
        self._spawn_piece()
        return 100 # Stage complete reward

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    self._draw_block(x, y, self.grid[y, x])
        
        # Draw ghost piece
        if self.current_piece:
            ghost_piece = self.current_piece.copy()
            dy = 0
            while self._is_valid_position({
                'x': ghost_piece['x'], 
                'y': ghost_piece['y'] + dy + 1, 
                'shape': ghost_piece['shape'], 
                'rotation': ghost_piece['rotation']
            }):
                dy += 1
            ghost_piece['y'] += dy
            coords = self._get_piece_coords(ghost_piece)
            for x, y in coords:
                self._draw_block(x, y, ghost_piece['color_index'], is_ghost=True)

        # Draw current piece
        if self.current_piece:
            coords = self._get_piece_coords()
            for x, y in coords:
                self._draw_block(x, y, self.current_piece['color_index'])

        # Draw line clear animation
        if self.line_clear_animation:
            new_anim_list = []
            for y, duration in self.line_clear_animation:
                rect = (self.GRID_X, self.GRID_Y + y * self.BLOCK_SIZE, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_FLASH, rect)
                if duration > 1:
                    new_anim_list.append((y, duration - 1))
            self.line_clear_animation = new_anim_list

    def _draw_block(self, grid_x, grid_y, color_index, is_ghost=False):
        px = self.GRID_X + grid_x * self.BLOCK_SIZE
        py = self.GRID_Y + grid_y * self.BLOCK_SIZE
        color = self.COLORS[color_index]
        
        if is_ghost:
            alpha_color = (*color, 60)
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            s.fill(alpha_color)
            self.screen.blit(s, (px, py))
        else:
            rect = (px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            # Add a slight 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, (px, py), (px + self.BLOCK_SIZE - 1, py))
            pygame.draw.line(self.screen, highlight, (px, py), (px, py + self.BLOCK_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (px + self.BLOCK_SIZE - 1, py), (px + self.BLOCK_SIZE - 1, py + self.BLOCK_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (px, py + self.BLOCK_SIZE - 1), (px + self.BLOCK_SIZE - 1, py + self.BLOCK_SIZE - 1))
        
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Time
        time_str = f"TIME: {max(0, self.timer // self.FPS)}"
        time_text = self.font_small.render(time_str, True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(time_text, time_rect)

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(center=(self.screen_width // 2, self.screen_height - 30))
        self.screen.blit(stage_text, stage_rect)

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
        elif self.game_won:
            msg = "YOU WIN!"
        else:
            return

        msg_text = self.font_large.render(msg, True, self.COLOR_FLASH)
        msg_rect = msg_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(msg_text, msg_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset(seed=42)
    terminated = False
    
    # Pygame setup for human play
    # This will re-initialize pygame with a visible display.
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    total_reward = 0
    
    while not terminated:
        movement = 0
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
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()