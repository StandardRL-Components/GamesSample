import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:56:48.578977
# Source Brief: brief_02005.md
# Brief Index: 2005
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A strategic match-3 puzzle game. Shift entire rows and columns to align three identical blocks and score points."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to shift rows or columns. Hold space, shift, or both to select which row/column to move."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 3
        self.CELL_SIZE = 90
        self.GRID_LINE_WIDTH = 4
        self.BLOCK_SIZE = self.CELL_SIZE - 10
        self.WIN_SCORE = 1000
        self.MAX_TURNS = 20
        self.MAX_STEPS = 3000 # Safety net

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_SELECTOR = (255, 255, 0, 100)
        self.COLORS = {
            1: (220, 50, 50),   # Red
            2: (50, 220, 50),   # Green
            3: (50, 100, 220),  # Blue
        }
        self.COLOR_MATCH_FLASH = (255, 255, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 255, 255, 180)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Calculate grid offset for centering
        self.grid_width = self.GRID_SIZE * self.CELL_SIZE
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_width) // 2

        # Animation settings
        self.animation_state = 'IDLE'
        self.animation_progress = 0.0
        self.ANIMATION_SPEED = {
            'SHIFTING': 8.0,
            'MATCHING': 15.0,
            'FALLING': 10.0,
        }
        
        # Initialize state variables
        self.grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turns_remaining = 0
        self.pending_reward = 0
        self.chain_length = 0
        self.shift_details = None
        self.match_details = []
        self.fall_details = []
        self.score_popups = []
        self.selector_info = {'type': None, 'index': 0}

        # self.reset() is called by the wrapper, but for standalone use it's good practice.
        # self.validate_implementation() # Commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.turns_remaining = self.MAX_TURNS
        self.game_over = False
        self.animation_state = 'IDLE'
        self.animation_progress = 0.0
        self.pending_reward = 0
        self.chain_length = 0
        self.shift_details = None
        self.match_details = []
        self.fall_details = []
        self.score_popups = []

        self._generate_initial_grid()
        
        return self._get_observation(), self._get_info()

    def _generate_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(1, len(self.COLORS) + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_matches():
                break
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # State machine for animations
        if self.animation_state != 'IDLE':
            self._update_animation()
        elif not self.game_over:
            self._handle_player_action(action)

        self._update_score_popups()

        reward = self.pending_reward
        self.pending_reward = 0
        
        if (self.turns_remaining <= 0 or self.score >= self.WIN_SCORE) and self.animation_state == 'IDLE':
            if not self.game_over: # First time termination condition is met
                self.game_over = True
                if self.score >= self.WIN_SCORE:
                    reward += 100 # Win bonus
            terminated = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        target_type, target_index, direction = self._decode_action(movement, space_held, shift_held)

        self.selector_info = self._get_selector_info(space_held, shift_held, movement)

        if target_type is not None:
            # SFX: Play shift sound
            self.turns_remaining -= 1
            self.chain_length = 0
            self.shift_details = {'type': target_type, 'index': target_index, 'direction': direction}
            self.animation_state = 'SHIFTING'
            self.animation_progress = 0.0

    def _decode_action(self, movement, space_held, shift_held):
        if movement == 0: return None, None, None # No movement direction

        direction_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        direction = direction_map.get(movement)

        target_type = 'col' if direction in ['up', 'down'] else 'row'
        
        if not space_held and not shift_held: return None, None, None
        if space_held and not shift_held: target_index = 0
        elif not space_held and shift_held: target_index = 1
        elif space_held and shift_held: target_index = 2
        else: return None, None, None

        return target_type, target_index, direction

    def _get_selector_info(self, space_held, shift_held, movement):
        if not space_held and not shift_held: return {'type': None, 'index': 0}
        
        direction_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        direction = direction_map.get(movement)
        
        target_type = 'col' if direction in ['up', 'down'] else 'row'
        if movement == 0: target_type = None

        if space_held and not shift_held: target_index = 0
        elif not space_held and shift_held: target_index = 1
        elif space_held and shift_held: target_index = 2
        else: return {'type': None, 'index': 0}

        return {'type': target_type, 'index': target_index}


    def _update_animation(self):
        self.animation_progress += 1.0 / self.ANIMATION_SPEED[self.animation_state]
        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0 # Clamp to prevent overshooting
            
            if self.animation_state == 'SHIFTING':
                self._apply_shift_logic()
                self._check_for_matches()
            elif self.animation_state == 'MATCHING':
                self._apply_match_removal()
                self._initiate_fall()
            elif self.animation_state == 'FALLING':
                self._apply_fall_logic()
                self._check_for_matches() # Check for cascades

    def _apply_shift_logic(self):
        t = self.shift_details['type']
        idx = self.shift_details['index']
        d = self.shift_details['direction']
        
        if t == 'row':
            row = self.grid[idx, :].tolist()
            if d == 'left':
                self.grid[idx, :] = np.array(row[1:] + row[:1])
            else: # right
                self.grid[idx, :] = np.array(row[-1:] + row[:-1])
        elif t == 'col':
            col = self.grid[:, idx].tolist()
            if d == 'up':
                self.grid[:, idx] = np.array(col[1:] + col[:1])
            else: # down
                self.grid[:, idx] = np.array(col[-1:] + col[:-1])
        self.shift_details = None

    def _check_for_matches(self):
        matches = self._find_matches()
        if matches:
            # SFX: Play match anticipation sound
            self.animation_state = 'MATCHING'
            self.animation_progress = 0.0
            self.match_details = list(set(matches)) # Unique coordinates
            self.chain_length += 1
        else:
            self.animation_state = 'IDLE'
            self.chain_length = 0

    def _find_matches(self):
        matches = []
        # Check rows
        for r in range(self.GRID_SIZE):
            if self.grid[r, 0] == self.grid[r, 1] == self.grid[r, 2] != 0:
                matches.extend([(r, 0), (r, 1), (r, 2)])
        # Check columns
        for c in range(self.GRID_SIZE):
            if self.grid[0, c] == self.grid[1, c] == self.grid[2, c] != 0:
                matches.extend([(0, c), (1, c), (2, c)])
        return matches

    def _apply_match_removal(self):
        if not self.match_details: return
        
        # SFX: Play match clear sound
        score_gain = len(self.match_details)
        if self.chain_length > 1:
            score_gain += 10 * self.chain_length
        
        self.score += score_gain
        self.pending_reward += score_gain

        # Create score popups
        for r, c in self.match_details:
            x = self.grid_offset_x + c * self.CELL_SIZE + self.CELL_SIZE // 2
            y = self.grid_offset_y + r * self.CELL_SIZE + self.CELL_SIZE // 2
            self.score_popups.append({'text': f"+{score_gain}", 'pos': [x, y], 'life': 30, 'color': self.COLOR_MATCH_FLASH})

        for r, c in self.match_details:
            self.grid[r, c] = 0 # 0 represents an empty cell
    
    def _initiate_fall(self):
        self.fall_details = []
        for c in range(self.GRID_SIZE):
            empty_count = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.fall_details.append({'from': (r, c), 'to': (r + empty_count, c), 'color': self.grid[r,c]})
        
        if self.fall_details or np.any(self.grid == 0):
            self.animation_state = 'FALLING'
            self.animation_progress = 0.0
        else:
            self.animation_state = 'IDLE'
    
    def _apply_fall_logic(self):
        # Create a new grid and populate it
        new_grid = np.zeros_like(self.grid)
        for c in range(self.GRID_SIZE):
            write_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    new_grid[write_row, c] = self.grid[r, c]
                    write_row -= 1
        
        # Fill empty top cells with new random blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = self.np_random.integers(1, len(self.COLORS) + 1)
        
        self.grid = new_grid
        self.fall_details = []

    def _update_score_popups(self):
        for popup in self.score_popups[:]:
            popup['pos'][1] -= 1 # Move up
            popup['life'] -= 1
            if popup['life'] <= 0:
                self.score_popups.remove(popup)

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_grid_lines()
        self._render_selector()

        # Create a copy of the grid to render from, which can be modified for animations
        render_grid = np.copy(self.grid)
        
        # Handle SHIFTING animation
        if self.animation_state == 'SHIFTING' and self.shift_details:
            self._render_shifting_animation(render_grid)
        
        # Handle FALLING animation
        if self.animation_state == 'FALLING':
            self._render_falling_animation(render_grid)

        # Render all blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_id = render_grid[r, c]
                if color_id != 0:
                    self._render_block(r, c, color_id)

    def _render_shifting_animation(self, render_grid):
        t = self.shift_details['type']
        idx = self.shift_details['index']
        d = self.shift_details['direction']
        prog = self.animation_progress
        
        indices = list(range(self.GRID_SIZE))
        if d in ['right', 'down']:
            prog = -prog
            indices.reverse()

        if t == 'row':
            for c_idx, c in enumerate(indices):
                color_id = self.grid[idx, c]
                if color_id == 0: continue
                
                next_c = (c - 1 + self.GRID_SIZE) % self.GRID_SIZE if d == 'left' else (c + 1) % self.GRID_SIZE
                x_start = self.grid_offset_x + c * self.CELL_SIZE
                x_end = self.grid_offset_x + next_c * self.CELL_SIZE
                # Handle wrap-around screen jump
                if abs(x_end - x_start) > self.CELL_SIZE:
                    if d == 'left': x_end += self.grid_width
                    else: x_start += self.grid_width
                
                x = self._lerp(x_start, x_end, abs(prog))
                y = self.grid_offset_y + idx * self.CELL_SIZE
                self._render_block_at_pixel(x, y, color_id)
                render_grid[idx, c] = 0 # Don't draw it again
        else: # 'col'
            for r_idx, r in enumerate(indices):
                color_id = self.grid[r, idx]
                if color_id == 0: continue
                
                next_r = (r - 1 + self.GRID_SIZE) % self.GRID_SIZE if d == 'up' else (r + 1) % self.GRID_SIZE
                y_start = self.grid_offset_y + r * self.CELL_SIZE
                y_end = self.grid_offset_y + next_r * self.CELL_SIZE
                if abs(y_end - y_start) > self.CELL_SIZE:
                    if d == 'up': y_end += self.grid_width
                    else: y_start += self.grid_width

                x = self.grid_offset_x + idx * self.CELL_SIZE
                y = self._lerp(y_start, y_end, abs(prog))
                self._render_block_at_pixel(x, y, color_id)
                render_grid[r, idx] = 0

    def _render_falling_animation(self, render_grid):
        prog = self.animation_progress
        # Temporarily remove falling blocks from grid to prevent double drawing
        for fall in self.fall_details:
            r, c = fall['from']
            render_grid[r, c] = 0
        
        # Draw falling blocks at interpolated positions
        for fall in self.fall_details:
            from_r, from_c = fall['from']
            to_r, to_c = fall['to']
            color_id = fall['color']
            
            y_start = self.grid_offset_y + from_r * self.CELL_SIZE
            y_end = self.grid_offset_y + to_r * self.CELL_SIZE
            x = self.grid_offset_x + to_c * self.CELL_SIZE
            y = self._lerp(y_start, y_end, prog)
            self._render_block_at_pixel(x, y, color_id)

        # Draw new blocks appearing from top
        for c in range(self.GRID_SIZE):
            empty_count = sum(1 for r in range(self.GRID_SIZE) if self.grid[r, c] == 0)
            for r in range(empty_count):
                # Use a consistent pseudo-random choice for rendering new blocks during animation
                # This ensures the color doesn't flicker between frames
                temp_rng = np.random.default_rng(seed=(self.steps, c, r))
                color_id = temp_rng.choice(list(self.COLORS.keys()))
                y_start = self.grid_offset_y + (r - empty_count) * self.CELL_SIZE
                y_end = self.grid_offset_y + r * self.CELL_SIZE
                x = self.grid_offset_x + c * self.CELL_SIZE
                y = self._lerp(y_start, y_end, prog)
                self._render_block_at_pixel(x, y, color_id)


    def _render_block(self, r, c, color_id):
        x = self.grid_offset_x + c * self.CELL_SIZE
        y = self.grid_offset_y + r * self.CELL_SIZE
        self._render_block_at_pixel(x, y, color_id)

    def _render_block_at_pixel(self, x, y, color_id):
        color = self.COLORS[color_id]
        
        # Pulsing flash for matched blocks
        if self.animation_state == 'MATCHING' and self._is_matching(self._pixel_to_grid(x, y)):
            pulse = abs(math.sin(self.animation_progress * math.pi * 4))
            color = tuple(int(self._lerp(c, m, pulse)) for c, m in zip(color, self.COLOR_MATCH_FLASH))
        
        block_rect = pygame.Rect(
            int(x + (self.CELL_SIZE - self.BLOCK_SIZE) / 2),
            int(y + (self.CELL_SIZE - self.BLOCK_SIZE) / 2),
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, color, block_rect, border_radius=8)
        # Add a subtle highlight
        highlight_color = tuple(min(255, c+30) for c in color)
        pygame.draw.rect(self.screen, highlight_color, block_rect.inflate(-10,-10), border_radius=6)

    def _pixel_to_grid(self, px, py):
        c = (px - self.grid_offset_x + self.CELL_SIZE / 2) // self.CELL_SIZE
        r = (py - self.grid_offset_y + self.CELL_SIZE / 2) // self.CELL_SIZE
        return int(r), int(c)

    def _is_matching(self, pos):
        return pos in self.match_details

    def _render_grid_lines(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.CELL_SIZE, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.CELL_SIZE, self.grid_offset_y + self.grid_width)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.CELL_SIZE)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)

    def _render_selector(self):
        if self.animation_state != 'IDLE' or self.selector_info['type'] is None:
            return
        
        sel_type = self.selector_info['type']
        sel_index = self.selector_info['index']

        s = pygame.Surface((self.grid_width, self.grid_width), pygame.SRCALPHA)
        
        if sel_type == 'row':
            rect = pygame.Rect(0, sel_index * self.CELL_SIZE, self.grid_width, self.CELL_SIZE)
        elif sel_type == 'col':
            rect = pygame.Rect(sel_index * self.CELL_SIZE, 0, self.CELL_SIZE, self.grid_width)
        else:
            return
            
        pygame.draw.rect(s, self.COLOR_SELECTOR, rect, border_radius=10)
        self.screen.blit(s, (self.grid_offset_x, self.grid_offset_y))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Turns
        turns_text = self.font_medium.render(f"Turns: {self.turns_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (self.WIDTH - turns_text.get_width() - 20, 20))

        # Score popups
        for popup in self.score_popups:
            alpha = max(0, min(255, popup['life'] * 255 / 15)) # Fade out
            popup_text = self.font_small.render(popup['text'], True, popup['color'])
            popup_text.set_alpha(alpha)
            text_rect = popup_text.get_rect(center=popup['pos'])
            self.screen.blit(popup_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_GAMEOVER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turns_remaining": self.turns_remaining,
            "is_animating": self.animation_state != 'IDLE'
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    # Set a non-dummy driver for interactive mode
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    env.validate_implementation()
    
    # To run interactively
    interactive_mode = True
    if interactive_mode:
        pygame.display.set_caption("GridShift Puzzle")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
        running = True
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Continuous action polling for smooth interaction
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            # Step the environment. Because auto_advance is True, we call this every frame.
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward > 0:
                print(f"Reward: {reward}, Score: {info['score']}")

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(3000)
                obs, info = env.reset()

            clock.tick(30) # Run at 30 FPS
            
        env.close()

    else: # For testing the API
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()