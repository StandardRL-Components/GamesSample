
# Generated: 2025-08-28T04:08:34.663623
# Source Brief: brief_05156.md
# Brief Index: 5156

        
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
        "Controls: ←→ to select a column, ↑↓ to select a ball. "
        "Press space to pick up the selected ball. "
        "Select a destination column and press space again to place it."
    )

    game_description = (
        "A minimalist puzzle game. Sort the colored balls into their matching "
        "columns before you run out of moves. Plan your sequence carefully to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    COLOR_BG = (25, 28, 40)
    COLOR_COLUMN = (50, 55, 70)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_BALL_CURSOR = (255, 255, 0)
    
    BALL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
    ]
    
    NUM_COLUMNS = 4
    BALLS_PER_COLOR = 4
    MAX_COLUMN_HEIGHT = 6
    TOTAL_MOVES = 50
    MAX_STEPS = 1000

    BALL_RADIUS = 20
    COLUMN_WIDTH = 80
    COLUMN_SPACING = 40
    
    ANIMATION_SPEED = 0.1  # Progress per step

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 60)
        
        self.columns = []
        self.cursor_col = 0
        self.cursor_row = 0
        self.held_ball = None
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.animations = []
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.TOTAL_MOVES
        self.held_ball = None
        self.cursor_col = 0
        self.cursor_row = 0
        self.animations = []
        self.particles = []

        all_balls = [i for i in range(self.NUM_COLUMNS) for _ in range(self.BALLS_PER_COLOR)]
        self.np_random.shuffle(all_balls)
        
        self.columns = [[] for _ in range(self.NUM_COLUMNS)]
        for ball_color_idx in all_balls:
            placed = False
            # Attempt to place in a non-full column
            for i in range(self.NUM_COLUMNS):
                if len(self.columns[i]) < self.MAX_COLUMN_HEIGHT:
                    self.columns[i].append(ball_color_idx)
                    placed = True
                    break
            if not placed:
                # Fallback if initial random distribution is weird
                self.columns[self.np_random.integers(0, self.NUM_COLUMNS)].append(ball_color_idx)

        self._clamp_cursors()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0
        terminated = False

        self._update_animations_and_particles()

        # Only process new actions if no major animation is running
        if not any(anim['type'] == 'move_ball' for anim in self.animations):
            move_result = self._handle_input(movement, space_held)
            reward = self._calculate_reward(move_result)
        
        self.score += reward
        
        if not self.game_over:
            is_win = self._check_win_condition()
            is_loss = self.moves_left <= 0
            is_timeout = self.steps >= self.MAX_STEPS
            
            if is_win or is_loss or is_timeout:
                terminated = True
                self.game_over = True
                self.win_state = is_win
                if is_win:
                    reward += 100
                    # // SFX: win_fanfare
                else:
                    reward -= 100
                    # // SFX: loss_sound
                self.score += 100 if is_win else -100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        move_result = {"type": "none"}
        
        # --- Cursor Movement ---
        if movement in [1, 2, 3, 4]:  # up, down, left, right
            if movement == 1: self.cursor_row += 1  # Up
            if movement == 2: self.cursor_row -= 1  # Down
            if movement == 3: self.cursor_col = (self.cursor_col - 1 + self.NUM_COLUMNS) % self.NUM_COLUMNS # Left
            if movement == 4: self.cursor_col = (self.cursor_col + 1) % self.NUM_COLUMNS # Right
            self._clamp_cursors()

        # --- Action Button (Space) ---
        if space_held:
            if self.held_ball is None:
                # Pick up a ball
                current_col_stack = self.columns[self.cursor_col]
                if self.cursor_row < len(current_col_stack):
                    self.held_ball = current_col_stack.pop(self.cursor_row)
                    self._clamp_cursors()
                    move_result = {"type": "pickup"}
                    # // SFX: pickup_pop
            else:
                # Place a ball
                target_col_stack = self.columns[self.cursor_col]
                potential_before = self._calculate_potential()

                # Rule: Can only place in a column of the same color, or an empty column.
                can_place = (
                    len(target_col_stack) == 0 or
                    (len(target_col_stack) > 0 and target_col_stack[0] == self.held_ball)
                )

                if len(target_col_stack) < self.MAX_COLUMN_HEIGHT and can_place:
                    # Valid move
                    start_pos = self._get_held_ball_pos()
                    target_col_stack.append(self.held_ball)
                    end_pos = self._get_ball_pos(self.cursor_col, len(target_col_stack) - 1)
                    
                    self.animations.append({
                        "type": "move_ball", "color_idx": self.held_ball,
                        "start": start_pos, "end": end_pos, "progress": 0.0
                    })
                    if self.held_ball == self.cursor_col:
                        # // SFX: place_correct_chime
                        self._create_particles(end_pos, self.BALL_COLORS[self.held_ball])
                    else:
                        # // SFX: place_normal_click
                        pass

                    self.held_ball = None
                    self.moves_left -= 1
                    
                    potential_after = self._calculate_potential()
                    move_result = {
                        "type": "valid_move",
                        "potential_change": potential_after - potential_before
                    }
                else:
                    # Invalid move
                    self.animations.append({"type": "shake", "duration": 0.5, "magnitude": 5})
                    move_result = {"type": "invalid_move"}
                    # // SFX: error_buzz
        return move_result

    def _calculate_reward(self, move_result):
        if move_result["type"] == "valid_move":
            return move_result["potential_change"]
        if move_result["type"] == "invalid_move":
            return -0.1
        return 0
    
    def _calculate_potential(self):
        potential = 0
        for i, col in enumerate(self.columns):
            if not col: continue
            is_sorted = all(ball_idx == i for ball_idx in col)
            if is_sorted:
                potential += len(col)
        return potential

    def _check_win_condition(self):
        for i, col in enumerate(self.columns):
            if not col:
                continue
            if len(col) != self.BALLS_PER_COLOR:
                return False
            if not all(ball_idx == i for ball_idx in col):
                return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_win": self.win_state
        }

    def _render_game(self):
        self._render_columns()
        self._render_balls_in_stacks()
        if not self.game_over:
            self._render_cursors()
            self._render_held_ball()
        self._render_animations_and_particles()

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (20, 20))
        
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win_state else "GAME OVER"
            end_text_color = (100, 255, 100) if self.win_state else (255, 100, 100)
            end_text = self.font_title.render(end_text_str, True, end_text_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _render_columns(self):
        total_width = self.NUM_COLUMNS * self.COLUMN_WIDTH + (self.NUM_COLUMNS - 1) * self.COLUMN_SPACING
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        
        for i in range(self.NUM_COLUMNS):
            x = start_x + i * (self.COLUMN_WIDTH + self.COLUMN_SPACING)
            y = self.SCREEN_HEIGHT - 20 - self.MAX_COLUMN_HEIGHT * (self.BALL_RADIUS * 2)
            h = self.MAX_COLUMN_HEIGHT * (self.BALL_RADIUS * 2)
            
            # Column base color
            col_rect = pygame.Rect(x, y, self.COLUMN_WIDTH, h)
            pygame.draw.rect(self.screen, self.COLOR_COLUMN, col_rect, border_radius=5)
            
            # Goal color indicator
            indicator_rect = pygame.Rect(x, self.SCREEN_HEIGHT - 15, self.COLUMN_WIDTH, 10)
            pygame.draw.rect(self.screen, self.BALL_COLORS[i], indicator_rect, border_radius=3)

    def _render_balls_in_stacks(self):
        for col_idx, col in enumerate(self.columns):
            for row_idx, ball_color_idx in enumerate(col):
                pos = self._get_ball_pos(col_idx, row_idx)
                self._draw_ball(pos, self.BALL_COLORS[ball_color_idx])

    def _render_cursors(self):
        # Column Cursor
        col_pos = self._get_ball_pos(self.cursor_col, 0)
        col_x = col_pos[0] - self.COLUMN_WIDTH / 2
        col_y = self.SCREEN_HEIGHT - 20 - self.MAX_COLUMN_HEIGHT * (self.BALL_RADIUS * 2)
        col_h = self.MAX_COLUMN_HEIGHT * (self.BALL_RADIUS * 2)
        
        cursor_surface = pygame.Surface((self.COLUMN_WIDTH, col_h), pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, (int(col_x), int(col_y)))

        # Ball Cursor
        if self.held_ball is None and self.cursor_row < len(self.columns[self.cursor_col]):
            pos = self._get_ball_pos(self.cursor_col, self.cursor_row)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS + 3, self.COLOR_BALL_CURSOR)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS + 4, self.COLOR_BALL_CURSOR)

    def _render_held_ball(self):
        if self.held_ball is not None:
            pos = self._get_held_ball_pos()
            shake_offset = (0, 0)
            for anim in self.animations:
                if anim['type'] == 'shake':
                    angle = self.np_random.random() * 2 * math.pi
                    shake_offset = (math.cos(angle) * anim['magnitude'], math.sin(angle) * anim['magnitude'])
            
            final_pos = (pos[0] + shake_offset[0], pos[1] + shake_offset[1])
            self._draw_ball(final_pos, self.BALL_COLORS[self.held_ball])

    def _update_animations_and_particles(self):
        # Animations
        for anim in self.animations[:]:
            if anim['type'] in ['move_ball']:
                anim['progress'] += self.ANIMATION_SPEED
                if anim['progress'] >= 1.0:
                    self.animations.remove(anim)
            elif anim['type'] == 'shake':
                anim['duration'] -= 1/30.0 # Assuming 30fps for step-based time
                if anim['duration'] <= 0:
                    self.animations.remove(anim)
        # Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_animations_and_particles(self):
        # Render moving balls
        for anim in self.animations:
            if anim['type'] == 'move_ball':
                progress = min(1.0, anim['progress'])
                # Ease-out curve
                eased_progress = 1 - pow(1 - progress, 3)
                curr_x = anim['start'][0] + (anim['end'][0] - anim['start'][0]) * eased_progress
                curr_y = anim['start'][1] + (anim['end'][1] - anim['start'][1]) * eased_progress
                self._draw_ball((curr_x, curr_y), self.BALL_COLORS[anim['color_idx']])
        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                rect = pygame.Rect(p['pos'][0] - size/2, p['pos'][1] - size/2, size, size)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), border_radius=int(size/2))
                self.screen.blit(shape_surf, rect)

    # --- Utility Methods ---
    def _clamp_cursors(self):
        self.cursor_col %= self.NUM_COLUMNS
        max_row = len(self.columns[self.cursor_col])
        if max_row == 0:
            self.cursor_row = 0
        else:
            self.cursor_row = np.clip(self.cursor_row, 0, max_row - 1)

    def _get_ball_pos(self, col, row):
        total_width = self.NUM_COLUMNS * self.COLUMN_WIDTH + (self.NUM_COLUMNS - 1) * self.COLUMN_SPACING
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        
        center_x = start_x + col * (self.COLUMN_WIDTH + self.COLUMN_SPACING) + self.COLUMN_WIDTH / 2
        bottom_y = self.SCREEN_HEIGHT - 20 - self.BALL_RADIUS
        center_y = bottom_y - row * (self.BALL_RADIUS * 2)
        return (center_x, center_y)

    def _get_held_ball_pos(self):
        col_pos = self._get_ball_pos(self.cursor_col, 0)
        return (col_pos[0], 80)

    def _draw_ball(self, pos, color):
        x, y = int(pos[0]), int(pos[1])
        # Main ball color
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, color)
        # Highlight for 3D effect
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.filled_circle(self.screen, x - 6, y - 6, self.BALL_RADIUS // 3, highlight_color)
        pygame.gfxdraw.aacircle(self.screen, x - 6, y - 6, self.BALL_RADIUS // 3, highlight_color)
        
    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'size': self.np_random.random() * 5 + 3,
            })

    def validate_implementation(self):
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
    # It's not part of the Gymnasium environment but is useful for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Sorter")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0  # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = (movement, space_held, shift_held)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
        
        # In a real game loop, you'd only step when a key is pressed for a turn-based game
        # For this interactive test, we'll step on keydown to feel more responsive.
        # We need a latch to only process one action per key press for the main actions
        
        # Simplified for testing: we just pass the raw state. The env handles the logic.
        # The env is auto_advance=False, so we must call step() to see changes.
        # To make it playable, we only step on a key press.
        
        # A better playable loop for auto_advance=False
        action_taken = False
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                action_taken = True
            if event.key == pygame.K_r:
                obs, info = env.reset()
                action_taken = False # don't step on reset
            if event.key == pygame.K_q:
                running = False
                
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}")
            if terminated:
                print("Game Over!")

        # Draw the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    pygame.quit()