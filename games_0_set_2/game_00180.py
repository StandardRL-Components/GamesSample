
# Generated: 2025-08-27T12:50:54.827981
# Source Brief: brief_00180.md
# Brief Index: 180

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class FeedbackAnimation:
    """A simple class for a temporary visual effect."""
    def __init__(self, grid_pos, grid_offset, cell_size, color, duration=15):
        self.cx = grid_offset[0] + (grid_pos[0] + 0.5) * cell_size
        self.cy = grid_offset[1] + (grid_pos[1] + 0.5) * cell_size
        self.color = color
        self.duration = duration
        self.max_duration = duration
        self.max_radius = cell_size * 0.7

    def update(self):
        self.duration -= 1
        return self.duration > 0

    def draw(self, surface):
        progress = (self.max_duration - self.duration) / self.max_duration
        current_radius = int(self.max_radius * math.sin(progress * math.pi))
        alpha = int(255 * (1 - progress))
        
        # Create a temporary surface for transparency
        temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.color + (alpha,), (current_radius, current_radius), current_radius)
        surface.blit(temp_surf, (self.cx - current_radius, self.cy - current_radius))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle the selected color. Press Space to fill the selected square."
    )

    game_description = (
        "A pixel art puzzle game. Fill the grid with the correct colors to recreate a hidden image. You have a limited number of mistakes!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self._define_colors_and_fonts()
        self._define_game_constants()

        self.player_grid = None
        self.target_grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mistakes = 0
        self.cursor_pos = [0, 0]
        self.selected_color_index = 0
        self.last_shift_state = False
        self.last_space_state = False
        self.feedback_animations = []
        
        self.reset()
        self.validate_implementation()

    def _define_colors_and_fonts(self):
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 80, 80)
        self.COLOR_CORRECT = (80, 255, 80)
        self.COLOR_INCORRECT = (255, 80, 80)
        
        self.GAME_COLORS = [
            (224, 58, 62),   # Red
            (62, 137, 224),  # Blue
            (62, 224, 114),  # Green
            (234, 222, 64),  # Yellow
            (153, 62, 224),  # Purple
        ]
        
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_GAME_OVER = pygame.font.Font(None, 64)

    def _define_game_constants(self):
        self.GRID_SIZE = 10
        self.MAX_MISTAKES = 10
        self.MAX_STEPS = 1000
        self.WIN_ACCURACY_THRESHOLD = 0.90
        
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.width - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.height - self.GRID_HEIGHT) // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.mistakes = 0
        self.game_over = False
        
        self.target_grid = self.np_random.integers(0, len(self.GAME_COLORS), size=(self.GRID_SIZE, self.GRID_SIZE))
        self.player_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=int)
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_index = 0
        
        self.last_shift_state = False
        self.last_space_state = False
        
        self.feedback_animations = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_binary, shift_binary = action
        space_held = space_binary == 1
        shift_held = shift_binary == 1
        
        reward = 0
        
        if not self.game_over:
            # Handle input
            self._handle_movement(movement)
            self._handle_color_cycle(shift_held)
            action_taken, correct_fill = self._handle_fill(space_held)

            # Calculate immediate reward
            if action_taken:
                if correct_fill:
                    reward = 1
                    self.score += 1
                else:
                    reward = -1
                    self.score -= 1
                    self.mistakes += 1
        
            self.steps += 1
            
            # Update last states for press detection
            self.last_shift_state = shift_held
            self.last_space_state = space_held
        
        # Update animations regardless of game over state
        self._update_animations()

        # Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            reward += self._calculate_terminal_reward()
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        # Wraparound
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE
    
    def _handle_color_cycle(self, shift_held):
        is_shift_press = shift_held and not self.last_shift_state
        if is_shift_press:
            self.selected_color_index = (self.selected_color_index + 1) % len(self.GAME_COLORS)
            # sfx: color_cycle.wav
            
    def _handle_fill(self, space_held):
        is_space_press = space_held and not self.last_space_state
        if not is_space_press:
            return False, False

        cx, cy = self.cursor_pos
        if self.player_grid[cy, cx] != -1:
            # sfx: already_filled_error.wav
            return False, False # Already filled
            
        target_color = self.target_grid[cy, cx]
        player_color = self.selected_color_index
        self.player_grid[cy, cx] = player_color
        
        correct = (target_color == player_color)
        if correct:
            # sfx: correct_fill.wav
            anim_color = self.COLOR_CORRECT
        else:
            # sfx: incorrect_fill.wav
            anim_color = self.COLOR_INCORRECT
            
        self.feedback_animations.append(
            FeedbackAnimation(self.cursor_pos, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y), self.CELL_SIZE, anim_color)
        )
        return True, correct

    def _update_animations(self):
        self.feedback_animations = [anim for anim in self.feedback_animations if anim.update()]

    def _calculate_accuracy(self):
        filled_mask = self.player_grid != -1
        num_filled = np.sum(filled_mask)
        if num_filled == 0:
            return 1.0
        
        correct_mask = self.player_grid[filled_mask] == self.target_grid[filled_mask]
        num_correct = np.sum(correct_mask)
        return num_correct / num_filled

    def _check_termination(self):
        if self.mistakes >= self.MAX_MISTAKES:
            return True
        if np.all(self.player_grid != -1):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _calculate_terminal_reward(self):
        if self.mistakes >= self.MAX_MISTAKES:
            # sfx: game_lose.wav
            return -100
        
        if np.all(self.player_grid != -1):
            final_accuracy = self._calculate_accuracy()
            if final_accuracy >= self.WIN_ACCURACY_THRESHOLD:
                # sfx: game_win.wav
                return 100
            else:
                # sfx: game_lose.wav
                return -100
        return 0 # Timeout

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw target hints and player fills
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                
                if self.player_grid[y, x] == -1:
                    # Draw faint hint of target color
                    target_color = self.GAME_COLORS[self.target_grid[y, x]]
                    hint_color = tuple(c // 4 for c in target_color)
                    hint_rect = pygame.Rect(rect.x + rect.width // 4, rect.y + rect.height // 4, rect.width // 2, rect.height // 2)
                    pygame.draw.rect(self.screen, hint_color, hint_rect)
                else:
                    # Draw the player's filled color
                    color = self.GAME_COLORS[self.player_grid[y, x]]
                    pygame.draw.rect(self.screen, color, rect)

        # Draw animations
        for anim in self.feedback_animations:
            anim.draw(self.screen)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            
        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Draw color palette
        palette_x = 20
        palette_y = 20
        for i, color in enumerate(self.GAME_COLORS):
            rect = pygame.Rect(palette_x + i * 40, palette_y, 30, 30)
            pygame.draw.rect(self.screen, color, rect)
            if i == self.selected_color_index:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)
        
        # Draw score
        score_text = self.FONT_UI.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 20, 20))
        
        # Draw accuracy
        accuracy = self._calculate_accuracy()
        acc_text = self.FONT_UI.render(f"Accuracy: {accuracy:.1f}%", True, self.COLOR_TEXT)
        self.screen.blit(acc_text, (self.width - acc_text.get_width() - 20, 45))

        # Draw mistakes (hearts)
        mistakes_left = self.MAX_MISTAKES - self.mistakes
        for i in range(mistakes_left):
            self._draw_heart(self.screen, self.width - 30 - i * 25, 80, 20, self.COLOR_HEART)

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            final_accuracy = self._calculate_accuracy()
            if self.mistakes >= self.MAX_MISTAKES or (np.all(self.player_grid != -1) and final_accuracy < self.WIN_ACCURACY_THRESHOLD):
                msg = "GAME OVER"
                color = self.COLOR_INCORRECT
            else: # Win condition
                msg = "YOU WIN!"
                color = self.COLOR_CORRECT
            
            text_surf = self.FONT_GAME_OVER.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_heart(self, surface, x, y, size, color):
        """Draws a heart shape using polygons."""
        points = [
            (x, y + size // 4),
            (x - size // 2, y - size // 4),
            (x - size // 2, y + size // 2),
            (x, y + size),
            (x + size // 2, y + size // 2),
            (x + size // 2, y - size // 4),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mistakes": self.mistakes,
            "accuracy": self._calculate_accuracy(),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Pixel Art Painter")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # Since auto_advance is False, we only step when an action is taken
        # For a better human play experience, we'll step continuously
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward}, Score: {info['score']}, Mistakes: {info['mistakes']}")
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Accuracy: {info['accuracy']:.2f}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate for human play

    env.close()