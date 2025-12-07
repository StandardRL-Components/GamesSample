
# Generated: 2025-08-27T17:06:57.545858
# Source Brief: brief_01430.md
# Brief Index: 1430

        
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
        "Controls: Arrows to move the selector. Space to clear horizontally, Shift to clear vertically."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect adjacent same-colored blocks in a grid to clear lines and achieve the highest score before the board fills up."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 12
    GRID_HEIGHT = 9
    CELL_SIZE = 40
    NUM_COLORS = 5
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_TEXT = (220, 220, 230)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.combo = 0
        self.particles = []
        self.last_action_feedback = ""
        
        self.grid_offset_x = (640 - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.grid_offset_y = 400 - self.GRID_HEIGHT * self.CELL_SIZE

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.combo = 0
        self.game_over = False
        self.victory = False
        self.particles = []
        self.last_action_feedback = ""
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._generate_board()
        while not self._has_valid_moves():
            self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.last_action_feedback = ""

        # --- Action Handling ---
        clear_action = space_pressed or shift_pressed
        if clear_action:
            reward = self._handle_clear_action(space_pressed)
        else:
            self._move_cursor(movement)
            # Small penalty for non-clearing moves to encourage action
            reward = -0.01

        self._update_particles()
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if not self._has_valid_moves():
            self.game_over = True
            self.victory = False
            terminated = True
            reward -= 50 # Loss penalty
            self.last_action_feedback = "GAME OVER"
        elif np.sum(self.grid) == 0:
            self.game_over = True
            self.victory = True
            terminated = True
            reward += 100 # Victory bonus
            self.last_action_feedback = "YOU WIN!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.last_action_feedback = "TIME UP"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

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
            "combo": self.combo,
            "victory": self.victory,
        }

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _handle_clear_action(self, is_horizontal):
        cx, cy = self.cursor_pos
        color_to_match = self.grid[cx, cy]
        
        if color_to_match == 0:
            self.last_action_feedback = "Empty Cell"
            self.combo = 0
            return -0.2 # Penalty for trying to clear empty space

        if is_horizontal:
            to_clear = self._find_contiguous_blocks(cx, cy, True)
        else:
            to_clear = self._find_contiguous_blocks(cx, cy, False)

        if len(to_clear) < 2:
            self.last_action_feedback = "Invalid Move"
            self.combo = 0
            return -0.2 # Penalty for clearing less than 2 blocks

        # --- Successful Clear ---
        self.combo += 1
        num_cleared = len(to_clear)
        
        # Base reward
        reward = num_cleared
        
        # Penalty/Bonus based on count
        if num_cleared < 3:
            reward -= 0.2
        if num_cleared >= 4:
            reward += 5
        
        # Combo bonus
        reward += self.combo * 0.5

        self.last_action_feedback = f"CLEAR! +{num_cleared}"
        if self.combo > 1:
            self.last_action_feedback = f"COMBO x{self.combo}!"
        
        # Clear blocks and create particles
        for x, y in to_clear:
            self.grid[x, y] = 0
            self._create_particles((x, y), self.BLOCK_COLORS[color_to_match - 1])
        
        # Check for full line clears
        reward += self._check_line_clears()
        
        # Apply gravity and refill
        self._apply_gravity_and_refill()
        
        self.score += reward
        return reward

    def _find_contiguous_blocks(self, start_x, start_y, is_horizontal):
        color = self.grid[start_x, start_y]
        if color == 0:
            return []
        
        line = []
        if is_horizontal:
            # Scan left
            for x in range(start_x, -1, -1):
                if self.grid[x, start_y] == color:
                    line.append((x, start_y))
                else:
                    break
            # Scan right
            for x in range(start_x + 1, self.GRID_WIDTH):
                if self.grid[x, start_y] == color:
                    line.append((x, start_y))
                else:
                    break
        else: # Vertical
            # Scan up
            for y in range(start_y, -1, -1):
                if self.grid[start_x, y] == color:
                    line.append((start_x, y))
                else:
                    break
            # Scan down
            for y in range(start_y + 1, self.GRID_HEIGHT):
                if self.grid[start_x, y] == color:
                    line.append((start_x, y))
                else:
                    break
        return list(set(line))

    def _check_line_clears(self):
        bonus = 0
        # Check columns
        for x in range(self.GRID_WIDTH):
            if np.all(self.grid[x, :] == 0):
                bonus += 10
        # Check rows
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y] == 0):
                bonus += 10
        if bonus > 0:
            self.last_action_feedback = "LINE CLEAR!"
        return bonus

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[x, y + empty_count] = self.grid[x, y]
                    self.grid[x, y] = 0
            # Refill top
            for y in range(empty_count):
                self.grid[x, y] = self.np_random.integers(1, self.NUM_COLORS + 1)
        # sfx: block_fall.wav

    def _move_cursor(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
        # sfx: cursor_move.wav

    def _has_valid_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 0:
                    continue
                # Check horizontal
                if x < self.GRID_WIDTH - 1 and self.grid[x, y] == self.grid[x + 1, y]:
                    return True
                # Check vertical
                if y < self.GRID_HEIGHT - 1 and self.grid[x, y] == self.grid[x, y + 1]:
                    return True
        return False
    
    # --- Rendering ---

    def _render_game(self):
        self._draw_grid()
        self._draw_blocks()
        self._draw_cursor()
        self._draw_particles()

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.grid_offset_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.grid_offset_y), (px, 400))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.grid_offset_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, py), (self.grid_offset_x + self.GRID_WIDTH * self.CELL_SIZE, py))

    def _draw_blocks(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_idx = self.grid[x, y]
                if color_idx > 0:
                    px = self.grid_offset_x + x * self.CELL_SIZE
                    py = self.grid_offset_y + y * self.CELL_SIZE
                    color = self.BLOCK_COLORS[color_idx - 1]
                    
                    rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                    
                    # Draw block with a subtle 3D effect
                    highlight = tuple(min(255, c + 40) for c in color)
                    shadow = tuple(max(0, c - 40) for c in color)
                    
                    pygame.draw.rect(self.screen, shadow, rect.move(2, 2))
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, highlight, rect.inflate(-self.CELL_SIZE*0.8, -self.CELL_SIZE*0.8).move(-2,-2))


    def _draw_cursor(self):
        if self.game_over: return
        cx, cy = self.cursor_pos
        px = self.grid_offset_x + cx * self.CELL_SIZE
        py = self.grid_offset_y + cy * self.CELL_SIZE
        
        # Pulsing effect for the cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        width = int(2 + pulse * 2)

        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, width, border_radius=4)
        
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Combo
        if self.combo > 1:
            combo_text = self.font_small.render(f"Combo: x{self.combo}", True, self.BLOCK_COLORS[3])
            self.screen.blit(combo_text, (15, 40))

        # Last action feedback
        if self.last_action_feedback and not self.game_over:
            feedback_text = self.font_small.render(self.last_action_feedback, True, self.COLOR_TEXT)
            text_rect = feedback_text.get_rect(center=(640 // 2, 25))
            self.screen.blit(feedback_text, text_rect)

        # Game Over / Victory text
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text_str = "GAME OVER"
            if self.victory:
                end_text_str = "VICTORY!"
            
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(640 // 2, 400 // 2 - 20))
            self.screen.blit(end_text, text_rect)

            final_score_text = self.font_main.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(640 // 2, 400 // 2 + 30))
            self.screen.blit(final_score_text, score_rect)

    # --- Particle System ---
    def _create_particles(self, pos, color):
        # sfx: block_clear.wav
        px = self.grid_offset_x + pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.grid_offset_y + pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 5)
            if radius < 1: continue
            
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(life_ratio * 255)
            color = p['color']
            
            # Use gfxdraw for anti-aliased circles
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (color[0], color[1], color[2], alpha))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (color[0], color[1], color[2], alpha))

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Block Clear Game")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, no space, no shift
    
    while not done:
        # --- Event Handling (for manual play) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Reset action on key up
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0 # No movement
                if event.key == pygame.K_SPACE:
                    action[1] = 0 # Space released
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 0 # Shift released
            
            # Set action on key down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                
                # These actions are mutually exclusive for a better play experience
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1

        # --- Step the environment ---
        # For this turn-based game, we only step when an action is taken.
        # A real agent would decide when to act. For manual play, we act on every key press.
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")
            
            # Reset the action after it's been processed for turn-based control
            action = [0, 0, 0]
            
        # --- Rendering ---
        # The observation is already a rendered frame
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for manual play

    env.close()