
# Generated: 2025-08-27T16:22:02.283508
# Source Brief: brief_00017.md
# Brief Index: 17

        
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
        "Controls: Arrows to move cursor. Space to select letters. Shift to submit word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a grid against the clock. Select letters with Space and submit with Shift."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_DIM = 12
        self.GRID_TOP_LEFT = (120, 20)
        self.CELL_SIZE = 30
        self.GRID_WIDTH = self.GRID_HEIGHT = self.GRID_DIM * self.CELL_SIZE
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

        # Word bank
        self.WORD_BANK = [
            "PYTHON", "AGENT", "REWARD", "STATE", "ACTION", "POLICY", "GYM", 
            "LEARN", "DEEP", "GRID", "SEARCH", "PUZZLE", "VECTOR", "TENSOR", 
            "MODEL", "GAME", "FRAME", "PIXEL", "CODE", "SOLVE", "TASK"
        ]
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 20)
        except pygame.error:
            # Fallback if default font is not found (e.g., in minimal docker)
            self.font_large = pygame.font.SysFont("sans-serif", 36)
            self.font_medium = pygame.font.SysFont("sans-serif", 24)
            self.font_small = pygame.font.SysFont("sans-serif", 20)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINE = (40, 60, 80)
        self.COLOR_LETTER = (200, 220, 240)
        self.COLOR_LETTER_FOUND = (80, 100, 120)
        self.COLOR_CURSOR = (255, 200, 0, 100) # RGBA
        self.COLOR_SELECTION = (255, 165, 0, 180) # RGBA
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_WORD_FOUND = (0, 255, 127)
        self.COLOR_WORD_UNFOUND = (150, 160, 170)
        self.COLOR_FEEDBACK_GOOD = (0, 255, 127)
        self.COLOR_FEEDBACK_BAD = (255, 80, 80)

        # Initialize state variables
        self.grid = None
        self.words_to_find = None
        self.word_locations = None
        self.cursor_pos = None
        self.current_selection = None
        self.found_words = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.feedback_message = None
        self.feedback_timer = None
        self.feedback_color = None
        self.particles = None
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

        self.reset()
        self.validate_implementation()
    
    def _generate_puzzle(self):
        self.grid = np.full((self.GRID_DIM, self.GRID_DIM), ' ', dtype=str)
        
        # Filter word bank for suitable words
        possible_words = [w for w in self.WORD_BANK if 3 <= len(w) <= self.GRID_DIM]
        
        # Select 8-10 words
        num_words = self.np_random.integers(8, 11)
        words_for_puzzle = self.np_random.choice(possible_words, size=num_words, replace=False).tolist()
        
        self.words_to_find = sorted(words_for_puzzle, key=len, reverse=True)
        self.word_locations = {}

        directions = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]

        for word in self.words_to_find:
            placed = False
            for _ in range(100): # 100 placement attempts
                self.np_random.shuffle(directions)
                direction = directions[0]
                start_r = self.np_random.integers(0, self.GRID_DIM)
                start_c = self.np_random.integers(0, self.GRID_DIM)
                
                end_r = start_r + (len(word) - 1) * direction[0]
                end_c = start_c + (len(word) - 1) * direction[1]

                if not (0 <= end_r < self.GRID_DIM and 0 <= end_c < self.GRID_DIM):
                    continue

                can_place = True
                path = []
                for i in range(len(word)):
                    r, c = start_r + i * direction[0], start_c + i * direction[1]
                    path.append((r, c))
                    if self.grid[r, c] != ' ' and self.grid[r, c] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word)):
                        r, c = path[i]
                        self.grid[r, c] = word[i]
                    self.word_locations[word] = path
                    placed = True
                    break
        
        # Fill empty cells with random letters
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                if self.grid[r, c] == ' ':
                    self.grid[r, c] = chr(self.np_random.integers(65, 91)) # A-Z

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_puzzle()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.current_selection = []
        self.found_words = set()
        self.feedback_message = ""
        self.feedback_timer = 0
        self.feedback_color = self.COLOR_UI_TEXT
        self.particles = []
        self.space_pressed_last_frame = True
        self.shift_pressed_last_frame = True
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.space_pressed_last_frame
        shift_press = shift_held and not self.shift_pressed_last_frame
        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held
        
        # 1. Handle Movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Up
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_DIM - 1, self.cursor_pos[0] + 1) # Down
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Left
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_DIM - 1, self.cursor_pos[1] + 1) # Right
        
        # 2. Handle Selection (Space)
        if space_press:
            pos = tuple(self.cursor_pos)
            if pos in self.current_selection:
                self.current_selection.remove(pos)
                # sfx: deselect_letter
            else:
                self.current_selection.append(pos)
                # sfx: select_letter
            reward -= 0.01 # Small cost for any action

        # 3. Handle Submission (Shift)
        if shift_press and self.current_selection:
            selected_word_str = "".join([self.grid[r, c] for r, c in self.current_selection])
            is_correct = False
            
            if selected_word_str in self.words_to_find and selected_word_str not in self.found_words:
                is_correct = True
            elif selected_word_str[::-1] in self.words_to_find and selected_word_str[::-1] not in self.found_words:
                selected_word_str = selected_word_str[::-1]
                is_correct = True
            
            if is_correct:
                # sfx: word_found_positive
                word_len = len(selected_word_str)
                word_reward = word_len * 2
                self.score += word_reward
                reward += word_reward
                self.found_words.add(selected_word_str)
                self.feedback_message = f"FOUND: {selected_word_str}!"
                self.feedback_color = self.COLOR_FEEDBACK_GOOD
                self.feedback_timer = 90 # 3 seconds
                
                # Create particles for correct word
                for r, c in self.word_locations[selected_word_str]:
                    px = self.GRID_TOP_LEFT[0] + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    py = self.GRID_TOP_LEFT[1] + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    for _ in range(5):
                        self.particles.append([px, py, self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2), 30, self.COLOR_WORD_FOUND])
            else:
                # sfx: word_found_negative
                penalty = -5
                self.score += penalty
                reward += penalty
                self.feedback_message = "INCORRECT"
                self.feedback_color = self.COLOR_FEEDBACK_BAD
                self.feedback_timer = 60 # 2 seconds
            
            self.current_selection.clear()

        # Update feedback timer
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.feedback_message = ""

        # Update particles
        self.particles = [[p[0]+p[2], p[1]+p[3], p[2]*0.95, p[3]*0.95, p[4]-1, p[5]] for p in self.particles if p[4] > 0]

        # 4. Check for termination
        terminated = False
        if len(self.found_words) == len(self.words_to_find):
            terminated = True
            reward += 50
            self.score += 50
            self.feedback_message = "GRID COMPLETE!"
            self.feedback_color = self.COLOR_WORD_FOUND
            self.feedback_timer = self.MAX_STEPS # Persist until reset
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 50
            self.score -= 50
            self.feedback_message = "TIME'S UP!"
            self.feedback_color = self.COLOR_FEEDBACK_BAD
            self.feedback_timer = self.MAX_STEPS
        
        if terminated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
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
            "words_found": len(self.found_words),
            "words_total": len(self.words_to_find),
        }

    def _render_game(self):
        grid_surface = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
        
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            pygame.draw.line(grid_surface, self.COLOR_GRID_LINE, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.GRID_HEIGHT))
            pygame.draw.line(grid_surface, self.COLOR_GRID_LINE, (0, i * self.CELL_SIZE), (self.GRID_WIDTH, i * self.CELL_SIZE))

        # Check which cells belong to found words
        found_cells = set()
        for word in self.found_words:
            for pos in self.word_locations[word]:
                found_cells.add(pos)
        
        # Draw letters
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                letter = self.grid[r, c]
                color = self.COLOR_LETTER_FOUND if (r, c) in found_cells else self.COLOR_LETTER
                text_surf = self.font_large.render(letter, True, color)
                text_rect = text_surf.get_rect(center=(c * self.CELL_SIZE + self.CELL_SIZE // 2, r * self.CELL_SIZE + self.CELL_SIZE // 2))
                grid_surface.blit(text_surf, text_rect)
        
        # Draw selection highlights
        for r, c in self.current_selection:
            pygame.gfxdraw.box(grid_surface, (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), self.COLOR_SELECTION)

        # Draw cursor
        cursor_rect = (self.cursor_pos[1] * self.CELL_SIZE, self.cursor_pos[0] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.gfxdraw.rectangle(grid_surface, cursor_rect, self.COLOR_CURSOR)
        pygame.gfxdraw.rectangle(grid_surface, (cursor_rect[0]+1, cursor_rect[1]+1, cursor_rect[2]-2, cursor_rect[3]-2), self.COLOR_CURSOR)

        self.screen.blit(grid_surface, self.GRID_TOP_LEFT)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p[4] / 30 * 255)))
            color = p[5] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), int(p[4]/5), color)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.GRID_TOP_LEFT[0] + self.GRID_WIDTH + 20, 20))

        # Time
        time_left_sec = (self.MAX_STEPS - self.steps) / 30
        time_color = self.COLOR_FEEDBACK_BAD if time_left_sec < 10 else self.COLOR_UI_TEXT
        time_text = self.font_medium.render(f"TIME: {max(0, time_left_sec):.1f}", True, time_color)
        self.screen.blit(time_text, (self.GRID_TOP_LEFT[0] + self.GRID_WIDTH + 20, 50))
        
        # Words Found
        found_count_text = self.font_medium.render(f"FOUND: {len(self.found_words)}/{len(self.words_to_find)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(found_count_text, (self.GRID_TOP_LEFT[0] + self.GRID_WIDTH + 20, 80))
        
        # Feedback Message
        if self.feedback_timer > 0:
            alpha = min(255, self.feedback_timer * 5)
            feedback_surf = self.font_large.render(self.feedback_message, True, self.feedback_color)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
            self.screen.blit(feedback_surf, feedback_rect)

        # Word List
        word_list_y = self.GRID_TOP_LEFT[1]
        for word in sorted(self.words_to_find):
            is_found = word in self.found_words
            color = self.COLOR_WORD_FOUND if is_found else self.COLOR_WORD_UNFOUND
            word_surf = self.font_small.render(word, True, color)
            self.screen.blit(word_surf, (20, word_list_y))
            if is_found:
                pygame.draw.line(self.screen, self.COLOR_WORD_FOUND, (20, word_list_y + 10), (20 + word_surf.get_width(), word_list_y + 10), 1)
            word_list_y += 20
    
    def render(self):
        # This method is not used by the gym API directly but can be useful for human play
        if self.render_mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example of how to run the environment for human play
    env = GameEnv(render_mode="rgb_array")
    
    # Use a window to display the game
    pygame.display.set_caption("Word Search")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(30)
        
    env.close()