import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the worm. Press Space to submit your collected letters as a word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a glowing worm to collect letters on the grid. Spell words from the target list to score points before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    # The game is logically turn-based (move one cell per action) but has a real-time clock (decremented per step).
    # auto_advance=False is the correct choice for turn-based logic.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 60
    GRID_WIDTH = 20
    GRID_HEIGHT = 11  # (400-60) / 30 = 11.33, so 11 rows
    CELL_SIZE = 30

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 40, 60)
    COLOR_WORM = (50, 255, 50)
    COLOR_WORM_GLOW = (50, 255, 50, 50)  # RGBA for alpha
    COLOR_LETTER = (255, 255, 255)
    COLOR_UI_BG = (20, 30, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_HIGHLIGHT = (255, 255, 100)
    COLOR_UI_SUCCESS = (100, 255, 100)
    COLOR_UI_FAIL = (255, 100, 100)

    # Game parameters
    START_TIME = 60.0
    TIME_DECREMENT_PER_STEP = 0.1  # 10 steps per second
    MAX_STEPS = 1000
    WORDS_TO_WIN = 10
    WORM_START_LENGTH = 3

    # Word list
    WORD_POOL = [
        "CAT", "DOG", "SUN", "MOON", "STAR", "SKY", "RUN", "JUMP", "WORD", "GAME",
        "CODE", "GRID", "WORM", "PLAY", "FAST", "GLOW", "NEON", "BYTE", "PIXEL",
        "PYTHON", "AGENT", "REWARD", "ACTION", "STATE", "SPACE", "LEARN", "SOLVE"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_ui_large = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_ui_medium = pygame.font.SysFont("Consolas", 18)
            self.font_ui_small = pygame.font.SysFont("Consolas", 14)
            self.font_grid = pygame.font.SysFont("Consolas", 22, bold=True)
        except pygame.error:
            self.font_ui_large = pygame.font.SysFont("Courier New", 24, bold=True)
            self.font_ui_medium = pygame.font.SysFont("Courier New", 18)
            self.font_ui_small = pygame.font.SysFont("Courier New", 14)
            self.font_grid = pygame.font.SysFont("Courier New", 22, bold=True)

        # Game state variables are initialized in reset()
        self.worm_body = None
        self.direction = None
        self.grid_letters = None
        self.target_words = None
        self.spelled_words = None
        self.collected_letters = None
        self.score = 0
        self.time_remaining = 0.0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.word_feedback = None  # {'text', 'color', 'timer'}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.steps = 0
        self.time_remaining = self.START_TIME
        self.game_over = False

        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.worm_body = deque([(start_x, start_y - i) for i in range(self.WORM_START_LENGTH)], maxlen=50)
        self.direction = (0, -1)  # Start moving up

        self.collected_letters = []
        self.spelled_words = []
        self.word_feedback = None
        self.particles = []

        self._select_and_place_letters()

        return self._get_observation(), self._get_info()

    def _select_and_place_letters(self):
        self.target_words = self.np_random.choice(self.WORD_POOL, self.WORDS_TO_WIN, replace=False).tolist()

        required_letters = set("".join(self.target_words))
        all_letters = list(required_letters)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        decoy_pool = [l for l in alphabet if l not in required_letters]

        # Calculate the desired number of decoys (50% of required letters)
        num_decoys_desired = int(len(required_letters) * 0.5)

        # The actual number of decoys is limited by the number of available letters in the alphabet
        num_decoys_to_sample = min(num_decoys_desired, len(decoy_pool))

        # Only sample if there are decoys to sample from and we need to sample them
        if num_decoys_to_sample > 0:
            decoys = self.np_random.choice(decoy_pool, size=num_decoys_to_sample, replace=False)
            all_letters.extend(decoys)

        self.grid_letters = {}
        available_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]

        # Remove cells occupied by the worm
        for segment in self.worm_body:
            if segment in available_cells:
                available_cells.remove(segment)

        num_letters_to_place = min(len(all_letters), len(available_cells))
        letter_indices = self.np_random.choice(len(all_letters), num_letters_to_place, replace=False)
        cell_indices = self.np_random.choice(len(available_cells), num_letters_to_place, replace=False)

        for i in range(num_letters_to_place):
            letter = all_letters[letter_indices[i]]
            pos = available_cells[cell_indices[i]]
            self.grid_letters[pos] = letter

    def step(self, action):
        if self.game_over:
            # On subsequent steps after termination, just return the final state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement, space_press, _ = action
        reward = -0.1  # Cost of living/time

        # --- Handle Word Feedback Timer ---
        if self.word_feedback and self.word_feedback['timer'] > 0:
            self.word_feedback['timer'] -= 1
        elif self.word_feedback:
            self.word_feedback = None

        # --- Handle Movement ---
        moved = False
        if movement > 0:
            new_direction = self.direction
            if movement == 1: new_direction = (0, -1)  # Up
            elif movement == 2: new_direction = (0, 1)  # Down
            elif movement == 3: new_direction = (-1, 0)  # Left
            elif movement == 4: new_direction = (1, 0)  # Right

            # Prevent moving back on itself
            if len(self.worm_body) == 1 or (new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]):
                self.direction = new_direction

        head = self.worm_body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Wall collision check
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            new_head = head  # Don't move
        # Self collision check
        elif len(self.worm_body) > 1 and new_head in list(self.worm_body)[1:]:
            new_head = head  # Don't move
        else:
            moved = True
            self.worm_body.appendleft(new_head)
            # The deque's maxlen handles tail removal automatically

        # --- Handle Letter Collection ---
        if new_head in self.grid_letters:
            letter = self.grid_letters.pop(new_head)
            self.collected_letters.append(letter)
            # Check if letter is useful for any remaining word
            is_useful = any(letter in w for w in self.target_words if w not in self.spelled_words)
            if is_useful:
                reward += 1.0

            # Sound effect placeholder: # sfx_collect_letter()
            self._create_particles(new_head, self.COLOR_LETTER, 10)

        # --- Handle Word Submission ---
        if space_press:
            submitted_word = "".join(self.collected_letters)
            if submitted_word in self.target_words and submitted_word not in self.spelled_words:
                word_len = len(submitted_word)
                word_reward = 10 + max(0, word_len - 3) * 2
                self.score += word_reward
                reward += word_reward
                self.spelled_words.append(submitted_word)
                self.collected_letters = []
                self.worm_body.maxlen = min(50, self.WORM_START_LENGTH + len(self.spelled_words))
                self.word_feedback = {'text': f"'{submitted_word}' +{word_reward}", 'color': self.COLOR_UI_SUCCESS,
                                      'timer': 6}
                # Sound effect placeholder: # sfx_word_success()
                self._create_particles(head, self.COLOR_UI_SUCCESS, 30)
            else:
                # Penalty for wrong submission could be added, but for now just clear
                self.collected_letters = []
                self.word_feedback = {'text': f"'{submitted_word}' - Invalid", 'color': self.COLOR_UI_FAIL, 'timer': 6}
                # Sound effect placeholder: # sfx_word_fail()

        # --- Update State & Check Termination ---
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - self.TIME_DECREMENT_PER_STEP)

        terminated = False
        if len(self.spelled_words) >= self.WORDS_TO_WIN:
            reward += 50  # Win bonus
            self.game_over = True
            terminated = True
            self.word_feedback = {'text': "YOU WIN!", 'color': self.COLOR_UI_SUCCESS, 'timer': 1000}
        elif self.time_remaining <= 0:
            reward -= 10  # Loss penalty
            self.game_over = True
            terminated = True
            self.word_feedback = {'text': "TIME UP!", 'color': self.COLOR_UI_FAIL, 'timer': 1000}
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
             self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        grid_offset_y = self.UI_HEIGHT + (self.SCREEN_HEIGHT - self.UI_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start = (grid_offset_x + x * self.CELL_SIZE, grid_offset_y)
            end = (grid_offset_x + x * self.CELL_SIZE, grid_offset_y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (grid_offset_x, grid_offset_y + y * self.CELL_SIZE)
            end = (grid_offset_x + self.GRID_WIDTH * self.CELL_SIZE, grid_offset_y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw letters
        for pos, letter in self.grid_letters.items():
            text_surf = self.font_grid.render(letter, True, self.COLOR_LETTER)
            text_rect = text_surf.get_rect(center=(
                grid_offset_x + pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                grid_offset_y + pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
            ))
            self.screen.blit(text_surf, text_rect)

        # Draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['radius'] -= 0.2
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color']
                )

        # Draw worm
        for i, segment in enumerate(self.worm_body):
            center_x = grid_offset_x + segment[0] * self.CELL_SIZE + self.CELL_SIZE // 2
            center_y = grid_offset_y + segment[1] * self.CELL_SIZE + self.CELL_SIZE // 2
            radius = int(self.CELL_SIZE * 0.45)
            glow_radius = int(radius * 1.8)

            # Glow effect
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, self.COLOR_WORM_GLOW)
            self.screen.blit(s, (center_x - glow_radius, center_y - glow_radius))

            # Solid body
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_WORM)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_WORM)

            # Eyes on head
            if i == 0:
                eye_radius = 2
                dir_x, dir_y = self.direction
                eye_offset_x = dir_y * (radius // 2)
                eye_offset_y = dir_x * (radius // 2)
                eye1_pos = (
                center_x + dir_x * radius * 0.5 + eye_offset_x, center_y + dir_y * radius * 0.5 + eye_offset_y)
                eye2_pos = (
                center_x + dir_x * radius * 0.5 - eye_offset_x, center_y + dir_y * radius * 0.5 - eye_offset_y)
                pygame.gfxdraw.filled_circle(self.screen, int(eye1_pos[0]), int(eye1_pos[1]), eye_radius, self.COLOR_BG)
                pygame.gfxdraw.filled_circle(self.screen, int(eye2_pos[0]), int(eye2_pos[1]), eye_radius, self.COLOR_BG)

    def _create_particles(self, grid_pos, color, count):
        grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        grid_offset_y = self.UI_HEIGHT + (self.SCREEN_HEIGHT - self.UI_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        center_x = grid_offset_x + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = grid_offset_y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2

        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(3, 7),
                'lifetime': self.np_random.integers(10, 20),
                'color': color
            })

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT - 1), (self.SCREEN_WIDTH, self.UI_HEIGHT - 1), 2)

        # Score
        score_text = self.font_ui_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Time
        time_color = self.COLOR_UI_HIGHLIGHT if self.time_remaining > 10 else self.COLOR_UI_FAIL
        time_text = self.font_ui_large.render(f"TIME: {self.time_remaining:.1f}", True, time_color)
        time_rect = time_text.get_rect(right=self.SCREEN_WIDTH - 10, top=5)
        self.screen.blit(time_text, time_rect)

        # Current collected letters
        collected_str = "".join(self.collected_letters)
        collected_text = self.font_ui_medium.render(f"Current: {collected_str}", True, self.COLOR_UI_HIGHLIGHT)
        collected_rect = collected_text.get_rect(centerx=self.SCREEN_WIDTH // 2, top=5)
        self.screen.blit(collected_text, collected_rect)

        # Word feedback
        if self.word_feedback:
            feedback_surf = self.font_ui_medium.render(self.word_feedback['text'], True, self.word_feedback['color'])
            feedback_rect = feedback_surf.get_rect(centerx=self.SCREEN_WIDTH // 2, top=collected_rect.bottom)
            self.screen.blit(feedback_surf, feedback_rect)

        # Words to spell
        words_text = self.font_ui_small.render(f"Words ({len(self.spelled_words)}/{self.WORDS_TO_WIN}):", True,
                                               self.COLOR_UI_TEXT)
        self.screen.blit(words_text, (10, 35))

        x_offset = words_text.get_width() + 20
        for word in self.target_words:
            if x_offset > self.SCREEN_WIDTH - 50: break
            color = self.COLOR_UI_SUCCESS if word in self.spelled_words else self.COLOR_UI_TEXT
            word_surf = self.font_ui_small.render(word, True, color)
            self.screen.blit(word_surf, (x_offset, 35))
            x_offset += word_surf.get_width() + 10

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "words_spelled": len(self.spelled_words),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Worm Words")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    print(env.user_guide)

    while running:
        movement = 0  # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

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

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- New Game Started ---")

        clock.tick(10)  # Control the speed of the manual play

    env.close()