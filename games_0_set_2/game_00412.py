
# Generated: 2025-08-27T13:34:40.166441
# Source Brief: brief_00412.md
# Brief Index: 412

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a letter. "
        "Press Space on a non-adjacent letter to submit the current word. "
        "Press Shift to deselect the last letter."
    )

    game_description = (
        "Connect adjacent letters in the grid to form words. Reach the target score before time runs out. "
        "Longer words and bonuses grant more points."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = 4
        self.TARGET_SCORE = 500
        self.TIME_LIMIT_SECONDS = 120
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_LETTER = (220, 220, 230)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_SELECTED_BG = (30, 80, 150)
        self.COLOR_SELECTED_LETTER = (255, 255, 255)
        self.COLOR_LINE = (50, 150, 250)
        self.COLOR_UI_TEXT = (200, 200, 210)
        self.COLOR_MSG_GOOD = (100, 255, 100)
        self.COLOR_MSG_BAD = (255, 100, 100)
        self.COLOR_SCORE = (255, 230, 100)

        # Word list
        self.word_list = self._generate_word_list()
        self.letter_distribution = "AAAAAAAAABBBCCCCDDDEEEEEEEEEEEEFFGGGHHIIIIIIIIIJKLLLLMMNNNNNNOOOOOOOOPPPQRRRRRRSSSSSSTTTTTTTUUUUVVWWXYYZ"

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_path = []
        self.current_word = ""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.message = ""
        self.message_color = self.COLOR_UI_TEXT
        self.message_timer = 0
        
        self.reset()
        self.validate_implementation()

    def _generate_word_list(self):
        # Embedded dictionary to avoid external files
        return set([
            "cat", "dog", "sun", "run", "fun", "big", "red", "bed", "mad", "sad",
            "game", "play", "word", "list", "grid", "time", "win", "lose", "code",
            "python", "agent", "learn", "reward", "action", "state", "space",
            "quest", "query", "quick", "jump", "prize", "faze", "maze", "zone",
            "score", "goal", "team", "form", "line", "path", "move", "select",
            "expert", "visual", "quality", "design", "brief", "fast", "arcade",
            "drift", "boost", "fire", "drive", "turn", "brake", "shift",
            "connect", "letter", "puzzle", "player", "bonus", "target"
        ])

    def _generate_grid(self):
        self.grid = [
            [self.np_random.choice(list(self.letter_distribution)) for _ in range(self.GRID_SIZE)]
            for _ in range(self.GRID_SIZE)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_grid()
        self.cursor_pos = [0, 0]
        self.selected_path = []
        self.current_word = ""
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.message = ""
        self.message_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # Unpack action and detect presses
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        elif movement == 2: self.cursor_pos[0] += 1  # Down
        elif movement == 3: self.cursor_pos[1] -= 1  # Left
        elif movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # 2. Handle deselect/cancel
        if shift_pressed and self.selected_path:
            self.selected_path.pop()
            self._update_current_word()
            reward -= 0.1

        # 3. Handle select/submit
        if space_pressed:
            pos = tuple(self.cursor_pos)
            if not self.selected_path:
                self.selected_path.append(pos)
                self._update_current_word()
                reward += 0.1
            elif pos in self.selected_path:
                self._set_message("Letter already used!", self.COLOR_MSG_BAD, 60)
                reward -= 0.1
            elif self._is_adjacent(pos, self.selected_path[-1]):
                self.selected_path.append(pos)
                self._update_current_word()
                reward += 0.1
            else: # Not adjacent, treat as word submission
                reward += self._submit_word()
        
        # 4. Update game systems
        self._update_particles()
        if self.message_timer > 0:
            self.message_timer -= 1
        else:
            self.message = ""

        # 5. Check for termination
        terminated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -100.0
            self._set_message("Time's Up!", self.COLOR_MSG_BAD, 120)
        elif self.score >= self.TARGET_SCORE:
            terminated = True
            reward = 100.0
            self._set_message("Target Score Reached!", self.COLOR_MSG_GOOD, 120)
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _submit_word(self):
        word = self.current_word.lower()
        if len(word) >= 3 and word in self.word_list:
            # Valid word
            word_len = len(word)
            base_score = word_len * 10
            bonus_score = 20 if word_len > 4 else 0
            total_word_score = base_score + bonus_score
            self.score += total_word_score
            
            # Sound effect placeholder
            # pygame.mixer.Sound.play(word_success_sound)
            
            self._set_message(f"'{word.upper()}' +{total_word_score}", self.COLOR_MSG_GOOD, 90)
            self._create_word_particles()
            
            self.selected_path.clear()
            self._update_current_word()
            
            # Reward shaping
            reward = 10.0 + word_len + (2.0 if word_len > 4 else 0)
            return reward
        else:
            # Invalid word
            # Sound effect placeholder
            # pygame.mixer.Sound.play(word_fail_sound)
            self._set_message("Not a valid word", self.COLOR_MSG_BAD, 60)
            self.selected_path.clear()
            self._update_current_word()
            return -1.0

    def _update_current_word(self):
        self.current_word = "".join([self.grid[r][c] for r, c in self.selected_path])

    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1

    def _set_message(self, text, color, duration):
        self.message = text
        self.message_color = color
        self.message_timer = duration

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        grid_pixel_size = 280
        cell_size = grid_pixel_size / self.GRID_SIZE
        offset_x = (self.WIDTH - grid_pixel_size) / 2
        offset_y = (self.HEIGHT - grid_pixel_size) / 2 + 40

        # Draw connecting lines
        if len(self.selected_path) > 1:
            points = []
            for r, c in self.selected_path:
                x = offset_x + c * cell_size + cell_size / 2
                y = offset_y + r * cell_size + cell_size / 2
                points.append((int(x), int(y)))
            pygame.draw.lines(self.screen, self.COLOR_LINE, False, points, 5)

        # Draw grid cells and letters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    offset_x + c * cell_size,
                    offset_y + r * cell_size,
                    cell_size,
                    cell_size
                )
                
                is_selected = (r, c) in self.selected_path
                is_cursor = self.cursor_pos == [r, c]

                # Draw background
                if is_selected:
                    pygame.draw.rect(self.screen, self.COLOR_SELECTED_BG, rect.inflate(-4, -4), border_radius=8)
                
                # Draw cursor
                if is_cursor:
                    pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=10)

                # Draw letter
                letter_color = self.COLOR_SELECTED_LETTER if is_selected else self.COLOR_LETTER
                letter_surf = self.font_large.render(self.grid[r][c], True, letter_color)
                letter_rect = letter_surf.get_rect(center=rect.center)
                self.screen.blit(letter_surf, letter_rect)
        
        self._render_particles(offset_x, offset_y, cell_size)

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {int(time_left // 60):02}:{int(time_left % 60):02}"
        time_surf = self.font_medium.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_SCORE)
        score_rect = score_surf.get_rect(centerx=self.WIDTH / 2, y=10)
        self.screen.blit(score_surf, score_rect)

        # Target Score
        target_text = f"TARGET: {self.TARGET_SCORE}"
        target_surf = self.font_medium.render(target_text, True, self.COLOR_UI_TEXT)
        target_rect = target_surf.get_rect(right=self.WIDTH - 10, y=10)
        self.screen.blit(target_surf, target_rect)
        
        # Current Word Display
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.HEIGHT - 50, self.WIDTH - 20, 40), border_radius=8)
        word_surf = self.font_medium.render(self.current_word, True, self.COLOR_LETTER)
        word_rect = word_surf.get_rect(midleft=(20, self.HEIGHT - 30))
        self.screen.blit(word_surf, word_rect)

        # Message Display
        if self.message and self.message_timer > 0:
            alpha = min(255, int(255 * (self.message_timer / 30)))
            msg_surf = self.font_medium.render(self.message, True, self.message_color)
            msg_surf.set_alpha(alpha)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, 60))
            self.screen.blit(msg_surf, msg_rect)

    def _create_word_particles(self):
        grid_pixel_size = 280
        cell_size = grid_pixel_size / self.GRID_SIZE
        offset_x = (self.WIDTH - grid_pixel_size) / 2
        offset_y = (self.HEIGHT - grid_pixel_size) / 2 + 40
        
        for r, c in self.selected_path:
            for _ in range(15):
                x = offset_x + c * cell_size + cell_size / 2
                y = offset_y + r * cell_size + cell_size / 2
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(20, 40)
                color = random.choice([self.COLOR_SCORE, self.COLOR_LINE, (255,255,255)])
                self.particles.append([x, y, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1]  # x += vx
            p[1] += p[2]  # y += vy
            p[4] -= 1     # life -= 1

    def _render_particles(self, offset_x, offset_y, cell_size):
        for x, y, _, _, life, color in self.particles:
            radius = int(max(0, (life / 20) * 4))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
            "current_word": self.current_word,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Word Grid Game")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    while not done:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()