
# Generated: 2025-08-28T00:32:15.270283
# Source Brief: brief_03820.md
# Brief Index: 3820

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CURSOR_SPEED = 8
    INITIAL_FALL_SPEED = 1.0
    MAX_LIVES = 10
    WIN_WORDS = 50
    MAX_STEPS = 3000
    SPAWN_INTERVAL = 10 # frames

    # Rewards
    REWARD_CORRECT_LETTER = 0.1
    REWARD_INCORRECT_LETTER = -0.1
    REWARD_COMPLETE_WORD = 1.0
    REWARD_MISSED_WORD = -1.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_HOVER = (0, 255, 128)
    COLOR_LETTER = (220, 220, 240)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_ACTIVE_WORD_REMAINING = (255, 255, 255)
    COLOR_ACTIVE_WORD_TYPED = (100, 255, 150)
    COLOR_HEART = (255, 80, 80)
    COLOR_FLASH_BAD = (180, 20, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_letter = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_ui = pygame.font.SysFont("Lucida Console", 20)
            self.font_word = pygame.font.SysFont("Arial", 32, bold=True)
            self.font_gameover = pygame.font.SysFont("Arial", 48, bold=True)
        except pygame.error:
            self.font_letter = pygame.font.Font(None, 24)
            self.font_ui = pygame.font.Font(None, 20)
            self.font_word = pygame.font.Font(None, 32)
            self.font_gameover = pygame.font.Font(None, 48)

        # Word bank
        self.word_bank = [
            "PYTHON", "AGENT", "LEARN", "REWARD", "ACTION", "STATE", "POLICY",
            "DEEP", "NEURAL", "NETWORK", "GYM", "SPACE", "VECTOR", "TENSOR",
            "FLUID", "GAME", "VISUAL", "CODE", "EXPERT", "DESIGN", "LOOP",
            "FRAME", "EVENT", "STEP", "RESET", "RENDER", "PIXEL", "ARRAY"
        ]

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.lives = self.MAX_LIVES
        self.words_completed = 0
        self.fall_speed = self.INITIAL_FALL_SPEED

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.last_space_held = False
        self.last_shift_held = False

        self.falling_letters = []
        self.particles = []
        self.letters_to_spawn = []

        self.active_word = ""
        self.typed_progress = ""
        self._select_new_word()

        self.screen_flash_timer = 0
        self.spawn_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)

        # --- Game Logic ---
        reward += self._update_letters()
        self._update_particles()
        self._spawn_letters()
        self._update_difficulty()

        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1
        
        self.steps += 1

        # --- Termination Check ---
        terminated = False
        if self.words_completed >= self.WIN_WORDS:
            self.victory = True
            terminated = True
            reward += self.REWARD_WIN
        elif self.lives <= 0:
            terminated = True
            reward += self.REWARD_LOSS
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        if movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        if movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        if movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Space press (type action)
        if space_held and not self.last_space_held:
            reward += self._execute_type_action()

        # Shift press (skip word action)
        if shift_held and not self.last_shift_held:
            self._select_new_word()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _execute_type_action(self):
        hovered_letter, _ = self._get_hovered_letter()
        if hovered_letter and self.typed_progress < self.active_word:
            expected_char = self.active_word[len(self.typed_progress)]
            
            if hovered_letter['char'] == expected_char:
                # Correct letter
                self.typed_progress += expected_char
                self._create_particles(hovered_letter['pos'], self.COLOR_ACTIVE_WORD_TYPED)
                self.falling_letters.remove(hovered_letter)
                
                # Word completed
                if self.typed_progress == self.active_word:
                    self.words_completed += 1
                    self.score += 1
                    self._select_new_word()
                    return self.REWARD_CORRECT_LETTER + self.REWARD_COMPLETE_WORD
                
                return self.REWARD_CORRECT_LETTER
        
        # Incorrect letter or no letter hovered
        self.screen_flash_timer = 5
        # sfx: bad_beep
        return self.REWARD_INCORRECT_LETTER

    def _update_letters(self):
        reward = 0
        for letter in self.falling_letters[:]:
            letter['pos'][1] += self.fall_speed
            if letter['pos'][1] > self.HEIGHT:
                self.falling_letters.remove(letter)
                self.lives -= 1
                reward += self.REWARD_MISSED_WORD
                self.screen_flash_timer = 5
                # sfx: miss_word
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_letters(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.SPAWN_INTERVAL
            
            spawn_pos = [self.np_random.integers(20, self.WIDTH - 20), -20]
            
            # 70% chance to spawn a needed letter, if any
            if self.letters_to_spawn and self.np_random.random() < 0.7:
                char_to_spawn = self.letters_to_spawn.pop(0)
            else: # 30% chance or no needed letters left
                char_to_spawn = chr(self.np_random.integers(65, 91)) # A-Z

            new_letter = {
                'char': char_to_spawn,
                'pos': np.array(spawn_pos, dtype=np.float32),
                'surf': self.font_letter.render(char_to_spawn, True, self.COLOR_LETTER),
            }
            self.falling_letters.append(new_letter)

    def _select_new_word(self):
        if not self.game_over:
            self.active_word = self.np_random.choice(self.word_bank)
            self.typed_progress = ""
            self.letters_to_spawn = list(self.active_word)
            self.np_random.shuffle(self.letters_to_spawn)
            # sfx: new_word

    def _update_difficulty(self):
        self.fall_speed = self.INITIAL_FALL_SPEED + (self.words_completed // 5) * 0.1

    def _get_hovered_letter(self):
        for i, letter in enumerate(self.falling_letters):
            letter_rect = letter['surf'].get_rect(center=letter['pos'])
            if letter_rect.collidepoint(self.cursor_pos):
                return letter, i
        return None, -1

    def _create_particles(self, pos, color):
        # sfx: correct_letter_zap
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw falling letters
        for letter in self.falling_letters:
            self.screen.blit(letter['surf'], letter['surf'].get_rect(center=letter['pos']))

        # Draw particles
        for p in self.particles:
            size = max(1, p['life'] // 4)
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Draw cursor
        hovered_letter, _ = self._get_hovered_letter()
        cursor_color = self.COLOR_CURSOR_HOVER if hovered_letter else self.COLOR_CURSOR
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, cursor_color, (x - 8, y), (x + 8, y), 2)
        pygame.draw.line(self.screen, cursor_color, (x, y - 8), (x, y + 8), 2)
        pygame.gfxdraw.aacircle(self.screen, x, y, 10, cursor_color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Words Completed
        words_text = self.font_ui.render(f"WORDS: {self.words_completed}/{self.WIN_WORDS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(words_text, (10, 35))

        # Lives
        for i in range(self.lives):
            heart_pos = (self.WIDTH - 30 - i * 25, 20)
            pygame.draw.polygon(self.screen, self.COLOR_HEART, [
                (heart_pos[0], heart_pos[1] + 5), (heart_pos[0] + 10, heart_pos[1] + 15),
                (heart_pos[0] + 20, heart_pos[1] + 5), (heart_pos[0] + 10, heart_pos[1])
            ])
            pygame.gfxdraw.filled_circle(self.screen, heart_pos[0] + 5, heart_pos[1], 5, self.COLOR_HEART)
            pygame.gfxdraw.filled_circle(self.screen, heart_pos[0] + 15, heart_pos[1], 5, self.COLOR_HEART)

        # Active Word Display
        if self.active_word:
            typed_part = self.active_word[:len(self.typed_progress)]
            remaining_part = self.active_word[len(self.typed_progress):]
            
            typed_surf = self.font_word.render(typed_part, True, self.COLOR_ACTIVE_WORD_TYPED)
            remaining_surf = self.font_word.render(remaining_part, True, self.COLOR_ACTIVE_WORD_REMAINING)
            
            total_width = typed_surf.get_width() + remaining_surf.get_width()
            start_x = (self.WIDTH - total_width) // 2
            
            self.screen.blit(typed_surf, (start_x, self.HEIGHT - 50))
            self.screen.blit(remaining_surf, (start_x + typed_surf.get_width(), self.HEIGHT - 50))

        # Screen Flash
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.screen_flash_timer / 5))
            flash_surface.fill((*self.COLOR_FLASH_BAD, alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_ACTIVE_WORD_TYPED if self.victory else self.COLOR_HEART
            text_surf = self.font_gameover.render(msg, True, color)
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "words_completed": self.words_completed
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Typing Game")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
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
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS
        
    env.close()