
# Generated: 2025-08-27T23:35:38.618293
# Source Brief: brief_03516.md
# Brief Index: 3516

        
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
    user_guide = "Controls: Use ← and → arrow keys to move the basket."

    # Must be a short, user-facing description of the game:
    game_description = "Catch falling words in your basket. Longer words are worth more points. Clear 3 stages to win, but don't miss more than two words!"

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 3000
        self.INITIAL_LIVES = 3
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_BASKET = (230, 230, 250)
        self.COLOR_BASKET_HL = (255, 255, 255)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_PARTICLE_GOOD = (0, 255, 150)
        self.COLOR_PARTICLE_BAD = (255, 50, 50)
        self.COLOR_WORD_SHORT = (100, 180, 255)
        self.COLOR_WORD_MEDIUM = (255, 255, 100)
        self.COLOR_WORD_LONG = (255, 150, 50)
        
        # Fonts
        try:
            self.font_word = pygame.font.SysFont('Consolas', 24, bold=True)
            self.font_ui = pygame.font.SysFont('Arial', 20)
            self.font_msg = pygame.font.SysFont('Arial', 48, bold=True)
        except pygame.error:
            self.font_word = pygame.font.Font(None, 28)
            self.font_ui = pygame.font.Font(None, 24)
            self.font_msg = pygame.font.Font(None, 52)
            
        # Word list (no external files)
        self.word_list = [
            "AGENT", "REWARD", "STATE", "ACTION", "POLICY", "VALUE", "MODEL", "LEARN",
            "DEEP", "NEURAL", "NETWORK", "TRAIN", "EXPLORE", "EXPLOIT", "GAMMA", "ALPHA",
            "Q", "SARSA", "DQN", "A2C", "PPO", "HER", "GYM", "PYTHON", "NUMPY", "CODE",
            "BUG", "DEBUG", "PLAY", "WIN", "LOSS", "SCORE", "STEP", "RESET", "ENV"
        ]
        
        # Game element properties
        self.basket_width = 100
        self.basket_height = 20
        self.basket_speed = 10.0

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.words_caught_in_stage = 0
        self.game_over = False
        self.win = False
        self.basket_x = 0
        self.words = []
        self.particles = []
        self.message = ""
        self.message_timer = 0
        self.base_word_speed = 0.0
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.words_caught_in_stage = 0
        self.game_over = False
        self.win = False

        self.basket_x = self.SCREEN_WIDTH / 2 - self.basket_width / 2
        self.words = []
        self.particles = []
        
        self.message = ""
        self.message_timer = 0
        self.base_word_speed = 2.0
        
        self._spawn_word()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        
        # --- 1. Calculate Shaping Reward (before state change) ---
        prev_dist = self._get_closest_word_dist()

        # --- 2. Update Game Logic ---
        self._handle_input(movement)
        self._update_particles()
        event_reward = self._update_words()
        self._maybe_spawn_word()
        
        if self.message_timer > 0:
            self.message_timer -= 1
        else:
            self.message = ""

        # --- 3. Calculate Final Reward ---
        new_dist = self._get_closest_word_dist()
        shaping_reward = 0
        if new_dist < prev_dist:
            shaping_reward = 0.1  # Reward for moving closer
        elif new_dist > prev_dist:
            shaping_reward = -0.01 # Penalty for moving away

        reward = event_reward + shaping_reward

        # --- 4. Check Termination ---
        self.steps += 1
        terminated = self.game_over or self.win or (self.steps >= self.MAX_STEPS)
        
        if terminated:
            if self.win:
                reward += 100 # Win bonus
            elif self.game_over:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }
        
    def _get_closest_word_dist(self):
        if not self.words:
            return self.SCREEN_WIDTH
        
        basket_center_x = self.basket_x + self.basket_width / 2
        active_words = [w for w in self.words if 'fade' not in w]
        if not active_words:
            return self.SCREEN_WIDTH
            
        # Find the lowest word on screen
        lowest_word = min(active_words, key=lambda w: -w['rect'].bottom)
        
        word_center_x = lowest_word['rect'].centerx
        return abs(basket_center_x - word_center_x)

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.basket_x -= self.basket_speed
        elif movement == 4:  # Right
            self.basket_x += self.basket_speed
        self.basket_x = np.clip(self.basket_x, 0, self.SCREEN_WIDTH - self.basket_width)

    def _update_words(self):
        event_reward = 0
        basket_rect = pygame.Rect(int(self.basket_x), self.SCREEN_HEIGHT - self.basket_height - 10, self.basket_width, self.basket_height)
        
        words_to_remove = []
        for word in self.words:
            if 'fade' in word:
                word['fade'] -= 1
                if word['fade'] <= 0:
                    words_to_remove.append(word)
                continue

            word['pos'][1] += word['speed']
            word['rect'].top = int(word['pos'][1])

            if basket_rect.colliderect(word['rect']):
                # sound: catch_sfx
                points = len(word['text']) * 10
                self.score += points
                event_reward += points
                
                self.words_caught_in_stage += 1
                self._create_particles(word['rect'].center, self.COLOR_PARTICLE_GOOD, 30)
                word['fade'] = 15
                word['color'] = self.COLOR_PARTICLE_GOOD

                if self.words_caught_in_stage >= 15:
                    self.stage += 1
                    event_reward += 50
                    if self.stage > 3:
                        self.win = True
                        self.message = "YOU WIN!"
                        self.message_timer = 180
                    else:
                        self.words_caught_in_stage = 0
                        self.base_word_speed += 0.75
                        self.message = f"STAGE {self.stage}"
                        self.message_timer = 90
                        # sound: stage_clear_sfx

            elif word['rect'].top > self.SCREEN_HEIGHT:
                # sound: miss_sfx
                self.lives -= 1
                event_reward -= 5
                words_to_remove.append(word)
                self._create_particles((word['rect'].centerx, self.SCREEN_HEIGHT - 5), self.COLOR_PARTICLE_BAD, 50)
                if self.lives <= 0:
                    self.game_over = True
                    self.message = "GAME OVER"
                    self.message_timer = 180
                    # sound: game_over_sfx

        self.words = [w for w in self.words if w not in words_to_remove]
        return event_reward

    def _maybe_spawn_word(self):
        if self.win or self.game_over:
            return
        
        num_words_on_screen = len([w for w in self.words if 'fade' not in w])
        if num_words_on_screen < self.stage + 1:
            if not self.words or self.words[-1]['pos'][1] > self.np_random.integers(100, 150):
                self._spawn_word()

    def _spawn_word(self):
        text = self.np_random.choice(self.word_list)
        length = len(text)
        
        if length <= 4: color = self.COLOR_WORD_SHORT
        elif length <= 6: color = self.COLOR_WORD_MEDIUM
        else: color = self.COLOR_WORD_LONG
        
        surface = self.font_word.render(text, True, color)
        pos_x = self.np_random.integers(10, self.SCREEN_WIDTH - surface.get_width() - 10)
        
        self.words.append({
            'text': text,
            'pos': [float(pos_x), -30.0],
            'speed': self.base_word_speed + self.np_random.uniform(-0.2, 0.2),
            'rect': surface.get_rect(topleft=(pos_x, -30)),
            'color': color,
            'surface': surface
        })

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_background(self):
        top_color, bottom_color = self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio),
                int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio),
                int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = max(1, int(3 * (p['life'] / p['max_life'])))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, (*p['color'], alpha))

        for word in self.words:
            surface = word['surface']
            if 'fade' in word:
                alpha = max(0, int(255 * (word['fade'] / 15.0)))
                temp_surf = surface.copy()
                temp_surf.set_alpha(alpha)
                self.screen.blit(temp_surf, word['rect'])
            else:
                self.screen.blit(surface, word['rect'])
                
        basket_rect = pygame.Rect(int(self.basket_x), self.SCREEN_HEIGHT - self.basket_height - 10, self.basket_width, self.basket_height)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        highlight_rect = pygame.Rect(basket_rect.left + 3, basket_rect.top + 3, basket_rect.width - 6, 5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_HL, highlight_rect, border_radius=3)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"Lives: {self.lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, lives_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10)))

        stage_text = self.font_ui.render(f"Stage: {min(self.stage, 3)}/3", True, self.COLOR_UI)
        self.screen.blit(stage_text, stage_text.get_rect(midtop=(self.SCREEN_WIDTH / 2, 10)))

        if self.message and self.message_timer > 0:
            total_duration = 90 if "STAGE" in self.message else 180
            alpha_ratio = 1.0
            if self.message_timer > total_duration - 30:
                alpha_ratio = (total_duration - self.message_timer) / 30.0
            elif self.message_timer < 30:
                alpha_ratio = self.message_timer / 30.0
            alpha = min(255, int(255 * alpha_ratio))
            
            msg_surf = self.font_msg.render(self.message, True, self.COLOR_UI)
            msg_surf.set_alpha(alpha)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")