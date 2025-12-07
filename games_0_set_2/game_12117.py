import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


# Set Pygame to run in headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Since pygame.gfxdraw is not always available, we provide a fallback.
try:
    import pygame.gfxdraw
    GFXDRAW_AVAILABLE = True
except ImportError:
    GFXDRAW_AVAILABLE = False


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Type the displayed words as quickly as possible using the on-screen keyboard to score points before time runs out."
    user_guide = "Controls: Use arrow keys (↑↓←→) to navigate the on-screen keyboard. Press space to type the selected letter and shift to submit the word."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    WIN_SCORE = 100
    MAX_EPISODE_STEPS = 1500

    # --- COLORS ---
    COLOR_BG_START = (10, 5, 25)
    COLOR_BG_END = (30, 10, 60)
    COLOR_TEXT = (220, 220, 255)
    COLOR_NEUTRAL = (100, 100, 150)
    COLOR_CORRECT = (0, 255, 150)
    COLOR_INCORRECT = (255, 80, 80)
    COLOR_CURSOR = (0, 200, 255)
    COLOR_PARTICLE_SUCCESS = [(0, 255, 150), (100, 255, 200), (200, 255, 255)]
    COLOR_PARTICLE_FAIL = [(255, 80, 80), (255, 150, 100), (255, 200, 150)]

    # --- WORD LIST (EMBEDDED) ---
    WORD_LIST = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go",
        "me", "when", "make", "can", "like", "time", "no", "just", "him",
        "know", "take", "people", "into", "year", "your", "good", "some",
        "could", "them", "see", "other", "than", "then", "now", "look",
        "only", "come", "its", "over", "think", "also", "back", "after",
        "use", "two", "how", "our", "work", "first", "well", "way", "even",
        "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "system", "agent", "reward", "state", "action", "policy", "value",
        "future", "model", "learn", "deep", "neural", "network", "data"
    ]

    # --- ON-SCREEN KEYBOARD ---
    KEYBOARD_LAYOUT = [
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm",
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_word = pygame.font.Font(None, 72)
        self.font_keyboard = pygame.font.Font(None, 28)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.current_word = ""
        self.typed_word = ""
        self.word_length = 3
        self.kb_cursor = [0, 0]
        self.particles = []
        self.screen_shake = 0
        self.last_action_time = {'move': -10, 'type': -10, 'submit': -10}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.word_length = 3
        self.typed_word = ""
        self.kb_cursor = [0, 0]
        self.particles = []
        self.screen_shake = 0
        self._generate_word()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        movement, space_action, shift_action = action[0], action[1], action[2]
        
        step_reward = self._handle_input(movement, space_action, shift_action)
        
        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1

        terminated, terminal_reward = self._check_termination()
        self.game_over = terminated
        
        reward = step_reward + terminal_reward

        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, set terminated=True if truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_action, shift_action):
        reward = -0.01 # Small penalty for existing

        # --- Movement ---
        if movement != 0 and self.steps > self.last_action_time['move'] + 3: # 3-frame cooldown
            self.last_action_time['move'] = self.steps
            if movement == 1: self.kb_cursor[1] = max(0, self.kb_cursor[1] - 1)
            elif movement == 2: self.kb_cursor[1] = min(len(self.KEYBOARD_LAYOUT) - 1, self.kb_cursor[1] + 1)
            elif movement == 3: self.kb_cursor[0] = max(0, self.kb_cursor[0] - 1)
            elif movement == 4: self.kb_cursor[0] = min(len(self.KEYBOARD_LAYOUT[self.kb_cursor[1]]) - 1, self.kb_cursor[0] + 1)
            # Ensure cursor x is valid for the new row
            self.kb_cursor[0] = min(self.kb_cursor[0], len(self.KEYBOARD_LAYOUT[self.kb_cursor[1]]) - 1)

        # --- Type a character (Space button) ---
        if space_action == 1 and self.steps > self.last_action_time['type'] + 5: # 5-frame cooldown
            self.last_action_time['type'] = self.steps
            char_to_type = self.KEYBOARD_LAYOUT[self.kb_cursor[1]][self.kb_cursor[0]]
            
            if len(self.typed_word) < len(self.current_word):
                self.typed_word += char_to_type
                typed_idx = len(self.typed_word) - 1
                if self.typed_word[typed_idx] == self.current_word[typed_idx]:
                    reward += 0.1 # Correct character
                else:
                    reward -= 0.2 # Incorrect character
            else:
                reward -= 0.2 # Typed past the end

        # --- Submit word (Shift button) ---
        if shift_action == 1 and self.steps > self.last_action_time['submit'] + 10: # 10-frame cooldown
            self.last_action_time['submit'] = self.steps
            if self.typed_word == self.current_word:
                word_points = len(self.current_word)
                self.score += word_points
                reward += 2.0
                self._create_particles(self.SCREEN_WIDTH // 2, 200, 50, self.COLOR_PARTICLE_SUCCESS)
                self._generate_word()
                self.typed_word = ""
            else:
                self.score = max(0, self.score - 1)
                reward -= 1.0
                self.screen_shake = 5
                self.typed_word = ""

        return reward

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True, 100.0
        if self.time_remaining <= 0:
            return True, -50.0
        return False, 0.0

    def _generate_word(self):
        self.word_length = 3 + (self.score // 20)
        possible_words = [w for w in self.WORD_LIST if len(w) == self.word_length]
        if not possible_words:
            # Fallback if no words of the required length are available
            possible_words = [w for w in self.WORD_LIST if len(w) >= 3]

        self.current_word = self.np_random.choice(possible_words)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "current_word_length": self.word_length
        }
        
    def _render_all(self):
        offset = (0, 0)
        if self.screen_shake > 0:
            offset = (self.np_random.integers(-5, 6), self.np_random.integers(-5, 6))

        self._render_background(offset)
        self._render_particles(offset)
        self._render_game_elements(offset)
        self._render_ui(offset)

    def _render_background(self, offset):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = [
                int(self.COLOR_BG_START[i] * (1 - interp) + self.COLOR_BG_END[i] * interp)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (offset[0], y + offset[1]), (self.SCREEN_WIDTH + offset[0], y + offset[1]))

    def _render_game_elements(self, offset):
        # --- Render Current and Typed Words ---
        word_y = self.SCREEN_HEIGHT // 2 - 40
        
        # Render target word with a subtle glow
        self._draw_text(self.current_word, self.font_word, self.COLOR_NEUTRAL, (self.SCREEN_WIDTH // 2, word_y + offset[1]), shadow=True)

        # Render typed word with character-by-character color
        word_width, _ = self.font_word.size(self.current_word)
        typed_x_start = self.SCREEN_WIDTH // 2 - word_width // 2
        for i, char in enumerate(self.typed_word):
            color = self.COLOR_INCORRECT
            if i < len(self.current_word) and char == self.current_word[i]:
                color = self.COLOR_CORRECT
            
            char_surf = self.font_word.render(char, True, color)
            char_rect = char_surf.get_rect(topleft=(typed_x_start + offset[0], word_y + offset[1]))
            self.screen.blit(char_surf, char_rect)
            typed_x_start += char_surf.get_width()

        # --- Render Keyboard ---
        self._render_keyboard(offset)

    def _render_keyboard(self, offset):
        kb_y_start = self.SCREEN_HEIGHT - 100
        key_size = 30
        key_margin = 5
        
        for r_idx, row in enumerate(self.KEYBOARD_LAYOUT):
            row_width = len(row) * (key_size + key_margin) - key_margin
            kb_x_start = (self.SCREEN_WIDTH - row_width) // 2
            for c_idx, char in enumerate(row):
                key_x = kb_x_start + c_idx * (key_size + key_margin)
                key_y = kb_y_start + r_idx * (key_size + key_margin)
                
                is_cursor_on = (c_idx == self.kb_cursor[0] and r_idx == self.kb_cursor[1])
                
                # Draw key
                key_color = self.COLOR_CURSOR if is_cursor_on else self.COLOR_NEUTRAL
                rect = pygame.Rect(key_x + offset[0], key_y + offset[1], key_size, key_size)
                pygame.draw.rect(self.screen, key_color, rect, border_radius=5)
                
                # Draw character on key
                char_surf = self.font_keyboard.render(char.upper(), True, self.COLOR_BG_START)
                char_rect = char_surf.get_rect(center=rect.center)
                self.screen.blit(char_surf, char_rect)

    def _render_ui(self, offset):
        # Score
        self._draw_text(f"SCORE: {self.score}", self.font_ui, self.COLOR_TEXT, (10 + offset[0], 10 + offset[1]), align="topleft")
        # Timer
        time_str = f"TIME: {max(0, int(self.time_remaining))}"
        self._draw_text(time_str, self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 10 + offset[0], 10 + offset[1]), align="topright")

    def _draw_text(self, text, font, color, pos, align="center", shadow=False):
        text_surf = font.render(text, True, color)
        if shadow:
            shadow_color = tuple(c * 0.5 for c in color)
            shadow_surf = font.render(text, True, shadow_color)
            shadow_rect = shadow_surf.get_rect(**{align: (pos[0]+2, pos[1]+2)})
            self.screen.blit(shadow_surf, shadow_rect)
        
        text_rect = text_surf.get_rect(**{align: pos})
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, x, y, count, colors):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 41),
                'color': colors[self.np_random.integers(len(colors))]
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self, offset):
        if not GFXDRAW_AVAILABLE:
            return
        for p in self.particles:
            x, y = int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1])
            size = int(p['life'] * 0.15)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, p['color'])

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will open a window and let you control the agent.
    
    # Unset the headless environment variable to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Typing Game")

    running = True
    
    print("\n--- Manual Control Instructions ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q/ESC: Quit")
    print("-----------------------------------\n")

    while running:
        action = [0, 0, 0] # [movement, space, shift] (no-op)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                # Map keys to actions
                elif event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1

        # Only step if an action was taken in manual mode
        if any(a != 0 for a in action):
             obs, reward, terminated, truncated, info = env.step(action)
        else: # Or just update the frame for smooth animation
             obs = env._get_observation()
             terminated, truncated = env.game_over, env.steps >= env.MAX_EPISODE_STEPS
        
        if 'reward' in locals() and reward != 0:
             print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
        
        if terminated or truncated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']}")
            obs, info = env.reset()

        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_surf.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()