
# Generated: 2025-08-27T18:13:00.711676
# Source Brief: brief_01762.md
# Brief Index: 1762

        
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
        "Controls: Use arrow keys to move the selector. Press space to select a fraction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fraction Frenzy: A fast-paced puzzle game. Match the target fraction at the top "
        "with its equivalent in the grid before the timer runs out. You have 3 attempts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.FPS = 30

        # --- Visuals ---
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        self.COLOR_BG = (15, 23, 42)
        self.COLOR_GRID = (30, 41, 59)
        self.COLOR_TEXT = (226, 232, 240)
        self.COLOR_TARGET = (59, 130, 246)
        self.COLOR_CURSOR = (250, 204, 21, 100)
        self.COLOR_CORRECT = (34, 197, 94)
        self.COLOR_INCORRECT = (239, 68, 68)
        self.COLOR_TIMER = (253, 224, 71)

        # --- Game Constants ---
        self.GRID_ROWS, self.GRID_COLS = 2, 4
        self.MAX_ATTEMPTS = 3
        self.WIN_CONDITION = 5
        self.MAX_EPISODE_STEPS = 1000
        self.TIME_PER_TARGET = 15 * self.FPS # 15 seconds
        self.MOVE_COOLDOWN_FRAMES = 5

        # Etc...        
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminated = False
        self.attempts_left = 0
        self.correct_matches = 0
        self.cursor_pos = [0, 0]
        self.prev_space_held = False
        self.move_cooldown = 0
        self.particles = []
        self.feedback_effect = {"type": None, "timer": 0}
        self.round_timer = 0
        self.target_fraction = (0, 0)
        self.grid_fractions = []
        self.grid_positions = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminated = False

        self.attempts_left = self.MAX_ATTEMPTS
        self.correct_matches = 0
        self.cursor_pos = [0, 0]
        self.prev_space_held = False
        self.move_cooldown = 0
        self.particles = []
        self.feedback_effect = {"type": None, "timer": 0}

        self._generate_new_round()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.terminated = False

        if self.game_over:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        # Update game logic
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)

        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        # 1. Handle Movement
        if movement != 0 and self.move_cooldown <= 0:
            prev_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[0] -= 1  # Up
            elif movement == 2: self.cursor_pos[0] += 1  # Down
            elif movement == 3: self.cursor_pos[1] -= 1  # Left
            elif movement == 4: self.cursor_pos[1] += 1  # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_ROWS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_COLS - 1)

            if prev_pos != self.cursor_pos:
                reward -= 0.1
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
                # Sound: "cursor_move.wav"

        # 2. Handle Selection (on key press)
        selection_made = space_held and not self.prev_space_held
        if selection_made:
            reward += 0.1
            selected_index = self.cursor_pos[0] * self.GRID_COLS + self.cursor_pos[1]
            selected_fraction = self.grid_fractions[selected_index]
            
            target_val = self.target_fraction[0] / self.target_fraction[1]
            selected_val = selected_fraction[0] / selected_fraction[1]

            cursor_center = self.grid_positions[selected_index]

            if abs(target_val - selected_val) < 1e-6:  # Correct
                # Sound: "correct.wav"
                self.score += 10 + int(self.round_timer / self.FPS)
                reward += 10
                self.correct_matches += 1
                self._spawn_particles(cursor_center, self.COLOR_CORRECT)
                self.feedback_effect = {"type": "correct", "timer": 10}

                if self.correct_matches >= self.WIN_CONDITION:
                    self.game_over = True
                    reward += 100
                else:
                    self._generate_new_round()
            else:  # Incorrect
                # Sound: "incorrect.wav"
                reward -= 5
                self.attempts_left -= 1
                self._spawn_particles(cursor_center, self.COLOR_INCORRECT)
                self.feedback_effect = {"type": "incorrect", "timer": 10}
                if self.attempts_left <= 0:
                    self.game_over = True
                    reward += -100
        
        self.prev_space_held = space_held

        # 3. Update Game Logic
        if not self.game_over:
            self.round_timer -= 1
            if self.round_timer <= 0:
                # Sound: "timeout.wav"
                reward -= 5
                self.attempts_left -= 1
                self.feedback_effect = {"type": "incorrect", "timer": 10}
                if self.attempts_left <= 0:
                    self.game_over = True
                    reward += -100
                else:
                    self._generate_new_round()
        
        self._update_particles()
        if self.feedback_effect["timer"] > 0:
            self.feedback_effect["timer"] -= 1

        # Check termination conditions
        terminated = self._check_termination()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _generate_new_round(self):
        self.round_timer = self.TIME_PER_TARGET
        
        # 1. Generate a simplified base fraction
        n = random.randint(1, 9)
        d = random.randint(n + 1, 10)
        common_divisor = math.gcd(n, d)
        self.target_fraction = (n // common_divisor, d // common_divisor)

        fractions_to_display = []

        # 2. Create the correct, un-simplified equivalent
        multiplier = random.randint(2, 5)
        correct_fraction = (self.target_fraction[0] * multiplier, self.target_fraction[1] * multiplier)
        fractions_to_display.append(correct_fraction)
        
        # 3. Generate distractors
        target_value = self.target_fraction[0] / self.target_fraction[1]
        while len(fractions_to_display) < self.GRID_ROWS * self.GRID_COLS:
            dist_n = random.randint(1, 20)
            dist_d = random.randint(dist_n + 1, 25)
            
            # Ensure no division by zero and create a valid value
            if dist_d == 0: continue
            dist_val = dist_n / dist_d
            
            is_duplicate = any(abs(dist_val - (f[0] / f[1])) < 1e-6 for f in fractions_to_display if f[1] != 0)
            
            if not is_duplicate and abs(dist_val - target_value) > 1e-6:
                fractions_to_display.append((dist_n, dist_d))
        
        random.shuffle(fractions_to_display)
        self.grid_fractions = fractions_to_display

    def _check_termination(self):
        if self.game_over:
            self.terminated = True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            self.terminated = True
        return self.terminated
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_feedback_effect()
        self._render_game()
        self._render_particles()

        # Render UI overlay
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "attempts_left": self.attempts_left,
            "correct_matches": self.correct_matches
        }

    def _render_fraction(self, surface, fraction, pos, font, color):
        num_str, den_str = str(fraction[0]), str(fraction[1])
        num_surf = font.render(num_str, True, color)
        den_surf = font.render(den_str, True, color)
        
        line_width = max(num_surf.get_width(), den_surf.get_width()) + 10
        line_height = 3
        total_height = num_surf.get_height() + den_surf.get_height() + line_height + 4

        cx, cy = pos
        
        num_pos = (cx - num_surf.get_width() / 2, cy - total_height / 2)
        den_pos = (cx - den_surf.get_width() / 2, cy + total_height / 2 - den_surf.get_height())
        line_pos_start = (cx - line_width / 2, cy)
        line_pos_end = (cx + line_width / 2, cy)

        surface.blit(num_surf, num_pos)
        surface.blit(den_surf, den_pos)
        pygame.draw.line(surface, color, line_pos_start, line_pos_end, line_height)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Attempts
        attempts_text = self.font_small.render(f"ATTEMPTS: {self.attempts_left}", True, self.COLOR_TEXT)
        self.screen.blit(attempts_text, (self.width - attempts_text.get_width() - 15, 10))

        # Timer bar
        timer_width = self.width * (self.round_timer / self.TIME_PER_TARGET)
        pygame.draw.rect(self.screen, self.COLOR_TIMER, (0, 0, max(0, timer_width), 5))

        # Target Fraction
        target_title = self.font_small.render("TARGET", True, self.COLOR_TARGET)
        self.screen.blit(target_title, (self.width / 2 - target_title.get_width() / 2, 45))
        self._render_fraction(self.screen, self.target_fraction, (self.width / 2, 100), self.font_large, self.COLOR_TARGET)
        
    def _render_game(self):
        # Background Grid
        for i in range(1, self.GRID_COLS):
            x = i * self.width / self.GRID_COLS
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 150), (x, self.height), 2)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 150), (self.width, 150), 2)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 150 + (self.height - 150)/2), (self.width, 150 + (self.height - 150)/2), 2)

        # Fractions and Cursor
        cell_w = self.width / self.GRID_COLS
        cell_h = (self.height - 150) / self.GRID_ROWS
        self.grid_positions = []
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                cx = c * cell_w + cell_w / 2
                cy = 150 + r * cell_h + cell_h / 2
                self.grid_positions.append((cx, cy))
                
                idx = r * self.GRID_COLS + c
                if idx < len(self.grid_fractions):
                    fraction = self.grid_fractions[idx]
                    
                    # Render cursor highlight
                    if r == self.cursor_pos[0] and c == self.cursor_pos[1] and not self.game_over:
                        cursor_rect = pygame.Rect(c * cell_w, 150 + r * cell_h, cell_w, cell_h)
                        s = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
                        s.fill(self.COLOR_CURSOR)
                        self.screen.blit(s, cursor_rect.topleft)

                    self._render_fraction(self.screen, fraction, (cx, cy), self.font_medium, self.COLOR_TEXT)

    def _spawn_particles(self, pos, color):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "radius": random.uniform(3, 8),
                "color": color,
                "life": self.FPS / 2, # half a second
            })

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
            p["radius"] *= 0.97
            if p["life"] > 0 and p["radius"] > 0.5:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p["color"])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, p["color"])

    def _render_feedback_effect(self):
        if self.feedback_effect["timer"] > 0:
            alpha = int(100 * (self.feedback_effect["timer"] / 10))
            if self.feedback_effect["type"] == "correct":
                color = self.COLOR_CORRECT + (alpha,)
            elif self.feedback_effect["type"] in ["incorrect", "timeout"]:
                color = self.COLOR_INCORRECT + (alpha,)
            else:
                return
            
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (0, 0))

    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 220))
        self.screen.blit(overlay, (0, 0))

        if self.correct_matches >= self.WIN_CONDITION:
            end_text = "YOU WIN!"
            color = self.COLOR_CORRECT
        else:
            end_text = "GAME OVER"
            color = self.COLOR_INCORRECT
        
        text_surf = self.font_large.render(end_text, True, color)
        text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2 - 30))
        self.screen.blit(text_surf, text_rect)

        score_surf = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.width / 2, self.height / 2 + 30))
        self.screen.blit(score_surf, score_rect)
        
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Fraction Frenzy")
    clock = pygame.time.Clock()
    running = True
    game_active = True

    while running:
        # --- Action mapping for human play ---
        action = [0, 0, 0] # Default no-op
        if game_active:
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and (terminated or truncated):
                    print("Resetting environment.")
                    obs, info = env.reset()
                    game_active = True

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if (terminated or truncated) and game_active:
            print(f"Episode finished. Score: {info['score']}. Press 'R' to restart.")
            game_active = False

        clock.tick(env.FPS)

    env.close()
    pygame.quit()