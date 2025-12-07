import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:12:54.170298
# Source Brief: brief_00332.md
# Brief Index: 332
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Banish spectral words from a haunted library by finding their constituent letters "
        "in a shifting shadow lexicon before they fully materialize."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to select a letter. "
        "Hold shift to slow down time."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_SHELF = (30, 20, 40)
        self.COLOR_BOOK = (40, 30, 50)
        self.COLOR_SPECTRAL = (200, 220, 255)
        self.COLOR_SPECTRAL_GLOW = (100, 120, 200)
        self.COLOR_SHADOW = (255, 200, 100)
        self.COLOR_SHADOW_GLOW = (180, 140, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 200)
        self.COLOR_TIME_ORB = (0, 150, 255)
        self.COLOR_TIME_ORB_EMPTY = (50, 50, 80)
        self.COLOR_TIME_SLOW_VIGNETTE = (50, 100, 200)

        # Game Data: Lexicon
        self.LEXICON = [
            ["SKY", "SUN", "DEW"],
            ["WATER", "EARTH", "EMBER"],
            ["FOREST", "SPIRIT", "SHADOW"],
            ["CRYSTAL", "ETHEREAL", "ANCIENT"],
            ["CELESTIAL", "PHANTASM", "FORGOTTEN"],
        ]
        self.ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_word = pygame.font.Font(None, 64)
        self.font_letter = pygame.font.Font(None, 48)
        self.font_ui = pygame.font.Font(None, 28)

        # --- State Variables ---
        # These are reset in self.reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lexicon_page = 0
        self.active_words = []
        self.shadow_letters = []
        self.particles = []
        
        self.cursor_index = 0
        self.last_movement_action = 0
        self.last_space_held = False
        
        self.time_slow_charges = 0
        self.time_slow_recharge_progress = 0
        self.time_slow_active = False
        self.incorrect_guess_penalty_timer = 0
        
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_index = 0
        self.last_movement_action = 0
        self.last_space_held = False
        
        self.time_slow_charges = 3
        self.time_slow_recharge_progress = 0
        self.time_slow_active = False
        self.incorrect_guess_penalty_timer = 0
        
        self.particles = []
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        page_index = self.lexicon_page % len(self.LEXICON)
        words_to_spawn = self.LEXICON[page_index]
        self.active_words = []
        
        for i, word_str in enumerate(words_to_spawn):
            angle = (i / len(words_to_spawn)) * 2 * math.pi + math.pi / 4
            start_x = self.WIDTH / 2 + math.cos(angle) * self.WIDTH * 0.4
            start_y = self.HEIGHT / 2 + math.sin(angle) * self.HEIGHT * 0.4
            
            self.active_words.append({
                "original_text": word_str,
                "text": list(word_str),
                "pos": pygame.Vector2(start_x, start_y),
                "target_pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2 - 50 + i * 70),
                "materialization": 0.0,
                "pulse": random.uniform(0, math.pi * 2)
            })
        self._regenerate_shadow_letters()

    def _regenerate_shadow_letters(self):
        required_chars = set()
        for word in self.active_words:
            required_chars.update(word["text"])
        
        num_required = len(required_chars)
        num_decoys = max(2, 8 - num_required)
        
        decoys = set()
        while len(decoys) < num_decoys:
            char = random.choice(self.ALPHABET)
            if char not in required_chars:
                decoys.add(char)
        
        all_chars = list(required_chars) + list(decoys)
        random.shuffle(all_chars)
        
        self.shadow_letters = []
        grid_w, grid_h = self.WIDTH - 100, 120
        start_x, start_y = 50, self.HEIGHT - grid_h - 10
        cols = 4
        rows = 2
        
        for i, char in enumerate(all_chars):
            col = i % cols
            row = i // cols
            x = start_x + (grid_w / (cols -1)) * col if cols > 1 else start_x + grid_w / 2
            y = start_y + (grid_h / (rows -1)) * row if rows > 1 else start_y + grid_h / 2
            
            self.shadow_letters.append({
                "char": char,
                "pos": pygame.Vector2(x, y),
                "vel": pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
                "bounds": pygame.Rect(x-15, y-15, 30, 30)
            })
        
        self.cursor_index = min(self.cursor_index, len(self.shadow_letters) - 1)

    def step(self, action):
        self.reward_this_step = 0
        self.game_over = self._check_termination()
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
        
        terminated = self._check_termination()
        reward = self._calculate_reward(terminated)
        self.score += reward
        
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        # Only move on new non-zero input
        if movement != 0 and movement != self.last_movement_action:
            cols = 4
            if movement == 1 and self.cursor_index >= cols: # Up
                self.cursor_index -= cols
            elif movement == 2 and self.cursor_index < len(self.shadow_letters) - cols: # Down
                self.cursor_index += cols
            elif movement == 3 and self.cursor_index % cols != 0: # Left
                self.cursor_index -= 1
            elif movement == 4 and (self.cursor_index + 1) % cols != 0 and self.cursor_index + 1 < len(self.shadow_letters): # Right
                self.cursor_index += 1
        self.last_movement_action = movement

        # --- Letter Selection ---
        # On button press (rising edge)
        if space_held and not self.last_space_held:
            self._handle_selection()
        self.last_space_held = space_held

        # --- Time Slow ---
        if shift_held and self.time_slow_charges > 0:
            self.time_slow_active = True
            # Sound: Time slow activate
        else:
            self.time_slow_active = False

    def _handle_selection(self):
        if not self.shadow_letters: return
        
        selected_char = self.shadow_letters[self.cursor_index]["char"]
        
        match_found = False
        for word in self.active_words:
            if selected_char in word["text"]:
                word["text"].remove(selected_char)
                self.reward_this_step += 0.1 # Correct letter
                # Sound: Correct match
                
                if not word["text"]:
                    self.reward_this_step += 2.0 # Banished word
                    self._create_banish_particles(word["pos"])
                    # Sound: Word banish
                
                match_found = True
                break
        
        if not match_found:
            self.reward_this_step -= 0.5 # Incorrect letter
            self.incorrect_guess_penalty_timer = self.FPS // 2 # 0.5 second penalty
            # Sound: Incorrect match
        
        # Remove completed words after iterating
        self.active_words = [w for w in self.active_words if w["text"]]
        
        self._regenerate_shadow_letters()

    def _update_game_state(self):
        # Time Slow Logic
        if self.time_slow_active:
            if self.time_slow_charges > 0:
                self.time_slow_charges -= 1/self.FPS # Deplete one charge per second
        else:
            self.time_slow_recharge_progress += 1 / (self.FPS * 5) # 5 seconds to recharge
            if self.time_slow_recharge_progress >= 1:
                self.time_slow_charges = min(3, self.time_slow_charges + 1)
                self.time_slow_recharge_progress = 0
                # Sound: Charge gained

        # Word materialization
        base_rate = 0.05 + self.lexicon_page * 0.02
        rate_multiplier = 0.2 if self.time_slow_active else 1.0
        if self.incorrect_guess_penalty_timer > 0:
            self.incorrect_guess_penalty_timer -= 1
            rate_multiplier *= 3.0 # Penalty makes words materialize 3x faster
            
        for word in self.active_words:
            word["materialization"] += (base_rate / self.FPS) * rate_multiplier
            word["materialization"] = min(1.0, word["materialization"])
            word["pos"] = word["pos"].lerp(word["target_pos"], 0.01)
            word["pulse"] += 0.1

        # Shadow letter drifting
        for letter in self.shadow_letters:
            letter["pos"] += letter["vel"]
            if not letter["bounds"].collidepoint(letter["pos"]):
                if letter["pos"].x < letter["bounds"].left or letter["pos"].x > letter["bounds"].right:
                    letter["vel"].x *= -1
                if letter["pos"].y < letter["bounds"].top or letter["pos"].y > letter["bounds"].bottom:
                    letter["vel"].y *= -1
        
        # Particle updates
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.1)

    def _check_termination(self):
        if any(word["materialization"] >= 1.0 for word in self.active_words):
            return True # Loss
        if not self.active_words and self.steps > 0:
            return True # Win
        return False

    def _calculate_reward(self, terminated):
        final_reward = self.reward_this_step
        if terminated:
            if not self.active_words: # Win condition
                final_reward += 50.0
                self.lexicon_page += 1
            else: # Loss condition
                final_reward -= 100.0
        return final_reward
        
    def _create_banish_particles(self, pos):
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": random.randint(20, 40),
                "size": random.uniform(2, 6),
                "color": random.choice([self.COLOR_SPECTRAL, self.COLOR_SHADOW, (255, 255, 255)])
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_spectral_words()
        self._render_shadow_letters()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Bookshelves
        for i in range(8):
            pygame.draw.rect(self.screen, self.COLOR_SHELF, (i * 90 - 20, 0, 10, self.HEIGHT))
            for j in range(10):
                pygame.draw.rect(self.screen, self.COLOR_BOOK, (i*90-15, j*40+10, 70, 35))
        # Candlelight glow
        self._draw_glowing_circle(self.screen, (50, 50), 100, (255, 180, 50), 0.1)
        self._draw_glowing_circle(self.screen, (self.WIDTH - 50, self.HEIGHT-50), 120, (255, 180, 50), 0.15)
    
    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["size"]), p["color"] + (int(255 * (p["life"]/40.0)),))

    def _render_spectral_words(self):
        for word in self.active_words:
            text = "".join(word["text"])
            alpha = int(word["materialization"] * 200 + 55)
            glow_size = 1 + math.sin(word["pulse"]) * 2
            self._draw_text_with_glow(self.font_word, text, word["pos"], self.COLOR_SPECTRAL, self.COLOR_SPECTRAL_GLOW, alpha, glow_size)

    def _render_shadow_letters(self):
        # Draw letters
        for i, letter in enumerate(self.shadow_letters):
            alpha = 255
            glow_size = 2
            self._draw_text_with_glow(self.font_letter, letter["char"], letter["pos"], self.COLOR_SHADOW, self.COLOR_SHADOW_GLOW, alpha, glow_size)

        # Draw cursor
        if self.shadow_letters:
            cursor_pos = self.shadow_letters[self.cursor_index]["pos"]
            rect = pygame.Rect(cursor_pos.x - 25, cursor_pos.y - 25, 50, 50)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=5)
            # Add a soft glow to the cursor
            glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_CURSOR + (50,), (0,0,60,60), 10, border_radius=10)
            self.screen.blit(glow_surf, (rect.x-5, rect.y-5))

    def _render_ui(self):
        # Time slow vignette
        if self.time_slow_active:
            vignette = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(vignette, self.COLOR_TIME_SLOW_VIGNETTE + (0,), (50, 50, self.WIDTH-100, self.HEIGHT-100))
            pygame.gfxdraw.filled_circle(vignette, self.WIDTH//2, self.HEIGHT//2, self.WIDTH, (0,0,0,0))
            vignette.set_alpha(80)
            self.screen.blit(vignette, (0,0))

        # Time slow orbs
        for i in range(3):
            pos = (30 + i * 30, self.HEIGHT - 170)
            if i < self.time_slow_charges:
                self._draw_glowing_circle(self.screen, pos, 10, self.COLOR_TIME_ORB, 1.0)
            else:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_TIME_ORB_EMPTY)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_TIME_ORB_EMPTY)
        
        # Score and Page
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        page_text = self.font_ui.render(f"Lexicon Page: {self.lexicon_page + 1}", True, self.COLOR_UI_TEXT)
        self.screen.blit(page_text, (self.WIDTH - page_text.get_width() - 10, 10))
        
        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "PAGE CLEARED" if not self.active_words else "THE LIBRARY IS CONSUMED"
            color = (100, 255, 100) if not self.active_words else (255, 100, 100)
            self._draw_text_with_glow(self.font_word, msg, (self.WIDTH/2, self.HEIGHT/2), color, color, 255, 3)

    def _draw_text_with_glow(self, font, text, pos, text_color, glow_color, alpha, glow_size):
        text_surf = font.render(text, True, text_color)
        glow_surf = font.render(text, True, glow_color)
        
        text_surf.set_alpha(alpha)
        glow_surf.set_alpha(int(alpha * 0.5))

        # Blit glow in multiple directions
        for dx in [-glow_size, glow_size]:
            for dy in [-glow_size, glow_size]:
                self.screen.blit(glow_surf, (pos[0] - glow_surf.get_width()/2 + dx, pos[1] - glow_surf.get_height()/2 + dy))
        
        self.screen.blit(text_surf, (pos[0] - text_surf.get_width()/2, pos[1] - text_surf.get_height()/2))

    def _draw_glowing_circle(self, surface, pos, radius, color, intensity):
        c = color + (int(255 * intensity * 0.1),)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), c)
        c = color + (int(255 * intensity * 0.2),)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.7), c)
        c = color + (int(255 * intensity),)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.4), c)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lexicon_page": self.lexicon_page,
            "words_left": len(self.active_words),
            "time_slow_charges": self.time_slow_charges
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-comment the line below to run with a visible display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Spectral Lexicon")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
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
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()