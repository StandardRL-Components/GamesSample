import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:18:18.804789
# Source Brief: brief_02233.md
# Brief Index: 2233
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import time

class GameEnv(gym.Env):
    """
    A retro Western-themed word game where players defend their saloon from outlaws
    by forming words. Faster word creation powers special attacks.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your saloon from outlaws in this retro Western-themed word game by "
        "quickly forming words to shoot them down."
    )
    user_guide = (
        "Use ↑↓ to select letters and ←→ to move the cursor. "
        "Press space to add a letter. Press shift to fire the word at an outlaw or use a special attack."
    )
    auto_advance = True

    # A curated list of words for the game.
    WORD_LIST = [
        "ALE", "ARM", "BAD", "BAR", "BAT", "BIG", "CAN", "CAR", "CAT", "COW", "CRY", "CUP", "CUT", "DAY", "DOG",
        "DRY", "EAT", "FAN", "FAR", "FAT", "FEW", "FLY", "FOG", "FOR", "FRY", "FUN", "GUN", "GUY", "HAY", "HEN",
        "HIT", "HOG", "HOT", "HOW", "LAW", "LAY", "LEG", "LET", "LID", "LOG", "LOT", "LOW", "MAD", "MAN", "MAP",
        "MAT", "MEN", "MUD", "NET", "NOW", "ODD", "OIL", "OLD", "ONE", "OWN", "PAD", "PAN", "PAR", "PAT", "PAW",
        "PAY", "PEN", "PET", "PIG", "PIN", "PIT", "POT", "PRO", "RAN", "RAT", "RAW", "RED", "ROB", "ROD", "ROT",
        "ROW", "RUN", "SAD", "SAW", "SAY", "SEE", "SET", "SHY", "SIN", "SIP", "SIR", "SIT", "SIX", "SKY", "SLY",
        "SON", "SUN", "TAN", "TAP", "TAR", "TEA", "TEN", "THE", "TIE", "TIN", "TIP", "TOE", "TOP", "TOY", "TRY",
        "WAR", "WAS", "WAY", "WHO", "WHY", "WIN", "YES", "YET", "YOU", "ZAP", "GOLD", "DUEL", "WEST", "TOWN",
        "BOOT", "HAT", "DUST", "SHOT", "BANG", "WILD", "FAST", "DRAW", "RIFLE", "HORSE", "CACTUS", "SALOON",
        "BOUNTY", "OUTLAW", "SHERIFF", "WANTED"
    ]
    COMMON_LETTERS = "AEIOURSTLN"

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_SKY = (135, 206, 235)
        self.COLOR_GROUND = (181, 147, 112)
        self.COLOR_SALOON = (139, 69, 19)
        self.COLOR_UI_BG = (0, 0, 0, 150)
        self.COLOR_TEXT = (255, 255, 240)
        self.COLOR_HEALTH = (118, 204, 102)
        self.COLOR_HEALTH_DMG = (220, 50, 50)
        self.COLOR_MOMENTUM = (255, 180, 0)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_PROJECTILE_GLOW = (255, 200, 0, 60)
        self.COLOR_OUTLAW = (40, 40, 40)
        self.COLOR_OUTLAW_HAT = (80, 80, 80)
        self.COLOR_INVALID = (255, 50, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_LETTER_SELECT = (100, 255, 100)
        self.COLOR_DAMAGE_TEXT = (255, 200, 0)

        # --- Game Parameters ---
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 5
        self.SALOON_START_HEALTH = 100.0
        self.OUTLAW_SPAWN_POINTS = [(100, 100), (250, 80), (400, 100), (550, 80), (180, 150), (470, 150)]
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.saloon_health = 0.0
        self.momentum = 0.0
        self.wave = 0
        self.outlaws = []
        self.projectiles = []
        self.particles = []
        self.available_letters = []
        self.current_word = ""
        self.letter_selection_index = 0
        self.word_cursor_index = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.time_since_last_word = 0
        self.invalid_word_timer = 0
        self.screen_shake = 0
        self.reward_buffer = 0.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.saloon_health = self.SALOON_START_HEALTH
        self.momentum = 0.0
        self.wave = 0
        
        self.outlaws = []
        self.projectiles = []
        self.particles = []
        
        self.current_word = ""
        self.letter_selection_index = 0
        self.word_cursor_index = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.time_since_last_word = 0
        self.invalid_word_timer = 0
        self.screen_shake = 0
        self.reward_buffer = 0.0

        self._refill_letters()
        self._spawn_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_since_last_word += 1
        self.reward_buffer = 0.0 # Reset per-step reward
        
        self._handle_input(action)
        self._update_game_state()
        
        # Calculate reward
        reward = self.reward_buffer - 0.01 # Small time penalty
        self.score += reward
        
        # Check for termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # --- Movement: Cycle through letters and move cursor ---
        if movement == 1 and self.available_letters: # Up
            self.letter_selection_index = (self.letter_selection_index - 1) % len(self.available_letters)
        elif movement == 2 and self.available_letters: # Down
            self.letter_selection_index = (self.letter_selection_index + 1) % len(self.available_letters)
        elif movement == 3: # Left
            self.word_cursor_index = max(0, self.word_cursor_index - 1)
        elif movement == 4: # Right
            self.word_cursor_index = min(len(self.current_word), self.word_cursor_index + 1)
        
        # --- Space: Add selected letter to word ---
        if space_pressed and self.available_letters:
            # sfx: type_key.wav
            letter = self.available_letters.pop(self.letter_selection_index)
            self.current_word = self.current_word[:self.word_cursor_index] + letter + self.current_word[self.word_cursor_index:]
            self.word_cursor_index += 1
            if self.available_letters:
                self.letter_selection_index = min(self.letter_selection_index, len(self.available_letters) - 1)

        # --- Shift: Submit word or use special attack ---
        if shift_pressed:
            if self.current_word == "":
                self._try_special_attack()
            else:
                self._submit_word()
        
        self.last_space_held, self.last_shift_held = space_held, shift_held

    def _submit_word(self):
        if self.current_word in self.WORD_LIST and self.outlaws:
            # sfx: word_shoot.wav
            target = random.choice(self.outlaws)
            self.projectiles.append({
                "text": self.current_word,
                "pos": np.array([self.WIDTH / 2, self.HEIGHT - 80], dtype=float),
                "target_pos": np.array(target["pos"], dtype=float),
                "target_obj": target,
                "speed": 5 + len(self.current_word) * 0.5,
                "damage": len(self.current_word) ** 1.5
            })
            
            # Momentum gain based on speed
            momentum_gain = max(0, 10 - self.time_since_last_word / 6)
            self.momentum = min(100, self.momentum + momentum_gain)
            self.time_since_last_word = 0
            
            self.current_word = ""
            self.word_cursor_index = 0
            if not self.available_letters:
                self._refill_letters()
        else:
            # sfx: invalid_word.wav
            self.invalid_word_timer = 15 # frames
            # Return letters to pool
            for char in self.current_word:
                self.available_letters.append(char)
            self.current_word = ""
            self.word_cursor_index = 0

    def _try_special_attack(self):
        if self.momentum >= 100 and self.outlaws: # Gatling Gun
            # sfx: gatling_gun.wav
            self.momentum = 0
            self.reward_buffer += 5
            for outlaw in self.outlaws[:5]: # Target up to 5 outlaws
                word = random.choice([w for w in self.WORD_LIST if len(w) == 3])
                self.projectiles.append({
                    "text": word, "pos": np.array([self.WIDTH / 2, self.HEIGHT - 80], dtype=float),
                    "target_pos": np.array(outlaw["pos"], dtype=float), "target_obj": outlaw,
                    "speed": 15, "damage": 10
                })
        elif self.momentum >= 50 and self.outlaws: # Fan the Hammer
            # sfx: fan_the_hammer.wav
            self.momentum -= 50
            self.reward_buffer += 2
            for _ in range(3):
                if not self.outlaws: break
                target = random.choice(self.outlaws)
                word = random.choice([w for w in self.WORD_LIST if len(w) == 3])
                self.projectiles.append({
                    "text": word, "pos": np.array([self.WIDTH / 2, self.HEIGHT - 80], dtype=float),
                    "target_pos": np.array(target["pos"], dtype=float), "target_obj": target,
                    "speed": 10, "damage": 5
                })

    def _update_game_state(self):
        # Update timers
        if self.invalid_word_timer > 0: self.invalid_word_timer -= 1
        if self.screen_shake > 0: self.screen_shake -= 1

        # Update projectiles and check collisions
        for proj in self.projectiles[:]:
            direction = proj["target_pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            if dist < 10:
                self._handle_hit(proj)
                self.projectiles.remove(proj)
                continue
            
            proj["pos"] += (direction / dist) * proj["speed"]
        
        # Update outlaws
        for outlaw in self.outlaws[:]:
            outlaw["attack_cooldown"] -= 1
            if outlaw["attack_cooldown"] <= 0:
                # sfx: enemy_shoot.wav
                self.saloon_health -= outlaw["damage"]
                self.screen_shake = 10
                self.particles.append({"type": "muzzle_flash", "pos": outlaw["pos"], "lifespan": 5})
                self.reward_buffer -= 1 # Penalty for getting hit
                outlaw["attack_cooldown"] = outlaw["attack_speed"] + random.randint(-20, 20)

        # Update particles
        for p in self.particles[:]:
            p["lifespan"] -= 1
            if "vel" in p: p["pos"] += p["vel"]
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # Check for wave completion
        if not self.outlaws and not self.game_over:
            if self.wave >= self.MAX_WAVES:
                self.game_over = True # Win condition
                self.reward_buffer += 100
            else:
                self.reward_buffer += 10
                self._spawn_next_wave()
                self._refill_letters()

    def _handle_hit(self, projectile):
        # sfx: hit_confirm.wav
        target = projectile["target_obj"]
        damage = projectile["damage"]
        
        if target in self.outlaws:
            target["health"] -= damage
            self.reward_buffer += damage * 0.1 # Reward for damage
            
            # Damage text particle
            self.particles.append({
                "type": "text", "text": f"{int(damage)}", "pos": np.copy(target["pos"]),
                "vel": np.array([random.uniform(-0.5, 0.5), -1.5]), "lifespan": 30, "color": self.COLOR_DAMAGE_TEXT
            })
            # Explosion particle
            self.particles.append({"type": "explosion", "pos": target["pos"], "radius": 1, "max_radius": 20 + damage, "lifespan": 10})
            
            if target["health"] <= 0:
                # sfx: outlaw_death.wav
                self.outlaws.remove(target)
                self.reward_buffer += 5 # Reward for kill
                self.score += 100 # Direct score bonus

    def _spawn_next_wave(self):
        self.wave += 1
        num_outlaws = min(len(self.OUTLAW_SPAWN_POINTS), 2 + self.wave)
        spawn_points = random.sample(self.OUTLAW_SPAWN_POINTS, num_outlaws)
        
        for i in range(num_outlaws):
            difficulty_mod = 1 + (self.wave - 1) * 0.1
            self.outlaws.append({
                "pos": spawn_points[i],
                "health": 20 * difficulty_mod,
                "max_health": 20 * difficulty_mod,
                "attack_speed": 150 / difficulty_mod,
                "attack_cooldown": 100 + random.randint(0, 50),
                "damage": 5 * difficulty_mod
            })

    def _refill_letters(self):
        self.available_letters.clear()
        base_word = random.choice(self.WORD_LIST)
        for char in base_word:
            self.available_letters.append(char)
        
        for _ in range(max(0, 10 - len(base_word))):
            self.available_letters.append(random.choice(self.COMMON_LETTERS))
        
        random.shuffle(self.available_letters)
        self.letter_selection_index = 0

    def _check_termination(self):
        if self.saloon_health <= 0:
            self.game_over = True
            self.reward_buffer -= 50 # Penalty for losing
        if self.wave > self.MAX_WAVES and not self.outlaws:
             self.game_over = True # Win condition already handled
        
        return self.game_over

    def _get_observation(self):
        # Apply screen shake
        shake_offset = (0, 0)
        if self.screen_shake > 0:
            shake_offset = (random.randint(-5, 5), random.randint(-5, 5))

        # --- Render everything ---
        self._render_background(shake_offset)
        self._render_saloon(shake_offset)
        self._render_outlaws(shake_offset)
        self._render_projectiles(shake_offset)
        self._render_particles(shake_offset)
        self._render_ui() # UI is not affected by shake
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, offset):
        self.screen.fill(self.COLOR_SKY)
        ground_rect = pygame.Rect(0, self.HEIGHT * 0.6, self.WIDTH, self.HEIGHT * 0.4)
        ground_rect.move_ip(offset)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)
        # Draw some cacti
        for x, y, h in [(50, 280, 40), (580, 260, 60), (320, 350, 20)]:
            pygame.draw.rect(self.screen, (0, 100, 0), (x+offset[0], y-h+offset[1], 10, h))

    def _render_saloon(self, offset):
        saloon_rect = pygame.Rect(self.WIDTH/2 - 150, self.HEIGHT * 0.6 - 180, 300, 180)
        saloon_rect.move_ip(offset)
        pygame.draw.rect(self.screen, self.COLOR_SALOON, saloon_rect)
        pygame.draw.rect(self.screen, (0,0,0), saloon_rect, 3)

    def _render_outlaws(self, offset):
        for outlaw in self.outlaws:
            x, y = int(outlaw["pos"][0] + offset[0]), int(outlaw["pos"][1] + offset[1])
            # Body
            pygame.draw.rect(self.screen, self.COLOR_OUTLAW, (x-7, y, 14, 30))
            # Hat
            pygame.draw.rect(self.screen, self.COLOR_OUTLAW_HAT, (x-12, y-5, 24, 8))
            pygame.draw.rect(self.screen, self.COLOR_OUTLAW_HAT, (x-7, y-12, 14, 8))
            # Health bar
            health_pct = max(0, outlaw["health"] / outlaw["max_health"])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_DMG, (x - 20, y - 25, 40, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (x - 20, y - 25, int(40 * health_pct), 5))

    def _render_projectiles(self, offset):
        for proj in self.projectiles:
            x, y = int(proj["pos"][0] + offset[0]), int(proj["pos"][1] + offset[1])
            text_surf = self.font_m.render(proj["text"], True, self.COLOR_PROJECTILE)
            text_rect = text_surf.get_rect(center=(x, y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, text_rect.centerx, text_rect.centery, int(text_rect.width / 2 + 10), self.COLOR_PROJECTILE_GLOW)
            self.screen.blit(text_surf, text_rect)

    def _render_particles(self, offset):
        for p in self.particles:
            x, y = int(p["pos"][0] + offset[0]), int(p["pos"][1] + offset[1])
            if p["type"] == "text":
                alpha = int(255 * (p["lifespan"] / 30))
                text_surf = self.font_s.render(p["text"], True, p["color"])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, (x, y))
            elif p["type"] == "explosion":
                alpha = int(150 * (p["lifespan"] / 10))
                radius = int(p["max_radius"] * (1 - p["lifespan"] / 10))
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (*self.COLOR_PROJECTILE, alpha))
            elif p["type"] == "muzzle_flash":
                alpha = int(255 * (p["lifespan"] / 5))
                radius = 10
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (*self.COLOR_PROJECTILE, alpha))
                pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius*0.5), (*self.COLOR_TEXT, alpha))

    def _render_ui(self):
        # --- Top Bar (Health, Momentum, Score) ---
        ui_bar = pygame.Surface((self.WIDTH, 50), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        # Health
        health_pct = max(0, self.saloon_health / self.SALOON_START_HEALTH)
        pygame.draw.rect(ui_bar, self.COLOR_HEALTH_DMG, (10, 10, 200, 30))
        pygame.draw.rect(ui_bar, self.COLOR_HEALTH, (10, 10, int(200 * health_pct), 30))
        # Momentum
        momentum_pct = self.momentum / 100.0
        pygame.draw.rect(ui_bar, (80,80,80), (220, 10, 150, 30))
        pygame.draw.rect(ui_bar, self.COLOR_MOMENTUM, (220, 10, int(150 * momentum_pct), 30))
        if self.momentum >= 100:
             special_text = "GATLING!"
        elif self.momentum >= 50:
             special_text = "FAN HAMMER"
        else:
             special_text = "MOMENTUM"
        text = self.font_s.render(special_text, True, self.COLOR_TEXT)
        ui_bar.blit(text, text.get_rect(center=(220+75, 25)))
        # Score & Wave
        score_text = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        ui_bar.blit(score_text, (self.WIDTH - 180, 15))
        wave_text = self.font_s.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        ui_bar.blit(wave_text, (self.WIDTH - 300, 18))
        self.screen.blit(ui_bar, (0, 0))

        # --- Bottom Bar (Word building) ---
        bottom_bar = pygame.Surface((self.WIDTH, 80), pygame.SRCALPHA)
        bottom_bar.fill(self.COLOR_UI_BG)
        
        # Available Letters
        for i, letter in enumerate(self.available_letters):
            color = self.COLOR_LETTER_SELECT if i == self.letter_selection_index else self.COLOR_TEXT
            text_surf = self.font_m.render(letter, True, color)
            bottom_bar.blit(text_surf, (20 + i * 30, 45))
        
        # Current Word
        word_color = self.COLOR_INVALID if self.invalid_word_timer > 0 else self.COLOR_TEXT
        word_surf = self.font_l.render(self.current_word, True, word_color)
        bottom_bar.blit(word_surf, (self.WIDTH / 2, 20))
        # Cursor
        if (self.steps // 15) % 2 == 0:
            cursor_sub = self.current_word[:self.word_cursor_index]
            cursor_x = self.font_l.size(cursor_sub)[0]
            pygame.draw.line(bottom_bar, self.COLOR_CURSOR, (self.WIDTH / 2 + cursor_x, 25), (self.WIDTH / 2 + cursor_x, 25 + 35), 2)

        self.screen.blit(bottom_bar, (0, self.HEIGHT - 80))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "saloon_health": self.saloon_health,
            "momentum": self.momentum,
            "outlaws_remaining": len(self.outlaws)
        }

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # This block is for human play and visualization, and is not used by the tests.
    # It has been modified to use pygame.display for rendering.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Word Saloon")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Mapping from keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_w: 1,
        pygame.K_DOWN: 2,
        pygame.K_s: 2,
        pygame.K_LEFT: 3,
        pygame.K_a: 3,
        pygame.K_RIGHT: 4,
        pygame.K_d: 4,
    }

    while running:
        movement = 0 
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement = key_to_action[event.key]

        # Continuous key presses
        keys = pygame.key.get_pressed()
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # If no keydown event for movement, check continuous press for arrows
        if movement == 0:
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            time.sleep(2)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()