import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:18:34.467795
# Source Brief: brief_00335.md
# Brief Index: 335
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Scribe:
    """Represents a single spectral scribe entity."""
    def __init__(self, word, initial_pos, screen_dims):
        self.word = word
        self.pos = np.array(initial_pos, dtype=float)
        self.target_pos = np.array(initial_pos, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.screen_width, self.screen_height = screen_dims
        self.writing_progress = 0
        self.is_banished = False
        self.banish_timer = 0
        self.radius = 15
        self.last_move_time = 0

    def update(self, writing_speed, time_delta):
        if self.is_banished:
            self.banish_timer += time_delta
            return

        # Update writing progress
        self.writing_progress += writing_speed * time_delta
        self.writing_progress = min(self.writing_progress, len(self.word))

        # Update movement
        self.last_move_time += time_delta
        if self.last_move_time > 2.0: # Change target every 2 seconds
            self.last_move_time = 0
            self.target_pos = np.array([
                random.uniform(self.radius * 2, self.screen_width - self.radius * 2),
                random.uniform(self.radius * 2, self.screen_height / 2)
            ], dtype=float)

        # Smooth movement towards target
        acceleration = (self.target_pos - self.pos) * 0.1 - self.velocity * 0.2
        self.velocity += acceleration * time_delta
        self.pos += self.velocity * time_delta

    def get_visible_text(self):
        return self.word[:int(self.writing_progress)]

    def banish(self):
        self.is_banished = True
        self.banish_timer = 0

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, color, lifespan, radius):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.radius = radius

    def update(self, time_delta):
        self.pos += self.vel * time_delta
        self.lifespan -= time_delta
        # Add some gravity/drag
        self.vel[1] += 10 * time_delta
        self.vel *= 0.99

    def is_alive(self):
        return self.lifespan > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Banish spectral scribes by matching their words from your word bank. "
        "Drag and drop the correct word onto a scribe before time runs out."
    )
    user_guide = (
        "Use ←→ to select a word. Press space to pick it up, then use ↑↓←→ to move it and release space to drop. "
        "Press shift to rewind time."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    COLOR_BG = (10, 20, 35)
    COLOR_BG_ACCENT = (20, 40, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_SCRIBE = (255, 255, 255)
    COLOR_SCRIBE_GLOW = (200, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SUCCESS = (50, 255, 50)
    COLOR_FAIL = (255, 50, 50)
    COLOR_REWIND = (100, 100, 255)
    COLOR_UI_PANEL = (25, 50, 80, 150)
    COLOR_FRAGMENT_SLOT = (40, 70, 110)
    COLOR_FRAGMENT_FILLED = (100, 180, 255)

    WORD_LIST = ["SPECTER", "GHOST", "PHANTOM", "WRAITH", "SPIRIT"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("georgia", 22, bold=True)
        self.font_scribe = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_ui = pygame.font.SysFont("tahoma", 16)

        # Game state variables are initialized in reset()
        self.scribes = []
        self.word_bank = []
        self.particles = []
        self.flash_effect = {"color": None, "timer": 0}
        self.cursor_index = 0
        self.drag_state = None
        self.last_space_held = False
        self.last_shift_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.fragments_collected = 0
        self.writing_speed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.fragments_collected = 0
        self.writing_speed = 1.0

        self.word_bank = random.sample(self.WORD_LIST, len(self.WORD_LIST))
        self.scribes = []
        for i, word in enumerate(self.WORD_LIST):
            x = self.WIDTH * (i + 1) / (len(self.WORD_LIST) + 1)
            y = self.HEIGHT * 0.3 + random.uniform(-30, 30)
            self.scribes.append(Scribe(word, (x, y), (self.WIDTH, self.HEIGHT)))

        self.particles.clear()
        self.flash_effect = {"color": None, "timer": 0}
        self.cursor_index = 0
        self.drag_state = None
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        time_delta = 1 / self.FPS
        reward = 0

        # --- UPDATE GAME LOGIC ---
        self.steps += 1
        self.timer -= 1

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.writing_speed += 0.05

        # Update Scribes
        for scribe in self.scribes:
            scribe.update(self.writing_speed, time_delta)

        # Update Particles
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update(time_delta)
        
        # Update Flash Effect
        if self.flash_effect["timer"] > 0:
            self.flash_effect["timer"] -= time_delta

        # --- HANDLE PLAYER ACTIONS ---
        # Time Rewind (Shift)
        if shift_held and not self.last_shift_held:
            self.timer = min(self.MAX_STEPS, self.timer + 60) # Rewind 2 seconds
            self._create_particles( (self.WIDTH/2, self.HEIGHT/2), self.COLOR_REWIND, 30, is_implosion=True)
            # Sfx: time_rewind.wav
        
        # Drag and Drop (Space)
        space_pressed = space_held and not self.last_space_held
        space_released = not space_held and self.last_space_held

        if self.drag_state is None:
            # Move cursor in word bank
            if movement == 1: self.cursor_index = max(0, self.cursor_index - 1)
            if movement == 2: self.cursor_index = min(len(self.word_bank) - 1, self.cursor_index + 1)
            if movement == 3: self.cursor_index = max(0, self.cursor_index - 1)
            if movement == 4: self.cursor_index = min(len(self.word_bank) - 1, self.cursor_index + 1)

            # Pick up a word
            if space_pressed:
                word_to_drag = self.word_bank[self.cursor_index]
                if word_to_drag is not None:
                    self.drag_state = {
                        "word": word_to_drag,
                        "pos": np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
                    }
                    # Sfx: word_pickup.wav
        else: # Word is being dragged
            # Move dragged word
            move_speed = 10
            if movement == 1: self.drag_state["pos"][1] -= move_speed
            if movement == 2: self.drag_state["pos"][1] += move_speed
            if movement == 3: self.drag_state["pos"][0] -= move_speed
            if movement == 4: self.drag_state["pos"][0] += move_speed
            self.drag_state["pos"][0] = np.clip(self.drag_state["pos"][0], 0, self.WIDTH)
            self.drag_state["pos"][1] = np.clip(self.drag_state["pos"][1], 0, self.HEIGHT)

            # Drop the word
            if space_released:
                # Sfx: word_drop.wav
                dropped_pos = self.drag_state["pos"]
                dropped_word = self.drag_state["word"]
                self.drag_state = None

                # Find closest non-banished scribe
                target_scribe = None
                min_dist = float('inf')
                for scribe in self.scribes:
                    if not scribe.is_banished:
                        dist = np.linalg.norm(dropped_pos - scribe.pos)
                        if dist < min_dist:
                            min_dist = dist
                            target_scribe = scribe
                
                if target_scribe and min_dist < 50: # Dropped close enough
                    if dropped_word == target_scribe.word:
                        # Correct match
                        reward += 10
                        target_scribe.banish()
                        self.fragments_collected += 1
                        self.word_bank[self.word_bank.index(dropped_word)] = None # Remove word from bank
                        self._create_particles(target_scribe.pos, self.COLOR_SUCCESS, 50)
                        # Sfx: correct_match.wav
                    else:
                        # Incorrect match
                        reward -= 5
                        self.flash_effect = {"color": self.COLOR_FAIL, "timer": 0.2}
                        # Sfx: incorrect_match.wav
                
                    # Partial reward for letter matching
                    for c1, c2 in zip(dropped_word, target_scribe.word):
                        if c1 == c2:
                            reward += 1


        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- CHECK TERMINATION ---
        terminated = (self.timer <= 0) or (self.fragments_collected == len(self.WORD_LIST))
        truncated = self.steps >= self.MAX_STEPS
        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.fragments_collected == len(self.WORD_LIST):
                reward += 100 # Victory bonus
                # Sfx: victory.wav
            else:
                reward -= 100 # Failure penalty (optional)
                # Sfx: failure.wav

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "fragments_collected": self.fragments_collected,
        }

    def _render_text_with_glow(self, text, font, pos, color, glow_color):
        text_surf = font.render(text, True, glow_color)
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                self.screen.blit(text_surf, (pos[0] + dx, pos[1] + dy))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        x, y = int(pos[0]), int(pos[1])
        # Draw multiple transparent circles for glow effect
        for i in range(int(radius * 1.5), radius, -2):
            alpha = int(80 * (1 - (i - radius) / (radius * 0.5)))
            pygame.gfxdraw.filled_circle(self.screen, x, y, i, (*glow_color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _create_particles(self, pos, color, count, is_implosion=False):
        for _ in range(count):
            if is_implosion:
                angle = random.uniform(0, 2 * math.pi)
                start_pos = pos + np.array([math.cos(angle), math.sin(angle)]) * 150
                vel = (pos - start_pos) * random.uniform(2, 4)
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(50, 150)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                start_pos = pos
            
            lifespan = random.uniform(0.5, 1.2)
            radius = random.uniform(1, 4)
            self.particles.append(Particle(start_pos, vel, color, lifespan, radius))

    def _render_game(self):
        # Background Bookshelves
        for i in range(0, self.WIDTH, 80):
            pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (i, 0, 60, self.HEIGHT))

        # Render Particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.initial_lifespan))
            color = (*p.color, alpha)
            pos = (int(p.pos[0]), int(p.pos[1]))
            radius = int(p.radius * (p.lifespan / p.initial_lifespan))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, radius), color)

        # Render Scribes
        for scribe in self.scribes:
            if not scribe.is_banished:
                self._draw_glowing_circle(scribe.pos, scribe.radius, self.COLOR_SCRIBE, self.COLOR_SCRIBE_GLOW)
                text_surf = self.font_scribe.render(scribe.get_visible_text(), True, self.COLOR_SCRIBE)
                text_pos = (
                    int(scribe.pos[0] + scribe.radius * 1.5),
                    int(scribe.pos[1] - text_surf.get_height() / 2)
                )
                self.screen.blit(text_surf, text_pos)
            elif scribe.banish_timer < 0.5: # Fade out effect
                alpha = int(255 * (1 - scribe.banish_timer / 0.5))
                pos = (int(scribe.pos[0]), int(scribe.pos[1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(scribe.radius * (1 - scribe.banish_timer / 0.5)), (*self.COLOR_SUCCESS, alpha))

    def _render_ui(self):
        # Word Bank Panel
        panel_height = 80
        panel_surf = pygame.Surface((self.WIDTH, panel_height), pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_PANEL)
        self.screen.blit(panel_surf, (0, self.HEIGHT - panel_height))
        
        # Render Words in Bank
        word_spacing = self.WIDTH / (len(self.word_bank) + 1)
        for i, word in enumerate(self.word_bank):
            if word is None: continue
            
            text_surf = self.font_main.render(word, True, self.COLOR_TEXT)
            text_pos = (
                int(word_spacing * (i + 1) - text_surf.get_width() / 2),
                self.HEIGHT - panel_height / 2 - text_surf.get_height() / 2
            )
            
            if i == self.cursor_index and self.drag_state is None:
                rect = text_surf.get_rect(center=(word_spacing * (i + 1), self.HEIGHT - panel_height / 2))
                rect.inflate_ip(20, 10)
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, 2, 5)

            self.screen.blit(text_surf, text_pos)

        # Render Dragged Word
        if self.drag_state:
            pos = (int(self.drag_state["pos"][0]), int(self.drag_state["pos"][1]))
            self._render_text_with_glow(self.drag_state["word"], self.font_main, pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

        # Top UI Bar (Fragments and Timer)
        ui_bar_height = 40
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (0, 0, self.WIDTH, ui_bar_height))
        
        # Fragments
        frag_text = f"FRAGMENTS: {self.fragments_collected} / {len(self.WORD_LIST)}"
        self._render_text_with_glow(frag_text, self.font_ui, (10, 10), self.COLOR_TEXT, self.COLOR_BG)
        
        # Timer Bar
        timer_width = 200
        timer_x = self.WIDTH - timer_width - 10
        timer_ratio = max(0, self.timer / self.MAX_STEPS)
        pygame.draw.rect(self.screen, self.COLOR_FRAGMENT_SLOT, (timer_x, 10, timer_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_FRAGMENT_FILLED, (timer_x, 10, timer_width * timer_ratio, 20))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (timer_x, 10, timer_width, 20), 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        # Render flash effect
        if self.flash_effect["timer"] > 0:
            flash_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.flash_effect["timer"] / 0.2))
            flash_surf.fill((*self.flash_effect["color"], alpha))
            self.screen.blit(flash_surf, (0,0))
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you must unset the dummy videodriver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Spectral Scribe")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Remove the validation call from the main loop
    # as it's only for initialization verification
    # env.validate_implementation() 
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward}")
            print("Press 'R' to reset.")

        clock.tick(GameEnv.FPS)

    env.close()