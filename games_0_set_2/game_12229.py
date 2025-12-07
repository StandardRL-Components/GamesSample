import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:17:37.770432
# Source Brief: brief_02229.md
# Brief Index: 2229
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your server from falling malware by typing to destroy them. "
        "Deploy powerful firewalls to survive increasingly difficult waves."
    )
    user_guide = (
        "Controls: Use ↑↓ to select a target. Press space to type. "
        "Press shift to cycle firewalls, and shift+space to deploy them."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (20, 30, 50)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_UI_TEXT = (200, 220, 255)
    COLOR_MOMENTUM = (255, 200, 0)
    COLOR_FIREWALL_SHIELD = (50, 150, 255)
    COLOR_FIREWALL_SLOW = (100, 100, 255)
    COLOR_FIREWALL_BOMB = (255, 100, 0)
    
    MAX_STEPS = 3000
    MAX_WAVES = 20
    
    WORD_LIST = [
        'root', 'kernel', 'exploit', 'buffer', 'malware', 'firewall', 'trojan', 
        'virus', 'worm', 'phishing', 'proxy', 'server', 'daemon', 'script', 
        'python', 'shell', 'binary', 'socket', 'packet', 'glitch', 'cypher',
        'access', 'denied', 'system', 'breach', 'secure', 'encrypt', 'decrypt',
        'network', 'protocol', 'admin', 'user', 'password', 'token', 'exploit'
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.server_health = 0
        self.momentum = 0
        self.wave = 0
        self.words_on_screen = []
        self.particles = []
        self.target_word_idx = -1
        self.firewalls = {}
        self.firewall_types = ['shield', 'slow', 'bomb']
        self.selected_firewall = 0
        self.active_effects = {}
        self.wave_progress_counter = 0
        self.words_to_clear_wave = 0
        self.word_spawn_timer = 0
        self.word_speed = 0
        self.word_length_min = 0
        self.word_length_max = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.server_health = 100
        self.momentum = 0
        self.wave = 1
        
        self.words_on_screen = []
        self.particles = []
        self.target_word_idx = -1
        
        self.firewalls = {'shield': 1, 'slow': 1, 'bomb': 0}
        self.selected_firewall = 0
        self.active_effects = {}

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._update_wave_difficulty()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- Handle Input and Actions ---
        reward += self._handle_input(action)
        
        # --- Update Game State ---
        self._update_effects()
        word_damage, word_reward = self._update_words()
        reward += word_reward
        
        if 'shield' in self.active_effects:
            if word_damage > 0:
                reward += 1.0 * word_damage # Reward for blocking damage
                # Sound: shield_block.wav
        else:
            if word_damage > 0:
                self.server_health -= word_damage
                self.active_effects['damage_flash'] = 10 # Flash for 10 frames
                # Sound: damage.wav

        self._spawn_words()
        self._update_particles()
        
        self.momentum = max(0, self.momentum - 0.1) # Momentum decay

        # --- Check Wave Completion ---
        if self.wave_progress_counter >= self.words_to_clear_wave:
            self.wave += 1
            reward += 100
            if self.wave > self.MAX_WAVES:
                self.game_over = True
            else:
                self._update_wave_difficulty()
                # Award a new firewall every 5 waves
                if self.wave % 5 == 0:
                    self.firewalls['bomb'] += 1
                if self.wave % 2 == 0:
                    self.firewalls['shield'] += 1
                    self.firewalls['slow'] += 1
                # Sound: wave_complete.wav

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.server_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100 # Penalty for losing
        elif self.wave > self.MAX_WAVES:
            self.game_over = True
            terminated = True
            reward += 500 # Bonus for winning
        
        if self.steps >= self.MAX_STEPS:
            truncated = True # Episode ends due to time limit
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_wave_difficulty(self):
        self.wave_progress_counter = 0
        self.words_to_clear_wave = 5 + self.wave * 2
        self.word_spawn_timer = 0
        
        self.word_speed = 0.5 + (self.wave - 1) * 0.1
        len_increase = (self.wave - 1) // 3
        self.word_length_min = 3 + len_increase
        self.word_length_max = 4 + len_increase

    def _handle_input(self, action):
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # Action: Change target word
        if movement in [1, 2]: # Up or Down
            self._change_target(1 if movement == 2 else -1)

        # Action: Cycle firewall
        if shift_press and not space_held:
            self.selected_firewall = (self.selected_firewall + 1) % len(self.firewall_types)
            # Sound: ui_cycle.wav

        # Action: Deploy firewall
        if space_press and shift_held:
            reward += self._deploy_firewall()

        # Action: Type letter
        if space_press and not shift_held:
            reward += self._type_letter()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward

    def _type_letter(self):
        if self.target_word_idx != -1 and self.target_word_idx < len(self.words_on_screen):
            word_obj = self.words_on_screen[self.target_word_idx]
            if word_obj['progress'] < len(word_obj['text']):
                word_obj['progress'] += 1
                # Sound: type_keypress.wav
                if word_obj['progress'] == len(word_obj['text']):
                    self.momentum = min(100, self.momentum + 20)
                    return 0.5 # Reward for completing word
                return 0.1 # Reward for typing a letter
        return -0.1 # Penalty for typing with no valid target

    def _deploy_firewall(self):
        firewall_name = self.firewall_types[self.selected_firewall]
        if self.firewalls.get(firewall_name, 0) > 0:
            self.firewalls[firewall_name] -= 1
            # Sound: firewall_activate.wav
            if firewall_name == 'shield':
                self.active_effects['shield'] = 300 # 10 seconds at 30fps
            elif firewall_name == 'slow':
                self.active_effects['slow'] = 450 # 15 seconds at 30fps
            elif firewall_name == 'bomb':
                reward = 0
                for _ in range(len(self.words_on_screen)):
                    word_obj = self.words_on_screen.pop(0)
                    self._create_particles(word_obj['pos'], self.COLOR_FIREWALL_BOMB, 30)
                    self.wave_progress_counter += 1
                    reward += 0.5 # Reward for each destroyed word
                self.target_word_idx = -1
                return reward
            return 0.5 # Small reward for deploying
        return 0

    def _update_effects(self):
        for effect in list(self.active_effects.keys()):
            self.active_effects[effect] -= 1
            if self.active_effects[effect] <= 0:
                del self.active_effects[effect]

    def _update_words(self):
        damage_taken = 0
        reward = 0
        
        current_speed = self.word_speed
        if 'slow' in self.active_effects:
            current_speed *= 0.4

        indices_to_remove = []
        for i, word_obj in enumerate(self.words_on_screen):
            # Check for completion and destruction
            if word_obj['progress'] >= len(word_obj['text']):
                word_obj['destroy_timer'] -= 1
                if word_obj['destroy_timer'] <= 0:
                    indices_to_remove.append(i)
                    self._create_particles(word_obj['pos'], self.COLOR_PLAYER, 20)
                    self.score += 10 * (1 + self.momentum / 100)
                    self.wave_progress_counter += 1
                    # Sound: word_destroy.wav
                    continue
            
            # Move word down
            word_obj['pos'][1] += current_speed

            # Check for reaching bottom
            if word_obj['pos'][1] > self.HEIGHT - 40:
                indices_to_remove.append(i)
                damage_taken += 10 # 10 damage per word
        
        # Remove words that are destroyed or have passed the bottom
        if indices_to_remove:
            # Sort indices in reverse to avoid index errors during removal
            for i in sorted(indices_to_remove, reverse=True):
                if i < len(self.words_on_screen):
                    del self.words_on_screen[i]
            
            # If the target was removed, find a new one
            if self.target_word_idx in indices_to_remove:
                self._find_new_target()
            else: # Adjust target index if words before it were removed
                removed_before_target = sum(1 for idx in indices_to_remove if idx < self.target_word_idx)
                self.target_word_idx -= removed_before_target
        
        return damage_taken, reward

    def _spawn_words(self):
        self.word_spawn_timer -= 1
        if self.word_spawn_timer <= 0 and len(self.words_on_screen) < 8:
            text = random.choice(self.WORD_LIST)
            while not (self.word_length_min <= len(text) <= self.word_length_max):
                text = random.choice(self.WORD_LIST)
            
            pos_x = random.randint(50, self.WIDTH - 50 - len(text) * 15)
            new_word = {
                'text': text,
                'progress': 0,
                'pos': [pos_x, 0],
                'destroy_timer': 10 # frames to linger after completion
            }
            self.words_on_screen.append(new_word)
            
            # If no word is targeted, target the new one
            if self.target_word_idx == -1:
                self.target_word_idx = len(self.words_on_screen) - 1

            self.word_spawn_timer = random.randint(60, 100) - self.wave * 2

    def _change_target(self, direction):
        if not self.words_on_screen:
            self.target_word_idx = -1
            return
        
        # Sort words by y-position to make up/down intuitive
        self.words_on_screen.sort(key=lambda w: w['pos'][1])
        
        if self.target_word_idx == -1:
            self.target_word_idx = 0
        else:
            self.target_word_idx = (self.target_word_idx + direction) % len(self.words_on_screen)
        # Sound: ui_select.wav

    def _find_new_target(self):
        if not self.words_on_screen:
            self.target_word_idx = -1
            return
        # Find the highest word on screen to target next
        self.words_on_screen.sort(key=lambda w: w['pos'][1])
        self.target_word_idx = 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        if 'damage_flash' in self.active_effects:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, 50))
            self.screen.blit(flash_surface, (0, 0))
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.server_health,
        }

    def _render_text_with_outline(self, font, text, pos, color, outline_color=(0,0,0)):
        text_surf = font.render(text, True, color)
        outline_surf = font.render(text, True, outline_color)
        self.screen.blit(outline_surf, (pos[0] - 1, pos[1] - 1))
        self.screen.blit(outline_surf, (pos[0] + 1, pos[1] - 1))
        self.screen.blit(outline_surf, (pos[0] - 1, pos[1] + 1))
        self.screen.blit(outline_surf, (pos[0] + 1, pos[1] + 1))
        self.screen.blit(text_surf, pos)
        return text_surf.get_width()

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)
        
        # Scanlines effect
        scanline = pygame.Surface((self.WIDTH, 2), pygame.SRCALPHA)
        scanline.fill((0, 0, 0, 30))
        for y in range(0, self.HEIGHT, 4):
            self.screen.blit(scanline, (0, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = p['life'] * 8
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], min(255, alpha)))

        # Render words
        for i, word_obj in enumerate(self.words_on_screen):
            x, y = int(word_obj['pos'][0]), int(word_obj['pos'][1])
            is_target = (i == self.target_word_idx)
            is_complete = (word_obj['progress'] >= len(word_obj['text']))
            
            if is_complete:
                alpha = max(0, word_obj['destroy_timer'] * 25)
                color = (*self.COLOR_PLAYER, alpha)
            elif is_target:
                color = self.COLOR_PLAYER
            else:
                color = self.COLOR_ENEMY

            typed_part = word_obj['text'][:word_obj['progress']]
            untyped_part = word_obj['text'][word_obj['progress']:]

            # Draw typed part
            typed_width = self._render_text_with_outline(self.font_main, typed_part, (x, y), self.COLOR_PLAYER)
            # Draw untyped part
            self._render_text_with_outline(self.font_main, untyped_part, (x + typed_width, y), color)
            
            if is_target and not is_complete:
                text_height = self.font_main.get_height()
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (x-4, y-2, len(word_obj['text'])*13 + 8, text_height+4), 1)

        # Render active effects
        if 'shield' in self.active_effects:
            shield_alpha = 50 + min(150, self.active_effects['shield'] * 2)
            pygame.draw.rect(self.screen, (*self.COLOR_FIREWALL_SHIELD, shield_alpha), (0, self.HEIGHT - 40, self.WIDTH, 10))
            pygame.draw.line(self.screen, self.COLOR_FIREWALL_SHIELD, (0, self.HEIGHT - 40), (self.WIDTH, self.HEIGHT - 40), 2)
        if 'slow' in self.active_effects:
            slow_alpha = 20 + min(40, self.active_effects['slow'])
            slow_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            slow_surf.fill((*self.COLOR_FIREWALL_SLOW, slow_alpha))
            self.screen.blit(slow_surf, (0,0))
            
    def _render_ui(self):
        # --- Top Bar ---
        self._render_text_with_outline(self.font_ui, f"SCORE: {int(self.score)}", (10, 5), self.COLOR_UI_TEXT)
        self._render_text_with_outline(self.font_ui, f"WAVE: {self.wave}/{self.MAX_WAVES}", (self.WIDTH - 150, 5), self.COLOR_UI_TEXT)
        
        # --- Bottom Bar ---
        ui_y = self.HEIGHT - 25
        # Health
        self._render_text_with_outline(self.font_ui, "SERVER HEALTH", (10, ui_y), self.COLOR_UI_TEXT)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (140, ui_y, 150, 15))
        health_w = max(0, 150 * (self.server_health / 100))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (140, ui_y, health_w, 15))
        
        # Momentum
        self._render_text_with_outline(self.font_ui, "MOMENTUM", (310, ui_y), self.COLOR_UI_TEXT)
        pygame.draw.rect(self.screen, (50,50,50), (400, ui_y, 100, 15))
        momentum_w = max(0, 100 * (self.momentum / 100))
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM, (400, ui_y, momentum_w, 15))

        # Firewalls
        fw_x = 520
        self._render_text_with_outline(self.font_ui, "FIREWALL:", (fw_x, ui_y-15), self.COLOR_UI_TEXT)
        for i, fw_name in enumerate(self.firewall_types):
            color = self.COLOR_FIREWALL_SHIELD if fw_name == 'shield' else (self.COLOR_FIREWALL_SLOW if fw_name == 'slow' else self.COLOR_FIREWALL_BOMB)
            is_selected = (i == self.selected_firewall)
            
            text_surf = self.font_small.render(f"{fw_name.upper()}: {self.firewalls.get(fw_name, 0)}", True, color if self.firewalls.get(fw_name, 0) > 0 else (100,100,100))
            self.screen.blit(text_surf, (fw_x, ui_y + i * 12 - 5))
            
            if is_selected:
                pygame.draw.rect(self.screen, color, (fw_x - 10, ui_y + i * 12 - 5, 5, 10))

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "SYSTEM OFFLINE" if self.server_health <= 0 else "CONNECTION SECURED"
            self._render_text_with_outline(pygame.font.SysFont("monospace", 40, bold=True), msg, (self.WIDTH/2 - 200, self.HEIGHT/2 - 50), self.COLOR_PLAYER)
            self._render_text_with_outline(self.font_ui, f"FINAL SCORE: {int(self.score)}", (self.WIDTH/2 - 100, self.HEIGHT/2 + 10), self.COLOR_UI_TEXT)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # To use this, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cyber Sentinel")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Remove the validation check from the interactive loop
    # as it's meant for init-time verification.
    
    while running:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3 # Unused in this mapping
        elif keys[pygame.K_RIGHT]: movement = 4 # Unused in this mapping
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()