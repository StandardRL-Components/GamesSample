import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:14:31.704474
# Source Brief: brief_00972.md
# Brief Index: 972
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend your guild from waves of enemies by strategically placing magical runes on the battlefield."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to place the selected rune and 'shift' to cycle between available runes."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_WIDTH = 160
    GAME_WIDTH = SCREEN_WIDTH - UI_WIDTH
    GRID_SIZE = 40
    GRID_W, GRID_H = GAME_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE

    # Colors
    COLOR_BG = (15, 19, 25)
    COLOR_GRID = (30, 35, 45)
    COLOR_GUILD = (60, 180, 75)
    COLOR_GUILD_HEALTH = (100, 220, 115)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_ENEMY_HEALTH = (255, 100, 125)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (25, 30, 40)
    
    RUNE_COLORS = {
        "slow": (66, 135, 245),
        "damage": (250, 190, 15),
        "block": (170, 170, 170)
    }

    # Game Parameters
    MAX_STEPS = 30 * 60 # 60 seconds at 30 FPS
    TOTAL_WAVES = 20
    GUILD_START_HEALTH = 100
    GUILD_POS = (GAME_WIDTH // 2, SCREEN_HEIGHT // 2)
    GUILD_SIZE = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_medium = pygame.font.SysFont("Arial", 20)
        self.font_large = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        # Initialize state variables
        self._initialize_state()
        

    def _initialize_state(self):
        self.guild_health = self.GUILD_START_HEALTH
        self.guild_level = 1
        self.wave_number = 0
        self.wave_in_progress = False
        self.inter_wave_timer = 0
        self.enemies = []
        self.runes = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_rune_idx = 0
        self.unlocked_runes = ["slow"]
        
        self.rune_cooldowns = {"slow": 0, "damage": 0, "block": 0}
        self.rune_cooldown_max = {"slow": 150, "damage": 240, "block": 450}

        self.previous_space_held = False
        self.previous_shift_held = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_state()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self._spawn_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.previous_space_held
        shift_pressed = shift_held and not self.previous_shift_held
        self._handle_input(movement, space_pressed, shift_pressed)
        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

        # --- Update Game State ---
        self._update_cooldowns()
        self._update_runes()
        self._move_enemies()
        self._apply_rune_effects()
        self._update_particles()
        
        # --- Check for wave completion ---
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.inter_wave_timer = 90 # 3 second delay
            self.reward_this_step += 1.0
            self.score += 10
            self.guild_level += 1
            if self.guild_level >= 5 and "damage" not in self.unlocked_runes:
                self.unlocked_runes.append("damage")
            if self.guild_level >= 10 and "block" not in self.unlocked_runes:
                self.unlocked_runes.append("block")
        
        if not self.wave_in_progress and self.inter_wave_timer > 0:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer == 0 and self.wave_number < self.TOTAL_WAVES:
                self._spawn_next_wave()

        # --- Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.guild_health <= 0:
                self.reward_this_step = -100
            elif self.wave_number >= self.TOTAL_WAVES and not self.enemies:
                self.reward_this_step = 100
                self.score += 1000
            self.game_over = True

        return self._get_observation(), self.reward_this_step, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)
        
        # Cycle rune
        if shift_pressed:
            self.selected_rune_idx = (self.selected_rune_idx + 1) % len(self.unlocked_runes)
            # sfx: UI_Cycle.wav

        # Deploy rune
        if space_pressed:
            rune_type = self.unlocked_runes[self.selected_rune_idx]
            if self.rune_cooldowns[rune_type] == 0:
                is_occupied = any(r['grid_pos'] == self.cursor_pos for r in self.runes)
                if not is_occupied:
                    self.runes.append({
                        "type": rune_type,
                        "grid_pos": list(self.cursor_pos),
                        "pos": (self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2, self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2),
                        "timer": 0,
                    })
                    self.rune_cooldowns[rune_type] = self.rune_cooldown_max[rune_type]
                    # sfx: Rune_Deploy.wav

    def _update_cooldowns(self):
        for r_type in self.rune_cooldowns:
            self.rune_cooldowns[r_type] = max(0, self.rune_cooldowns[r_type] - 1)

    def _update_runes(self):
        for rune in self.runes:
            rune['timer'] += 1

    def _move_enemies(self):
        occupied_cells = {tuple(r['grid_pos']) for r in self.runes if r['type'] == 'block'}

        for enemy in self.enemies:
            target_pos = np.array(self.GUILD_POS, dtype=float)
            enemy_pos = np.array(enemy['pos'], dtype=float)
            
            direction = target_pos - enemy_pos
            dist = np.linalg.norm(direction)
            if dist < enemy['speed']:
                enemy['pos'] = tuple(target_pos)
            else:
                direction /= dist
                
                # Simple pathfinding around block runes
                next_pos = enemy_pos + direction * enemy['speed']
                next_grid_pos = (int(next_pos[0] // self.GRID_SIZE), int(next_pos[1] // self.GRID_SIZE))

                if next_grid_pos in occupied_cells:
                    # Try moving perpendicularly
                    perp_dir1 = np.array([-direction[1], direction[0]])
                    perp_dir2 = np.array([direction[1], -direction[0]])
                    
                    next_pos1 = enemy_pos + perp_dir1 * enemy['speed']
                    next_grid_pos1 = (int(next_pos1[0] // self.GRID_SIZE), int(next_pos1[1] // self.GRID_SIZE))
                    
                    if next_grid_pos1 not in occupied_cells:
                        direction = perp_dir1
                    else:
                        direction = perp_dir2

                enemy['pos'] = tuple(enemy_pos + direction * enemy['speed'])

            # Check for collision with guild
            if np.linalg.norm(np.array(enemy['pos']) - np.array(self.GUILD_POS)) < self.GUILD_SIZE / 2:
                self.guild_health = max(0, self.guild_health - enemy['damage'])
                self.reward_this_step -= 0.1 * enemy['damage']
                enemy['health'] = 0 # Mark for removal
                # sfx: Guild_Damage.wav
                self._create_particles(self.GUILD_POS, self.COLOR_ENEMY, 20)

        self.enemies = [e for e in self.enemies if e['health'] > 0]

    def _apply_rune_effects(self):
        for rune in self.runes:
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(rune['pos']) - np.array(enemy['pos']))
                
                if rune['type'] == 'slow' and dist < self.GRID_SIZE * 1.5:
                    enemy['speed'] = enemy['base_speed'] * 0.5
                
                if rune['type'] == 'damage' and dist < self.GRID_SIZE * 1.2:
                    if rune['timer'] % 30 == 0: # Damage every second
                        enemy['health'] -= 1
                        # sfx: Damage_Tick.wav
                        self._create_particles(enemy['pos'], self.RUNE_COLORS['damage'], 5, 0.5)
                        if enemy['health'] <= 0:
                            self.reward_this_step += 0.1
                            self.score += 1
                            # sfx: Enemy_Die.wav
                            self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _spawn_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            return

        self.wave_in_progress = True
        
        num_enemies = 2 + self.wave_number + (self.wave_number // 5) * 2
        enemy_health = 1 + (self.wave_number // 10)
        enemy_speed = 1.0 + (self.wave_number // 5) * 0.25
        enemy_damage = 1

        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: # top
                pos = (self.np_random.uniform(0, self.GAME_WIDTH), -20)
            elif side == 1: # bottom
                pos = (self.np_random.uniform(0, self.GAME_WIDTH), self.SCREEN_HEIGHT + 20)
            elif side == 2: # left
                pos = (-20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # right
                pos = (self.GAME_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            self.enemies.append({
                "pos": pos,
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed,
                "base_speed": enemy_speed,
                "damage": enemy_damage
            })

    def _check_termination(self):
        win = self.wave_number > self.TOTAL_WAVES and not self.enemies
        loss = self.guild_health <= 0
        return win or loss

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "guild_health": self.guild_health}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.GAME_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.GAME_WIDTH, y))
        
        # Draw Guild
        guild_rect = pygame.Rect(0, 0, self.GUILD_SIZE, self.GUILD_SIZE)
        guild_rect.center = self.GUILD_POS
        pygame.draw.rect(self.screen, self.COLOR_GUILD, guild_rect, border_radius=4)
        
        # Draw Runes
        for rune in self.runes:
            self._render_rune(rune)

        # Draw Enemies
        for enemy in self.enemies:
            # Reset speed before drawing, effects will re-apply next step
            enemy['speed'] = enemy['base_speed']
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            # Health bar
            if enemy['health'] < enemy['max_health']:
                hp_ratio = enemy['health'] / enemy['max_health']
                bar_w = 16
                pygame.draw.rect(self.screen, (50,0,0), (pos[0] - bar_w/2, pos[1] - 15, bar_w, 3))
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (pos[0] - bar_w/2, pos[1] - 15, bar_w * hp_ratio, 3))

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))


        # Draw Cursor
        cursor_world_x = self.cursor_pos[0] * self.GRID_SIZE
        cursor_world_y = self.cursor_pos[1] * self.GRID_SIZE
        cursor_rect = pygame.Rect(cursor_world_x, cursor_world_y, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

    def _render_rune(self, rune):
        pos = (int(rune['pos'][0]), int(rune['pos'][1]))
        color = self.RUNE_COLORS[rune['type']]
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        
        if rune['type'] == 'slow':
            radius = int(self.GRID_SIZE * 1.5)
            alpha = int(30 + pulse * 20)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, radius, radius, radius, (*color, alpha))
            pygame.gfxdraw.aacircle(s, radius, radius, radius, (*color, alpha*2))
            self.screen.blit(s, (pos[0]-radius, pos[1]-radius))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, color)
        
        elif rune['type'] == 'damage':
            glow_radius = int(8 + pulse * 4)
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, (*color, 80))
            pygame.gfxdraw.aacircle(s, glow_radius, glow_radius, glow_radius, (*color, 120))
            self.screen.blit(s, (pos[0]-glow_radius, pos[1]-glow_radius))
            pygame.draw.polygon(self.screen, color, [(pos[0], pos[1]-7), (pos[0]-7, pos[1]+7), (pos[0]+7, pos[1]+7)])

        elif rune['type'] == 'block':
            rect = pygame.Rect(0, 0, self.GRID_SIZE - 4, self.GRID_SIZE - 4)
            rect.center = pos
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255), rect, 1, border_radius=3)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(self.GAME_WIDTH, 0, self.UI_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.GAME_WIDTH, 0), (self.GAME_WIDTH, self.SCREEN_HEIGHT))

        # Guild Health Bar
        health_ratio = self.guild_health / self.GUILD_START_HEALTH
        bar_w, bar_h = self.GAME_WIDTH - 20, 15
        bar_x, bar_y = 10, self.SCREEN_HEIGHT - bar_h - 10
        pygame.draw.rect(self.screen, (50,0,0), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_GUILD_HEALTH, (bar_x, bar_y, bar_w * health_ratio, bar_h), border_radius=4)
        health_text = f"GUILD: {int(self.guild_health)} / {self.GUILD_START_HEALTH}"
        self._render_text(health_text, (bar_x + bar_w/2, bar_y + bar_h/2), self.font_small, self.COLOR_TEXT, align="center")

        # UI Text
        ui_x = self.GAME_WIDTH + self.UI_WIDTH / 2
        self._render_text(f"Wave {self.wave_number}/{self.TOTAL_WAVES}", (ui_x, 30), self.font_large, self.COLOR_TEXT, align="center")
        self._render_text(f"Score: {self.score}", (ui_x, 70), self.font_medium, self.COLOR_TEXT, align="center")
        self._render_text("Runes", (ui_x, 120), self.font_medium, self.COLOR_TEXT, align="center")
        
        # Rune selection UI
        y_offset = 150
        for i, r_type in enumerate(self.unlocked_runes):
            is_selected = i == self.selected_rune_idx
            color = self.RUNE_COLORS[r_type]
            
            # Selection box
            if is_selected:
                pygame.draw.rect(self.screen, (255,255,255), (self.GAME_WIDTH + 10, y_offset - 5, self.UI_WIDTH - 20, 50), 2, border_radius=5)

            # Rune icon and name
            self._render_text(r_type.upper(), (self.GAME_WIDTH + 20, y_offset + 15), self.font_medium, color, align="midleft")

            # Cooldown bar
            cooldown_ratio = self.rune_cooldowns[r_type] / self.rune_cooldown_max[r_type]
            if cooldown_ratio > 0:
                cd_rect = pygame.Rect(self.GAME_WIDTH + 15, y_offset, (self.UI_WIDTH - 30) * cooldown_ratio, 40)
                s = pygame.Surface(cd_rect.size, pygame.SRCALPHA)
                s.fill((50,50,50,180))
                self.screen.blit(s, cd_rect.topleft)

            y_offset += 60

        if self.game_over:
            s = pygame.Surface((self.GAME_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY!" if self.guild_health > 0 and self.wave_number > self.TOTAL_WAVES else "DEFEAT"
            color = self.COLOR_GUILD_HEALTH if self.guild_health > 0 else self.COLOR_ENEMY
            self._render_text(msg, (self.GAME_WIDTH/2, self.SCREEN_HEIGHT/2), self.font_large, color, align="center")

    def _render_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "midleft":
            text_rect.midleft = pos
        elif align == "midright":
            text_rect.midright = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": life,
                "max_life": life,
                "size": self.np_random.uniform(1, 3),
                "color": color
            })

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc. depending on your OS
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Guild Defense")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Manual Control ---
        movement = 0 # none
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        # Use get_just_pressed for shift to avoid rapid cycling
        current_shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if current_shift_pressed and not env.previous_shift_held:
            shift = 1
        
        current_space_pressed = keys[pygame.K_SPACE]
        if current_space_pressed:
            space = 1

        action = [movement, space, shift] # Note: action space expects held, not just pressed for space/shift
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step([movement, 1 if current_space_pressed else 0, 1 if current_shift_pressed else 0])
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()