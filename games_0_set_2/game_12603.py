import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:08:19.434941
# Source Brief: brief_02603.md
# Brief Index: 2603
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A stealth action/puzzle game where the player must eliminate waves of
    color-coded enemies by matching their weapon color. Mismatched attacks
    trigger an alarm and end the game. Chaining same-color eliminations
    builds a combo for bonus points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A stealth puzzle game where you must eliminate color-coded enemies. "
        "Match your weapon color to the target to score points and build combos, but a single mismatch will trigger the alarm!"
    )
    user_guide = (
        "Use ↑/↓ to select a weapon and space to confirm. Use arrow keys to select a target and space to execute. Press shift to cancel targeting."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_AREA_Y_START = 50
    UI_AREA_Y_END = SCREEN_HEIGHT - 80

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 35, 55)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_ALARM = (255, 0, 0)
    COLOR_SUCCESS = (0, 255, 128)
    COLOR_RETICLE = (0, 200, 255)

    ENEMY_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 80, 255),
        "yellow": (255, 255, 80),
        "purple": (200, 80, 255),
    }

    # Game Parameters
    MAX_STEPS = 1500
    ENEMY_RADIUS = 12
    WAVE_CLEAR_DELAY = 60  # frames
    EXECUTION_DELAY = 30   # frames
    COMBO_WINDOW = 150     # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces as required
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.unlocked_colors = []
        self.enemies = []
        self.particles = []
        self.game_phase = "" # e.g., 'WEAPON_SELECT', 'TARGET_SELECT', 'EXECUTE', 'WAVE_CLEAR'
        self.weapon_cards = []
        self.selected_weapon_idx = 0
        self.confirmed_weapon_color_name = ""
        self.selected_target_idx = -1
        self.last_space_held = False
        self.last_shift_held = False
        self.phase_timer = 0
        self.combo_counter = 0
        self.combo_state = {'color': None, 'timer': 0}
        self.execution_data = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.unlocked_colors = ["red", "green", "blue"]
        
        self._start_new_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small time penalty

        # --- Update Game State ---
        self._update_particles()
        self._update_enemies()
        if self.combo_state['timer'] > 0:
            self.combo_state['timer'] -= 1
        else:
            self.combo_counter = 0 # Combo breaks if timer runs out

        # --- Handle Input and Phase Transitions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        if self.phase_timer > 0:
            self.phase_timer -= 1
        else:
            if self.game_phase == 'WAVE_CLEAR':
                self._start_new_wave()
            elif self.game_phase == 'EXECUTE':
                reward += self._resolve_execution()
                self.execution_data = None
                if not self.game_over:
                    if not self.enemies:
                        self.game_phase = 'WAVE_CLEAR'
                        self.phase_timer = self.WAVE_CLEAR_DELAY
                        reward += 100 # Wave clear bonus
                    else:
                        self._transition_to_weapon_select()

            elif self.game_phase == 'WEAPON_SELECT':
                if movement == 1: # Up
                    self.selected_weapon_idx = (self.selected_weapon_idx - 1) % len(self.weapon_cards)
                elif movement == 2: # Down
                    self.selected_weapon_idx = (self.selected_weapon_idx + 1) % len(self.weapon_cards)
                
                if space_press:
                    self.confirmed_weapon_color_name = self.weapon_cards[self.selected_weapon_idx]
                    self._transition_to_target_select()
                    # SFX: UI_Confirm_A
            
            elif self.game_phase == 'TARGET_SELECT':
                num_enemies = len(self.enemies)
                if num_enemies > 0:
                    if movement == 1 or movement == 4: # Up or Right
                        self.selected_target_idx = (self.selected_target_idx + 1) % num_enemies
                    elif movement == 2 or movement == 3: # Down or Left
                        self.selected_target_idx = (self.selected_target_idx - 1) % num_enemies

                if shift_press:
                    self._transition_to_weapon_select()
                    # SFX: UI_Cancel
                elif space_press and self.selected_target_idx != -1:
                    self.game_phase = 'EXECUTE'
                    self.phase_timer = self.EXECUTION_DELAY
                    target_pos = self.enemies[self.selected_target_idx]['pos']
                    self.execution_data = {'target_pos': target_pos, 'color': self.ENEMY_COLORS[self.confirmed_weapon_color_name]}
                    # SFX: Execute_Start
        
        # --- Check Termination ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _start_new_wave(self):
        self.wave += 1
        self.enemies.clear()
        self.particles.clear()
        
        # Unlock new colors at specific waves
        if self.wave == 10 and "yellow" not in self.unlocked_colors:
            self.unlocked_colors.append("yellow")
        if self.wave == 20 and "purple" not in self.unlocked_colors:
            self.unlocked_colors.append("purple")

        num_enemies = min(3 + self.wave // 2, 10)
        enemy_speed = 0.5 + (self.wave // 5) * 0.1

        enemy_colors_in_wave = set()
        for _ in range(num_enemies):
            color_name = self.np_random.choice(self.unlocked_colors)
            enemy_colors_in_wave.add(color_name)
            self.enemies.append({
                'pos': pygame.Vector2(
                    self.np_random.uniform(self.ENEMY_RADIUS, self.SCREEN_WIDTH - self.ENEMY_RADIUS),
                    self.np_random.uniform(self.GAME_AREA_Y_START + self.ENEMY_RADIUS, self.UI_AREA_Y_END - self.ENEMY_RADIUS)
                ),
                'color_name': color_name,
                'target_pos': self._get_random_patrol_point(),
                'speed': enemy_speed,
                'pulse': self.np_random.uniform(0, 2 * math.pi)
            })
        
        # Anti-softlock: Ensure weapon cards are useful
        card_pool = list(enemy_colors_in_wave)
        while len(card_pool) < 3 and len(self.unlocked_colors) > len(card_pool):
            c = self.np_random.choice(self.unlocked_colors)
            if c not in card_pool:
                card_pool.append(c)
        self.np_random.shuffle(card_pool)
        self.weapon_cards = card_pool[:3]

        self._transition_to_weapon_select()
    
    def _transition_to_weapon_select(self):
        self.game_phase = 'WEAPON_SELECT'
        self.selected_weapon_idx = 0
        self.selected_target_idx = -1
        self.confirmed_weapon_color_name = ""
        self.combo_state['timer'] = 0 # Combo breaks between turns

    def _transition_to_target_select(self):
        self.game_phase = 'TARGET_SELECT'
        # Auto-select the first enemy
        self.selected_target_idx = 0 if self.enemies else -1
    
    def _resolve_execution(self):
        reward = 0
        target_enemy = self.enemies[self.selected_target_idx]
        
        if target_enemy['color_name'] == self.confirmed_weapon_color_name:
            # --- SUCCESS ---
            # SFX: Success_Elimination
            self.score += 10
            reward += 1

            # Combo check
            if self.combo_state['color'] == self.confirmed_weapon_color_name and self.combo_state['timer'] > 0:
                self.combo_counter += 1
                combo_bonus = 5 * self.combo_counter
                self.score += combo_bonus
                reward += 5
                # SFX: Combo_Chain
            else:
                self.combo_counter = 1
            
            self.combo_state = {'color': self.confirmed_weapon_color_name, 'timer': self.COMBO_WINDOW}

            self._create_particles(target_enemy['pos'], self.ENEMY_COLORS[target_enemy['color_name']], 30)
            self.enemies.pop(self.selected_target_idx)
        else:
            # --- FAILURE ---
            # SFX: Alarm_Trigger
            self.game_over = True
            self.game_phase = 'GAME_OVER'
            reward -= 50
        
        return reward

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['pulse'] = (enemy['pulse'] + 0.1) % (2 * math.pi)
            if enemy['pos'].distance_to(enemy['target_pos']) < 5:
                enemy['target_pos'] = self._get_random_patrol_point()
            
            direction = (enemy['target_pos'] - enemy['pos']).normalize()
            enemy['pos'] += direction * enemy['speed']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.2

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'radius': self.np_random.uniform(3, 8),
                'color': color,
                'lifespan': self.np_random.integers(20, 40)
            })
    
    def _get_random_patrol_point(self):
        return pygame.Vector2(
            self.np_random.uniform(self.ENEMY_RADIUS, self.SCREEN_WIDTH - self.ENEMY_RADIUS),
            self.np_random.uniform(self.GAME_AREA_Y_START + self.ENEMY_RADIUS, self.UI_AREA_Y_END - self.ENEMY_RADIUS)
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_enemies()
        self._render_ui()
        
        if self.game_phase == 'GAME_OVER':
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_alpha = 150 if (self.steps // 5) % 2 == 0 else 50
            s.fill((self.COLOR_ALARM[0], self.COLOR_ALARM[1], self.COLOR_ALARM[2], flash_alpha))
            self.screen.blit(s, (0,0))
            self._draw_text("ALARM TRIPPED", self.font_large, self.COLOR_TEXT, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GAME_AREA_Y_START), (x, self.UI_AREA_Y_END))
        for y in range(self.GAME_AREA_Y_START, self.UI_AREA_Y_END + 1, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, 0, self.SCREEN_WIDTH, self.GAME_AREA_Y_START), 1)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, self.UI_AREA_Y_END, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.UI_AREA_Y_END), 1)

    def _render_enemies(self):
        for i, enemy in enumerate(self.enemies):
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            color = self.ENEMY_COLORS[enemy['color_name']]
            
            # Pulsing effect
            pulse_factor = 0.8 + 0.2 * (math.sin(enemy['pulse']) + 1) / 2
            inner_radius = int(self.ENEMY_RADIUS * 0.7 * pulse_factor)
            
            # Draw anti-aliased circles
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, (color[0]//2, color[1]//2, color[2]//2))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], inner_radius, color)
            
            # Agent icon (triangle)
            p1 = (pos[0], pos[1] - 4)
            p2 = (pos[0] - 3.5, pos[1] + 2)
            p3 = (pos[0] + 3.5, pos[1] + 2)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_BG)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_BG)

    def _render_particles(self):
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'].x), int(p['pos'].y))
                # Fade color with lifespan
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 40))))
                color = (p['color'][0], p['color'][1], p['color'][2], alpha)
                
                # Create a temporary surface for the particle to handle alpha
                radius = int(p['radius'])
                surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(surf, radius, radius, radius, color)
                self.screen.blit(surf, (pos[0] - radius, pos[1] - radius))

    def _render_ui(self):
        # Top Bar
        self._draw_text(f"SCORE: {self.score}", self.font_medium, self.COLOR_TEXT, 10, 15, align="left")
        self._draw_text(f"WAVE: {self.wave}", self.font_medium, self.COLOR_TEXT, self.SCREEN_WIDTH / 2, 15)
        if self.combo_counter > 1:
            self._draw_text(f"COMBO x{self.combo_counter}", self.font_medium, self.COLOR_SUCCESS, self.SCREEN_WIDTH - 10, 15, align="right")
        
        # Bottom Bar (Weapon Cards)
        card_width, card_height = 100, 50
        total_width = len(self.weapon_cards) * (card_width + 10) - 10
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        
        for i, color_name in enumerate(self.weapon_cards):
            card_x = start_x + i * (card_width + 10)
            card_y = self.SCREEN_HEIGHT - 65
            rect = pygame.Rect(card_x, card_y, card_width, card_height)
            color = self.ENEMY_COLORS[color_name]

            is_selected = (self.game_phase == 'WEAPON_SELECT' and i == self.selected_weapon_idx)
            is_confirmed = (self.game_phase != 'WEAPON_SELECT' and color_name == self.confirmed_weapon_color_name)

            if is_selected or is_confirmed:
                pygame.draw.rect(self.screen, self.COLOR_RETICLE, rect.inflate(6, 6), 2, border_radius=5)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, 2, border_radius=5)
            self._draw_text(color_name.upper(), self.font_small, self.COLOR_BG, rect.centerx, rect.centery)
        
        # Render Reticle and Execution Line
        if self.game_phase == 'TARGET_SELECT' and self.selected_target_idx != -1 and self.selected_target_idx < len(self.enemies):
            self._render_reticle(self.enemies[self.selected_target_idx]['pos'])
        elif self.game_phase == 'EXECUTE' and self.execution_data:
            self._render_reticle(self.execution_data['target_pos'])
            
            # Execution line animation
            progress = 1 - (self.phase_timer / self.EXECUTION_DELAY)
            card_center = (start_x + self.weapon_cards.index(self.confirmed_weapon_color_name) * (card_width + 10) + card_width/2, self.SCREEN_HEIGHT - 40)
            target_pos = self.execution_data['target_pos']
            
            interp_pos = card_center + (target_pos - pygame.Vector2(card_center)) * progress
            pygame.draw.line(self.screen, self.execution_data['color'], card_center, interp_pos, 3)
            pygame.gfxdraw.filled_circle(self.screen, int(interp_pos.x), int(interp_pos.y), 5, self.execution_data['color'])
            pygame.gfxdraw.aacircle(self.screen, int(interp_pos.x), int(interp_pos.y), 5, self.execution_data['color'])

        # Phase instructions
        if self.game_phase == 'WEAPON_SELECT':
            self._draw_text("SELECT WEAPON [↑/↓] + [SPACE]", self.font_small, self.COLOR_TEXT, self.SCREEN_WIDTH/2, self.UI_AREA_Y_END + 10)
        elif self.game_phase == 'TARGET_SELECT':
            self._draw_text("SELECT TARGET [ARROWS] + [SPACE] | CANCEL [SHIFT]", self.font_small, self.COLOR_TEXT, self.SCREEN_WIDTH/2, self.UI_AREA_Y_END + 10)

    def _render_reticle(self, pos):
        radius = self.ENEMY_RADIUS + 8
        angle = (self.steps * 4) % 360  # Rotation effect
        for i in range(4):
            start_angle = math.radians(angle + i * 90 + 20)
            end_angle = math.radians(angle + i * 90 + 70)
            p1 = pos + pygame.Vector2(radius, 0).rotate_rad(start_angle)
            p2 = pos + pygame.Vector2(radius, 0).rotate_rad(end_angle)
            pygame.draw.line(self.screen, self.COLOR_RETICLE, p1, p2, 2)

    def _draw_text(self, text, font, color, x, y, align="center"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = (x, y)
        elif align == "left":
            text_rect.topleft = (x, y)
        elif align == "right":
            text_rect.topright = (x, y)
        
        self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "game_phase": self.game_phase,
            "combo": self.combo_counter,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Testing ---
    # To run this, you'll need to `pip install pygame`
    # and remove/comment out the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Check if we are in a headless environment
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    
    if not is_headless:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Stealth Assassin")
    
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        if not is_headless:
            # Convert observation for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # --- Action Mapping for Human Play ---
        # 0=none, 1=up, 2=down, 3=left, 4=right
        movement = 0
        space = 0
        shift = 0

        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
        else: # In headless mode, just take random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            if not is_headless:
                # Show final frame
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(3000)

        clock.tick(30) # Run at 30 FPS

    env.close()