import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:10:42.205324
# Source Brief: brief_00858.md
# Brief Index: 858
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An endless, visually stunning strategic defense game where the player expands
    color-coded fractal defenses to repel waves of anti-fractal entities.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Expand your color-coded fractal defenses to repel waves of anti-fractal entities in this endless strategic defense game."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to build or upgrade. Press shift to cycle through actions."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game Constants ---
        self.CENTER = (self.width // 2, self.height // 2)
        self.MAX_STEPS = 10000
        self.CURSOR_SPEED = 10
        self.ENEMY_BASE_SPEED = 0.5
        self.ENEMY_BASE_HEALTH = 100
        self.WAVE_COOLDOWN = 150 # steps between waves

        # --- Color Palette ---
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_BG_GLOW = (20, 30, 45)
        self.COLOR_CORE = (255, 255, 255)
        self.COLOR_RED = (255, 50, 50)
        self.COLOR_GREEN = (50, 255, 50)
        self.COLOR_BLUE = (50, 100, 255)
        self.COLOR_GRAY = (150, 150, 150)
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_HIGHLIGHT = (255, 200, 0)
        self.COLORS = {'RED': self.COLOR_RED, 'GREEN': self.COLOR_GREEN, 'BLUE': self.COLOR_BLUE, 'GRAY': self.COLOR_GRAY}

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.previous_space_held = False
        self.previous_shift_held = False
        self.fractal_nodes = []
        self.enemies = []
        self.particles = []
        self.core_health = 0
        self.resources = 0
        self.wave = 0
        self.wave_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemies_killed_in_wave = 0
        self.current_action_mode_idx = 0
        self.action_modes = []
        self.upgrades = {}
        
        # self.reset() # This is called by the wrapper/runner
        # self.validate_implementation() # This is for dev, not production

    def _init_action_modes_and_upgrades(self):
        self.action_modes = [
            {'type': 'BUILD', 'color': 'RED'},
            {'type': 'BUILD', 'color': 'GREEN'},
            {'type': 'BUILD', 'color': 'BLUE'},
            {'type': 'UPGRADE', 'name': 'POWER'},
            {'type': 'UPGRADE', 'name': 'RANGE'},
            {'type': 'UPGRADE', 'name': 'ECONOMY'},
        ]
        self.upgrades = {
            'POWER': {'level': 1, 'cost': 50, 'value': 1.0},
            'RANGE': {'level': 1, 'cost': 50, 'value': 60},
            'ECONOMY': {'level': 1, 'cost': 75, 'value': 1.0},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = list(self.CENTER)
        self.previous_space_held = False
        self.previous_shift_held = False
        
        self.fractal_nodes = [{
            'pos': self.CENTER, 'color': 'CORE', 'parent': None, 'level': 1, 'id': 0
        }]
        self.enemies = []
        self.particles = []
        
        self.core_health = 100
        self.resources = 60
        self.wave = 0
        self.wave_timer = self.WAVE_COOLDOWN 
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemies_killed_in_wave = 0

        self._init_action_modes_and_upgrades()
        self.current_action_mode_idx = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.previous_space_held
        shift_press = shift_held and not self.previous_shift_held
        
        reward += self._handle_input(movement, space_press, shift_press)
        
        wave_cleared = self._update_wave_logic()
        if wave_cleared:
            reward += 1.0
            if self.wave % 10 == 0:
                reward += 10.0

        enemy_reward, core_damage = self._update_enemies()
        reward += enemy_reward
        self.core_health -= core_damage
        if core_damage > 0:
            self._create_particles(self.CENTER, self.COLOR_RED, 20, 5, 20)

        self._update_particles()
        
        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

        terminated = self.core_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.core_health <= 0:
                reward = -100.0
            else: # Max steps reached
                reward = 100.0
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        # 1. Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.height)

        # 2. Cycle action mode
        if shift_press:
            self.current_action_mode_idx = (self.current_action_mode_idx + 1) % len(self.action_modes)
            # sfx: UI_Bleep

        # 3. Execute action
        if space_press:
            mode = self.action_modes[self.current_action_mode_idx]
            if mode['type'] == 'BUILD':
                return self._try_build_node(mode['color'])
            elif mode['type'] == 'UPGRADE':
                return self._try_buy_upgrade(mode['name'])
        return 0

    def _try_build_node(self, color):
        cost = int(30 * self.upgrades['ECONOMY']['value'])
        if self.resources < cost:
            # sfx: Error_Sound
            return 0

        # Find closest existing node to cursor to be the parent
        parent_node = min(self.fractal_nodes, key=lambda n: self._dist_sq(n['pos'], self.cursor_pos))
        
        dist_to_parent = math.sqrt(self._dist_sq(parent_node['pos'], self.cursor_pos))
        
        # Must build within a certain range of a parent
        if dist_to_parent < 20 or dist_to_parent > self.upgrades['RANGE']['value']:
            # sfx: Error_Sound
            return 0

        # Prevent building on top of other nodes
        for node in self.fractal_nodes:
            if self._dist_sq(node['pos'], self.cursor_pos) < 20**2:
                # sfx: Error_Sound
                return 0

        self.resources -= cost
        new_node = {
            'pos': tuple(self.cursor_pos),
            'color': color,
            'parent': parent_node['id'],
            'level': 1,
            'id': len(self.fractal_nodes)
        }
        self.fractal_nodes.append(new_node)
        self._create_particles(new_node['pos'], self.COLORS[color], 15, 3, 15)
        # sfx: Build_Success
        return 0.05 # Small reward for building

    def _try_buy_upgrade(self, name):
        upgrade = self.upgrades[name]
        if self.resources >= upgrade['cost']:
            self.resources -= upgrade['cost']
            upgrade['level'] += 1
            upgrade['cost'] = int(upgrade['cost'] * 1.5)
            
            if name == 'POWER': upgrade['value'] *= 1.25
            elif name == 'RANGE': upgrade['value'] *= 1.2
            elif name == 'ECONOMY': upgrade['value'] *= 0.9

            # sfx: Upgrade_Success
            return 0.5 # Small reward for upgrading
        # sfx: Error_Sound
        return 0

    def _update_wave_logic(self):
        if self.enemies_spawned >= self.enemies_in_wave and self.enemies_killed_in_wave >= self.enemies_in_wave:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave += 1
                self.wave_timer = self.WAVE_COOLDOWN
                
                base_enemies = 5 + self.wave * 2
                density_mod = 1.1 ** (self.wave // 5)
                self.enemies_in_wave = int(base_enemies * density_mod)
                
                self.enemies_spawned = 0
                self.enemies_killed_in_wave = 0
                return True # Wave cleared
        elif self.enemies_spawned < self.enemies_in_wave:
            # Spawn enemies periodically
            if self.steps % 10 == 0:
                self._spawn_enemy()
                self.enemies_spawned += 1
        return False

    def _spawn_enemy(self):
        edge = self.np_random.integers(4)
        if edge == 0: pos = [self.np_random.uniform(0, self.width), -10]
        elif edge == 1: pos = [self.np_random.uniform(0, self.width), self.height + 10]
        elif edge == 2: pos = [-10, self.np_random.uniform(0, self.height)]
        else: pos = [self.width + 10, self.np_random.uniform(0, self.height)]
        
        colors = ['RED', 'GREEN', 'BLUE']
        if self.wave >= 20:
            colors.append('GRAY')
        color = self.np_random.choice(colors)

        speed_mod = 0.05 * (self.wave // 10)
        speed = self.ENEMY_BASE_SPEED + speed_mod
        
        self.enemies.append({
            'pos': pos, 'color': color, 'health': self.ENEMY_BASE_HEALTH, 'speed': speed
        })

    def _update_enemies(self):
        reward = 0
        core_damage = 0
        power = self.upgrades['POWER']['value']
        
        for enemy in self.enemies[:]:
            # Move towards center
            direction = np.array(self.CENTER) - np.array(enemy['pos'])
            dist_to_center = np.linalg.norm(direction)
            if dist_to_center > 0:
                direction = direction / dist_to_center
            enemy['pos'][0] += direction[0] * enemy['speed']
            enemy['pos'][1] += direction[1] * enemy['speed']

            # Check for core collision
            if dist_to_center < 15:
                core_damage += 10
                self.enemies.remove(enemy)
                self.enemies_killed_in_wave += 1 # Count as "killed" for wave progression
                # sfx: Core_Damage
                continue

            # Take damage from fractal nodes
            total_damage = 0
            for node in self.fractal_nodes:
                if node['color'] == 'CORE': continue
                
                dist_sq = self._dist_sq(enemy['pos'], node['pos'])
                range_sq = self.upgrades['RANGE']['value'] ** 2
                
                if dist_sq < range_sq:
                    damage = (power * 5000) / (dist_sq + 100) # Inverse square falloff
                    color_mult = 2.0 if enemy['color'] == node['color'] else (0.75 if enemy['color'] != 'GRAY' else 1.0)
                    total_damage += damage * color_mult
            
            enemy['health'] -= total_damage
            if enemy['health'] <= 0:
                reward += 0.1
                self.resources += 5
                self._create_particles(enemy['pos'], self.COLORS[enemy['color']], 10, 2, 10)
                self.enemies.remove(enemy)
                self.enemies_killed_in_wave += 1
                # sfx: Enemy_Destroyed
        
        return reward, core_damage

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_glow()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "core_health": self.core_health,
            "resources": self.resources,
        }

    # --- Helper & Rendering Methods ---

    def _dist_sq(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def _create_particles(self, pos, color, count, speed_max, life_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(10, life_max), 'color': color
            })

    def _render_text(self, text, pos, font, color, shadow_color=(0,0,0), align="topleft"):
        text_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        setattr(text_rect, align, (pos[0]+1, pos[1]+1))
        self.screen.blit(text_surf, text_rect)
        
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surf, text_rect)

    def _render_background_glow(self):
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER[0], self.CENTER[1], 200, self.COLOR_BG_GLOW)

    def _render_game(self):
        # Render connections
        parent_map = {node['id']: node for node in self.fractal_nodes}
        for node in self.fractal_nodes:
            if node['parent'] is not None and node['parent'] in parent_map:
                parent_node = parent_map[node['parent']]
                color = self.COLORS.get(node['color'], self.COLOR_CORE)
                pygame.draw.aaline(self.screen, color, node['pos'], parent_node['pos'], 1)

        # Render fractal nodes and their range
        for node in self.fractal_nodes:
            pos_int = (int(node['pos'][0]), int(node['pos'][1]))
            if node['color'] == 'CORE':
                color = self.COLOR_CORE
                radius = 12
                # Core health pulse
                pulse = abs(math.sin(self.steps * 0.1))
                glow_radius = int(radius * (1.2 + pulse * (self.core_health / 100)))
                glow_color = tuple(int(c * (self.core_health/100)) for c in color)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, (*glow_color, 50))
            else:
                color = self.COLORS[node['color']]
                radius = 8
                # Range indicator
                range_radius = int(self.upgrades['RANGE']['value'])
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], range_radius, (*color, 20))
                # Glow
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius * 3, (*color, 30))

            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

        # Render enemies
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            color = self.COLORS[enemy['color']]
            size = 8
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size * 2, (*color, 80))
            # Body
            points = [
                (pos_int[0], pos_int[1] - size),
                (pos_int[0] - size * 0.866, pos_int[1] + size * 0.5),
                (pos_int[0] + size * 0.866, pos_int[1] + size * 0.5)
            ]
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 15.0)) # Assuming max life is around 15
            color = (*p['color'][:3], max(0, min(255, alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life']/5 + 1), color)

        # Render cursor and potential build
        self._render_cursor()
    
    def _render_cursor(self):
        cursor_pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        mode = self.action_modes[self.current_action_mode_idx]
        
        if mode['type'] == 'BUILD':
            cost = int(30 * self.upgrades['ECONOMY']['value'])
            can_afford = self.resources >= cost
            
            # Find closest parent to show connection
            parent_node = min(self.fractal_nodes, key=lambda n: self._dist_sq(n['pos'], self.cursor_pos))
            dist_to_parent = math.sqrt(self._dist_sq(parent_node['pos'], self.cursor_pos))
            is_valid_dist = 20 < dist_to_parent < self.upgrades['RANGE']['value']

            color = self.COLORS[mode['color']]
            if can_afford and is_valid_dist:
                cursor_color = color
                pygame.draw.aaline(self.screen, (*color, 100), parent_node['pos'], self.cursor_pos)
            else:
                cursor_color = self.COLOR_GRAY

            pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 8, (*cursor_color, 150))
        
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (cursor_pos_int[0] - 10, cursor_pos_int[1]), (cursor_pos_int[0] - 5, cursor_pos_int[1]))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (cursor_pos_int[0] + 5, cursor_pos_int[1]), (cursor_pos_int[0] + 10, cursor_pos_int[1]))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (cursor_pos_int[0], cursor_pos_int[1] - 10), (cursor_pos_int[0], cursor_pos_int[1] - 5))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (cursor_pos_int[0], cursor_pos_int[1] + 5), (cursor_pos_int[0], cursor_pos_int[1] + 10))


    def _render_ui(self):
        # Top bar
        self._render_text(f"WAVE: {self.wave}", (10, 10), self.font_medium, self.COLOR_UI_TEXT)
        self._render_text(f"RESOURCES: {self.resources}", (self.width/2, 10), self.font_medium, self.COLOR_UI_TEXT, align="midtop")
        self._render_text(f"SCORE: {int(self.score)}", (self.width - 10, 10), self.font_medium, self.COLOR_UI_TEXT, align="topright")
        
        # Core Health
        health_percent = max(0, self.core_health / 100)
        health_color = (255 * (1 - health_percent), 255 * health_percent, 50)
        self._render_text(f"CORE: {int(self.core_health)}%", (self.width/2, self.height - 30), self.font_medium, health_color, align="midbottom")
        
        # Wave progress/cooldown
        if self.enemies_killed_in_wave < self.enemies_in_wave and self.wave > 0:
            progress = self.enemies_killed_in_wave / self.enemies_in_wave
            text = f"WAVE IN PROGRESS ({int(progress*100)}%)"
            self._render_text(text, (self.width/2, 40), self.font_small, self.COLOR_UI_TEXT, align="midtop")
        elif self.wave > 0:
            text = f"NEXT WAVE IN {self.wave_timer}"
            self._render_text(text, (self.width/2, 40), self.font_small, self.COLOR_UI_HIGHLIGHT, align="midtop")

        # Action Mode/Upgrade Panel
        panel_rect = pygame.Rect(10, self.height - 115, 230, 105)
        panel_surf = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(panel_surf, panel_rect.topleft)

        for i, mode in enumerate(self.action_modes):
            y_pos = panel_rect.top + 5 + i * 16
            is_selected = i == self.current_action_mode_idx
            color = self.COLOR_UI_HIGHLIGHT if is_selected else self.COLOR_UI_TEXT
            prefix = "> " if is_selected else "  "
            
            if mode['type'] == 'BUILD':
                cost = int(30 * self.upgrades['ECONOMY']['value'])
                text = f"{prefix}BUILD {mode['color']} ({cost} res)"
            else: # UPGRADE
                upgrade = self.upgrades[mode['name']]
                text = f"{prefix}UPG {mode['name']} L{upgrade['level']} ({upgrade['cost']} res)"
            
            self._render_text(text, (panel_rect.left + 5, y_pos), self.font_small, color)

        if self.game_over:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._render_text("GAME OVER", self.CENTER, self.font_large, self.COLOR_RED, align="center")

    def validate_implementation(self):
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
    # --- Manual Play Example ---
    # This block is for local testing and will not be executed by the evaluation server.
    # It allows you to play the game with keyboard controls.
    
    # Un-comment the line below to run with display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Fractal Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        movement = 0 # none
        space = 0
        shift = 0

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            # Detect key presses for actions that should only happen once
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift = 1

        # Hold-down key detection for movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        # The action space is MultiDiscrete, but for human play, we use key presses.
        # The step function logic is based on press events (e.g., space_press),
        # so we need to manage the state of keys between frames.
        # For simplicity in this manual test, we'll just send the current state.
        # The environment correctly handles the press logic internally.
        
        # We need to manage the held state for space and shift for the env's logic
        is_space_held = keys[pygame.K_SPACE]
        is_shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        # The action is [movement, space_held, shift_held]
        action = [movement, 1 if is_space_held else 0, 1 if is_shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Waves Survived: {info['wave']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    pygame.quit()