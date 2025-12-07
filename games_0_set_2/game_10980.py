import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:15:33.517597
# Source Brief: brief_00980.md
# Brief Index: 980
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Rune Guardian: A Tower Defense Gymnasium Environment.

    The player defends against waves of demonic entities by strategically
    crafting and placing magical rune portals. The goal is to survive all
    20 waves.

    Action Space (MultiDiscrete([5, 2, 2])):
    - actions[0]: Cursor Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Action Button (Space) (0=released, 1=held) - Crafts or places a portal.
    - actions[2]: Cycle Button (Shift) (0=released, 1=held) - Cycles through available runes.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend against waves of demonic entities by strategically crafting and placing magical rune portals."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press Space to craft a rune or place a portal. Press Shift to cycle between available runes."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    GRID_CELL_SIZE = 30
    PLAY_AREA_X_OFFSET = (SCREEN_WIDTH - GRID_COLS * GRID_CELL_SIZE) // 2
    PLAY_AREA_Y_OFFSET = (SCREEN_HEIGHT - GRID_ROWS * GRID_CELL_SIZE) // 2
    MAX_STEPS = 5000 # Increased to allow for longer games
    TOTAL_WAVES = 20
    FPS = 30

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_GRID = (40, 30, 60)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (25, 20, 45, 200)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PORTAL_HEALTH = (60, 200, 255)
    COLOR_PORTAL_HEALTH_BG = (255, 255, 255, 50)

    # Rune Definitions
    RUNE_DATA = {
        "arcane": {
            "name": "Arcane", "color": (150, 100, 255), "cost": 40,
            "stats": {"damage": 2, "range": 80, "fire_rate": 30} # fire_rate in frames
        },
        "fire": {
            "name": "Fire", "color": (255, 120, 50), "cost": 75,
            "stats": {"damage": 4, "range": 60, "fire_rate": 45}
        },
        "frost": {
            "name": "Frost", "color": (100, 200, 255), "cost": 100,
            "stats": {"damage": 1, "range": 90, "fire_rate": 20} # High fire rate
        }
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 20, bold=True)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.cursor_pos = (0, 0)
        self.inventory = {}
        self.portals = []
        self.enemies = []
        self.particles = []
        
        self.wave_number = 0
        self.wave_cooldown = 0
        self.enemy_spawn_cooldown = 0
        self.enemies_to_spawn = 0
        
        self.available_runes = []
        self.selected_rune_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        
        self.inventory = {"components": 100, "runes": {}}
        for rune_id in self.RUNE_DATA:
            self.inventory["runes"][rune_id] = 0
            
        self.portals = []
        self.enemies = []
        self.particles = []
        
        self.wave_number = 0
        self.wave_cooldown = self.FPS * 3 # 3 seconds until first wave
        
        self.available_runes = ["arcane"]
        self.selected_rune_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1

        self._handle_input(action)
        self._update_game_logic()
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            # First frame of termination
            self.game_over = True
            is_win = self.wave_number > self.TOTAL_WAVES and not self.portals_destroyed()
            self.reward_this_step += 100 if is_win else -100

        self.score += self.reward_this_step
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            self.cursor_pos = (
                np.clip(self.cursor_pos[0] + dx, 0, self.GRID_COLS - 1),
                np.clip(self.cursor_pos[1] + dy, 0, self.GRID_ROWS - 1)
            )

        # --- Cycle Rune (Shift) ---
        if shift_held and not self.last_shift_held:
            self.selected_rune_idx = (self.selected_rune_idx + 1) % len(self.available_runes)

        # --- Action (Space) ---
        if space_held and not self.last_space_held:
            self._perform_action()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _perform_action(self):
        selected_rune_id = self.available_runes[self.selected_rune_idx]
        
        # Try to place a portal first
        if self.inventory["runes"][selected_rune_id] > 0:
            is_occupied = any(p['grid_pos'] == self.cursor_pos for p in self.portals)
            if not is_occupied:
                self.inventory["runes"][selected_rune_id] -= 1
                
                # Convert grid pos to screen pos
                screen_x = self.PLAY_AREA_X_OFFSET + self.cursor_pos[0] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2
                screen_y = self.PLAY_AREA_Y_OFFSET + self.cursor_pos[1] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2
                
                self.portals.append({
                    "grid_pos": self.cursor_pos,
                    "pos": (screen_x, screen_y),
                    "rune_id": selected_rune_id,
                    "health": 100, "max_health": 100,
                    "attack_cooldown": 0,
                    "anim_phase": self.np_random.uniform(0, 2 * math.pi)
                })
                self.reward_this_step += 5 # Reward for placing a portal
                self._create_particles(20, (screen_x, screen_y), self.RUNE_DATA[selected_rune_id]['color'], 3)
                return

        # If placing failed, try to craft a rune
        cost = self.RUNE_DATA[selected_rune_id]["cost"]
        if self.inventory["components"] >= cost:
            self.inventory["components"] -= cost
            self.inventory["runes"][selected_rune_id] += 1
            self.reward_this_step += 1 # Reward for crafting

    def _update_game_logic(self):
        # Automatic component generation
        if self.steps % (self.FPS // 2) == 0:
            self.inventory["components"] += 1
            
        self._update_wave_spawner()
        self._update_portals()
        self._update_enemies()
        self._update_particles()
        
        # Clean up dead entities
        self.enemies = [e for e in self.enemies if e['health'] > 0]
        self.portals = [p for p in self.portals if p['health'] > 0]

    def _update_wave_spawner(self):
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self.wave_number += 1
                self._start_wave(self.wave_number)
        elif self.enemies_to_spawn > 0:
            self.enemy_spawn_cooldown -= 1
            if self.enemy_spawn_cooldown <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.enemy_spawn_cooldown = self.FPS // 2 # 0.5 sec between spawns
        elif not self.enemies: # Wave cleared
            self.reward_this_step += 10
            self.wave_cooldown = self.FPS * 5 # 5 seconds between waves
            if self.wave_number == 5: self.available_runes.append("fire")
            if self.wave_number == 10: self.available_runes.append("frost")

    def _start_wave(self, wave_num):
        self.enemies_to_spawn = 5 + wave_num * 2
        self.enemy_spawn_cooldown = 0

    def _spawn_enemy(self):
        side = self.np_random.integers(0, 4)
        if side == 0: x, y = -20, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        elif side == 1: x, y = self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        elif side == 2: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), -20
        else: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20
        
        speed = 1.0 + (self.wave_number // 5) * 0.25
        health = 10 + (self.wave_number // 2) * 2
        
        self.enemies.append({"pos": np.array([x, y], dtype=float), "health": health, "max_health": health, "speed": speed})

    def _update_portals(self):
        for portal in self.portals:
            portal['attack_cooldown'] = max(0, portal['attack_cooldown'] - 1)
            portal['anim_phase'] += 0.05
            
            if portal['attack_cooldown'] == 0 and self.enemies:
                rune_stats = self.RUNE_DATA[portal['rune_id']]['stats']
                target = None
                min_dist = float('inf')
                
                for enemy in self.enemies:
                    dist = np.linalg.norm(np.array(portal['pos']) - enemy['pos'])
                    if dist < rune_stats['range'] and dist < min_dist:
                        min_dist = dist
                        target = enemy
                        
                if target:
                    target['health'] -= rune_stats['damage']
                    self.reward_this_step += 0.1 * rune_stats['damage']
                    portal['attack_cooldown'] = rune_stats['fire_rate']
                    self._create_beam_effect(portal['pos'], target['pos'], self.RUNE_DATA[portal['rune_id']]['color'])
                    if target['health'] <= 0:
                        self._create_particles(15, target['pos'], self.COLOR_ENEMY, 2)

    def _update_enemies(self):
        for enemy in self.enemies:
            if not self.portals: break # No portals to move towards
            
            target_portal = min(self.portals, key=lambda p: np.linalg.norm(enemy['pos'] - np.array(p['pos'])))
            direction = np.array(target_portal['pos']) - enemy['pos']
            dist = np.linalg.norm(direction)
            
            if dist > 10: # Move towards portal
                direction = direction / dist
                enemy['pos'] += direction * enemy['speed']
            else: # Attack portal
                target_portal['health'] -= 1
                self.reward_this_step -= 0.1
                enemy['health'] = 0 # Enemy sacrifices itself
                self._create_particles(10, enemy['pos'], self.COLOR_PORTAL_HEALTH, 2)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_termination(self):
        if self.portals_destroyed() and self.inventory['components'] < self.RUNE_DATA['arcane']['cost'] and self.inventory['runes']['arcane'] == 0:
            return True
        if self.wave_number > self.TOTAL_WAVES and not self.enemies:
            return True
        return False
        
    def portals_destroyed(self):
        return not self.portals

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "portals": len(self.portals),
            "enemies": len(self.enemies),
            "components": self.inventory["components"]
        }

    # --- Rendering Methods ---
    
    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = self.PLAY_AREA_Y_OFFSET + r * self.GRID_CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAY_AREA_X_OFFSET, y), (self.SCREEN_WIDTH - self.PLAY_AREA_X_OFFSET, y))
        for c in range(self.GRID_COLS + 1):
            x = self.PLAY_AREA_X_OFFSET + c * self.GRID_CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAY_AREA_Y_OFFSET), (x, self.SCREEN_HEIGHT - self.PLAY_AREA_Y_OFFSET))
            
        # Draw portals
        for portal in self.portals:
            self._draw_glowing_portal(portal)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))

            if p.get('type') == 'beam':
                end_pos_np = p['end_pos']
                end_pos = (int(end_pos_np[0]), int(end_pos_np[1]))
                pygame.draw.line(self.screen, color_with_alpha, pos, end_pos, 2)
            else:
                size = int(p['size'])
                if size > 0:
                    pygame.draw.circle(self.screen, color_with_alpha, pos, size)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.PLAY_AREA_X_OFFSET + self.cursor_pos[0] * self.GRID_CELL_SIZE,
            self.PLAY_AREA_Y_OFFSET + self.cursor_pos[1] * self.GRID_CELL_SIZE,
            self.GRID_CELL_SIZE, self.GRID_CELL_SIZE
        )
        s = pygame.Surface((self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)

    def _draw_glowing_portal(self, portal):
        pos = (int(portal['pos'][0]), int(portal['pos'][1]))
        rune_id = portal['rune_id']
        color = self.RUNE_DATA[rune_id]['color']
        
        # Pulsing glow effect
        glow_size = 10 + 3 * math.sin(portal['anim_phase'])
        for i in range(int(glow_size), 0, -2):
            alpha = 80 * (1 - i / glow_size)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8 + i, (*color, int(alpha)))
        
        # Core circle
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)
        
        # Health bar
        health_pct = max(0, portal['health'] / portal['max_health'])
        bar_width = 30
        bar_height = 4
        bar_x = pos[0] - bar_width // 2
        bar_y = pos[1] - 25
        pygame.draw.rect(self.screen, self.COLOR_PORTAL_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PORTAL_HEALTH, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_ui(self):
        # Top panel
        top_panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        s = pygame.Surface(top_panel_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, top_panel_rect.topleft)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 39), (self.SCREEN_WIDTH, 39))

        # Bottom panel
        bottom_panel_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 60, self.SCREEN_WIDTH, 60)
        s = pygame.Surface(bottom_panel_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, bottom_panel_rect.topleft)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.SCREEN_HEIGHT - 60), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 60))

        # Top info text
        wave_text = f"Wave: {self.wave_number}/{self.TOTAL_WAVES}"
        if self.wave_cooldown > 0 and self.wave_number < self.TOTAL_WAVES:
            wave_text += f" (Next in {self.wave_cooldown // self.FPS + 1}s)"
        self._draw_text(wave_text, (10, 10), self.font_large)
        
        score_text = f"Score: {int(self.score)}"
        self._draw_text(score_text, (self.SCREEN_WIDTH - 150, 10), self.font_large, align="right")
        
        # Bottom info text (inventory and crafting)
        comp_text = f"Components: {self.inventory['components']}"
        self._draw_text(comp_text, (10, self.SCREEN_HEIGHT - 50), self.font_small)
        
        # Rune inventory
        rune_x = 10
        for i, (rune_id, count) in enumerate(self.inventory['runes'].items()):
            if rune_id not in self.available_runes: continue
            color = self.RUNE_DATA[rune_id]['color']
            pygame.draw.circle(self.screen, color, (rune_x + 10, self.SCREEN_HEIGHT - 25), 8)
            self._draw_text(f"x{count}", (rune_x + 22, self.SCREEN_HEIGHT - 32), self.font_small)
            rune_x += 60
            
        # Crafting selection
        craft_panel_x = self.SCREEN_WIDTH - 250
        self._draw_text("Craft/Place [SPACE]", (craft_panel_x, self.SCREEN_HEIGHT - 50), self.font_small)
        
        selected_rune_id = self.available_runes[self.selected_rune_idx]
        rune_info = self.RUNE_DATA[selected_rune_id]
        
        # Highlight selected rune
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (craft_panel_x - 5, self.SCREEN_HEIGHT - 35, 240, 25), 2, border_radius=3)
        
        self._draw_text(f"Selected: {rune_info['name']}", (craft_panel_x, self.SCREEN_HEIGHT - 32), self.font_small)
        self._draw_text(f"Cost: {rune_info['cost']}", (craft_panel_x + 150, self.SCREEN_HEIGHT - 32), self.font_small)
        
        self._draw_text("Cycle [SHIFT]", (craft_panel_x, self.SCREEN_HEIGHT - 15), self.font_small)
        
    def _draw_text(self, text, pos, font, color=COLOR_TEXT, align="left"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "left":
            text_rect.topleft = pos
        elif align == "right":
            text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, count, pos, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array(vel, dtype=float),
                'life': self.np_random.integers(10, 21),
                'max_life': 20,
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })
            
    def _create_beam_effect(self, start_pos, end_pos, color):
        self.particles.append({
            'pos': np.array(start_pos, dtype=float), 'vel': np.array((0,0), dtype=float), 'life': 2, 'max_life': 2,
            'color': color, 'size': 0, 'type': 'beam', 'end_pos': end_pos
        })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for human play and debugging, not used by the tests
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset(seed=42)
    terminated = False
    
    pygame.display.set_caption("Rune Guardian")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not terminated:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            print("Episode truncated due to time limit.")
            terminated = True
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                terminated = True
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print("\n--- Game Over ---")
    print(f"Final Info: {info}")
    env.close()