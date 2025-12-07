import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:37:51.184214
# Source Brief: brief_01173.md
# Brief Index: 1173
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a puzzle/strategy game.
    The player places bioluminescent creature components on a grid to form creatures,
    which can then be activated to attack enemy factions. The goal is to eliminate
    all enemies before the player's creatures are destroyed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Place bioluminescent components to form creatures on a grid. "
        "Activate your creatures to attack and eliminate enemy factions before they destroy you."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place a component and shift to activate the creature under the cursor."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_ORIGIN_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_ORIGIN_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2 + 20
        self.MAX_STEPS = 5000
        self.CARD_TYPES = 3

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (30, 50, 80)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.PLAYER_COLORS = {
            1: (0, 150, 255),  # Blue - Single Target
            2: (0, 255, 150),  # Cyan - Area of Effect
            3: (200, 100, 255), # Purple - Chain Attack
        }
        self.ENEMY_COLORS = [(255, 80, 80), (80, 255, 80), (255, 255, 80)]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("tahoma", 14, bold=True)
        self.font_medium = pygame.font.SysFont("tahoma", 20, bold=True)
        self.font_large = pygame.font.SysFont("tahoma", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.cursor_pos = None
        self.player_creatures = None
        self.enemy_factions = None
        self.current_card_type = None
        self.card_draw_timer = 0
        self.enemy_attack_timer = 0
        self.base_enemy_health = 0
        self.enemy_attack_rate = 0
        self.particles = []
        self.floating_texts = []
        self.previous_space_held = False
        self.previous_shift_held = False
        self.bg_stars = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.player_creatures = {}

        self.base_enemy_health = 100
        self.enemy_factions = [
            {
                'id': i,
                'color': self.ENEMY_COLORS[i],
                'health': self.base_enemy_health,
                'max_health': self.base_enemy_health,
                'pos': (self.WIDTH - 130, 60 + i * 45)
            } for i in range(len(self.ENEMY_COLORS))
        ]

        self.current_card_type = None
        self._draw_new_card()
        self.card_draw_timer = 150 # Steps until new card if not placed

        self.enemy_attack_rate = 0.005 # attacks per step
        self.enemy_attack_timer = 1 / self.enemy_attack_rate if self.enemy_attack_rate > 0 else float('inf')

        self.particles = []
        self.floating_texts = []
        self.previous_space_held = False
        self.previous_shift_held = False

        if not self.bg_stars:
            self.bg_stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.5, 1.5))
                for _ in range(100)
            ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        # 1. Handle Player Input
        step_reward += self._handle_input(action)

        # 2. Update Game World
        self._update_timers()
        step_reward += self._handle_enemy_attacks()
        self._update_difficulty()
        self._update_particles_and_texts()

        # 3. Check for Termination
        terminated, terminal_reward = self._check_termination()
        step_reward += terminal_reward
        if terminated:
            self.game_over = True

        self.score += step_reward
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Core Logic ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        # Actions on button press (rising edge)
        if space_held and not self.previous_space_held:
            reward += self._place_card()
        if shift_held and not self.previous_shift_held:
            reward += self._activate_creature()

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        return reward

    def _place_card(self):
        x, y = self.cursor_pos
        if self.current_card_type is not None and self.grid[y, x] == 0:
            # sfx: sfx_place_card.wav
            self.grid[y, x] = self.current_card_type
            self._create_particle_burst(self._grid_to_pixel(x, y), self.PLAYER_COLORS[self.current_card_type], 10)
            self.current_card_type = None
            self.card_draw_timer = 150 # Reset draw timer
            self._update_player_creatures()
            return 0.01 # Small reward for a valid action
        return 0

    def _activate_creature(self):
        x, y = self.cursor_pos
        if self.grid[y, x] == 0: return 0

        reward = 0
        creature_to_activate = None
        for cid, creature in self.player_creatures.items():
            if tuple(self.cursor_pos) in creature['cells_set']:
                creature_to_activate = creature
                break
        
        if not creature_to_activate: return 0

        # sfx: sfx_activate_creature.wav
        size = len(creature_to_activate['cells'])
        c_type = creature_to_activate['type']
        
        alive_enemies = [e for e in self.enemy_factions if e['health'] > 0]
        if not alive_enemies: return 0

        # --- Ability Logic ---
        if c_type == 1: # Single Target
            damage = size * 10
            target = random.choice(alive_enemies)
            reward += self._deal_damage(target, damage, creature_to_activate['center_pos'])
        elif c_type == 2: # Area of Effect
            damage = size * 4
            for target in alive_enemies:
                reward += self._deal_damage(target, damage, creature_to_activate['center_pos'])
        elif c_type == 3: # Chain Attack
            damage = size * 8
            targets = random.sample(alive_enemies, k=min(len(alive_enemies), 2))
            for i, target in enumerate(targets):
                chain_damage = damage if i == 0 else damage * 0.5
                reward += self._deal_damage(target, chain_damage, creature_to_activate['center_pos'])
        
        return reward

    def _deal_damage(self, enemy_faction, damage, origin_pos):
        reward = 0
        damage = int(damage)
        enemy_faction['health'] -= damage
        # sfx: sfx_enemy_hit.wav
        self._create_particle_burst(enemy_faction['pos'], enemy_faction['color'], 15, (0, -20))
        self._create_floating_text(f"-{damage}", (enemy_faction['pos'][0], enemy_faction['pos'][1] - 20), (255, 180, 0))

        reward += 0.01 * damage # Reward for damage dealt
        
        if enemy_faction['health'] <= 0 and enemy_faction['max_health'] > 0:
            enemy_faction['health'] = 0
            enemy_faction['max_health'] = 0 # Mark as defeated
            reward += 1.0 # Reward for eliminating faction
            # sfx: sfx_enemy_destroyed.wav
            self._create_floating_text("FACTION ELIMINATED", (self.WIDTH//2, self.HEIGHT//2), (255, 50, 50), 60, self.font_large)
            self._create_particle_burst(enemy_faction['pos'], enemy_faction['color'], 100)

        return reward

    def _update_player_creatures(self):
        self.player_creatures.clear()
        visited = np.zeros_like(self.grid, dtype=bool)
        creature_id_counter = 1

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y, x] > 0 and not visited[y, x]:
                    card_type = self.grid[y, x]
                    component_cells = []
                    q = deque([(x, y)])
                    visited[y, x] = True
                    
                    sum_x, sum_y = 0, 0
                    while q:
                        cx, cy = q.popleft()
                        component_cells.append((cx, cy))
                        sum_x += cx
                        sum_y += cy
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and \
                               not visited[ny, nx] and self.grid[ny, nx] == card_type:
                                visited[ny, nx] = True
                                q.append((nx, ny))
                    
                    center_x = self.GRID_ORIGIN_X + (sum_x / len(component_cells) + 0.5) * self.CELL_SIZE
                    center_y = self.GRID_ORIGIN_Y + (sum_y / len(component_cells) + 0.5) * self.CELL_SIZE
                    
                    self.player_creatures[creature_id_counter] = {
                        'id': creature_id_counter,
                        'type': card_type,
                        'cells': component_cells,
                        'cells_set': set(component_cells),
                        'center_pos': (center_x, center_y)
                    }
                    creature_id_counter += 1

    def _update_timers(self):
        # Card draw timer
        if self.current_card_type is None:
            self.card_draw_timer -= 1
            if self.card_draw_timer <= 0:
                self._draw_new_card()

        # Enemy attack timer
        self.enemy_attack_timer -= 1

    def _handle_enemy_attacks(self):
        if self.enemy_attack_timer > 0:
            return 0
        
        self.enemy_attack_timer = 1 / self.enemy_attack_rate if self.enemy_attack_rate > 0 else float('inf')

        player_cards_pos = np.argwhere(self.grid > 0)
        if len(player_cards_pos) == 0:
            return 0

        # sfx: sfx_player_hit.wav
        pos_to_destroy_idx = self.np_random.integers(len(player_cards_pos))
        pos_to_destroy = player_cards_pos[pos_to_destroy_idx]
        y, x = pos_to_destroy
        self.grid[y, x] = 0
        self._update_player_creatures()
        
        px, py = self._grid_to_pixel(x, y)
        self._create_particle_burst((px, py), (255, 100, 0), 20)
        self._create_floating_text("-1", (px, py), (255, 100, 0))

        return -0.1 # Penalty for losing a card

    def _update_difficulty(self):
        # Increase enemy aggression
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_attack_rate += 0.001
        
        # Increase enemy health
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_enemy_health += 10
            for e in self.enemy_factions:
                if e['max_health'] > 0: # Don't resurrect dead factions
                    e['max_health'] = self.base_enemy_health
                    e['health'] = self.base_enemy_health
    
    def _draw_new_card(self):
        self.current_card_type = self.np_random.integers(1, self.CARD_TYPES + 1)
        # sfx: sfx_new_card.wav

    def _check_termination(self):
        win = all(e['health'] <= 0 for e in self.enemy_factions)
        if win:
            return True, 100.0

        num_player_cards = np.count_nonzero(self.grid)
        loss = num_player_cards == 0 and self.current_card_type is None
        if loss:
            return True, -100.0

        if self.steps >= self.MAX_STEPS:
            return True, 0.0
            
        return False, 0.0

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_grid()
        self._render_creatures()
        self._render_cursor()
        self._render_particles_and_texts()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.bg_stars:
            pulse = (math.sin(self.steps * 0.02 + x) + 1) / 2
            alpha = int(30 + pulse * 40)
            color = (alpha, alpha, int(alpha * 1.2))
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), color)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_v = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y)
            end_v = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v, 1)
            # Horizontal
            start_h = (self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            end_h = (self.GRID_ORIGIN_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h, 1)

    def _render_creatures(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                card_type = self.grid[y, x]
                if card_type > 0:
                    px, py = self._grid_to_pixel(x, y)
                    color = self.PLAYER_COLORS[card_type]
                    
                    pulse = (math.sin(self.steps * 0.1 + x * 0.5) + 1) / 2
                    radius = int(self.CELL_SIZE * 0.35 + pulse * 2)
                    
                    self._draw_glow_circle(self.screen, color, (px, py), radius, 5)
                    pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, px, py, radius, (255,255,255))


    def _render_cursor(self):
        x, y = self.cursor_pos
        px, py = self._grid_to_pixel(x, y)
        size = self.CELL_SIZE // 2
        rect = pygame.Rect(px - size, py - size, self.CELL_SIZE, self.CELL_SIZE)
        
        alpha = 100 + (math.sin(self.steps * 0.2) + 1) * 75
        color = (*self.COLOR_CURSOR[:3], int(alpha))
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, 3)
        
        # Draw corners
        corner_len = 8
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.left, rect.top), (rect.left + corner_len, rect.top), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.left, rect.top), (rect.left, rect.top + corner_len), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.right - 1, rect.top), (rect.right - 1 - corner_len, rect.top), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.right - 1, rect.top), (rect.right - 1, rect.top + corner_len), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.left, rect.bottom - 1), (rect.left + corner_len, rect.bottom - 1), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.left, rect.bottom - 1), (rect.left, rect.bottom - 1 - corner_len), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.right - 1, rect.bottom - 1), (rect.right - 1- corner_len, rect.bottom - 1), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.right - 1, rect.bottom - 1), (rect.right - 1, rect.bottom - 1 - corner_len), 2)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (15, 40))

        # Enemy Faction Health Bars
        for faction in self.enemy_factions:
            x, y = faction['pos']
            w, h = 110, 20
            
            health_ratio = faction['health'] / faction['max_health'] if faction['max_health'] > 0 else 0
            
            # Background bar
            pygame.draw.rect(self.screen, (50,50,50), (x, y, w, h), border_radius=3)
            # Health fill
            if health_ratio > 0:
                pygame.draw.rect(self.screen, faction['color'], (x, y, int(w * health_ratio), h), border_radius=3)
            # Border
            pygame.draw.rect(self.screen, self.COLOR_GRID, (x, y, w, h), 2, border_radius=3)
            # Text
            health_text = self.font_small.render(f"{int(faction['health'])}/{int(faction['max_health'])}", True, self.COLOR_TEXT)
            self.screen.blit(health_text, (x + w/2 - health_text.get_width()/2, y + h/2 - health_text.get_height()/2))
        
        # Current Card Display
        card_area_rect = pygame.Rect(15, self.HEIGHT - 60, 100, 50)
        pygame.draw.rect(self.screen, self.COLOR_GRID, card_area_rect, 2, border_radius=5)
        title = self.font_small.render("NEXT CARD", True, self.COLOR_TEXT)
        self.screen.blit(title, (card_area_rect.centerx - title.get_width()//2, card_area_rect.top - 20))

        if self.current_card_type is not None:
            color = self.PLAYER_COLORS[self.current_card_type]
            self._draw_glow_circle(self.screen, color, card_area_rect.center, 12, 4)
            pygame.gfxdraw.filled_circle(self.screen, card_area_rect.centerx, card_area_rect.centery, 12, color)
            pygame.gfxdraw.aacircle(self.screen, card_area_rect.centerx, card_area_rect.centery, 12, (255,255,255))
        else:
            # Draw timer bar
            timer_ratio = (150 - self.card_draw_timer) / 150
            bar_width = int((card_area_rect.width - 6) * timer_ratio)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (card_area_rect.x+3, card_area_rect.y+card_area_rect.height - 10, bar_width, 5), border_radius=2)


    # --- VFX Helpers ---

    def _update_particles_and_texts(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['vy'] += 0.05 # gravity
        
        self.floating_texts = [t for t in self.floating_texts if t['life'] > 0]
        for t in self.floating_texts:
            t['y'] -= 0.5
            t['life'] -= 1

    def _render_particles_and_texts(self):
        # Particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            color = (*p['color'], int(alpha * 255))
            radius = int(p['radius'] * alpha)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, color)
        
        # Floating Texts
        for t in self.floating_texts:
            alpha = min(1, t['life'] / (t['max_life'] / 2))
            color = (*t['color'], int(alpha * 255))
            text_surf = t['font'].render(t['text'], True, color)
            text_surf.set_alpha(int(alpha * 255))
            self.screen.blit(text_surf, (t['x'] - text_surf.get_width()//2, t['y'] - text_surf.get_height()//2))

    def _create_particle_burst(self, pos, color, count, offset=(0,0)):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': pos[0] + offset[0], 'y': pos[1] + offset[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'radius': self.np_random.integers(2, 6),
                'color': color,
                'life': self.np_random.integers(20, 41), 'max_life': 40
            })

    def _create_floating_text(self, text, pos, color, lifetime=45, font=None):
        if font is None: font = self.font_medium
        self.floating_texts.append({
            'text': text, 'font': font,
            'x': pos[0], 'y': pos[1],
            'color': color,
            'life': lifetime, 'max_life': lifetime
        })

    def _draw_glow_circle(self, surface, color, center, radius, intensity):
        for i in range(intensity, 0, -1):
            alpha = int(80 * (1 - i / intensity))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius + i*2), glow_color)

    # --- Utility ---

    def _grid_to_pixel(self, x, y):
        px = self.GRID_ORIGIN_X + int((x + 0.5) * self.CELL_SIZE)
        py = self.GRID_ORIGIN_Y + int((y + 0.5) * self.CELL_SIZE)
        return px, py

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_creatures": len(self.player_creatures),
            "enemies_alive": sum(1 for e in self.enemy_factions if e['health'] > 0),
        }

    def close(self):
        pygame.quit()


# Example usage for visualization
if __name__ == '__main__':
    # Set the video driver to a real one for visualization
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display window
    pygame.display.set_caption("Bioluminescent Abyss")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        # --- Human Input ---
        # This part is for human play, not for the agent
        action = [0, 0, 0] # no-op, released, released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display it
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        
        if done:
            font = pygame.font.SysFont("tahoma", 50, bold=True)
            win_status = "YOU WIN" if info['enemies_alive'] == 0 else "GAME OVER"
            text = font.render(win_status, True, (255, 255, 255))
            text_rect = text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 20))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.SysFont("tahoma", 20)
            text_reset = font_small.render("Press 'R' to Reset", True, (200, 200, 200))
            text_reset_rect = text_reset.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 30))
            screen.blit(text_reset, text_reset_rect)

        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS

    env.close()