import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:23:07.068111
# Source Brief: brief_00955.md
# Brief Index: 955
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a physics-based tower defense game.
    The player places blocks and portals to defend a central core from
    waves of nightmare creatures. The visual style is dark and dreamlike
    with glowing neon elements.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your core from nightmare creatures by strategically placing blocks and portals "
        "in this physics-based tower defense game."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place an item or start the wave. "
        "Use shift to cycle through placement modes (Block, Portal A, Portal B)."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20

    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (30, 20, 50)
    COLOR_PLAYER_BLOCK = (0, 150, 255)
    COLOR_PORTAL = (255, 150, 0)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_CORE = (50, 255, 100)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_CURSOR = (255, 255, 255)

    MAX_WAVES = 20
    MAX_STEPS_PER_EPISODE = 5000
    CORE_MAX_HEALTH = 100
    BLOCK_MAX_HEALTH = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.phase = 'BUILD' # 'BUILD' or 'WAVE'
        self.wave_num = 0
        self.core_health = 0
        self.core_pos_grid = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 3)
        
        self.cursor_pos_grid = [0, 0]
        self.placement_mode = 'BLOCK'
        self.placement_modes = ['BLOCK']
        
        self.blocks = {} # {(x,y): health}
        self.portals = {'A': None, 'B': None}
        self.enemies = []
        self.particles = []
        self.nebula_particles = []
        
        self.blocks_to_place = 0
        
        self.previous_actions = [0, 0] # [space, shift]

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed) # Standard library random
            # self.np_random = np.random.default_rng(seed=seed) # for numpy randomness

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.phase = 'BUILD'
        self.wave_num = 1
        self.core_health = self.CORE_MAX_HEALTH
        
        self.cursor_pos_grid = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.placement_mode = 'BLOCK'
        self.placement_modes = ['BLOCK']
        
        self.blocks = {}
        self.portals = {'A': None, 'B': None}
        self.enemies = []
        self.particles = []
        
        self.blocks_to_place = 5
        self.previous_actions = [0, 0]

        # Background nebula effect
        self.nebula_particles = [
            {
                "pos": [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                "radius": random.uniform(50, 150),
                "color": (random.randint(15, 30), random.randint(10, 20), random.randint(25, 40), random.randint(5, 15)),
                "vel": [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
            } for _ in range(15)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.previous_actions[0]
        shift_pressed = shift_held and not self.previous_actions[1]
        self.previous_actions = [space_held, shift_held]

        reward = 0
        
        if self.phase == 'BUILD':
            self._handle_build_phase(movement, space_pressed, shift_pressed)
        elif self.phase == 'WAVE':
            reward += self._handle_wave_phase()
        
        self._update_particles()
        
        self.steps += 1
        
        terminated = False
        truncated = False
        
        if self.core_health <= 0 and not self.game_over:
            # Sfx: Game_over
            self.game_over = True
            terminated = True
            reward -= 100 # Terminal penalty
        
        if self.wave_num > self.MAX_WAVES and not self.game_over:
            # Sfx: Game_win
            self.game_over = True
            terminated = True
            reward += 100 # Terminal bonus
            
        if self.steps >= self.MAX_STEPS_PER_EPISODE:
            truncated = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_build_phase(self, movement, space_pressed, shift_pressed):
        if movement == 1: self.cursor_pos_grid[1] -= 1 # Up
        if movement == 2: self.cursor_pos_grid[1] += 1 # Down
        if movement == 3: self.cursor_pos_grid[0] -= 1 # Left
        if movement == 4: self.cursor_pos_grid[0] += 1 # Right
        
        self.cursor_pos_grid[0] = np.clip(self.cursor_pos_grid[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos_grid[1] = np.clip(self.cursor_pos_grid[1], 0, self.GRID_HEIGHT - 1)

        if shift_pressed:
            # Sfx: UI_switch
            current_index = self.placement_modes.index(self.placement_mode)
            self.placement_mode = self.placement_modes[(current_index + 1) % len(self.placement_modes)]

        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos_grid)
            
            if self._is_on_start_button(cursor_tuple):
                self._start_wave()
                return

            core_area = {self.core_pos_grid, (self.core_pos_grid[0]-1, self.core_pos_grid[1])}
            if cursor_tuple in core_area:
                 # Sfx: UI_error
                 return

            if self.placement_mode == 'BLOCK':
                if self.blocks_to_place > 0 and cursor_tuple not in self.blocks:
                    # Sfx: Block_place
                    self.blocks[cursor_tuple] = self.BLOCK_MAX_HEALTH
                    self.blocks_to_place -= 1
            elif self.placement_mode == 'PORTAL_A':
                # Sfx: Portal_place
                if self.portals['B'] == cursor_tuple: self.portals['B'] = None
                self.portals['A'] = cursor_tuple
            elif self.placement_mode == 'PORTAL_B':
                # Sfx: Portal_place
                if self.portals['A'] == cursor_tuple: self.portals['A'] = None
                self.portals['B'] = cursor_tuple
    
    def _is_on_start_button(self, cursor_tuple):
        return self.GRID_HEIGHT-2 <= cursor_tuple[1] <= self.GRID_HEIGHT-1 and \
               self.GRID_WIDTH-6 <= cursor_tuple[0] <= self.GRID_WIDTH-1

    def _start_wave(self):
        # Sfx: Wave_start
        self.phase = 'WAVE'
        num_enemies = 4 + self.wave_num
        enemy_speed = 0.5 + self.wave_num * 0.05
        
        for _ in range(num_enemies):
            spawn_x = random.randint(0, self.GRID_WIDTH - 1) * self.CELL_SIZE + self.CELL_SIZE / 2
            self.enemies.append({
                "pos": np.array([spawn_x, -self.CELL_SIZE], dtype=float),
                "speed": enemy_speed,
                "teleport_cooldown": 0
            })

    def _handle_wave_phase(self):
        reward = 0
        enemies_to_remove = []
        blocks_to_damage = []

        for i, enemy in enumerate(self.enemies):
            if enemy["teleport_cooldown"] > 0:
                enemy["teleport_cooldown"] -= 1

            core_pixel_pos = np.array([(self.core_pos_grid[0] - 0.5) * self.CELL_SIZE, self.core_pos_grid[1] * self.CELL_SIZE])
            direction = core_pixel_pos - enemy["pos"]
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist
            
            next_pos = enemy["pos"] + direction * enemy["speed"]
            next_grid_pos = (int(next_pos[0] / self.CELL_SIZE), int(next_pos[1] / self.CELL_SIZE))
            
            if next_grid_pos in self.blocks:
                 # Sfx: Impact_soft
                blocks_to_damage.append(next_grid_pos)
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 10)
                enemies_to_remove.append(i)
                reward += 0.1 # Enemy destroyed
                continue
            
            enemy["pos"] = next_pos
            
            if enemy["teleport_cooldown"] == 0 and self.portals['A'] and self.portals['B']:
                current_grid_pos = (int(enemy["pos"][0] / self.CELL_SIZE), int(enemy["pos"][1] / self.CELL_SIZE))
                if current_grid_pos == self.portals['A']:
                    # Sfx: Portal_whoosh
                    exit_pos = self.portals['B']
                    enemy["pos"] = np.array([exit_pos[0] * self.CELL_SIZE + self.CELL_SIZE/2, exit_pos[1] * self.CELL_SIZE + self.CELL_SIZE/2])
                    enemy["teleport_cooldown"] = 30
                    self._create_particles(enemy["pos"], self.COLOR_PORTAL, 20)
                elif current_grid_pos == self.portals['B']:
                    # Sfx: Portal_whoosh
                    exit_pos = self.portals['A']
                    enemy["pos"] = np.array([exit_pos[0] * self.CELL_SIZE + self.CELL_SIZE/2, exit_pos[1] * self.CELL_SIZE + self.CELL_SIZE/2])
                    enemy["teleport_cooldown"] = 30
                    self._create_particles(enemy["pos"], self.COLOR_PORTAL, 20)
            
            core_rect = pygame.Rect((self.core_pos_grid[0]-1) * self.CELL_SIZE, self.core_pos_grid[1] * self.CELL_SIZE, self.CELL_SIZE*2, self.CELL_SIZE)
            if core_rect.collidepoint(enemy["pos"]):
                # Sfx: Core_damage
                self.core_health -= 10
                self._create_particles(enemy["pos"], self.COLOR_CORE, 30)
                enemies_to_remove.append(i)

        for pos in blocks_to_damage:
            if pos in self.blocks:
                self.blocks[pos] -= 1
                if self.blocks[pos] <= 0:
                    del self.blocks[pos]
                    reward -= 0.1 # Block lost

        for i in sorted(list(set(enemies_to_remove)), reverse=True):
            del self.enemies[i]
            
        if not self.enemies and self.phase == 'WAVE':
            # Sfx: Wave_complete
            reward += 1 # Wave survived
            self.wave_num += 1
            
            if 2 <= self.wave_num < 5 and 'PORTAL_A' not in self.placement_modes:
                self.placement_modes.extend(['PORTAL_A', 'PORTAL_B'])
            
            self.phase = 'BUILD'
            self.blocks_to_place = 5 + self.wave_num // 5

        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [random.uniform(-2, 2), random.uniform(-2, 2)],
                "life": random.randint(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_num,
            "core_health": self.core_health,
            "phase": self.phase,
        }

    def _render_background(self):
        for p in self.nebula_particles:
            p["pos"][0] = (p["pos"][0] + p["vel"][0]) % self.SCREEN_WIDTH
            p["pos"][1] = (p["pos"][1] + p["vel"][1]) % self.SCREEN_HEIGHT
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), p["color"])
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
    
    def _draw_neon_rect(self, surface, color, rect, width=1, glow_size=5, border_radius=3):
        for i in range(glow_size, 0, -1):
            alpha = int(150 / (i**0.8))
            glow_color = (*color, alpha)
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color, s.get_rect(), 0, border_radius=border_radius)
            surface.blit(s, rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(surface, color, rect, width, border_radius=border_radius)

    def _draw_neon_circle(self, surface, color, center, radius, width=1, glow_size=5):
        center_int = (int(center[0]), int(center[1]))
        for i in range(glow_size, 0, -1):
            alpha = int(150 / (i**0.8))
            glow_color = (*color, alpha)
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius + i, glow_color)
        if width == 0:
             pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def _render_game(self):
        core_rect = pygame.Rect((self.core_pos_grid[0]-1) * self.CELL_SIZE, self.core_pos_grid[1] * self.CELL_SIZE, self.CELL_SIZE*2, self.CELL_SIZE)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 5
        self._draw_neon_rect(self.screen, self.COLOR_CORE, core_rect, 0, glow_size=int(5+pulse), border_radius=5)

        for pos, health in self.blocks.items():
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            color = tuple(c * (health/self.BLOCK_MAX_HEALTH) for c in self.COLOR_PLAYER_BLOCK)
            self._draw_neon_rect(self.screen, color, rect, width=2, glow_size=3)

        for key, pos in self.portals.items():
            if pos:
                rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                self._draw_neon_rect(self.screen, self.COLOR_PORTAL, rect, width=2, glow_size=5)
                text = self.font_small.render(key, True, self.COLOR_PORTAL)
                self.screen.blit(text, text.get_rect(center=rect.center))
        
        if self.portals['A'] and self.portals['B']:
            start = (self.portals['A'][0]*self.CELL_SIZE+self.CELL_SIZE//2, self.portals['A'][1]*self.CELL_SIZE+self.CELL_SIZE//2)
            end = (self.portals['B'][0]*self.CELL_SIZE+self.CELL_SIZE//2, self.portals['B'][1]*self.CELL_SIZE+self.CELL_SIZE//2)
            pygame.draw.aaline(self.screen, (*self.COLOR_PORTAL, 50), start, end)

        for enemy in self.enemies:
            self._draw_neon_circle(self.screen, self.COLOR_ENEMY, enemy["pos"], self.CELL_SIZE // 3, width=0, glow_size=4)

        for p in self.particles:
            size = max(0, int(p["life"] / 10))
            pygame.draw.rect(self.screen, p["color"], (p["pos"][0]-size//2, p["pos"][1]-size//2, size, size))

        if self.phase == 'BUILD':
            cursor_rect = pygame.Rect(self.cursor_pos_grid[0] * self.CELL_SIZE, self.cursor_pos_grid[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            ghost_color = (255, 255, 255, 50)
            if self.placement_mode == 'BLOCK': ghost_color = (*self.COLOR_PLAYER_BLOCK, 100)
            elif self.placement_mode.startswith('PORTAL'): ghost_color = (*self.COLOR_PORTAL, 100)
            pygame.draw.rect(self.screen, ghost_color, cursor_rect, 0, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 1, border_radius=3)

    def _render_ui(self):
        wave_text = self.font_small.render(f"WAVE: {min(self.wave_num, self.MAX_WAVES)}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 5))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 5))

        bar_x = (self.core_pos_grid[0]-1) * self.CELL_SIZE
        bar_y = (self.core_pos_grid[1]+1) * self.CELL_SIZE + 2
        bar_w = self.CELL_SIZE * 2
        bar_h = 8
        health_ratio = max(0, self.core_health / self.CORE_MAX_HEALTH)
        pygame.draw.rect(self.screen, (255,0,0,100), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_CORE, (bar_x, bar_y, bar_w * health_ratio, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

        if self.phase == 'BUILD':
            mode_text = self.font_small.render(f"MODE: {self.placement_mode}", True, self.COLOR_UI_TEXT)
            self.screen.blit(mode_text, (10, 25))
            if self.placement_mode == 'BLOCK':
                blocks_text = self.font_small.render(f"BLOCKS: {self.blocks_to_place}", True, self.COLOR_UI_TEXT)
                self.screen.blit(blocks_text, (10, 45))
            
            start_rect = pygame.Rect((self.GRID_WIDTH-6)*self.CELL_SIZE, (self.GRID_HEIGHT-2)*self.CELL_SIZE, 6*self.CELL_SIZE, 2*self.CELL_SIZE)
            cursor_tuple = tuple(self.cursor_pos_grid)
            button_color = (0, 200, 0) if self._is_on_start_button(cursor_tuple) else (0, 100, 0)
            self._draw_neon_rect(self.screen, button_color, start_rect, 0, glow_size=5, border_radius=5)
            start_text = self.font_large.render("START", True, self.COLOR_UI_TEXT)
            self.screen.blit(start_text, start_text.get_rect(center=start_rect.center))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            msg = "VICTORY" if self.wave_num > self.MAX_WAVES else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It will be ignored and does not need to be fixed.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Nightmare Fortress")
    clock = pygame.time.Clock()
    
    running = True
    paused = False
    
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p: paused = not paused
                if event.key == pygame.K_r: 
                    obs, info = env.reset()
                    paused = False
        if paused:
            pygame.display.flip()
            clock.tick(30)
            continue

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            paused = True
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)
        
    env.close()