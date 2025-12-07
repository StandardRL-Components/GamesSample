import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move. Space to attack. Shift for special attack. No input to defend."
    )

    game_description = (
        "A turn-based dungeon crawler. Navigate the grid, battle enemies, and descend to level 10 to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.TILE_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.MAX_STEPS = 1000
        self.VICTORY_LEVEL = 10

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_FLOOR = (40, 40, 50)
        self.COLOR_GRID_LINE = (50, 50, 60)
        self.COLOR_HERO = (60, 180, 255)
        self.COLOR_EXIT = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BG = (80, 20, 20)
        self.COLOR_HEALTH_FG = (20, 200, 20)
        self.COLOR_SHIELD = (100, 150, 255, 100)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.hero_pos = [0, 0]
        self.hero_max_health = 100
        self.hero_health = self.hero_max_health
        self.exit_pos = [0, 0]
        self.grid = []
        self.enemies = []
        self.particles = []
        self.combat_log = []
        self.is_defending = False
        self.exit_pulse = 0.0

        # This call is needed to initialize the state for the first observation
        # It needs a seed to be reproducible, so we will call super().reset() first
        super().reset(seed=0)
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.hero_health = self.hero_max_health
        self.particles = []
        self.combat_log = ["Welcome to the dungeon!"]
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.is_defending = False
        
        # --- Player Turn ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_taken = False

        if shift_held: # Priority 1: Special Attack
            # SFX: Special attack woosh
            if self.hero_health > 10:
                self.hero_health -= 10
                reward -= 0.5 # Cost for using special
                target_enemy = self._get_adjacent_enemy()
                if target_enemy:
                    damage = 30
                    target_enemy['health'] -= damage
                    self._add_log(f"Hero used Special! Dealt {damage} dmg.")
                    self._add_particle(target_enemy['pos'], (255, 100, 0), 20, 20)
                    if target_enemy['health'] <= 0:
                        reward += self._kill_enemy(target_enemy)
                else:
                    self._add_log("Special attack missed!")
                action_taken = True
            else:
                self._add_log("Not enough health for Special!")

        elif space_held: # Priority 2: Attack
            # SFX: Sword swing
            target_enemy = self._get_adjacent_enemy()
            if target_enemy:
                damage = self.np_random.integers(10, 21)
                target_enemy['health'] -= damage
                self._add_log(f"Hero attacked! Dealt {damage} dmg.")
                self._add_particle(target_enemy['pos'], (255, 255, 255), 10, 15)
                if target_enemy['health'] <= 0:
                    reward += self._kill_enemy(target_enemy)
            else:
                self._add_log("Attack missed! No enemy nearby.")
            action_taken = True

        elif movement > 0: # Priority 3: Movement
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            next_pos = [self.hero_pos[0] + dx, self.hero_pos[1] + dy]
            
            if self._is_valid(next_pos) and self.grid[next_pos[1]][next_pos[0]] == 0:
                if any(e['pos'] == next_pos for e in self.enemies):
                    self._add_log("Enemy blocks the way!")
                elif next_pos == self.exit_pos:
                    # SFX: Level up fanfare
                    self.level += 1
                    reward += 10
                    if self.level >= self.VICTORY_LEVEL:
                        self.game_over = True
                        reward += 100
                        self._add_log("VICTORY! You escaped the dungeon!")
                    else:
                        self._add_log(f"Descended to level {self.level}!")
                        self._generate_level()
                else:
                    # SFX: Footstep
                    self.hero_pos = next_pos
                    self._add_particle(self.hero_pos, (100, 100, 100), 5, 5, is_grid_pos=True, count=3)
            action_taken = True
        
        if not action_taken: # Priority 4: Defend
            self.is_defending = True
            self._add_log("Hero is defending.")
            action_taken = True

        # --- Enemy Turn ---
        if not self.game_over:
            for enemy in self.enemies:
                if self._is_adjacent(self.hero_pos, enemy['pos']):
                    # SFX: Enemy attack
                    damage = int(enemy['base_damage'] * (1 + (self.level - 1) * 0.1))
                    if self.is_defending:
                        damage //= 2
                        self._add_log(f"Blocked! Took {damage} dmg.")
                    else:
                        self._add_log(f"Hero took {damage} dmg from {enemy['name']}.")
                    
                    self.hero_health -= damage
                    reward -= 0.1
                    self.hero_health = max(0, self.hero_health)
                    self._add_particle(self.hero_pos, (255, 50, 50), 10, 10)
        
        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.hero_health <= 0:
            # SFX: Player death
            self.game_over = True
            terminated = True
            reward -= 100
            self._add_log("You have been defeated.")
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            self._add_log("Ran out of time!")
        elif self.game_over: # Victory case
            terminated = True
            
        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _kill_enemy(self, enemy):
        # SFX: Enemy death
        self._add_log(f"{enemy['name']} defeated! +1 reward.")
        self._add_particle(enemy['pos'], (200, 0, 0), 30, 30, is_grid_pos=True, count=20)
        self.enemies.remove(enemy)
        return 1

    def _generate_level(self):
        # 1. Create open grid
        self.grid = [[0] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
        
        # 2. Place walls, ensuring solvability
        for _ in range(self.GRID_SIZE * 2):
            x, y = self.np_random.integers(0, self.GRID_SIZE, size=2)
            self.grid[y][x] = 1
        
        # 3. Set hero and exit, ensuring they are not in walls
        self.hero_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.grid[self.hero_pos[1]][self.hero_pos[0]] = 0

        self.exit_pos = self.hero_pos
        while self._dist(self.hero_pos, self.exit_pos) < self.GRID_SIZE // 2 or self.grid[self.exit_pos[1]][self.exit_pos[0]] == 1:
            self.exit_pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
        
        # 4. Flood fill to ensure path exists
        q = [self.hero_pos]
        visited = {tuple(self.hero_pos)}
        while q:
            x, y = q.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if self._is_valid([nx, ny]) and self.grid[ny][nx] == 0 and tuple([nx, ny]) not in visited:
                    visited.add(tuple([nx, ny]))
                    q.append([nx, ny])
        
        # If exit is not reachable, clear a path or regenerate
        if tuple(self.exit_pos) not in visited:
            # Simple fix: clear all walls and try again
            return self._generate_level()

        # 5. Spawn enemies
        self.enemies.clear()
        num_enemies = self.np_random.integers(2, 5) + self.level // 2
        enemy_types = [
            {'name': 'Goblin', 'health': 20, 'damage': 5, 'color': (50, 200, 50)},
            {'name': 'Orc', 'health': 40, 'damage': 7, 'color': (100, 150, 50)},
            {'name': 'Troll', 'health': 60, 'damage': 9, 'color': (150, 120, 50)},
            {'name': 'Ogre', 'health': 80, 'damage': 11, 'color': (200, 100, 50)},
            {'name': 'Demon', 'health': 100, 'damage': 13, 'color': (220, 50, 50)},
        ]
        
        for i in range(num_enemies):
            pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
            while pos == self.hero_pos or pos == self.exit_pos or self.grid[pos[1]][pos[0]] == 1 or any(e['pos'] == pos for e in self.enemies):
                pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
            
            enemy_type_idx = min(len(enemy_types) - 1, self.np_random.integers(0, self.level // 2 + 2))
            base_stat = enemy_types[enemy_type_idx]
            
            health_mod = 1 + (self.level - 1) * 0.1
            
            self.enemies.append({
                'pos': pos,
                'name': base_stat['name'],
                'max_health': int(base_stat['health'] * health_mod),
                'health': int(base_stat['health'] * health_mod),
                'base_damage': base_stat['damage'],
                'color': base_stat['color']
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(self.GRID_OFFSET_X + x * self.TILE_SIZE,
                                   self.GRID_OFFSET_Y + y * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[y][x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

        # Draw exit portal
        self.exit_pulse = (self.exit_pulse + 0.1) % (2 * math.pi)
        pulse_val = int((math.sin(self.exit_pulse) + 1) / 2 * 55)
        # FIX: Pulse the green component instead of blue to avoid negative values
        exit_color = (self.COLOR_EXIT[0], self.COLOR_EXIT[1] - pulse_val, self.COLOR_EXIT[2])
        exit_rect = pygame.Rect(self.GRID_OFFSET_X + self.exit_pos[0] * self.TILE_SIZE,
                                self.GRID_OFFSET_Y + self.exit_pos[1] * self.TILE_SIZE,
                                self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, exit_color, exit_rect)
        pygame.gfxdraw.filled_circle(self.screen, exit_rect.centerx, exit_rect.centery, self.TILE_SIZE // 3, (255, 255, 150, 100))

        # Draw enemies
        for enemy in self.enemies:
            rect = self._get_grid_rect(enemy['pos'])
            pygame.draw.rect(self.screen, enemy['color'], rect.inflate(-4, -4))
            self._render_health_bar(rect.centerx, rect.top, enemy['health'], enemy['max_health'])

        # Draw hero
        hero_rect = self._get_grid_rect(self.hero_pos)
        pygame.draw.rect(self.screen, self.COLOR_HERO, hero_rect.inflate(-4, -4))
        self._render_health_bar(hero_rect.centerx, hero_rect.top, self.hero_health, self.hero_max_health)
        if self.is_defending:
            pygame.gfxdraw.filled_circle(self.screen, hero_rect.centerx, hero_rect.centery, self.TILE_SIZE // 2, self.COLOR_SHIELD)

        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
                color = p['color'] + (alpha,)
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Render level and score
        self._render_text(f"Level: {self.level}", (10, 10), self.font_small)
        self._render_text(f"Score: {self.score:.1f}", (10, 30), self.font_small)

        # Render combat log
        for i, msg in enumerate(self.combat_log[-4:]):
            alpha = 255 - (len(self.combat_log[-4:]) - 1 - i) * 60
            self._render_text(msg, (self.GRID_OFFSET_X, self.HEIGHT - 60 + i * 15), self.font_small, color=(200, 200, 200, alpha))
        
        # Render game over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.level >= self.VICTORY_LEVEL else "GAME OVER"
            color = (50, 255, 50) if self.level >= self.VICTORY_LEVEL else (255, 50, 50)
            self._render_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_large, color=color, center=True)

    def _render_health_bar(self, x, y, health, max_health):
        if health < 0: health = 0
        bar_width = self.TILE_SIZE * 0.8
        bar_height = 5
        x_pos = x - bar_width / 2
        y_pos = y - bar_height - 2
        
        fill_ratio = health / max_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x_pos, y_pos, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (x_pos, y_pos, bar_width * fill_ratio, bar_height))

    def _render_text(self, text, pos, font, color=None, center=False):
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        if len(color) == 4: # Handle alpha
            text_surface.set_alpha(color[3])
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "health": self.hero_health}

    def close(self):
        pygame.font.quit()
        pygame.quit()
    
    # --- Helper methods ---
    def _is_valid(self, pos):
        return 0 <= pos[0] < self.GRID_SIZE and 0 <= pos[1] < self.GRID_SIZE

    def _dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _is_adjacent(self, p1, p2):
        return self._dist(p1, p2) == 1

    def _get_adjacent_enemy(self):
        for enemy in self.enemies:
            if self._is_adjacent(self.hero_pos, enemy['pos']):
                return enemy
        return None
    
    def _get_grid_rect(self, grid_pos):
        return pygame.Rect(self.GRID_OFFSET_X + grid_pos[0] * self.TILE_SIZE,
                           self.GRID_OFFSET_Y + grid_pos[1] * self.TILE_SIZE,
                           self.TILE_SIZE, self.TILE_SIZE)
    
    def _grid_to_screen(self, grid_pos):
        rect = self._get_grid_rect(grid_pos)
        return list(rect.center)
    
    def _add_log(self, message):
        self.combat_log.append(message)
        if len(self.combat_log) > 20:
            self.combat_log.pop(0)

    def _add_particle(self, pos, color, size, life, is_grid_pos=False, count=10):
        screen_pos = self._grid_to_screen(pos) if is_grid_pos else pos
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(screen_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': self.np_random.integers(1, size + 1),
                'life': self.np_random.integers(life // 2, life + 1),
                'max_life': life
            })