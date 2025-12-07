import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from collections import deque
import pygame.gfxdraw
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment set in a decaying clockwork labyrinth.
    The player must navigate levels, match colors to activate portals,
    and use a time-rewind ability to survive against clockwork enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = "Navigate a clockwork labyrinth, match colors to activate portals, and rewind time to evade enemies."
    user_guide = "Use arrow keys to move. Press space to activate portals. Press shift to rewind time."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_FPS = 30

    # Colors (Steampunk Theme)
    COLOR_BG = (20, 15, 10)
    COLOR_WALL = (60, 50, 40)
    COLOR_WALL_ACCENT = (80, 70, 60)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 128)
    COLOR_ENEMY_PATROLLER = (255, 50, 50)
    COLOR_ENEMY_SPINNER = (255, 100, 0)
    COLOR_ENEMY_SHOOTER = (200, 0, 100)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_PORTAL_FRAME = (212, 175, 55) # Gold
    COLOR_UI_TEXT = (220, 220, 200)
    COLOR_HEALTH_BAR = (0, 255, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_REWIND_COOLDOWN = (0, 150, 255)

    PORTAL_COLORS = {
        "RED": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "BLUE": (50, 50, 255),
        "GOLD": (255, 215, 0),
    }

    # Game Parameters
    MAX_STEPS = 2500
    PLAYER_SPEED = 4
    PLAYER_SIZE = 8
    PLAYER_MAX_HEALTH = 100
    REWIND_COOLDOWN_STEPS = 90  # 3 seconds
    REWIND_DURATION_FRAMES = 15 # 0.5 seconds
    ENEMY_HISTORY_LENGTH = 60 # Store 2 seconds of history

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.render_mode = render_mode
        
        # Initialize state variables to avoid attribute errors
        self._initialize_state()

    def _initialize_state(self):
        self.player_pos = np.array([0.0, 0.0])
        self.player_health = 0
        self.player_active_color = None
        self.level = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.rewind_cooldown = 0
        self.is_rewinding = False
        self.rewind_timer = 0
        
        self.walls = []
        self.enemies = []
        self.projectiles = []
        self.portals = []
        self.color_pads = []
        self.particles = []
        self.visited_tiles = set()
        self.background_gears = []
        self.unlocked_colors = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()

        self.player_health = self.PLAYER_MAX_HEALTH
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.level = 1
        
        self._setup_level(self.level)
        
        return self._get_observation(), self._get_info()

    def _setup_level(self, level_num):
        self.walls = []
        self.enemies = []
        self.projectiles = []
        self.portals = []
        self.color_pads = []
        self.particles = []
        self.visited_tiles.clear()
        self.player_active_color = None
        self.unlocked_colors = set()

        # --- Procedural Level Generation ---
        grid_w, grid_h = self.SCREEN_WIDTH // 20, self.SCREEN_HEIGHT // 20
        grid = np.ones((grid_w, grid_h)) # 1 = wall, 0 = floor

        # Carve out paths
        def carve(x, y):
            grid[x, y] = 0
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.np_random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx*2, y + dy*2
                if 0 < nx < grid_w-1 and 0 < ny < grid_h-1 and grid[nx, ny] == 1:
                    grid[x+dx, y+dy] = 0
                    carve(nx, ny)
        
        carve(1, 1)

        # Place player
        self.player_pos = np.array([30.0, 30.0])
        self.visited_tiles.add((1,1))

        # Create wall rects from grid
        for x in range(grid_w):
            for y in range(grid_h):
                if grid[x, y] == 1:
                    self.walls.append(pygame.Rect(x * 20, y * 20, 20, 20))

        # Generate background gears
        self.background_gears = []
        for _ in range(20):
            self.background_gears.append({
                "pos": (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                "radius": self.np_random.integers(20, 81),
                "speed": self.np_random.uniform(-0.1, 0.1),
                "angle": self.np_random.uniform(0, 360)
            })

        # Place entities
        floor_tiles = [(x, y) for x in range(grid_w) for y in range(grid_h) if grid[x, y] == 0]
        self.np_random.shuffle(floor_tiles)

        # Place color pads and portals
        num_colors = min(3, level_num)
        level_colors = self.np_random.choice(list(self.PORTAL_COLORS.keys())[:3], num_colors, replace=False).tolist()

        for i, color_name in enumerate(level_colors):
            pad_pos = floor_tiles.pop()
            self.color_pads.append({"pos": (pad_pos[0]*20+10, pad_pos[1]*20+10), "color": color_name})
            
            portal_pos = floor_tiles.pop()
            self.portals.append({"pos": (portal_pos[0]*20+10, portal_pos[1]*20+10), "color": color_name, "is_level_exit": False})

        # Place level exit portal
        exit_pos = floor_tiles.pop()
        self.portals.append({"pos": (exit_pos[0]*20+10, exit_pos[1]*20+10), "color": "GOLD", "is_level_exit": True})

        # Place enemies
        enemy_speed = 1.0 + (level_num // 2) * 0.2
        num_enemies = min(10, 2 + level_num)
        for _ in range(num_enemies):
            if not floor_tiles: break
            enemy_pos_grid = floor_tiles.pop()
            enemy_pos = np.array([enemy_pos_grid[0]*20+10, enemy_pos_grid[1]*20+10], dtype=float)
            enemy_type = self.np_random.choice(["patroller", "spinner", "shooter"])
            
            if enemy_type == "patroller" and level_num >= 1:
                self.enemies.append(Patroller(enemy_pos, enemy_speed, self.np_random))
            elif enemy_type == "spinner" and level_num >= 2:
                self.enemies.append(Spinner(enemy_pos, enemy_speed, self.np_random))
            elif enemy_type == "shooter" and level_num >= 3:
                self.enemies.append(Shooter(enemy_pos, enemy_speed, self.np_random))

    def step(self, action):
        self.steps += 1
        reward = 0.0
        self.game_over = self.player_health <= 0

        if self.game_over:
            return self._get_observation(), -100.0, True, False, self._get_info()

        # -- Action Handling --
        movement_action = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1

        # -- Rewind Mechanic --
        if self.is_rewinding:
            self.rewind_timer -= 1
            if self.rewind_timer <= 0:
                self.is_rewinding = False
        elif shift_pressed and self.rewind_cooldown <= 0:
            self.is_rewinding = True
            self.rewind_timer = self.REWIND_DURATION_FRAMES
            self.rewind_cooldown = self.REWIND_COOLDOWN_STEPS
            for e in self.enemies: e.rewind(self.REWIND_DURATION_FRAMES)
            self.projectiles = [p for p in self.projectiles if p['age'] > self.REWIND_DURATION_FRAMES]
        
        if self.rewind_cooldown > 0:
            self.rewind_cooldown -= 1

        # -- Player Movement --
        move_vec = np.array([0.0, 0.0])
        if movement_action == 1: move_vec[1] = -1
        elif movement_action == 2: move_vec[1] = 1
        elif movement_action == 3: move_vec[0] = -1
        elif movement_action == 4: move_vec[0] = 1

        if np.any(move_vec):
            new_pos = self.player_pos + move_vec * self.PLAYER_SPEED
            player_rect = pygame.Rect(new_pos[0] - self.PLAYER_SIZE, new_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
            
            if not any(wall.colliderect(player_rect) for wall in self.walls):
                self.player_pos = new_pos
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        current_tile = (int(self.player_pos[0] // 20), int(self.player_pos[1] // 20))
        if current_tile not in self.visited_tiles:
            self.visited_tiles.add(current_tile)
            reward += 0.1

        # -- Update Game Entities --
        if not self.is_rewinding:
            for enemy in self.enemies:
                new_projectiles = enemy.update(self.player_pos, self.walls)
                if new_projectiles:
                    self.projectiles.extend(new_projectiles)
        
        for p in self.projectiles:
            p['pos'] += p['vel']
            p['age'] += 1
        self.projectiles = [p for p in self.projectiles if p['age'] < 150 and not any(w.collidepoint(p['pos']) for w in self.walls)]

        # -- Collisions and Interactions --
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)

        for enemy in self.enemies:
            if player_rect.colliderect(enemy.get_rect()):
                self.player_health -= 10
                reward -= 1.0
                self._spawn_particles(self.player_pos, self.COLOR_ENEMY_PATROLLER, 20)
                self.player_health = max(0, self.player_health)
                break

        for p in self.projectiles[:]:
            if player_rect.collidepoint(p['pos']):
                self.player_health -= 10
                reward -= 1.0
                self._spawn_particles(self.player_pos, self.COLOR_PROJECTILE, 20)
                self.projectiles.remove(p)
                self.player_health = max(0, self.player_health)

        for pad in self.color_pads:
            if np.linalg.norm(self.player_pos - pad['pos']) < self.PLAYER_SIZE + 10:
                if self.player_active_color != pad['color']:
                    self.player_active_color = pad['color']
                    reward += 1.0
                    self._spawn_particles(pad['pos'], self.PORTAL_COLORS[pad['color']], 30)

        if space_pressed:
            for portal in self.portals:
                if np.linalg.norm(self.player_pos - portal['pos']) < self.PLAYER_SIZE + 15:
                    num_base_colors = len([c for c in self.color_pads])
                    is_gold_unlocked = len(self.unlocked_colors) == num_base_colors
                    
                    if portal['color'] == self.player_active_color:
                        reward += 5.0
                        self.score += 5
                        self.unlocked_colors.add(portal['color'])
                        self._spawn_particles(portal['pos'], self.PORTAL_COLORS[portal['color']], 50)
                        other_portals = [p for p in self.portals if p['color'] == portal['color'] and p is not portal]
                        if other_portals:
                            self.player_pos = np.array(random.choice(other_portals)['pos'])
                        break
                    
                    elif portal['is_level_exit'] and is_gold_unlocked:
                        self.level += 1
                        self.score += 100
                        if self.level >= 10:
                            reward += 1000.0
                            self.game_over = True
                        else:
                            reward += 100.0
                            self._setup_level(self.level)
                        break

        self.particles = [p for p in self.particles if p.update()]

        terminated = self.game_over or self.player_health <= 0
        truncated = self.steps >= self.MAX_STEPS
        if self.player_health <= 0:
            reward = -10.0
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "health": self.player_health,
        }

    def _render_game(self):
        for gear in self.background_gears:
            gear['angle'] += gear['speed'] * (1 if not self.is_rewinding else -5)
            self._draw_gear(self.screen, (40, 30, 25), gear['pos'], gear['radius'], 12, gear['angle'])

        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            pygame.draw.rect(self.screen, self.COLOR_WALL_ACCENT, wall.inflate(-4, -4))

        for pad in self.color_pads:
            color = self.PORTAL_COLORS[pad['color']]
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 5
            pygame.gfxdraw.filled_circle(self.screen, int(pad['pos'][0]), int(pad['pos'][1]), int(12 + pulse), (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, int(pad['pos'][0]), int(pad['pos'][1]), 10, color)

        for portal in self.portals:
            self._render_portal(portal)

        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, p['pos'], 3)
            pygame.draw.circle(self.screen, (255,255,255), p['pos'], 1)

        for enemy in self.enemies:
            enemy.draw(self.screen, self.steps)
        
        for p in self.particles:
            p.draw(self.screen)

        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        glow_size = int(self.PLAYER_SIZE * 1.8 + (math.sin(self.steps * 0.2) + 1) * 2)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, glow_size, (*self.COLOR_PLAYER_GLOW, 100))
        
        if self.player_active_color:
            active_color_rgb = self.PORTAL_COLORS[self.player_active_color]
            pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_SIZE + 3, active_color_rgb)
            pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_SIZE + 4, active_color_rgb)

        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_SIZE, self.COLOR_PLAYER)

        if self.is_rewinding:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            factor = self.rewind_timer / self.REWIND_DURATION_FRAMES
            alpha = int(100 * factor)
            for i in range(0, self.SCREEN_HEIGHT, 10):
                y_offset = int(math.sin(i * 0.1 + self.steps * 0.5) * 10 * factor)
                pygame.draw.line(overlay, (255, 255, 255, alpha), (0, i), (self.SCREEN_WIDTH, i + y_offset), 2)
            self.screen.blit(overlay, (0, 0))

    def _render_portal(self, portal):
        pos = (int(portal['pos'][0]), int(portal['pos'][1]))
        num_base_colors = len([c for c in self.color_pads])
        is_gold_unlocked = len(self.unlocked_colors) == num_base_colors
        
        is_active = (portal['color'] == self.player_active_color) or (portal['is_level_exit'] and is_gold_unlocked)

        for i in range(20):
            angle = (self.steps * 0.05) + (i * 2 * math.pi / 20)
            radius = (math.sin(angle * 3 + i) + 1.5) * 5
            px = pos[0] + math.cos(angle) * radius
            py = pos[1] + math.sin(angle) * radius
            color = self.PORTAL_COLORS[portal['color']]
            if not is_active:
                color = tuple(c // 3 for c in color)
            pygame.draw.circle(self.screen, color, (px, py), 2)

        frame_color = self.COLOR_PORTAL_FRAME if is_active else (80, 70, 60)
        self._draw_gear(self.screen, frame_color, pos, 18, 8, self.steps if is_active else 0)

    def _render_ui(self):
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"HP: {int(self.player_health)}/{self.PLAYER_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 40))

        cooldown_ratio = 1.0 - (self.rewind_cooldown / self.REWIND_COOLDOWN_STEPS)
        if cooldown_ratio < 1.0:
            pygame.draw.rect(self.screen, (50, 50, 80), (10, 35, 150, 10))
            pygame.draw.rect(self.screen, self.COLOR_REWIND_COOLDOWN, (10, 35, 150 * cooldown_ratio, 10))

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color, self.np_random))

    def _draw_gear(self, surface, color, pos, radius, num_teeth, angle_deg):
        angle_rad = math.radians(angle_deg)
        tooth_angle = 2 * math.pi / (num_teeth * 2)
        
        points = []
        for i in range(num_teeth * 2):
            r = radius if i % 2 == 0 else radius * 0.8
            current_angle = angle_rad + i * tooth_angle
            x = pos[0] + r * math.cos(current_angle)
            y = pos[1] + r * math.sin(current_angle)
            points.append((int(x), int(y)))
        
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        
        inner_radius = int(radius * 0.5)
        if inner_radius > 0:
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), inner_radius, self.COLOR_BG)
            pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), inner_radius, color)


class Particle:
    def __init__(self, pos, color, rng):
        self.pos = list(pos)
        self.vel = [rng.uniform(-2, 2), rng.uniform(-2, 2)]
        self.color = color
        self.lifespan = rng.integers(15, 31)
        self.life = self.lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.lifespan))
        radius = int(3 * (self.life / self.lifespan))
        if radius > 0:
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (radius, radius), radius)
            surface.blit(s, (self.pos[0] - radius, self.pos[1] - radius))


class BaseEnemy:
    def __init__(self, pos, speed, rng):
        self.pos = pos
        self.speed = speed
        self.size = 7
        self.history = deque(maxlen=GameEnv.ENEMY_HISTORY_LENGTH)
        self.rng = rng

    def update(self, player_pos, walls):
        self.history.append(self.pos.copy())
        return None

    def rewind(self, frames):
        if len(self.history) > frames:
            self.pos = self.history[-frames]
            self.history = deque(list(self.history)[:-frames], maxlen=GameEnv.ENEMY_HISTORY_LENGTH)

    def get_rect(self):
        return pygame.Rect(self.pos[0] - self.size, self.pos[1] - self.size, self.size * 2, self.size * 2)

    def draw(self, surface, steps):
        raise NotImplementedError


class Patroller(BaseEnemy):
    def __init__(self, pos, speed, rng):
        super().__init__(pos, speed, rng)
        self.direction = self.rng.choice([-1.0, 1.0], size=2)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction /= norm

    def update(self, player_pos, walls):
        super().update(player_pos, walls)
        new_pos = self.pos + self.direction * self.speed
        
        rect = pygame.Rect(new_pos[0]-self.size, new_pos[1]-self.size, self.size*2, self.size*2)
        if any(w.colliderect(rect) for w in walls):
            self.direction *= -1
        else:
            self.pos = new_pos
        return None

    def draw(self, surface, steps):
        pygame.draw.rect(surface, GameEnv.COLOR_ENEMY_PATROLLER, self.get_rect())
        pygame.draw.rect(surface, (255,255,255), self.get_rect().inflate(-4,-4))


class Spinner(BaseEnemy):
    def __init__(self, pos, speed, rng):
        super().__init__(pos, speed, rng)
        self.angle = 0
        self.size = 9

    def rewind(self, frames):
        if len(self.history) > frames:
            pos_hist, angle_hist = self.history[-frames]
            self.pos = pos_hist
            self.angle = angle_hist
            self.history = deque(list(self.history)[:-frames], maxlen=GameEnv.ENEMY_HISTORY_LENGTH)

    def update(self, player_pos, walls):
        self.history.append((self.pos.copy(), self.angle))
        self.angle += self.speed * 2
        return None

    def draw(self, surface, steps):
        angle_rad = math.radians(self.angle)
        x1 = self.pos[0] + math.cos(angle_rad) * self.size
        y1 = self.pos[1] + math.sin(angle_rad) * self.size
        x2 = self.pos[0] - math.cos(angle_rad) * self.size
        y2 = self.pos[1] - math.sin(angle_rad) * self.size
        pygame.draw.line(surface, GameEnv.COLOR_ENEMY_SPINNER, (x1, y1), (x2, y2), 4)
        
        angle_rad += math.pi / 2
        x1 = self.pos[0] + math.cos(angle_rad) * self.size
        y1 = self.pos[1] + math.sin(angle_rad) * self.size
        x2 = self.pos[0] - math.cos(angle_rad) * self.size
        y2 = self.pos[1] - math.sin(angle_rad) * self.size
        pygame.draw.line(surface, GameEnv.COLOR_ENEMY_SPINNER, (x1, y1), (x2, y2), 4)


class Shooter(BaseEnemy):
    def __init__(self, pos, speed, rng):
        super().__init__(pos, speed, rng)
        self.shoot_cooldown = 90
        self.cooldown_timer = self.rng.integers(0, self.shoot_cooldown)

    def update(self, player_pos, walls):
        super().update(player_pos, walls)
        self.cooldown_timer -= 1
        if self.cooldown_timer <= 0:
            self.cooldown_timer = self.shoot_cooldown
            direction = player_pos - self.pos
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist
                return [{
                    'pos': self.pos.copy(),
                    'vel': direction * 3,
                    'age': 0
                }]
        return None

    def draw(self, surface, steps):
        points = []
        for i in range(3):
            angle = math.radians(self.cooldown_timer * 2 + i * 120)
            x = self.pos[0] + math.cos(angle) * self.size
            y = self.pos[1] + math.sin(angle) * self.size
            points.append((x, y))
        pygame.draw.polygon(surface, GameEnv.COLOR_ENEMY_SHOOTER, points)


if __name__ == '__main__':
    # This block will not be run by the tests, but is useful for human-play testing.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Clockwork Labyrinth")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0

    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward}, Score: {info['score']}, Level: {info['level']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.GAME_FPS)

    pygame.quit()