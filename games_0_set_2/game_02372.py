# Generated: 2025-08-27T20:10:43.245989
# Source Brief: brief_02372.md
# Brief Index: 2372

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to place a block. Defend the core!"
    )

    game_description = (
        "Build a block fortress to defend your core against waves of enemies in this isometric strategy game."
    )

    auto_advance = True

    # --- Constants ---
    # Game world
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 22, 18
    TILE_WIDTH = 32
    TILE_HEIGHT = TILE_WIDTH // 2
    ISO_ORIGIN_X = WIDTH // 2
    ISO_ORIGIN_Y = 60
    MAX_STEPS = 3000
    MAX_WAVES = 10
    BUILD_ZONE_Y_START = 8
    
    # Colors
    COLOR_BG = (34, 32, 52)
    COLOR_GRID = (60, 56, 82)
    COLOR_FORTRESS = (0, 150, 255)
    COLOR_BLOCK = (20, 200, 120)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_CURSOR_VALID = (0, 255, 0, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SHADOW = (24, 22, 42)

    # --- Helper Classes ---
    class Enemy:
        def __init__(self, grid_pos, speed, health, damage):
            self.grid_pos = list(grid_pos)
            self.pixel_pos = [0, 0]
            self.speed = speed
            self.health = health
            self.max_health = health
            self.damage = damage
            self.bob_offset = random.uniform(0, math.pi * 2)
            self.z_offset = 0

        def update_visuals(self, iso_origin_x, iso_origin_y, tile_width, tile_height):
            self.bob_offset += 0.15
            self.z_offset = (math.sin(self.bob_offset) + 1) * 3
            
            target_px = GameEnv._iso_to_screen(self.grid_pos[0], self.grid_pos[1], iso_origin_x, iso_origin_y, tile_width, tile_height)
            
            if self.pixel_pos == [0, 0]:
                self.pixel_pos = list(target_px)
            else:
                dx = target_px[0] - self.pixel_pos[0]
                dy = target_px[1] - self.pixel_pos[1]
                dist = math.hypot(dx, dy)
                if dist < self.speed:
                    self.pixel_pos = list(target_px)
                else:
                    self.pixel_pos[0] += (dx / dist) * self.speed
                    self.pixel_pos[1] += (dy / dist) * self.speed
    
    class Particle:
        def __init__(self, pos, vel, size, color, lifespan):
            self.pos = list(pos)
            self.vel = list(vel)
            self.size = size
            self.color = color
            self.lifespan = lifespan
            self.max_lifespan = lifespan

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.lifespan -= 1
            self.size = max(0, self.size - 0.1)

    # --- Initialization ---
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_m = pygame.font.SysFont(None, 22)
            self.font_l = pygame.font.SysFont(None, 30)

        # These attributes are defined here and initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fortress_health = 0
        self.fortress_pos = []
        self.fortress_blocks = []
        self.cursor_pos = []
        self.blocks = {}
        self.enemies = []
        self.particles = []
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = 0
        self.blocks_available = 0
        self.screen_shake = 0
        self.rng = None
        self.last_space_held = False

        self.reset()

    # --- Gymnasium API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.fortress_health = 500
        self.max_fortress_health = 500
        
        cx, cy = self.GRID_WIDTH // 2, self.GRID_HEIGHT - 3
        self.fortress_pos = [cx, cy]
        self.fortress_blocks = [
            (cx, cy), (cx-1, cy), (cx, cy-1), (cx-1, cy-1)
        ]

        self.cursor_pos = [self.GRID_WIDTH // 2, self.BUILD_ZONE_Y_START + 2]
        self.blocks = {}
        self.enemies = []
        self.particles = []
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = 90  # 3 seconds at 30fps
        self.blocks_available = 10
        self.screen_shake = 0
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not terminated:
            self._handle_input(action)
            
            wave_reward = self._update_waves()
            enemy_reward = self._update_enemies()
            
            reward += wave_reward + enemy_reward
            
            self._update_particles()
            self.screen_shake = max(0, self.screen_shake - 1)

            self.steps += 1
            
            terminated = self._check_termination()
            if terminated:
                self.game_over = True
                if self.fortress_health <= 0:
                    reward -= 100
                elif self.wave_number > self.MAX_WAVES:
                    reward += 100
            
            self.score += reward # Add terminal reward to final score

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Game Logic ---
    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], self.BUILD_ZONE_Y_START, self.GRID_HEIGHT - 1)

        # Place block
        is_valid_pos = self._is_valid_build_pos(tuple(self.cursor_pos))
        if space_held and not self.last_space_held and self.blocks_available > 0 and is_valid_pos:
            self.blocks[tuple(self.cursor_pos)] = {'health': 100, 'max_health': 100, 'damage_timer': 0}
            self.blocks_available -= 1
            # SFX: place_block.wav
            for _ in range(10):
                self._add_particle(self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1]), size=4, count=1, force=1.5)

        self.last_space_held = space_held

    def _update_waves(self):
        reward = 0
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.wave_cooldown = 150 # 5 seconds
            if self.wave_number <= self.MAX_WAVES:
                reward += 1
                self.blocks_available += 8 + self.wave_number
                # SFX: wave_complete.wav
        
        if not self.wave_in_progress and self.wave_number < self.MAX_WAVES:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.wave_number += 1
                self.wave_in_progress = True
                num_enemies = 5 + (self.wave_number - 1) * 2
                enemy_speed = 0.8 + (self.wave_number - 1) * 0.05
                enemy_health = 50 + self.wave_number * 10
                enemy_damage = 10 + self.wave_number * 2
                self._spawn_enemies(num_enemies, enemy_speed, enemy_health, enemy_damage)
                # SFX: new_wave.wav
        return reward

    def _update_enemies(self):
        total_reward = 0
        fortress_center = (self.fortress_pos[0] - 0.5, self.fortress_pos[1] - 0.5)

        for i in range(len(self.enemies) - 1, -1, -1):
            enemy = self.enemies[i]
            enemy.update_visuals(self.ISO_ORIGIN_X, self.ISO_ORIGIN_Y, self.TILE_WIDTH, self.TILE_HEIGHT)
            
            # Pathfinding
            gx, gy = enemy.grid_pos
            
            dist_to_fortress = math.hypot(gx - fortress_center[0], gy - fortress_center[1])
            if dist_to_fortress < 1.5: # Close enough to attack fortress
                self.fortress_health -= enemy.damage
                self.screen_shake = 15
                pos = self._iso_to_screen(gx, gy)
                self._add_particle(pos, size=8, count=30, force=3)
                # SFX: fortress_hit_explosion.wav
                del self.enemies[i]
                total_reward += 0.1 # "Killed" by reaching goal
                continue

            # Simple greedy pathing
            possible_moves = []
            if gy < self.GRID_HEIGHT -1: possible_moves.append((gx, gy + 1))
            if gx < fortress_center[0]: possible_moves.append((gx + 1, gy))
            elif gx > fortress_center[0]: possible_moves.append((gx - 1, gy))
            
            best_move = None
            min_dist = float('inf')
            for move in possible_moves:
                dist = math.hypot(move[0] - fortress_center[0], move[1] - fortress_center[1])
                if dist < min_dist:
                    min_dist = dist
                    best_move = move
            
            if best_move:
                if best_move in self.blocks:
                    # Attack block
                    block = self.blocks[best_move]
                    block['health'] -= enemy.damage
                    block['damage_timer'] = 5
                    self._add_particle(self._iso_to_screen(best_move[0], best_move[1]), size=3, count=3, force=1)
                    # SFX: hit_block.wav
                    if block['health'] <= 0:
                        del self.blocks[best_move]
                        total_reward -= 0.01
                        self._add_particle(self._iso_to_screen(best_move[0], best_move[1]), size=6, count=20, force=2)
                        # SFX: block_destroy.wav
                else:
                    enemy.grid_pos = list(best_move)
        return total_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    # --- Rendering ---
    def _get_observation(self):
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            render_offset_x = self.rng.integers(-5, 6)
            render_offset_y = self.rng.integers(-5, 6)
        
        temp_surf = self.screen.copy()
        temp_surf.fill(self.COLOR_BG)
        
        self._render_grid(temp_surf)
        
        # Depth sort and render entities
        renderables = []
        for pos, block in self.blocks.items():
            renderables.append(('block', pos, block))
        for pos in self.fortress_blocks:
            renderables.append(('fortress', pos, None))
        for enemy in self.enemies:
            renderables.append(('enemy', enemy.grid_pos, enemy))
        
        renderables.sort(key=lambda item: (item[1][0] + item[1][1], item[1][1]))
        
        for r_type, r_obj, r_data in renderables:
            if r_type == 'block':
                self._draw_iso_cube(temp_surf, r_obj, self.COLOR_BLOCK, 1, r_data['health']/r_data['max_health'], r_data['damage_timer'] > 0)
                r_data['damage_timer'] = max(0, r_data['damage_timer'] - 1)
            elif r_type == 'fortress':
                self._draw_iso_cube(temp_surf, r_obj, self.COLOR_FORTRESS, 1)
            elif r_type == 'enemy':
                self._draw_enemy(temp_surf, r_data)
        
        self._render_cursor(temp_surf)
        self._render_particles(temp_surf)
        
        self.screen.blit(temp_surf, (render_offset_x, render_offset_y))
        self._render_ui(self.screen)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self, surface):
        for y in range(self.BUILD_ZONE_Y_START, self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.draw.polygon(surface, self.COLOR_GRID, [p1, p2, p3, p4], 1)
    
    def _render_cursor(self, surface):
        is_valid = self._is_valid_build_pos(tuple(self.cursor_pos))
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        p1 = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        p2 = self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1])
        p3 = self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1)
        p4 = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1] + 1)
        
        cursor_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(cursor_surf, [p1, p2, p3, p4], color)
        surface.blit(cursor_surf, (0, 0))

    def _draw_iso_cube(self, surface, grid_pos, color, height_mult=1, health_pct=1.0, is_damaged=False):
        x, y = grid_pos
        h = self.TILE_HEIGHT * height_mult
        
        top_color = tuple(min(255, c * 1.1) for c in color)
        side_color = tuple(min(255, c * 0.8) for c in color)
        
        if is_damaged:
            top_color, side_color = (255,255,255), (220,220,220)
        
        base_x, base_y = self._iso_to_screen(x, y)
        
        top_points = [
            (base_x, base_y - h),
            (base_x + self.TILE_WIDTH // 2, base_y - h + self.TILE_HEIGHT // 2),
            (base_x, base_y - h + self.TILE_HEIGHT),
            (base_x - self.TILE_WIDTH // 2, base_y - h + self.TILE_HEIGHT // 2)
        ]

        shadow_points = [
            (p[0], p[1] + h + 2) for p in top_points
        ]
        pygame.gfxdraw.filled_polygon(surface, shadow_points, self.COLOR_SHADOW)
        
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)
        
        side_points_1 = [top_points[3], top_points[2], (top_points[2][0], top_points[2][1] + h), (top_points[3][0], top_points[3][1] + h)]
        side_points_2 = [top_points[2], top_points[1], (top_points[1][0], top_points[1][1] + h), (top_points[2][0], top_points[2][1] + h)]
        
        pygame.gfxdraw.filled_polygon(surface, side_points_1, side_color)
        pygame.gfxdraw.filled_polygon(surface, side_points_2, side_color)
        
        if health_pct < 1.0:
            health_bar_y = base_y - h - 10
            health_bar_w = self.TILE_WIDTH * 0.8
            pygame.draw.rect(surface, (50,0,0), (base_x - health_bar_w/2, health_bar_y, health_bar_w, 5))
            pygame.draw.rect(surface, (0,200,0), (base_x - health_bar_w/2, health_bar_y, health_bar_w * health_pct, 5))

    def _draw_enemy(self, surface, enemy):
        x, y = enemy.pixel_pos
        z = enemy.z_offset
        size = 8
        
        shadow_pos = (int(x), int(y + z + size * 0.8))
        pygame.gfxdraw.filled_ellipse(surface, shadow_pos[0], shadow_pos[1], size, size//2, self.COLOR_SHADOW)

        # Body
        body_color = self.COLOR_ENEMY
        pygame.draw.circle(surface, body_color, (int(x), int(y - z)), size)
        
        # Highlight
        highlight_color = (255, 150, 150)
        pygame.draw.circle(surface, highlight_color, (int(x + size*0.2), int(y - z - size*0.2)), size // 3)

    def _render_particles(self, surface):
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = p.color + (alpha,)
            pos = (int(p.pos[0]), int(p.pos[1]))
            size = int(p.size)
            if size > 0:
                # Use a temp surface for alpha blending
                part_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(part_surf, color, (size, size), size)
                surface.blit(part_surf, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self, surface):
        # Health bar
        bar_w, bar_h = 200, 20
        health_pct = max(0, self.fortress_health / self.max_fortress_health)
        pygame.draw.rect(surface, (80,0,0), (10, 10, bar_w, bar_h))
        pygame.draw.rect(surface, (200,0,0), (10, 10, bar_w * health_pct, bar_h))
        pygame.draw.rect(surface, self.COLOR_TEXT, (10, 10, bar_w, bar_h), 2)
        health_text = self.font_m.render(f"CORE: {self.fortress_health}/{self.max_fortress_health}", True, self.COLOR_TEXT)
        surface.blit(health_text, (15, 12))

        # Wave info
        wave_str = f"WAVE {self.wave_number}/{self.MAX_WAVES}" if self.wave_in_progress or self.enemies else f"NEXT WAVE IN {self.wave_cooldown//30+1}"
        if self.wave_number > self.MAX_WAVES: wave_str = "VICTORY!"
        if self.fortress_health <= 0: wave_str = "DEFEAT"
        wave_text = self.font_l.render(wave_str, True, self.COLOR_TEXT)
        surface.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 10))

        # Block count
        block_text = self.font_l.render(f"BLOCKS: {self.blocks_available}", True, self.COLOR_TEXT)
        surface.blit(block_text, (self.WIDTH - block_text.get_width() - 15, 10))
        
        # Score
        score_text = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        surface.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 40))

    # --- Helpers & Utils ---
    @staticmethod
    def _iso_to_screen(grid_x, grid_y, origin_x=ISO_ORIGIN_X, origin_y=ISO_ORIGIN_Y, tile_w=TILE_WIDTH, tile_h=TILE_HEIGHT):
        screen_x = origin_x + (grid_x - grid_y) * (tile_w / 2)
        screen_y = origin_y + (grid_x + grid_y) * (tile_h / 2)
        return int(screen_x), int(screen_y)

    def _is_valid_build_pos(self, pos):
        return (pos not in self.blocks and
                pos not in self.fortress_blocks and
                self.BUILD_ZONE_Y_START <= pos[1] < self.GRID_HEIGHT and
                0 <= pos[0] < self.GRID_WIDTH)

    def _spawn_enemies(self, num, speed, health, damage):
        for _ in range(num):
            spawn_x = self.rng.integers(0, self.GRID_WIDTH)
            spawn_y = self.rng.integers(0, 3)
            self.enemies.append(self.Enemy((spawn_x, spawn_y), speed, health, damage))

    def _add_particle(self, pos, size=5, count=10, force=2.0, color=(255, 200, 100)):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(0.5, 1.0) * force
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.rng.integers(20, 40)
            self.particles.append(self.Particle(pos, vel, size * self.rng.uniform(0.5, 1.2), color, lifespan))

    def _check_termination(self):
        return (self.fortress_health <= 0 or
                (self.wave_number > self.MAX_WAVES and not self.enemies) or
                self.steps >= self.MAX_STEPS)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "fortress_health": self.fortress_health,
            "blocks_available": self.blocks_available,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Fortress Defense")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move = 0 # no-op
        if keys[pygame.K_UP]: move = 1
        elif keys[pygame.K_DOWN]: move = 2
        elif keys[pygame.K_LEFT]: move = 3
        elif keys[pygame.K_RIGHT]: move = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()