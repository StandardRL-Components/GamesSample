
# Generated: 2025-08-27T18:24:56.559507
# Source Brief: brief_01824.md
# Brief Index: 1824

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place selected block. Shift to cycle block types."
    )

    game_description = (
        "Defend your isometric fortress from waves of enemies by placing defensive blocks."
    )

    auto_advance = True

    # --- Constants ---
    # Game Feel
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TARGET_FPS = 30
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps

    # Colors
    COLOR_BG = (18, 22, 33)
    COLOR_GRID = (30, 36, 50)
    COLOR_PATH = (45, 54, 75)
    COLOR_BASE = (50, 205, 50)
    COLOR_BASE_STROKE = (30, 120, 30)
    COLOR_ENEMY_A = (255, 70, 70)
    COLOR_ENEMY_B = (255, 140, 70)
    COLOR_ENEMY_C = (255, 200, 70)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CANNON = (0, 150, 255)
    COLOR_WALL = (150, 150, 150)
    COLOR_MINE = (255, 100, 255)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_UI_BG = (30, 36, 50, 180)

    # Isometric Grid
    GRID_SIZE_X, GRID_SIZE_Y = 20, 15
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    ISO_OFFSET_X = SCREEN_WIDTH // 2
    ISO_OFFSET_Y = 100

    # Game Logic
    BASE_START_HEALTH = 100
    STARTING_RESOURCES = 50
    MAX_WAVES = 20
    WAVE_COOLDOWN_FRAMES = 150 # 5 seconds

    # Block Types
    BLOCK_TYPES = ["CANNON", "WALL", "MINE"]
    BLOCK_COSTS = {"CANNON": 15, "WALL": 5, "MINE": 20}
    CANNON_RANGE = 4.5
    CANNON_COOLDOWN = 45 # 1.5 seconds
    CANNON_PROJECTILE_SPEED = 4.0
    MINE_RADIUS = 1.5
    MINE_DAMAGE = 30
    WALL_HEALTH = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_s = pygame.font.Font(None, 18)
        self.font_m = pygame.font.Font(None, 24)
        self.font_l = pygame.font.Font(None, 48)

        self.path = self._generate_path()
        
        # This will be initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave = 0
        self.wave_cooldown = 0
        self.enemies_in_wave = 0
        self.enemies = []
        self.blocks = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_block_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.base_damage_flash = 0
        
        self.reset()
        self.validate_implementation()

    def _generate_path(self):
        path = []
        x, y = 0, self.GRID_SIZE_Y // 2
        path.append((x, y))
        while x < self.GRID_SIZE_X - 2:
            move = random.choice(['r', 'r', 'r', 'd', 'u'])
            if move == 'r' and x < self.GRID_SIZE_X - 2:
                x += 1
            elif move == 'd' and y < self.GRID_SIZE_Y - 2:
                y += 1
            elif move == 'u' and y > 1:
                y -= 1
            if (x, y) not in path:
                path.append((x, y))
        
        while x < self.GRID_SIZE_X - 1:
            x += 1
            path.append((x, y))
        return path

    def _iso_to_screen(self, x, y):
        screen_x = self.ISO_OFFSET_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ISO_OFFSET_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.victory = False

        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.wave = 0
        self.wave_cooldown = self.WAVE_COOLDOWN_FRAMES
        self.enemies_in_wave = 0
        
        self.enemies = []
        self.blocks = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_SIZE_X // 2, self.GRID_SIZE_Y // 2]
        self.selected_block_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.base_damage_flash = 0

        # Regenerate path for variety
        if seed is not None:
            random.seed(seed)
        self.path = self._generate_path()

        return self._get_observation(), self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE_X - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE_Y - 1)

        # --- Cycle Block Type (on press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.BLOCK_TYPES)
            # sfx: ui_cycle

        # --- Place Block (on press) ---
        if space_held and not self.prev_space_held:
            self._place_block()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
    def _place_block(self):
        block_type = self.BLOCK_TYPES[self.selected_block_idx]
        cost = self.BLOCK_COSTS[block_type]
        cx, cy = self.cursor_pos
        
        is_on_path = (cx, cy) in self.path
        is_occupied = any(b['pos'] == [cx, cy] for b in self.blocks)
        is_base = (cx, cy) == self.path[-1]

        if self.resources >= cost and not is_occupied and not is_base:
            if block_type == "MINE" and not is_on_path: return # Mines must be on path
            if block_type != "MINE" and is_on_path: return # Others can't be on path

            self.resources -= cost
            self.reward_this_step -= 0.01

            new_block = {'pos': [cx, cy], 'type': block_type}
            if block_type == "CANNON":
                new_block['cooldown'] = 0
            if block_type == "WALL":
                new_block['health'] = self.WALL_HEALTH
            
            self.blocks.append(new_block)
            # sfx: place_block

    def _update_waves(self):
        if len(self.enemies) == 0 and self.enemies_in_wave == 0 and not self.game_over:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                if self.wave > 0: # Don't reward for surviving wave 0
                    self.score += 10
                    self.reward_this_step += 5
                
                self.wave += 1
                if self.wave > self.MAX_WAVES:
                    self.victory = True
                    return

                self.wave_cooldown = self.WAVE_COOLDOWN_FRAMES
                self._spawn_wave()

    def _spawn_wave(self):
        num_enemies = 3 + self.wave * 2
        self.enemies_in_wave = num_enemies
        
        base_speed = 0.5 + self.wave // 2 * 0.05
        base_health = 10 + self.wave // 3 * 5
        
        for i in range(num_enemies):
            enemy_type_roll = random.random()
            if self.wave > 10 and enemy_type_roll > 0.66:
                color, speed_mult, health_mult = self.COLOR_ENEMY_C, 1.2, 1.5
            elif self.wave > 5 and enemy_type_roll > 0.33:
                color, speed_mult, health_mult = self.COLOR_ENEMY_B, 1.1, 1.2
            else:
                color, speed_mult, health_mult = self.COLOR_ENEMY_A, 1.0, 1.0

            self.enemies.append({
                'pos': list(self.path[0]),
                'offset': [random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)],
                'screen_pos': self._iso_to_screen(*self.path[0]),
                'path_idx': 0,
                'health': int(base_health * health_mult),
                'max_health': int(base_health * health_mult),
                'speed': base_speed * speed_mult,
                'spawn_cooldown': i * (self.TARGET_FPS / 2),
                'color': color,
            })
        # sfx: wave_start

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy['spawn_cooldown'] > 0:
                enemy['spawn_cooldown'] -= 1
                continue

            if enemy['path_idx'] >= len(self.path) - 1:
                self.enemies.remove(enemy)
                self.enemies_in_wave -= 1
                self.base_health -= 1
                self.base_damage_flash = 5
                self.score -= 5
                # sfx: base_damage
                continue

            target_node = self.path[enemy['path_idx'] + 1]
            current_pos = enemy['pos']
            
            direction = [target_node[0] - current_pos[0], target_node[1] - current_pos[1]]
            dist = math.hypot(*direction)
            
            if dist < 0.1:
                enemy['path_idx'] += 1
            else:
                move = [d / dist * enemy['speed'] / self.TARGET_FPS for d in direction]
                enemy['pos'][0] += move[0]
                enemy['pos'][1] += move[1]

    def _update_blocks(self):
        for block in self.blocks:
            if block['type'] == 'CANNON':
                block['cooldown'] = max(0, block['cooldown'] - 1)
                if block['cooldown'] == 0:
                    target = self._find_target(block)
                    if target:
                        self._fire_projectile(block, target)
                        block['cooldown'] = self.CANNON_COOLDOWN
                        # sfx: cannon_fire

    def _find_target(self, cannon):
        closest_enemy = None
        min_dist = self.CANNON_RANGE ** 2
        
        for enemy in self.enemies:
            if enemy['spawn_cooldown'] > 0: continue
            dist_sq = (cannon['pos'][0] - enemy['pos'][0])**2 + (cannon['pos'][1] - enemy['pos'][1])**2
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_enemy = enemy
        return closest_enemy

    def _fire_projectile(self, cannon, target):
        start_pos = list(cannon['pos'])
        self.projectiles.append({
            'pos': start_pos,
            'target': target,
            'speed': self.CANNON_PROJECTILE_SPEED
        })

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            current_pos = proj['pos']
            
            direction = [target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]]
            dist = math.hypot(*direction)

            if dist < 0.2:
                self._hit_enemy(proj['target'], 10)
                self.projectiles.remove(proj)
                # sfx: projectile_hit
                continue

            move = [d / dist * proj['speed'] / self.TARGET_FPS for d in direction]
            proj['pos'][0] += move[0]
            proj['pos'][1] += move[1]
    
    def _update_collisions(self):
        for enemy in reversed(self.enemies):
            for block in reversed(self.blocks):
                if block['type'] == 'MINE':
                    dist_sq = (enemy['pos'][0] - block['pos'][0])**2 + (enemy['pos'][1] - block['pos'][1])**2
                    if dist_sq < 0.5**2: # Mine trigger radius
                        self._detonate_mine(block)
                        self.blocks.remove(block)
                        break # Mine is gone, move to next enemy

    def _detonate_mine(self, mine):
        # sfx: mine_explode
        mine_pos = mine['pos']
        self._create_explosion(mine_pos, self.MINE_RADIUS, self.COLOR_MINE)

        for enemy in self.enemies:
            dist_sq = (enemy['pos'][0] - mine_pos[0])**2 + (enemy['pos'][1] - mine_pos[1])**2
            if dist_sq < self.MINE_RADIUS ** 2:
                self._hit_enemy(enemy, self.MINE_DAMAGE)

    def _hit_enemy(self, enemy, damage):
        enemy['health'] -= damage
        self.reward_this_step += 0.1
        self._create_damage_particle(enemy, damage)
        
        if enemy['health'] <= 0:
            if enemy in self.enemies:
                self.enemies.remove(enemy)
                self.enemies_in_wave -= 1
                self.score += 5
                self.reward_this_step += 1
                self.resources += 5
                self._create_explosion(enemy['pos'], 0.8, enemy['color'])
                # sfx: enemy_die

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += p['gravity']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_damage_particle(self, enemy, damage):
        sx, sy = self._iso_to_screen(enemy['pos'][0] + enemy['offset'][0], enemy['pos'][1] + enemy['offset'][1])
        self.particles.append({
            'pos': [sx, sy - 15],
            'vel': [random.uniform(-0.5, 0.5), -1],
            'life': 20,
            'text': str(damage),
            'color': (255, 255, 100),
            'type': 'text',
            'gravity': 0.1
        })

    def _create_explosion(self, grid_pos, radius, color):
        sx, sy = self._iso_to_screen(*grid_pos)
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3 * radius)
            self.particles.append({
                'pos': [sx, sy],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'color': color,
                'type': 'spark',
                'gravity': 0.05
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.reward_this_step = 0

        self._handle_input(action)
        self._update_waves()
        self._update_enemies()
        self._update_blocks()
        self._update_projectiles()
        self._update_collisions()
        self._update_particles()
        
        if self.base_damage_flash > 0:
            self.base_damage_flash -= 1
        
        terminated = self._check_termination()
        reward = self._calculate_reward()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.victory:
            self.game_over = True
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _calculate_reward(self):
        reward = self.reward_this_step
        if self.game_over:
            if self.victory:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100
        return reward

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE_Y):
            for x in range(self.GRID_SIZE_X):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF),
                    (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF),
                    (sx - self.TILE_WIDTH_HALF, sy)
                ]
                color = self.COLOR_PATH if (x,y) in self.path else self.COLOR_GRID
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Draw base
        base_pos = self.path[-1]
        color = (255, 0, 0) if self.base_damage_flash > 0 else self.COLOR_BASE
        self._draw_iso_cube(base_pos[0], base_pos[1], color, self.COLOR_BASE_STROKE)
        
        # Draw blocks
        for block in self.blocks:
            color = getattr(self, f"COLOR_{block['type']}")
            self._draw_iso_cube(block['pos'][0], block['pos'][1], color)
            if block['type'] == 'WALL' and block['health'] < self.WALL_HEALTH:
                 # In a real game, would show damage.
                 pass

        # Draw projectiles
        for proj in self.projectiles:
            sx, sy = self._iso_to_screen(*proj['pos'])
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, 3, self.COLOR_PROJECTILE)

        # Draw enemies
        for enemy in self.enemies:
            if enemy['spawn_cooldown'] > 0: continue
            
            # Interpolate screen position for smoothness
            path_node_pos = self._iso_to_screen(*self.path[enemy['path_idx']])
            lerp_pos = self._iso_to_screen(*enemy['pos'])
            
            offset_x, offset_y = self._iso_to_screen(enemy['offset'][0], enemy['offset'][1])
            offset_x -= self.ISO_OFFSET_X
            offset_y -= self.ISO_OFFSET_Y
            
            sx, sy = lerp_pos[0] + offset_x, lerp_pos[1] + offset_y
            enemy['screen_pos'] = (sx, sy)

            self._draw_iso_cube(enemy['pos'][0], enemy['pos'][1], enemy['color'], offset=enemy['offset'])
            
            # Health bar
            bar_w = 20
            bar_h = 4
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, (50,0,0), (sx - bar_w/2, sy - 20, bar_w, bar_h))
            pygame.draw.rect(self.screen, (255,0,0), (sx - bar_w/2, sy - 20, bar_w * health_pct, bar_h))

        # Draw cursor
        cx, cy = self.cursor_pos
        sx, sy = self._iso_to_screen(cx, cy)
        points = [
            (sx, sy - self.TILE_HEIGHT_HALF),
            (sx + self.TILE_WIDTH_HALF, sy),
            (sx, sy + self.TILE_HEIGHT_HALF),
            (sx - self.TILE_WIDTH_HALF, sy)
        ]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points, 2)
        
        # Draw particles
        for p in self.particles:
            if p['type'] == 'spark':
                size = max(0, p['life'] / 15)
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))
            elif p['type'] == 'text':
                alpha = max(0, min(255, p['life'] * 20))
                text_surf = self.font_s.render(p['text'], True, p['color'])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, (int(p['pos'][0]), int(p['pos'][1])))

    def _draw_iso_cube(self, x, y, color, stroke=None, height=1, offset=(0,0)):
        sx, sy = self._iso_to_screen(x + offset[0], y + offset[1])
        
        top_points = [
            (sx, sy - self.TILE_HEIGHT_HALF * height),
            (sx + self.TILE_WIDTH_HALF, sy - self.TILE_HEIGHT_HALF * (height-1)),
            (sx, sy + self.TILE_HEIGHT_HALF),
            (sx - self.TILE_WIDTH_HALF, sy - self.TILE_HEIGHT_HALF * (height-1))
        ]
        
        side_color = tuple(max(0, c-40) for c in color[:3])
        side_color_dark = tuple(max(0, c-80) for c in color[:3])

        # Right face
        pygame.gfxdraw.filled_polygon(self.screen, [top_points[1], (top_points[1][0], top_points[1][1] + self.TILE_HEIGHT_HALF*2), (top_points[2][0], top_points[2][1] + self.TILE_HEIGHT_HALF*2), top_points[2]], side_color)
        # Left face
        pygame.gfxdraw.filled_polygon(self.screen, [top_points[3], (top_points[3][0], top_points[3][1] + self.TILE_HEIGHT_HALF*2), (top_points[2][0], top_points[2][1] + self.TILE_HEIGHT_HALF*2), top_points[2]], side_color_dark)
        
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        if stroke:
            pygame.gfxdraw.aapolygon(self.screen, top_points, stroke)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0]+1, pos[1]+1))
            self.screen.blit(content, pos)
            
        # Health
        draw_text(f"BASE HEALTH: {self.base_health}", self.font_m, self.COLOR_BASE, (10, 10))
        # Resources
        draw_text(f"RESOURCES: {self.resources}", self.font_m, (255, 223, 0), (10, 35))
        # Wave
        wave_text = f"WAVE {self.wave}/{self.MAX_WAVES}" if self.wave > 0 else "WAVE 0"
        draw_text(wave_text, self.font_m, self.COLOR_TEXT, (250, 10))
        
        if len(self.enemies) == 0 and self.wave > 0 and not self.game_over:
            countdown = self.wave_cooldown / self.TARGET_FPS
            draw_text(f"NEXT WAVE IN: {countdown:.1f}s", self.font_m, self.COLOR_TEXT, (250, 35))

        # Block Selector
        draw_text("BUILD:", self.font_m, self.COLOR_TEXT, (450, 10))
        for i, block_type in enumerate(self.BLOCK_TYPES):
            x_pos = 460 + i * 50
            is_selected = i == self.selected_block_idx
            
            rect = pygame.Rect(x_pos, 32, 30, 20)
            color = getattr(self, f"COLOR_{block_type}")
            pygame.draw.rect(self.screen, color, rect)
            
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)
            
            cost_text = self.font_s.render(f"${self.BLOCK_COSTS[block_type]}", True, self.COLOR_TEXT)
            self.screen.blit(cost_text, (x_pos + 5, 35))

        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            draw_text(msg, self.font_l, color, (self.SCREEN_WIDTH/2 - self.font_l.size(msg)[0]/2, 150))
            score_msg = f"Final Score: {self.score}"
            draw_text(score_msg, self.font_m, self.COLOR_TEXT, (self.SCREEN_WIDTH/2 - self.font_m.size(score_msg)[0]/2, 210))


    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# This block allows direct execution of the file for testing
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Fortress Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.TARGET_FPS)
        
    pygame.quit()