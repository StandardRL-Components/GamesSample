
# Generated: 2025-08-27T16:32:52.168692
# Source Brief: brief_01256.md
# Brief Index: 1256

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through tower types. Press Space to build a tower at the cursor."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers on a procedurally generated isometric map. Survive all 10 waves to win."
    )

    auto_advance = True

    # --- CONFIGURATION ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 22, 22
    ISO_TILE_W_HALF, ISO_TILE_H_HALF = 16, 8
    MAX_STEPS = 30 * 600  # 10 minutes at 30fps
    FPS = 30
    
    # --- COLORS ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_PATH = (60, 80, 110)
    COLOR_PATH_BORDER = (80, 100, 130)
    COLOR_BASE = (50, 150, 255)
    COLOR_BASE_GLOW = (100, 180, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 120, 120)
    COLOR_CURSOR_VALID = (100, 255, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_UI_BG = (40, 45, 55, 180)

    # --- TOWER STATS ---
    TOWER_TYPES = {
        0: {"name": "Cannon", "cost": 25, "range": 3.5, "damage": 10, "fire_rate": 0.8, "color": (0, 200, 0), "projectile_speed": 8},
        1: {"name": "Machine Gun", "cost": 40, "range": 2.5, "damage": 4, "fire_rate": 0.2, "color": (255, 255, 0), "projectile_speed": 12},
        2: {"name": "Sniper", "cost": 60, "range": 6.0, "damage": 35, "fire_rate": 2.0, "color": (200, 0, 200), "projectile_speed": 20},
    }

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
        self.font_s = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)
        
        self.iso_offset_x = self.SCREEN_WIDTH // 2
        self.iso_offset_y = 100

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.step_reward = 0.0

        self.base_health = 100
        self.resources = 80
        self.current_wave = 0
        self.wave_in_progress = False
        self.inter_wave_timer = 3 * self.FPS
        self.wave_damage_taken = False
        self.enemies_to_spawn = deque()
        self.last_spawn_time = 0

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self._generate_path()
        self.path_set = set(self.path_nodes)
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.step_reward = 0.0

        if not self.game_over:
            self._handle_actions(action)
            self._update_game_state()

        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Cycle Tower Type ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
            # sfx: UI_SELECT

        # --- Place Tower ---
        if space_held and not self.prev_space_held:
            self._try_place_tower()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _try_place_tower(self):
        cx, cy = self.cursor_pos
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        
        is_on_path = (cx, cy) in self.path_set
        is_occupied = any(t['grid_pos'] == [cx, cy] for t in self.towers)
        can_afford = self.resources >= tower_spec['cost']

        if not is_on_path and not is_occupied and can_afford:
            self.resources -= tower_spec['cost']
            new_tower = {
                "grid_pos": [cx, cy],
                "type": self.selected_tower_type,
                "spec": tower_spec,
                "cooldown": 0,
                "target": None
            }
            self.towers.append(new_tower)
            # sfx: TOWER_PLACE
            self._create_particles(self._grid_to_screen(cx, cy), 15, tower_spec['color'], 0.5, 3)

    def _update_game_state(self):
        self._update_wave_logic()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
    def _update_wave_logic(self):
        if self.wave_in_progress:
            if not self.enemies and not self.enemies_to_spawn:
                self.wave_in_progress = False
                self.inter_wave_timer = 5 * self.FPS # 5 sec break
                if not self.wave_damage_taken:
                    self.step_reward += 1.0
                    self.score += 100
                if self.current_wave == 10:
                    self.game_won = True
                # sfx: WAVE_CLEAR
        else:
            if self.current_wave < 10:
                self.inter_wave_timer -= 1
                if self.inter_wave_timer <= 0:
                    self._spawn_wave()

    def _spawn_wave(self):
        self.current_wave += 1
        self.wave_in_progress = True
        self.wave_damage_taken = False
        
        num_enemies = 3 + self.current_wave
        base_health = 20 + self.current_wave * 10
        base_speed = 0.5 + self.current_wave * 0.05
        
        for i in range(num_enemies):
            enemy_health = base_health * (1 + random.uniform(-0.1, 0.1))
            enemy_speed = base_speed * (1 + random.uniform(-0.1, 0.1))
            self.enemies_to_spawn.append({"health": enemy_health, "speed": enemy_speed})
        
        self.last_spawn_time = self.steps

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1 / self.FPS
                continue

            # Find a target
            target = None
            min_dist = tower['spec']['range'] ** 2
            tx, ty = tower['grid_pos']
            
            for enemy in self.enemies:
                ex, ey = enemy['pos']
                dist_sq = (tx - ex/self.ISO_TILE_W_HALF/2)**2 + (ty - ey/self.ISO_TILE_H_HALF/2)**2 # Approximation
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy

            if target:
                tower['cooldown'] = tower['spec']['fire_rate']
                tower_pos = self._grid_to_screen(tx, ty)
                self._create_particles(tower_pos, 3, (255,255,150), 0.2, 2) # Muzzle flash
                self.projectiles.append({
                    "pos": list(tower_pos),
                    "target": target,
                    "spec": tower['spec']
                })
                # sfx: TOWER_FIRE

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = self._grid_to_screen(*proj['target']['path_nodes'][proj['target']['node_idx']])
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['spec']['projectile_speed']:
                proj['target']['health'] -= proj['spec']['damage']
                self._create_particles(target_pos, 8, self.COLOR_ENEMY_GLOW, 0.4, 4)
                # sfx: HIT_IMPACT
                self.projectiles.remove(proj)
            else:
                proj['pos'][0] += (dx / dist) * proj['spec']['projectile_speed']
                proj['pos'][1] += (dy / dist) * proj['spec']['projectile_speed']

    def _update_enemies(self):
        # Spawn new enemies from queue
        if self.enemies_to_spawn and (self.steps - self.last_spawn_time) > 0.5 * self.FPS:
            self.last_spawn_time = self.steps
            enemy_spec = self.enemies_to_spawn.popleft()
            start_pos = self._grid_to_screen(self.path_nodes[0][0], self.path_nodes[0][1])
            self.enemies.append({
                "pos": list(start_pos),
                "health": enemy_spec['health'],
                "max_health": enemy_spec['health'],
                "speed": enemy_spec['speed'],
                "path_nodes": self.path_nodes,
                "node_idx": 1
            })

        # Move and update existing enemies
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.step_reward += 0.1
                self.score += 10
                self.resources += 5
                self._create_particles(enemy['pos'], 20, self.COLOR_ENEMY, 0.8, 5)
                # sfx: ENEMY_DEATH
                self.enemies.remove(enemy)
                continue

            target_node_pos = self._grid_to_screen(*enemy['path_nodes'][enemy['node_idx']])
            dx = target_node_pos[0] - enemy['pos'][0]
            dy = target_node_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['node_idx'] += 1
                if enemy['node_idx'] >= len(enemy['path_nodes']):
                    self.base_health -= 10
                    self.wave_damage_taken = True
                    self.step_reward -= 1.0
                    base_pos = self._grid_to_screen(*self.base_pos)
                    self._create_particles(base_pos, 30, self.COLOR_BASE, 1.0, 6)
                    # sfx: BASE_DAMAGE
                    self.enemies.remove(enemy)
                    continue
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, life_mult, speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * random.uniform(0.5, 1.0), 
                   math.sin(angle) * speed * random.uniform(0.5, 1.0)]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': int(random.uniform(10, 20) * life_mult),
                'color': color,
                'radius': random.uniform(1, 4)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            self.step_reward -= 10.0
            return True
        if self.game_won:
            self.game_over = True
            self.step_reward += 10.0
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._grid_to_screen(x, y)
                points = [
                    self._grid_to_screen(x, y),
                    self._grid_to_screen(x + 1, y),
                    self._grid_to_screen(x + 1, y + 1),
                    self._grid_to_screen(x, y + 1),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Render path
        for i in range(len(self.path_nodes) - 1):
            p1 = self._grid_to_screen(*self.path_nodes[i])
            p2 = self._grid_to_screen(*self.path_nodes[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.ISO_TILE_H_HALF * 3)
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, self.ISO_TILE_H_HALF * 3 + 2)

        # Render base
        base_screen_pos = self._grid_to_screen(*self.base_pos)
        base_rect = pygame.Rect(0, 0, 40, 40)
        base_rect.center = base_screen_pos
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        pygame.gfxdraw.aacircle(self.screen, int(base_screen_pos[0]), int(base_screen_pos[1]), 25, self.COLOR_BASE_GLOW)
        
        # Render towers
        for tower in self.towers:
            pos = self._grid_to_screen(*tower['grid_pos'])
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 8, tower['spec']['color'])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 8, (255,255,255))
            # Range indicator when placing or selected
            if self.cursor_pos == tower['grid_pos']:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(tower['spec']['range'] * self.ISO_TILE_W_HALF * 1.5), (*tower['spec']['color'], 100))

        # Render enemies
        for enemy in self.enemies:
            pos = enemy['pos']
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 6, self.COLOR_ENEMY_GLOW)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 12
            pygame.draw.rect(self.screen, (50,0,0), (pos[0] - bar_w/2, pos[1] - 12, bar_w, 3))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0] - bar_w/2, pos[1] - 12, bar_w * health_pct, 3))

        # Render projectiles
        for proj in self.projectiles:
            color = proj['spec']['color']
            pygame.draw.circle(self.screen, color, proj['pos'], 3)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']), special_flags=pygame.BLEND_RGBA_ADD)

        # Render cursor
        if not self.game_over:
            cursor_screen_pos = self._grid_to_screen(*self.cursor_pos)
            cx, cy = self.cursor_pos
            is_on_path = (cx, cy) in self.path_set
            is_occupied = any(t['grid_pos'] == [cx, cy] for t in self.towers)
            color = self.COLOR_CURSOR_VALID if not is_on_path and not is_occupied else self.COLOR_CURSOR_INVALID
            points = [
                self._grid_to_screen(cx, cy),
                self._grid_to_screen(cx + 1, cy),
                self._grid_to_screen(cx + 1, cy + 1),
                self._grid_to_screen(cx, cy + 1),
            ]
            pygame.draw.lines(self.screen, color, True, points, 2)
            # Range indicator for selected tower
            tower_spec = self.TOWER_TYPES[self.selected_tower_type]
            pygame.gfxdraw.aacircle(self.screen, int(cursor_screen_pos[0]), int(cursor_screen_pos[1]), int(tower_spec['range'] * self.ISO_TILE_W_HALF * 1.5), (*color, 100))


    def _render_ui(self):
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Helper to draw text with shadow
        def draw_text(text, pos, font, color=self.COLOR_TEXT):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(content, pos)

        # Top-left: Resources and Base Health
        draw_text(f"$: {int(self.resources)}", (10, 10), self.font_m, (255, 220, 100))
        draw_text(f"HP: {int(self.base_health)}", (10, 35), self.font_m, self.COLOR_BASE)
        
        # Top-right: Score and Wave
        score_text = f"Score: {self.score}"
        score_w = self.font_m.size(score_text)[0]
        draw_text(score_text, (self.SCREEN_WIDTH - score_w - 10, 10), self.font_m)

        wave_text = f"Wave: {self.current_wave}/10"
        wave_w = self.font_m.size(wave_text)[0]
        draw_text(wave_text, (self.SCREEN_WIDTH - wave_w - 10, 35), self.font_m)

        # Bottom-center: Selected Tower
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        tower_info = f"Build: {tower_spec['name']} (${tower_spec['cost']})"
        info_w, info_h = self.font_m.size(tower_info)
        draw_text(tower_info, (self.SCREEN_WIDTH/2 - info_w/2, self.SCREEN_HEIGHT - info_h - 5), self.font_m)

        # Center message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            msg_w, msg_h = self.font_l.size(msg)
            draw_text(msg, (self.SCREEN_WIDTH/2 - msg_w/2, self.SCREEN_HEIGHT/2 - msg_h/2), self.font_l, color)
        elif not self.wave_in_progress and self.inter_wave_timer > 0:
            msg = f"Wave {self.current_wave + 1} starting in {math.ceil(self.inter_wave_timer/self.FPS)}..."
            msg_w, msg_h = self.font_m.size(msg)
            draw_text(msg, (self.SCREEN_WIDTH/2 - msg_w/2, 70), self.font_m)


    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "resources": self.resources, "base_health": self.base_health}

    def _grid_to_screen(self, x, y):
        screen_x = self.iso_offset_x + (x - y) * self.ISO_TILE_W_HALF
        screen_y = self.iso_offset_y + (x + y) * self.ISO_TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _generate_path(self):
        path = []
        x, y = 0, self.np_random.integers(5, self.GRID_HEIGHT - 5)
        path.append((x, y))
        
        while x < self.GRID_WIDTH - 1:
            possible_moves = []
            # Strong bias to move right
            if (x + 1, y) not in path and 0 <= x + 1 < self.GRID_WIDTH:
                possible_moves.extend([(x + 1, y)] * 5)
            # Move up/down
            if (x, y - 1) not in path and y > 0:
                possible_moves.append((x, y - 1))
            if (x, y + 1) not in path and y < self.GRID_HEIGHT - 1:
                possible_moves.append((x, y + 1))
            
            if not possible_moves: # Stuck, backtrack (should be rare)
                if len(path) > 1: path.pop()
                else: break # Failsafe
                x, y = path[-1]
                continue
            
            x, y = random.choice(possible_moves)
            path.append((x, y))
        
        self.path_nodes = path
        self.base_pos = path[-1]

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Wave: {info['wave']}")

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"--- GAME OVER --- Final Score: {info['score']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(env.FPS)

    pygame.quit()