
# Generated: 2025-08-27T12:26:15.003814
# Source Brief: brief_00042.md
# Brief Index: 42

        
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
        "Controls: Arrow keys to move cursor, Shift to cycle tower type, Space to place tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 35, 50)
    COLOR_PATH = (50, 60, 80)
    COLOR_GRID = (40, 50, 70)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_DMG = (255, 80, 80)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_GOLD = (255, 200, 0)
    
    # Tower Stats: [cost, damage, range, cooldown, color, name]
    TOWER_STATS = [
        [100, 10, 80, 45, (0, 255, 100), "Cannon"],
        [150, 25, 60, 75, (255, 150, 0), "Heavy Cannon"],
        [200, 5, 120, 20, (200, 100, 255), "Gatling"],
        [250, 0, 100, 90, (100, 200, 255), "Frost Tower"], # Damage 0, but slows
        [350, 150, 90, 180, (255, 50, 50), "Rocket"],
    ]

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_W, GRID_H = 20, 12
    TILE_W, TILE_H = 32, 16
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 80

    # Game rules
    MAX_STEPS = 15000 # ~8 minutes at 30fps
    TOTAL_WAVES = 10
    INITIAL_BASE_HEALTH = 100
    INITIAL_RESOURCES = 300

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
        
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.path = []
        self.buildable_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=bool)
        self._generate_map()

        self.reset()
        self.validate_implementation()
    
    def _generate_map(self):
        # Create a winding path
        self.path = []
        waypoints = [(0, 5), (4, 5), (4, 2), (9, 2), (9, 8), (14, 8), (14, 4), (19, 4)]
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            dx = np.sign(p2[0] - p1[0])
            dy = np.sign(p2[1] - p1[1])
            if dx != 0:
                for x in range(p1[0], p2[0] + dx, dx):
                    self.path.append((x, p1[1]))
            if dy != 0:
                for y in range(p1[1], p2[1] + dy, dy):
                    self.path.append((p2[0], y))
        
        # Define buildable area
        self.buildable_grid.fill(True)
        for x, y in self.path:
            self.buildable_grid[x, y] = False
        self.buildable_grid[self.path[-1][0], self.path[-1][1]] = False # Base location

    def _grid_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * (self.TILE_W / 2)
        iso_y = self.ORIGIN_Y + (x + y) * (self.TILE_H / 2)
        return int(iso_x), int(iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.victory = False

        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        self.wave_timer = 150 # Time until first wave
        self.wave_spawning = False
        self.spawn_queue = []

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.placement_cursor = (self.GRID_W // 2, self.GRID_H // 2)
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_shake = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        self._handle_input(action)
        self._update_waves()
        self._update_enemies()
        self._update_towers()
        self._update_projectiles()
        self._update_particles()

        if self.screen_shake > 0:
            self.screen_shake -= 1

        self.steps += 1
        terminated = self._check_termination()

        if not self.game_over:
            self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement: Move placement cursor ---
        if movement != 0:
            px, py = self.placement_cursor
            if movement == 1: py -= 1 # Up
            elif movement == 2: py += 1 # Down
            elif movement == 3: px -= 1 # Left
            elif movement == 4: px += 1 # Right
            self.placement_cursor = (max(0, min(self.GRID_W - 1, px)), max(0, min(self.GRID_H - 1, py)))

        # --- Shift: Cycle tower type ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_STATS)
            # sfx: UI_CYCLE

        # --- Space: Place tower ---
        if space_held and not self.prev_space_held:
            self._place_tower()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        px, py = self.placement_cursor
        cost = self.TOWER_STATS[self.selected_tower_type][0]
        
        is_buildable = self.buildable_grid[px, py]
        is_occupied = any(t['grid_pos'] == (px, py) for t in self.towers)

        if self.resources >= cost and is_buildable and not is_occupied:
            self.resources -= cost
            self.towers.append({
                'grid_pos': (px, py),
                'type': self.selected_tower_type,
                'cooldown': 0,
                'target': None
            })
            # sfx: TOWER_PLACE
            self._create_particles(self._grid_to_iso(px, py), 15, self.TOWER_STATS[self.selected_tower_type][4])
        else:
            # sfx: ACTION_FAIL
            pass
    
    def _update_waves(self):
        if self.wave_spawning:
            if self.wave_timer > 0:
                self.wave_timer -= 1
            elif self.spawn_queue:
                self.enemies.append(self.spawn_queue.pop(0))
                self.wave_timer = 20 # Delay between enemies
                # sfx: ENEMY_SPAWN
            else:
                self.wave_spawning = False
        elif not self.enemies and not self.game_over and self.wave_number < self.TOTAL_WAVES:
            if self.wave_timer > 0:
                self.wave_timer -= 1
            else:
                self.wave_number += 1
                self.wave_spawning = True
                self.wave_timer = 0
                self._generate_spawn_queue()
                self.resources += 100 + self.wave_number * 10 # Wave clear bonus
                # sfx: WAVE_START

    def _generate_spawn_queue(self):
        num_enemies = 3 + self.wave_number * 2
        base_health = 50 + self.wave_number * 20
        base_speed = 0.5 + self.wave_number * 0.05
        
        for i in range(num_enemies):
            health = int(base_health * (1 + self.np_random.uniform(-0.1, 0.1)))
            speed = base_speed * (1 + self.np_random.uniform(-0.1, 0.1))
            self.spawn_queue.append({
                'pos': self._grid_to_iso(*self.path[0]),
                'path_index': 0,
                'health': health,
                'max_health': health,
                'speed': max(0.2, speed),
                'slow_timer': 0,
                'id': self.np_random.integers(1, 1_000_000)
            })

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy['path_index'] >= len(self.path) - 1:
                self.base_health -= 10
                self.reward_this_step -= 10
                self.enemies.remove(enemy)
                self.screen_shake = 10
                # sfx: BASE_DAMAGE
                continue

            if enemy['slow_timer'] > 0:
                enemy['slow_timer'] -= 1
                current_speed = enemy['speed'] * 0.5
            else:
                current_speed = enemy['speed']

            target_node = self.path[enemy['path_index'] + 1]
            target_pos = self._grid_to_iso(*target_node)
            
            direction = (target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1])
            dist = math.hypot(*direction)
            
            if dist < current_speed:
                enemy['path_index'] += 1
                enemy['pos'] = target_pos
            else:
                enemy['pos'] = (enemy['pos'][0] + direction[0] / dist * current_speed, 
                                enemy['pos'][1] + direction[1] / dist * current_speed)

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            stats = self.TOWER_STATS[tower['type']]
            tower_range = stats[2]
            
            # Find a target
            possible_targets = []
            tower_pos = self._grid_to_iso(*tower['grid_pos'])
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower_pos[0], enemy['pos'][1] - tower_pos[1])
                if dist <= tower_range:
                    possible_targets.append(enemy)
            
            if possible_targets:
                # Target first enemy on path
                target = min(possible_targets, key=lambda e: -e['path_index'])
                tower['target'] = target['id']
                
                # Fire
                tower['cooldown'] = stats[3]
                self.projectiles.append({
                    'start_pos': tower_pos,
                    'end_pos': target['pos'],
                    'target_id': target['id'],
                    'type': tower['type'],
                    'pos': tower_pos,
                    'progress': 0
                })
                # sfx: TOWER_FIRE

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            target = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target:
                self.projectiles.remove(proj)
                continue
            
            proj['end_pos'] = target['pos']
            
            start_x, start_y = proj['start_pos']
            end_x, end_y = proj['end_pos']
            
            proj['progress'] += 0.1 # Speed of projectile
            
            if proj['progress'] >= 1.0:
                # Hit!
                stats = self.TOWER_STATS[proj['type']]
                damage = stats[1]
                target['health'] -= damage
                self.reward_this_step += 0.1 # Damage reward
                
                # Frost tower effect
                if proj['type'] == 3:
                    target['slow_timer'] = 120 # 4 seconds

                self._create_particles(target['pos'], 5, (255, 255, 255))
                # sfx: ENEMY_HIT
                
                if target['health'] <= 0:
                    self.resources += 10
                    self.reward_this_step += 1.0
                    self._create_particles(target['pos'], 20, (255, 80, 80), 2)
                    self.enemies.remove(target)
                    # sfx: ENEMY_DEATH
                
                self.projectiles.remove(proj)
            else:
                # Interpolate position
                proj['pos'] = (
                    start_x + (end_x - start_x) * proj['progress'],
                    start_y + (end_y - start_y) * proj['progress']
                )

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['vel'] = (p['vel'][0] * 0.95, p['vel'][1] * 0.95) # Damping
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, speed_mult=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pos,
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.base_health <= 0:
            self.game_over = True
            self.reward_this_step -= 100
            return True
            
        if self.wave_number >= self.TOTAL_WAVES and not self.enemies and not self.wave_spawning:
            self.game_over = True
            self.victory = True
            self.reward_this_step += 100
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
        # Apply screen shake
        render_offset = (0,0)
        if self.screen_shake > 0:
            render_offset = (self.np_random.integers(-5, 6), self.np_random.integers(-5, 6))

        # Draw grid
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                iso_x, iso_y = self._grid_to_iso(x, y)
                iso_x += render_offset[0]
                iso_y += render_offset[1]
                
                # Draw grid lines
                points = [
                    self._grid_to_iso(x, y), self._grid_to_iso(x + 1, y),
                    self._grid_to_iso(x + 1, y + 1), self._grid_to_iso(x, y + 1)
                ]
                points = [(p[0] + render_offset[0], p[1] + render_offset[1]) for p in points]
                pygame.draw.lines(self.screen, self.COLOR_GRID, True, points, 1)

        # Draw path
        for i in range(len(self.path) - 1):
            p1 = self._grid_to_iso(*self.path[i])
            p2 = self._grid_to_iso(*self.path[i+1])
            p1 = (p1[0] + self.TILE_W/4, p1[1] + self.TILE_H/4)
            p2 = (p2[0] + self.TILE_W/4, p2[1] + self.TILE_H/4)
            p1 = (p1[0] + render_offset[0], p1[1] + render_offset[1])
            p2 = (p2[0] + render_offset[0], p2[1] + render_offset[1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.TILE_H + 4)

        # Draw base
        base_pos = self._grid_to_iso(*self.path[-1])
        base_pos = (base_pos[0] + render_offset[0], base_pos[1] + render_offset[1])
        base_color = self.COLOR_BASE if self.screen_shake == 0 else self.COLOR_BASE_DMG
        self._draw_iso_cube(base_pos[0], base_pos[1], self.TILE_W-4, self.TILE_H-2, 20, base_color)

        # Draw towers
        for tower in self.towers:
            pos = self._grid_to_iso(*tower['grid_pos'])
            pos = (pos[0] + render_offset[0], pos[1] + render_offset[1])
            stats = self.TOWER_STATS[tower['type']]
            self._draw_iso_cube(pos[0], pos[1], self.TILE_W/2, self.TILE_H/2, 10, stats[4])

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0] + render_offset[0]), int(enemy['pos'][1] + render_offset[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, (200, 50, 50))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, (255, 100, 100))
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 12
            pygame.draw.rect(self.screen, (50,0,0), (pos[0] - bar_w/2, pos[1] - 12, bar_w, 3))
            pygame.draw.rect(self.screen, (0,200,0), (pos[0] - bar_w/2, pos[1] - 12, bar_w * health_pct, 3))
            
        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0] + render_offset[0]), int(proj['pos'][1] + render_offset[1]))
            color = self.TOWER_STATS[proj['type']][4]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, (255,255,255))

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0] + render_offset[0]), int(p['pos'][1] + render_offset[1]))
            size = int(p['life'] / 5)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)

        # Draw placement cursor
        if not self.game_over:
            px, py = self.placement_cursor
            pos = self._grid_to_iso(px, py)
            pos = (pos[0] + render_offset[0], pos[1] + render_offset[1])
            
            is_buildable = self.buildable_grid[px, py]
            is_occupied = any(t['grid_pos'] == (px, py) for t in self.towers)
            
            if not is_buildable or is_occupied:
                cursor_color = (255, 0, 0)
            else:
                cursor_color = (255, 255, 255)

            points = [
                self._grid_to_iso(px, py), self._grid_to_iso(px + 1, py),
                self._grid_to_iso(px + 1, py + 1), self._grid_to_iso(px, py + 1)
            ]
            points = [(p[0] + render_offset[0], p[1] + render_offset[1]) for p in points]
            pygame.draw.polygon(self.screen, cursor_color, points, 2)
            
            # Show tower range
            stats = self.TOWER_STATS[self.selected_tower_type]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], stats[2], (*cursor_color, 100))
    
    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow=True):
        if shadow:
            text_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf, (pos[0] + 1, pos[1] + 1))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Top Bar
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.SCREEN_WIDTH, 40))
        
        # Base Health
        self._draw_text(f"♥ {max(0, self.base_health)}", (10, 10), self.font_m)
        
        # Resources
        self._draw_text(f"♦ {self.resources}", (130, 10), self.font_m, self.COLOR_GOLD)
        
        # Wave Info
        if not self.game_over:
            wave_text = f"Wave {self.wave_number}/{self.TOTAL_WAVES}"
            if not self.wave_spawning and not self.enemies:
                 wave_text += f" (Next in {self.wave_timer // 30 + 1}s)"
            self._draw_text(wave_text, (self.SCREEN_WIDTH - 200, 10), self.font_m)
        
        # Bottom Bar (Selected Tower)
        pygame.draw.rect(self.screen, (0,0,0,150), (0, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH, 50))
        stats = self.TOWER_STATS[self.selected_tower_type]
        cost, dmg, rng, cd, color, name = stats
        
        self._draw_text(f"Selected: {name}", (10, self.SCREEN_HEIGHT - 40), self.font_m, color)
        info_text = f"Cost: {cost} | Dmg: {dmg} | Range: {rng} | CD: {cd/30:.1f}s"
        self._draw_text(info_text, (250, self.SCREEN_HEIGHT - 35), self.font_s)
        
        # Game Over / Victory Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            self._draw_text(msg, (self.SCREEN_WIDTH/2 - self.font_l.size(msg)[0]/2, 150), self.font_l, color)
            score_msg = f"Final Score: {self.score:.0f}"
            self._draw_text(score_msg, (self.SCREEN_WIDTH/2 - self.font_m.size(score_msg)[0]/2, 210), self.font_m)

    def _draw_iso_cube(self, x, y, w, h, z, color):
        darker = tuple(max(0, c-40) for c in color)
        darkest = tuple(max(0, c-80) for c in color)
        
        # Top face
        points = [ (x, y - z), (x + w/2, y - h/2 - z), (x, y - h - z), (x - w/2, y - h/2 - z) ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        
        # Left face
        points = [ (x - w/2, y - h/2 - z), (x - w/2, y - h/2), (x, y), (x, y - z) ]
        pygame.gfxdraw.filled_polygon(self.screen, points, darker)
        pygame.gfxdraw.aapolygon(self.screen, points, darker)

        # Right face
        points = [ (x + w/2, y - h/2 - z), (x + w/2, y - h/2), (x, y), (x, y - z) ]
        pygame.gfxdraw.filled_polygon(self.screen, points, darkest)
        pygame.gfxdraw.aapolygon(self.screen, points, darkest)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "enemies": len(self.enemies),
            "towers": len(self.towers)
        }

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over. Total Reward: {total_reward}")
            print(f"Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()