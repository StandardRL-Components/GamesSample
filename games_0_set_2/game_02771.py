import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press space to attack in your last moved direction. "
        "Reach the boss room at the far end of the dungeon and defeat the boss to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric dungeon crawler. Explore a procedurally generated dungeon, fight monsters, "
        "collect gold, and defeat the final boss."
    )

    # Should frames auto-advance or wait for user input?
    # Game is turn-based, so it only advances on action.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.MAP_SIZE = (30, 30)
        self.TILE_WIDTH_HALF = 18
        self.TILE_HEIGHT_HALF = 9

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_FLOOR = (50, 60, 70)
        self.COLOR_FLOOR_BORDER = (70, 80, 90)
        self.COLOR_WALL_TOP = (80, 90, 100)
        self.COLOR_WALL_SIDE = (60, 70, 80)
        self.COLOR_PLAYER = (50, 200, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 255)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_ENEMY_OUTLINE = (255, 150, 150)
        self.COLOR_BOSS = (150, 0, 200)
        self.COLOR_BOSS_OUTLINE = (255, 100, 255)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (0, 200, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_PORTAL = (100, 0, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 18)
            self.font_large = pygame.font.Font(None, 28)

        # Etc...        
        self.player = {}
        self.enemies = []
        self.boss = {}
        self.gold_items = []
        self.particles = []
        self.map_grid = np.zeros(self.MAP_SIZE)
        self.boss_room_pos = (0, 0)
        self.player_start_pos = (0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_respawn_timer = 0
        self.difficulty_tier = 0
        self.np_random = None

        # Initialize state variables
        # This is called in __init__ to set up the initial state for validation
        self.reset(seed=random.randint(0, 1_000_000))

        self.validate_implementation()
    
    def _world_to_screen(self, x, y):
        """Converts grid coordinates to screen coordinates."""
        screen_x = self.WIDTH // 2 + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.HEIGHT // 4 + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _generate_map(self):
        """Generates a dungeon map using a random walk."""
        self.map_grid = np.zeros(self.MAP_SIZE, dtype=int)
        max_tiles = int(self.MAP_SIZE[0] * self.MAP_SIZE[1] * 0.4) # 40% floor
        
        px, py = self.np_random.integers(1, self.MAP_SIZE[0]-1), self.np_random.integers(1, self.MAP_SIZE[1]-1)
        self.map_grid[px, py] = 1
        num_tiles = 1

        while num_tiles < max_tiles:
            dx, dy = self.np_random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            px, py = px + dx, py + dy
            
            if 0 < px < self.MAP_SIZE[0] -1 and 0 < py < self.MAP_SIZE[1] -1:
                if self.map_grid[px, py] == 0:
                    self.map_grid[px, py] = 1
                    num_tiles += 1
            else:
                floor_tiles_arr = np.argwhere(self.map_grid == 1)
                if len(floor_tiles_arr) > 0:
                    px, py = floor_tiles_arr[self.np_random.integers(len(floor_tiles_arr))]

        floor_tiles = list(zip(*np.where(self.map_grid == 1)))
        shuffled_floor_tiles = list(floor_tiles)
        self.np_random.shuffle(shuffled_floor_tiles)

        self.player_start_pos = shuffled_floor_tiles.pop(0)

        queue = [(self.player_start_pos, 0)]
        visited = {self.player_start_pos}
        farthest_point = self.player_start_pos
        max_dist = 0
        head = 0
        while head < len(queue):
            (cx, cy), dist = queue[head]; head += 1
            if dist > max_dist:
                max_dist = dist; farthest_point = (cx, cy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.MAP_SIZE[0] and 0 <= ny < self.MAP_SIZE[1] and
                        self.map_grid[nx, ny] == 1 and (nx, ny) not in visited):
                    visited.add((nx, ny)); queue.append(((nx, ny), dist + 1))
        
        self.boss_room_pos = farthest_point

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_map()
        self.MAP_SIZE = (30, 30)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_respawn_timer = 50
        self.difficulty_tier = 0

        self.player = {
            'pos': list(self.player_start_pos), 'health': 100, 'max_health': 100,
            'attack_cooldown': 0, 'anim_offset': 0.0, 'anim_dir': 1, 'facing': (0, 1)
        }

        self.enemies = []; self.boss = {}; self.gold_items = []; self.particles = []
        self._spawn_entities(initial_spawn=True)
        
        return self._get_observation(), self._get_info()
    
    def _get_base_enemy_stats(self):
        scaling_factor = 1 + (self.difficulty_tier * 0.05)
        return int(20 * scaling_factor), int(5 * scaling_factor)
        
    def _spawn_entities(self, initial_spawn=False):
        floor_tiles = list(zip(*np.where(self.map_grid == 1)))
        occupied = {tuple(self.player['pos'])}
        if self.boss: occupied.add(tuple(self.boss['pos']))
        for e in self.enemies: occupied.add(tuple(e['pos']))
        for g in self.gold_items: occupied.add(tuple(g['pos']))

        valid_spawns = [t for t in floor_tiles if t not in occupied and self._manhattan_distance(t, self.player['pos']) > 3]
        shuffled_valid_spawns = list(valid_spawns)
        self.np_random.shuffle(shuffled_valid_spawns)

        num_to_spawn = 3 if initial_spawn else 1
        health, damage = self._get_base_enemy_stats()
        for _ in range(num_to_spawn):
            if not shuffled_valid_spawns: break
            pos = shuffled_valid_spawns.pop()
            self.enemies.append({
                'pos': list(pos), 'health': health, 'max_health': health, 'damage': damage,
                'attack_cooldown': 0, 'anim_offset': self.np_random.random() * 5, 'anim_dir': 1,
            })
        
        for _ in range(num_to_spawn):
            if not shuffled_valid_spawns: break
            pos = shuffled_valid_spawns.pop()
            self.gold_items.append({'pos': list(pos), 'anim_offset': self.np_random.random() * math.pi * 2})

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.1
        
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        prev_pos = tuple(self.player['pos'])
        
        if movement != 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            self.player['facing'] = (dx, dy)
            next_pos = (self.player['pos'][0] + dx, self.player['pos'][1] + dy)
            if self._is_valid_move(next_pos): self.player['pos'] = list(next_pos)

        if space_pressed and self.player['attack_cooldown'] == 0:
            self.player['attack_cooldown'] = 10 # sfx: player_attack_swing
            attack_pos = (self.player['pos'][0] + self.player['facing'][0], self.player['pos'][1] + self.player['facing'][1])
            for enemy in self.enemies:
                if tuple(enemy['pos']) == attack_pos:
                    enemy['health'] -= 25; self._create_particles(self._world_to_screen(*attack_pos), self.COLOR_ENEMY, 10); break # sfx: enemy_hit
            if self.boss and tuple(self.boss['pos']) == attack_pos:
                self.boss['health'] -= 25; self._create_particles(self._world_to_screen(*attack_pos), self.COLOR_BOSS, 20, 3) # sfx: boss_hit

        if self.player['attack_cooldown'] > 0: self.player['attack_cooldown'] -= 1

        initial_gold_count = len(self.gold_items)
        self.gold_items = [g for g in self.gold_items if tuple(g['pos']) != tuple(self.player['pos'])]
        collected_count = initial_gold_count - len(self.gold_items)
        if collected_count > 0: self.score += collected_count; reward += 1 * collected_count # sfx: gold_pickup
            
        if tuple(self.player['pos']) == self.boss_room_pos and not self.boss:
            self._enter_boss_room(); reward += 10
        
        goal_pos = tuple(self.boss['pos']) if self.boss else self.boss_room_pos
        if self._manhattan_distance(self.player['pos'], goal_pos) < self._manhattan_distance(prev_pos, goal_pos): reward += 1
        
        dead_enemies = []
        for enemy in self.enemies:
            if enemy['health'] <= 0: dead_enemies.append(enemy); reward += 5; continue # sfx: enemy_death
            if enemy['attack_cooldown'] > 0: enemy['attack_cooldown'] -= 1
            elif self._manhattan_distance(enemy['pos'], self.player['pos']) == 1:
                self.player['health'] -= enemy['damage']; enemy['attack_cooldown'] = 5; self._create_particles(self._world_to_screen(*self.player['pos']), self.COLOR_PLAYER, 15) # sfx: player_hit
            else:
                ex, ey = enemy['pos']; px, py = self.player['pos']; dx, dy = np.sign(px - ex), np.sign(py - ey)
                if self.np_random.random() > 0.5:
                    if dx != 0 and self._is_valid_move((ex + dx, ey), True): enemy['pos'][0] += dx
                    elif dy != 0 and self._is_valid_move((ex, ey + dy), True): enemy['pos'][1] += dy
                else:
                    if dy != 0 and self._is_valid_move((ex, ey + dy), True): enemy['pos'][1] += dy
                    elif dx != 0 and self._is_valid_move((ex + dx, ey), True): enemy['pos'][0] += dx
        self.enemies = [e for e in self.enemies if e not in dead_enemies]
            
        if self.boss:
            if self.boss['health'] <= 0: self.game_over = True; reward += 100
            elif self.boss['attack_cooldown'] > 0: self.boss['attack_cooldown'] -= 1
            elif self._manhattan_distance(self.boss['pos'], self.player['pos']) <= self.boss['attack_range']:
                self.player['health'] -= self.boss['damage']; self.boss['attack_cooldown'] = 8; self._create_particles(self._world_to_screen(*self.player['pos']), self.COLOR_PLAYER, 25, 3) # sfx: player_hit_heavy
        
        self.steps += 1
        if self.steps > 0 and self.steps % 200 == 0: self.difficulty_tier += 1
        self.enemy_respawn_timer -= 1
        if self.enemy_respawn_timer <= 0 and not self.boss: self._spawn_entities(); self.enemy_respawn_timer = 50

        self._update_animations()

        terminated = False
        truncated = False
        if self.player['health'] <= 0: self.player['health'] = 0; self.game_over = terminated = True; reward += -100
        elif self.boss and self.boss['health'] <= 0: self.game_over = terminated = True
        elif self.steps >= self.MAX_STEPS: self.game_over = truncated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _is_valid_move(self, pos, exclude_player=False):
        if not (0 <= pos[0] < self.MAP_SIZE[0] and 0 <= pos[1] < self.MAP_SIZE[1]): return False
        if self.map_grid[pos[0], pos[1]] == 0: return False
        if not exclude_player and tuple(self.player['pos']) == pos: return False
        if any(tuple(e['pos']) == pos for e in self.enemies): return False
        if self.boss and tuple(self.boss['pos']) == pos: return False
        return True

    def _enter_boss_room(self): # sfx: portal_enter
        self.map_grid = np.ones((15, 15), dtype=int)
        self.map_grid[0, :] = self.map_grid[-1, :] = self.map_grid[:, 0] = self.map_grid[:, -1] = 0
        self.MAP_SIZE = (15, 15); self.player['pos'] = [2, 7]; self.enemies = []; self.gold_items = []
        boss_health = 200 + 50 * self.difficulty_tier; boss_damage = 20 + 5 * self.difficulty_tier
        self.boss = {
            'pos': [12, 7], 'health': boss_health, 'max_health': boss_health, 'damage': boss_damage,
            'attack_cooldown': 0, 'attack_range': 10, 'anim_offset': 0.0, 'anim_dir': 1,
        }

    def _update_animations(self):
        self.player['anim_offset'] += 0.2 * self.player['anim_dir']
        if abs(self.player['anim_offset']) > 2: self.player['anim_dir'] *= -1
        for e in self.enemies: e['anim_offset'] += 0.15 * e['anim_dir']; e['anim_dir'] *= -1 if abs(e['anim_offset']) > 1.5 else 1
        if self.boss: self.boss['anim_offset'] += 0.1 * self.boss['anim_dir']; self.boss['anim_dir'] *= -1 if abs(self.boss['anim_offset']) > 3 else 1
        for g in self.gold_items: g['anim_offset'] = (g['anim_offset'] + 0.1) % (2 * math.pi)
        self.particles = [p for p in self.particles if p['lifespan'] > 1]
        for p in self.particles: p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]; p['vel'][1] += 0.1; p['lifespan'] -= 1
        
    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi; speed = (self.np_random.random() * 2 + 1) * speed_mult
            self.particles.append({'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed], 'lifespan': self.np_random.integers(15, 30), 'color': color})
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_offset_x = self.WIDTH // 2 - (self.player['pos'][0] - self.player['pos'][1]) * self.TILE_WIDTH_HALF
        cam_offset_y = self.HEIGHT // 2 - (self.player['pos'][0] + self.player['pos'][1]) * self.TILE_HEIGHT_HALF
        
        entities_to_render = []
        for y in range(self.MAP_SIZE[1]):
            for x in range(self.MAP_SIZE[0]):
                sx, sy = cam_offset_x + (x - y) * self.TILE_WIDTH_HALF, cam_offset_y + (x + y) * self.TILE_HEIGHT_HALF
                points = [(sx, sy), (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF), (sx, sy + self.TILE_HEIGHT_HALF * 2), (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF)]
                if self.map_grid[x, y] == 1:
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FLOOR); pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_FLOOR_BORDER)
                    if (x, y) == self.boss_room_pos and not self.boss:
                        c = (int(self.COLOR_PORTAL[0]*(0.75+0.25*math.sin(self.steps*0.1))), int(self.COLOR_PORTAL[1]*(0.75+0.25*math.sin(self.steps*0.1+2))), int(self.COLOR_PORTAL[2]*(0.75+0.25*math.sin(self.steps*0.1+4))))
                        pygame.gfxdraw.filled_circle(self.screen, sx, sy + self.TILE_HEIGHT_HALF, 8, c); pygame.gfxdraw.aacircle(self.screen, sx, sy + self.TILE_HEIGHT_HALF, 8, (255,255,255))
                else:
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_WALL_TOP)
                    p_l = [(p[0], p[1] + 20) for p in points[2:]] + points[2:]; pygame.gfxdraw.filled_polygon(self.screen, [points[3], points[2], p_l[0], p_l[1]], self.COLOR_WALL_SIDE)
                    p_r = [(p[0], p[1] + 20) for p in points[1:3]] + points[1:3]; pygame.gfxdraw.filled_polygon(self.screen, [points[2], points[1], p_r[1], p_r[0]], self.COLOR_WALL_SIDE)

                pos = (x, y)
                if tuple(self.player['pos']) == pos: entities_to_render.append(('player', self.player, sx, sy))
                if self.boss and tuple(self.boss['pos']) == pos: entities_to_render.append(('boss', self.boss, sx, sy))
                for e in self.enemies:
                    if tuple(e['pos']) == pos: entities_to_render.append(('enemy', e, sx, sy))
                for g in self.gold_items:
                    if tuple(g['pos']) == pos: entities_to_render.append(('gold', g, sx, sy))
        
        for type, data, sx, sy in entities_to_render:
            if type == 'player': self._render_character(sx, sy - 10, data, self.COLOR_PLAYER, self.COLOR_PLAYER_OUTLINE, 10, 15)
            elif type == 'enemy': self._render_character(sx, sy - 5, data, self.COLOR_ENEMY, self.COLOR_ENEMY_OUTLINE, 8, 12)
            elif type == 'boss': self._render_character(sx, sy - 20, data, self.COLOR_BOSS, self.COLOR_BOSS_OUTLINE, 20, 30)
            elif type == 'gold':
                anim = (1 + math.sin(data['anim_offset']))/2; w = int(8*anim+2); c = (min(255,self.COLOR_GOLD[0]+30*(1-anim)), min(255,self.COLOR_GOLD[1]+30*(1-anim)), self.COLOR_GOLD[2])
                pygame.draw.ellipse(self.screen, c, (sx - w // 2, sy + 5, w, 5))

        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0)); size = int(max(1, 3 * (p['lifespan'] / 30.0)))
            if alpha > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
                self.screen.blit(s, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

        for type, data, sx, sy in entities_to_render:
            if type in ['player', 'enemy', 'boss']:
                self._render_health_bar(sx, sy - (35 if type == 'boss' else 25), data['health'], data['max_health'])

    def _render_character(self, sx, sy, data, color, outline, width, height):
        y_pos = sy + int(data['anim_offset']); points = [(sx, y_pos - height//2), (sx + width//2, y_pos), (sx, y_pos + height//2), (sx - width//2, y_pos)]
        pygame.gfxdraw.filled_polygon(self.screen, points, color); pygame.gfxdraw.aapolygon(self.screen, points, outline)

    def _render_health_bar(self, sx, sy, health, max_health):
        w = 30; ratio = max(0, health / max_health); bg = pygame.Rect(sx-w//2, sy, w, 5); fg = pygame.Rect(sx-w//2, sy, int(w*ratio), 5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg); pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, fg); pygame.draw.rect(self.screen, (255,255,255), bg, 1)

    def _render_ui(self):
        self.screen.blit(self.font_large.render(f"HP: {self.player['health']}/{self.player['max_health']}", True, self.COLOR_TEXT), (10, 10))
        self.screen.blit(self.font_large.render(f"GOLD: {self.score}", True, self.COLOR_GOLD), (10, 40))
        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
            msg, color = ("VICTORY", (100, 255, 100)) if self.boss and self.boss['health'] <= 0 else ("GAME OVER", (255, 100, 100))
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(reward, (int, float)) and isinstance(term, bool) and isinstance(trunc, bool) and isinstance(info, dict)
        self.reset()
        assert self.player['health'] <= self.player['max_health']
        assert self.score >= 0
        original_pos = list(self.player['pos'])
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            check_pos = (original_pos[0] + dx, original_pos[1] + dy)
            if not (0 <= check_pos[0] < self.MAP_SIZE[0] and 0 <= check_pos[1] < self.MAP_SIZE[1]) or self.map_grid[check_pos[0], check_pos[1]] == 0:
                move_action = {(0,-1): 1, (0,1): 2, (-1,0): 3, (1,0): 4}[(dx, dy)]
                self.step([move_action, 0, 0]); assert self.player['pos'] == original_pos
                # print("✓ Implementation validated successfully") # Commented out print
                return
        raise AssertionError("Validation failed to find a wall to test collision")

if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or other backend
    env = GameEnv()
    obs, info = env.reset()
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Dungeon Crawler")
    terminated = False
    truncated = False
    clock = pygame.time.Clock()
    
    while not terminated and not truncated:
        movement_action, space_action = 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: terminated = True
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        if keys[pygame.K_SPACE]: space_action = 1
        
        obs, reward, terminated, truncated, info = env.step([movement_action, space_action, 0])
        
        # Blit the env's screen (which is a pygame.Surface) to the display screen
        display_screen.blit(pygame.transform.scale(env.screen, display_screen.get_rect().size), (0, 0))
        pygame.display.flip()
        clock.tick(15)
        
    env.close()
    pygame.quit()