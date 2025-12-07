
# Generated: 2025-08-28T02:18:20.180233
# Source Brief: brief_01661.md
# Brief Index: 1661

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys for isometric movement. "
        "Space to attack nearby enemies."
    )

    game_description = (
        "Explore a procedurally generated isometric dungeon, battling enemies and "
        "collecting gold to defeat the final boss."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_FLOOR = (40, 40, 60)
    COLOR_WALL = (70, 70, 90)
    COLOR_WALL_TOP = (90, 90, 110)
    COLOR_GOLD = (255, 223, 0)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_SHADOW = (0, 180, 180)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_STAIRS = (180, 100, 255)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STEPS = 1000
    PLAYER_MAX_HEALTH = 100
    PLAYER_ATTACK_DAMAGE = 25
    PLAYER_ATTACK_RANGE = 2.0  # In grid units

    # World Parameters
    TILE_WIDTH = 32
    TILE_HEIGHT = 16
    TILE_DEPTH = 12
    MAP_SIZE = 40  # Dungeon grid size (MAP_SIZE x MAP_SIZE)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.dungeon_level = 0
        self.grid = []
        self.player = None
        self.enemies = []
        self.gold_piles = []
        self.stairs_pos = None
        self.particles = deque()
        self.steps = 0
        self.game_over_message = ""
        self.game_over = False
        
        self.validate_implementation()

    def _get_enemy_stats(self, enemy_type, level):
        base_stats = {
            "goblin": {"health": 30, "damage": 5, "range": 1.5, "color": (50, 180, 50)},
            "skeleton": {"health": 50, "damage": 10, "range": 1.5, "color": (200, 200, 200)},
            "orc": {"health": 80, "damage": 15, "range": 1.5, "color": (0, 120, 0)},
            "mage": {"health": 40, "damage": 20, "range": 5.0, "color": (180, 0, 180)},
            "bat": {"health": 20, "damage": 8, "range": 1.5, "color": (100, 80, 60)},
            "boss": {"health": 300, "damage": 25, "range": 2.5, "color": (255, 50, 50)},
        }
        stats = base_stats[enemy_type].copy()
        # Scale stats with dungeon level
        scale_factor = 1 + (level - 1) * 0.1
        stats["health"] = int(stats["health"] * scale_factor)
        stats["damage"] = int(stats["damage"] * scale_factor)
        return stats

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dungeon_level = 1
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""
        self.particles.clear()

        self.player = {
            "x": 0, "y": 0,
            "health": self.PLAYER_MAX_HEALTH,
            "max_health": self.PLAYER_MAX_HEALTH,
            "gold": 0,
        }

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.enemies.clear()
        self.gold_piles.clear()
        self.particles.clear()

        # 1. Generate dungeon layout
        self.grid = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=int)
        start_pos = (self.MAP_SIZE // 2, self.MAP_SIZE // 2)
        px, py = start_pos
        self.grid[px, py] = 1
        num_tiles = int(self.MAP_SIZE * self.MAP_SIZE * 0.3)
        for _ in range(num_tiles):
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            px, py = max(1, min(self.MAP_SIZE - 2, px + dx)), max(1, min(self.MAP_SIZE - 2, py + dy))
            self.grid[px, py] = 1
        
        floor_tiles = list(zip(*np.where(self.grid == 1)))
        
        # 2. Place player
        player_start_pos = random.choice(floor_tiles)
        self.player["x"], self.player["y"] = player_start_pos

        # 3. Place stairs
        possible_stairs = [pos for pos in floor_tiles if self._grid_dist(pos, player_start_pos) > 15]
        self.stairs_pos = random.choice(possible_stairs) if possible_stairs else random.choice(floor_tiles)

        # 4. Place gold
        for _ in range(10):
            pos = random.choice(floor_tiles)
            if pos != player_start_pos and pos != self.stairs_pos:
                self.gold_piles.append(list(pos))

        # 5. Place enemies
        if self.dungeon_level == 3: # Boss level
            boss_pos = random.choice([p for p in floor_tiles if self._grid_dist(p, player_start_pos) > 10])
            stats = self._get_enemy_stats("boss", self.dungeon_level)
            self.enemies.append({
                "x": boss_pos[0], "y": boss_pos[1], "type": "boss",
                "health": stats["health"], "max_health": stats["health"],
                "damage": stats["damage"], "range": stats["range"], "color": stats["color"]
            })

        enemy_types = ["goblin", "bat"]
        if self.dungeon_level >= 2:
            enemy_types.extend(["skeleton", "mage"])
        if self.dungeon_level >= 3:
            enemy_types.append("orc")

        num_enemies = 5 + self.dungeon_level * 2
        for _ in range(num_enemies):
            pos = random.choice(floor_tiles)
            if self._grid_dist(pos, player_start_pos) > 5:
                enemy_type = random.choice(enemy_types)
                stats = self._get_enemy_stats(enemy_type, self.dungeon_level)
                self.enemies.append({
                    "x": pos[0], "y": pos[1], "type": enemy_type,
                    "health": stats["health"], "max_health": stats["health"],
                    "damage": stats["damage"], "range": stats["range"], "color": stats["color"]
                })

    def _grid_dist(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Cost of taking a step
        self.steps += 1

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Player Turn ---
        # 1. Player Movement
        if movement != 0:
            px, py = self.player["x"], self.player["y"]
            dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(movement, (0, 0))
            nx, ny = px + dx, py + dy

            if 0 <= nx < self.MAP_SIZE and 0 <= ny < self.MAP_SIZE and self.grid[nx, ny] == 1:
                self.player["x"], self.player["y"] = nx, ny

        # 2. Player Attack
        if space_held:
            attacked = False
            # Find closest enemy in range
            targets = [e for e in self.enemies if self._grid_dist((self.player['x'], self.player['y']), (e['x'], e['y'])) <= self.PLAYER_ATTACK_RANGE]
            if targets:
                target = min(targets, key=lambda e: self._grid_dist((self.player['x'], self.player['y']), (e['x'], e['y'])))
                
                target["health"] -= self.PLAYER_ATTACK_DAMAGE
                attacked = True
                
                # Sound placeholder: # sfx_player_attack()
                p_pos = self._iso_to_screen(self.player['x'], self.player['y'])
                t_pos = self._iso_to_screen(target['x'], target['y'])
                self._create_hit_effect(t_pos)
                self._create_line_particle(p_pos, t_pos, self.COLOR_PLAYER, 10)

                if target["health"] <= 0:
                    if target["type"] == "boss":
                        reward += 10.0 # Event reward
                    else:
                        reward += 1.0
                    self.enemies.remove(target)
        
        # --- Post-Player Action Updates ---
        # 1. Gold Collection
        for gold_pos in self.gold_piles[:]:
            if gold_pos[0] == self.player["x"] and gold_pos[1] == self.player["y"]:
                self.player["gold"] += 1
                reward += 0.1
                self.gold_piles.remove(gold_pos)
                # Sound placeholder: # sfx_gold_pickup()
                g_screen_pos = self._iso_to_screen(gold_pos[0], gold_pos[1])
                self._create_sparkle_effect(g_screen_pos, self.COLOR_GOLD)

        # 2. Level Progression
        if self.player["x"] == self.stairs_pos[0] and self.player["y"] == self.stairs_pos[1]:
            self.dungeon_level += 1
            if self.dungeon_level > 3: # Should be handled by boss defeat, but as a fallback
                self.game_over = True
                self.game_over_message = "YOU WIN!"
            else:
                self._generate_level()
                # Sound placeholder: # sfx_level_up()

        # --- Enemy Turn ---
        for enemy in self.enemies:
            dist_to_player = self._grid_dist((self.player['x'], self.player['y']), (enemy['x'], enemy['y']))
            
            # 1. Enemy Attack
            if dist_to_player <= enemy["range"]:
                self.player["health"] -= enemy["damage"]
                # Sound placeholder: # sfx_player_hit()
                p_pos = self._iso_to_screen(self.player['x'], self.player['y'])
                self._create_hit_effect(p_pos, (255,0,0))
            
            # 2. Enemy Movement
            else:
                ex, ey = enemy["x"], enemy["y"]
                px, py = self.player["x"], self.player["y"]
                
                # Simple greedy movement towards player
                best_move = (ex, ey)
                min_dist = dist_to_player
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.MAP_SIZE and 0 <= ny < self.MAP_SIZE and self.grid[nx, ny] == 1:
                        # Check if another enemy is there
                        if not any(e['x'] == nx and e['y'] == ny for e in self.enemies):
                            new_dist = self._grid_dist((px, py), (nx, ny))
                            if new_dist < min_dist:
                                min_dist = new_dist
                                best_move = (nx, ny)
                enemy["x"], enemy["y"] = best_move

        # --- Termination Check ---
        terminated = False
        boss_alive = any(e['type'] == 'boss' for e in self.enemies)
        
        if self.player["health"] <= 0:
            self.player["health"] = 0
            reward = -100.0
            terminated = True
            self.game_over = True
            self.game_over_message = "YOU DIED"
        
        elif self.dungeon_level == 3 and not boss_alive:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.game_over_message = "YOU DEFEATED THE BOSS!"

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME'S UP"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y):
        screen_x = (x - y) * self.TILE_WIDTH // 2
        screen_y = (x + y) * self.TILE_HEIGHT // 2
        return screen_x, screen_y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera offset to center on player
        cam_offset_x = self.SCREEN_WIDTH // 2 - self._iso_to_screen(self.player["x"], self.player["y"])[0]
        cam_offset_y = self.SCREEN_HEIGHT // 2 - self._iso_to_screen(self.player["x"], self.player["y"])[1]

        # Get visible tile range
        # This is an approximation but good enough
        min_vis_x = self.player['x'] - 25
        max_vis_x = self.player['x'] + 25
        min_vis_y = self.player['y'] - 40
        max_vis_y = self.player['y'] + 40

        # Collect all entities to draw and sort by y-order
        render_queue = []

        for y in range(max(0, min_vis_y), min(self.MAP_SIZE, max_vis_y)):
            for x in range(max(0, min_vis_x), min(self.MAP_SIZE, max_vis_x)):
                if self.grid[x, y] == 1:
                    render_queue.append({'type': 'floor', 'x': x, 'y': y})
                else: # Wall
                    # Check if wall is visible (next to a floor)
                    is_visible = False
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.MAP_SIZE and 0 <= ny < self.MAP_SIZE and self.grid[nx,ny] == 1:
                            is_visible = True
                            break
                    if is_visible:
                        render_queue.append({'type': 'wall', 'x': x, 'y': y})

        # Add dynamic entities to the queue
        for gold_pos in self.gold_piles:
            render_queue.append({'type': 'gold', 'x': gold_pos[0], 'y': gold_pos[1]})
        
        if self.stairs_pos:
            render_queue.append({'type': 'stairs', 'x': self.stairs_pos[0], 'y': self.stairs_pos[1]})

        for enemy in self.enemies:
            render_queue.append({'type': 'enemy', 'entity': enemy})

        render_queue.append({'type': 'player', 'entity': self.player})
        
        # Sort by y-grid, then x-grid for correct isometric overlap
        def sort_key(item):
            if item['type'] in ['floor', 'wall', 'gold', 'stairs']:
                return (item['y'], item['x'], 0)
            else: # player, enemy
                return (item['entity']['y'], item['entity']['x'], 1)
        
        render_queue.sort(key=sort_key)

        # Draw everything in order
        for item in render_queue:
            item_type = item['type']
            if item_type in ['floor', 'wall']:
                sx, sy = self._iso_to_screen(item['x'], item['y'])
                sx += cam_offset_x
                sy += cam_offset_y
                if item_type == 'floor':
                    self._draw_iso_tile(self.screen, (sx, sy), self.COLOR_FLOOR)
                else: # wall
                    self._draw_iso_cube(self.screen, (sx, sy), self.COLOR_WALL, self.COLOR_WALL_TOP)
            
            elif item_type == 'gold':
                sx, sy = self._iso_to_screen(item['x'], item['y'])
                sx += cam_offset_x
                sy += cam_offset_y
                self._draw_gold(self.screen, (sx, sy))

            elif item_type == 'stairs':
                sx, sy = self._iso_to_screen(item['x'], item['y'])
                sx += cam_offset_x
                sy += cam_offset_y
                self._draw_iso_tile(self.screen, (sx, sy), self.COLOR_STAIRS)
                text = self.font_s.render("V", True, (255,255,255))
                self.screen.blit(text, (sx - text.get_width()//2, sy - text.get_height()//2 - 2))

            elif item_type in ['player', 'enemy']:
                entity = item['entity']
                sx, sy = self._iso_to_screen(entity['x'], entity['y'])
                sx += cam_offset_x
                sy += cam_offset_y
                
                # Shadow
                shadow_pos = (sx, sy + 4)
                if item_type == 'player':
                    self._draw_character(self.screen, shadow_pos, self.COLOR_PLAYER_SHADOW, is_shadow=True)
                    self._draw_character(self.screen, (sx, sy), self.COLOR_PLAYER)
                else: # enemy
                    self._draw_character(self.screen, shadow_pos, (0,0,0,100), is_shadow=True)
                    self._draw_character(self.screen, (sx, sy), entity['color'])
                
                # Health bar for enemies
                if item_type == 'enemy':
                    self._draw_health_bar(self.screen, (sx, sy - 25), entity['health'], entity['max_health'], 30)

        # Update and draw particles
        self._update_and_draw_particles(cam_offset_x, cam_offset_y)

    def _draw_iso_tile(self, surface, pos, color):
        x, y = int(pos[0]), int(pos[1])
        points = [
            (x, y),
            (x + self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2),
            (x, y + self.TILE_HEIGHT),
            (x - self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_iso_cube(self, surface, pos, side_color, top_color):
        x, y = int(pos[0]), int(pos[1])
        # Top face
        top_points = [
            (x, y - self.TILE_DEPTH),
            (x + self.TILE_WIDTH // 2, y - self.TILE_DEPTH + self.TILE_HEIGHT // 2),
            (x, y - self.TILE_DEPTH + self.TILE_HEIGHT),
            (x - self.TILE_WIDTH // 2, y - self.TILE_DEPTH + self.TILE_HEIGHT // 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)
        # Left face
        left_points = [
            (x - self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2),
            (x, y + self.TILE_HEIGHT),
            (x, y + self.TILE_HEIGHT - self.TILE_DEPTH),
            (x - self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2 - self.TILE_DEPTH)
        ]
        pygame.gfxdraw.filled_polygon(surface, left_points, side_color)
        # Right face
        right_points = [
            (x + self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2),
            (x, y + self.TILE_HEIGHT),
            (x, y + self.TILE_HEIGHT - self.TILE_DEPTH),
            (x + self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2 - self.TILE_DEPTH)
        ]
        pygame.gfxdraw.filled_polygon(surface, right_points, side_color)

    def _draw_character(self, surface, pos, color, is_shadow=False):
        x, y = int(pos[0]), int(pos[1])
        h = 20
        w = 10
        points = [
            (x, y - h),
            (x + w//2, y - h//2),
            (x, y),
            (x - w//2, y - h//2)
        ]
        if is_shadow:
            pygame.gfxdraw.filled_ellipse(surface, x, y, 8, 4, color)
        else:
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_gold(self, surface, pos):
        x, y = int(pos[0]), int(pos[1])
        size = 3 + math.sin(self.steps * 0.2) * 1.5
        pygame.draw.rect(surface, self.COLOR_GOLD, (x - size, y - size, size*2, size*2))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player["health"] / self.player["max_health"]
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_s.render(f"HP: {self.player['health']}/{self.player['max_health']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Gold Count
        gold_text = self.font_m.render(f"Gold: {self.player['gold']}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 40))

        # Dungeon Level
        level_text = self.font_m.render(f"Level: {self.dungeon_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_l.render(self.game_over_message, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_health_bar(self, surface, pos, current, maximum, width):
        if maximum <= 0: return
        ratio = max(0, min(1, current / maximum))
        x, y = int(pos[0]), int(pos[1])
        bg_rect = pygame.Rect(x - width // 2, y, width, 5)
        fg_rect = pygame.Rect(x - width // 2, y, int(width * ratio), 5)
        pygame.draw.rect(surface, self.COLOR_HEALTH_BAR_BG, bg_rect)
        pygame.draw.rect(surface, self.COLOR_HEALTH_BAR, fg_rect)

    def _create_hit_effect(self, pos, color=(255, 255, 255)):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            dx, dy = math.cos(angle) * speed, math.sin(angle) * speed
            self.particles.append({'pos': list(pos), 'vel': [dx, dy], 'life': 10, 'color': color, 'type': 'spark'})
    
    def _create_sparkle_effect(self, pos, color):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            dx, dy = math.cos(angle) * speed, math.sin(angle) * speed
            self.particles.append({'pos': list(pos), 'vel': [dx, dy], 'life': 15, 'color': color, 'type': 'spark'})

    def _create_line_particle(self, start_pos, end_pos, color, life):
        self.particles.append({'start': start_pos, 'end': end_pos, 'life': life, 'color': color, 'type': 'line'})

    def _update_and_draw_particles(self, cam_offset_x, cam_offset_y):
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p['life'] -= 1
            if p['life'] > 0:
                if p['type'] == 'spark':
                    p['pos'][0] += p['vel'][0]
                    p['pos'][1] += p['vel'][1]
                    px = int(p['pos'][0] + cam_offset_x)
                    py = int(p['pos'][1] + cam_offset_y)
                    size = max(1, int(p['life'] / 3))
                    pygame.draw.circle(self.screen, p['color'], (px, py), size)
                elif p['type'] == 'line':
                    alpha = int(255 * (p['life'] / 10.0))
                    color = (*p['color'], alpha)
                    start_x = p['start'][0] + cam_offset_x
                    start_y = p['start'][1] + cam_offset_y
                    end_x = p['end'][0] + cam_offset_x
                    end_y = p['end'][1] + cam_offset_y
                    
                    temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(temp_surf, color, (start_x, start_y), (end_x, end_y), 2)
                    self.screen.blit(temp_surf, (0,0))
                self.particles.append(p)

    def _get_info(self):
        return {
            "score": self.player["gold"] if self.player else 0,
            "steps": self.steps,
            "health": self.player["health"] if self.player else 0,
            "level": self.dungeon_level
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space from a direct call
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    # It's not part of the required class structure but is useful for development.
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Dungeon Crawler")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAY TEST")
    print("="*30)
    print(env.user_guide)
    print("Press ESC or close window to quit.")

    # Game loop for manual play
    running = True
    while running:
        movement_action = 0  # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if terminated:
            # On game over, wait for reset key
            pass
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4
            
            if keys[pygame.K_SPACE]: space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
            action = [movement_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
        
        # In a real-time loop, we need a clock tick. 
        # For turn-based, this just controls how fast you can press keys.
        env.clock.tick(10) # Limit manual input speed

    env.close()