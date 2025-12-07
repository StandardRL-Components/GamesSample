
# Generated: 2025-08-27T14:15:49.183157
# Source Brief: brief_00628.md
# Brief Index: 628

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Press space to attack nearby enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated isometric dungeon, battling enemies to reach and defeat the boss."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_FLOOR = (50, 45, 65)
    COLOR_WALL = (90, 85, 115)
    COLOR_WALL_TOP = (120, 115, 145)

    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_SHADOW = (30, 100, 30)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_SHADOW = (100, 30, 30)
    COLOR_BOSS = (250, 100, 100)
    COLOR_BOSS_SHADOW = (120, 50, 50)
    COLOR_GOLD = (255, 215, 0)

    COLOR_HEALTH_BAR_BG = (70, 70, 70)
    COLOR_HEALTH_BAR_PLAYER = (0, 255, 0)
    COLOR_HEALTH_BAR_ENEMY = (255, 0, 0)

    COLOR_ATTACK_FLASH = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)

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
        self.font_ui = pygame.font.SysFont('Consolas', 18, bold=True)
        self.font_damage = pygame.font.SysFont('Arial', 14, bold=True)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT))
        self.player = {}
        self.enemies = []
        self.boss = {}
        self.gold_piles = []
        self.damage_texts = []
        self.attack_effects = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.damage_texts.clear()
        self.attack_effects.clear()

        self._generate_dungeon()
        
        start_pos = self._find_valid_spawn_point()
        self.player = {
            "pos": start_pos,
            "health": 100,
            "max_health": 100,
            "damage_flash": 0
        }

        boss_pos = self._find_valid_spawn_point(min_dist_from=start_pos, dist=10)
        self.boss = {
            "pos": boss_pos,
            "health": 50,
            "max_health": 50,
            "damage_flash": 0
        }

        self.enemies = []
        for _ in range(5):
            self._spawn_enemy()
            
        self.gold_piles = []
        for _ in range(10):
            pos = self._find_valid_spawn_point()
            if pos:
                self.gold_piles.append({"pos": pos})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.1  # Per-step penalty
        self.steps += 1
        
        # --- Update Effects ---
        self._update_effects()

        # --- Player Action ---
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        dist_to_boss_before = self._manhattan_distance(self.player["pos"], self.boss["pos"])

        # Player Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = [self.player["pos"][0] + dx, self.player["pos"][1] + dy]
            if self._is_valid_and_walkable(new_pos):
                self.player["pos"] = new_pos

        dist_to_boss_after = self._manhattan_distance(self.player["pos"], self.boss["pos"])
        reward += (dist_to_boss_before - dist_to_boss_after)

        # Gold Collection
        for gold in self.gold_piles[:]:
            if self.player["pos"] == gold["pos"]:
                self.gold_piles.remove(gold)
                self.score += 1
                reward += 1
                # SFX: gold_pickup.wav
                
        # Player Attack
        if space_pressed:
            target = self._find_nearest_target()
            if target:
                damage = self.np_random.integers(8, 13)
                target["health"] -= damage
                target["damage_flash"] = 3
                self._create_damage_text(str(damage), target["pos"])
                self._create_attack_effect(target["pos"])
                # SFX: sword_swing.wav, hit_flesh.wav
                if target["health"] <= 0:
                    if target is self.boss:
                        reward += 100
                        self.score += 50
                    else:
                        reward += 10
                        self.score += 10
                        self.enemies.remove(target)
                        self._spawn_enemy(respawn=True) # Respawn one after a delay

        # --- Enemy & Boss Turn ---
        # Enemies
        for enemy in self.enemies:
            if self._manhattan_distance(enemy["pos"], self.player["pos"]) <= 1:
                # Attack player
                damage = self.np_random.integers(3, 6)
                self.player["health"] -= damage
                self.player["damage_flash"] = 3
                self._create_damage_text(str(damage), self.player["pos"], is_player=True)
                # SFX: enemy_attack.wav, player_hurt.wav
            else:
                # Move
                if enemy.get("patrol_timer", 0) <= 0:
                    px, py = self.player["pos"]
                    ex, ey = enemy["pos"]
                    # Simple move towards player
                    if abs(px - ex) > abs(py - ey):
                        new_pos = [ex + np.sign(px - ex), ey]
                    else:
                        new_pos = [ex, ey + np.sign(py - ey)]
                    
                    if self._is_valid_and_walkable(new_pos):
                        enemy["pos"] = new_pos
                    enemy["patrol_timer"] = 2 # Move every 2 steps
                else:
                    enemy["patrol_timer"] -= 1

        # Boss
        if self._manhattan_distance(self.boss["pos"], self.player["pos"]) <= 3:
            # Attack player
            damage = self.np_random.integers(8, 15)
            self.player["health"] -= damage
            self.player["damage_flash"] = 3
            self._create_damage_text(str(damage), self.player["pos"], is_player=True)
            # SFX: boss_attack.wav, player_hurt.wav

        # --- Check Termination ---
        terminated = False
        if self.player["health"] <= 0:
            reward -= 100
            terminated = True
            # SFX: game_over.wav
        elif self.boss["health"] <= 0:
            # Reward already given on kill
            terminated = True
            # SFX: victory.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper & Rendering Methods ---

    def _world_to_iso(self, x, y):
        iso_x = (x - y) * self.TILE_WIDTH_HALF + self.SCREEN_WIDTH / 2
        iso_y = (x + y) * self.TILE_HEIGHT_HALF + self.SCREEN_HEIGHT / 2 - 100
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, pos, color_top, color_side, size=1.0, height=1.0):
        x, y = pos
        iso_x, iso_y = self._world_to_iso(x, y)
        
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        cube_height = h * 2 * height
        
        # Top face
        top_points = [
            (iso_x, iso_y - cube_height),
            (iso_x + w * size, iso_y + h * size - cube_height),
            (iso_x, iso_y + h * 2 * size - cube_height),
            (iso_x - w * size, iso_y + h * size - cube_height)
        ]
        pygame.gfxdraw.filled_polygon(surface, top_points, color_top)
        pygame.gfxdraw.aapolygon(surface, top_points, color_top)

        # Left face
        left_points = [
            (iso_x - w * size, iso_y + h * size - cube_height),
            (iso_x, iso_y + h * 2 * size - cube_height),
            (iso_x, iso_y + h * 2 * size),
            (iso_x - w * size, iso_y + h * size)
        ]
        pygame.gfxdraw.filled_polygon(surface, left_points, color_side)
        
        # Right face
        right_points = [
            (iso_x + w * size, iso_y + h * size - cube_height),
            (iso_x, iso_y + h * 2 * size - cube_height),
            (iso_x, iso_y + h * 2 * size),
            (iso_x + w * size, iso_y + h * size)
        ]
        pygame.gfxdraw.filled_polygon(surface, right_points, tuple(c * 0.7 for c in color_side))
    
    def _render_game(self):
        # Sort all entities by y-position for correct draw order
        render_queue = []
        
        # Floor and Walls
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                iso_x, iso_y = self._world_to_iso(x, y)
                floor_points = [
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF * 2),
                    (iso_x + self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF),
                    (iso_x, iso_y),
                    (iso_x - self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, floor_points, self.COLOR_FLOOR)
                
                if self.grid[x, y] == 1:
                    render_queue.append(({'type': 'wall', 'pos': [x, y]}, y))

        # Gold
        for gold in self.gold_piles:
            render_queue.append(({'type': 'gold', 'pos': gold['pos']}, gold['pos'][1] + 0.1))
            
        # Entities
        render_queue.append(({'type': 'player', **self.player}, self.player['pos'][1] + 0.5))
        for enemy in self.enemies:
            render_queue.append(({'type': 'enemy', **enemy}, enemy['pos'][1] + 0.5))
        if self.boss['health'] > 0:
            render_queue.append(({'type': 'boss', **self.boss}, self.boss['pos'][1] + 0.5))
            
        render_queue.sort(key=lambda item: self._world_to_iso(*item[0]['pos'])[1])

        # Draw from queue
        for item, _ in render_queue:
            if item['type'] == 'wall':
                self._draw_iso_cube(self.screen, item['pos'], self.COLOR_WALL_TOP, self.COLOR_WALL)
            elif item['type'] == 'gold':
                iso_x, iso_y = self._world_to_iso(*item['pos'])
                pygame.draw.circle(self.screen, self.COLOR_GOLD, (iso_x, iso_y + self.TILE_HEIGHT_HALF), 5)
            else:
                self._render_entity(item)
        
        # Render effects on top
        for effect in self.attack_effects:
            iso_x, iso_y = self._world_to_iso(*effect['pos'])
            radius = int(15 * effect['life'])
            alpha = int(255 * effect['life'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, iso_x, iso_y, radius, (*self.COLOR_ATTACK_FLASH, alpha))
        
        for text in self.damage_texts:
            iso_x, iso_y = self._world_to_iso(*text['pos'])
            color = (255, 80, 80) if not text['is_player'] else (80, 255, 80)
            text_surf = self.font_damage.render(text['text'], True, color)
            self.screen.blit(text_surf, (iso_x - text_surf.get_width()//2, iso_y - 20 - (1-text['life'])*20))

    def _render_entity(self, entity):
        size, color, shadow_color, height = 1.0, (0,0,0), (0,0,0), 1.0
        if entity['type'] == 'player':
            size, color, shadow_color, height = 0.8, self.COLOR_PLAYER, self.COLOR_PLAYER_SHADOW, 1.0
        elif entity['type'] == 'enemy':
            size, color, shadow_color, height = 0.7, self.COLOR_ENEMY, self.COLOR_ENEMY_SHADOW, 0.8
        elif entity['type'] == 'boss':
            size, color, shadow_color, height = 1.2, self.COLOR_BOSS, self.COLOR_BOSS_SHADOW, 1.5

        if entity.get('damage_flash', 0) > 0:
            color = (255, 255, 255)

        # Shadow
        iso_x, iso_y = self._world_to_iso(*entity['pos'])
        shadow_rect = pygame.Rect(0, 0, self.TILE_WIDTH_HALF * 1.5 * size, self.TILE_HEIGHT_HALF * 1.5 * size)
        shadow_rect.center = (iso_x, iso_y + self.TILE_HEIGHT_HALF * 1.8)
        pygame.draw.ellipse(self.screen, shadow_color, shadow_rect)
        
        # Body
        self._draw_iso_cube(self.screen, entity['pos'], color, tuple(c * 0.7 for c in color), size=size*0.8, height=height)
        
        # Health bar
        bar_width = 30
        bar_height = 4
        health_pct = max(0, entity['health'] / entity['max_health'])
        bar_x = iso_x - bar_width // 2
        bar_y = iso_y - int(self.TILE_HEIGHT_HALF * 2 * height) - 10
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        bar_color = self.COLOR_HEALTH_BAR_PLAYER if entity['type'] == 'player' else self.COLOR_HEALTH_BAR_ENEMY
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_ui(self):
        health_text = self.font_ui.render(f"HP: {max(0, self.player['health'])} / {self.player['max_health']}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"Gold: {self.score}", True, self.COLOR_TEXT)
        steps_text = self.font_ui.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(score_text, (10, 30))
        self.screen.blit(steps_text, (10, 50))
        
    def _generate_dungeon(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        # Create borders
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        # Add some random pillars
        num_pillars = self.np_random.integers(10, 20)
        for _ in range(num_pillars):
            x = self.np_random.integers(2, self.GRID_WIDTH - 2)
            y = self.np_random.integers(2, self.GRID_HEIGHT - 2)
            self.grid[x, y] = 1

    def _find_valid_spawn_point(self, min_dist_from=None, dist=0):
        for _ in range(100): # Max attempts
            x = self.np_random.integers(1, self.GRID_WIDTH - 1)
            y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
            if self.grid[x, y] == 0:
                if min_dist_from:
                    if self._manhattan_distance([x, y], min_dist_from) >= dist:
                        return [x, y]
                else:
                    return [x, y]
        return None # Could not find a point

    def _spawn_enemy(self, respawn=False):
        if len(self.enemies) >= 5: return
        pos = self._find_valid_spawn_point(min_dist_from=self.player["pos"], dist=5)
        if pos:
            self.enemies.append({
                "pos": pos,
                "health": 10,
                "max_health": 10,
                "damage_flash": 0,
                "patrol_timer": self.np_random.integers(0, 3)
            })

    def _is_valid_and_walkable(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        if self.grid[x, y] == 1:
            return False
        return True

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _find_nearest_target(self):
        targets = [e for e in self.enemies if e['health'] > 0]
        if self.boss['health'] > 0:
            targets.append(self.boss)
        
        nearest_target = None
        min_dist = float('inf')

        for target in targets:
            dist = self._manhattan_distance(self.player['pos'], target['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest_target = target
        
        if nearest_target and min_dist <= 1.5: # Attack adjacent (including diagonals)
            return nearest_target
        return None

    def _create_damage_text(self, text, pos, is_player=False):
        self.damage_texts.append({'text': text, 'pos': list(pos), 'life': 1.0, 'is_player': is_player})

    def _create_attack_effect(self, pos):
        self.attack_effects.append({'pos': list(pos), 'life': 1.0})

    def _update_effects(self):
        # Damage text
        for text in self.damage_texts[:]:
            text['life'] -= 0.05
            if text['life'] <= 0:
                self.damage_texts.remove(text)
        # Attack effect
        for effect in self.attack_effects[:]:
            effect['life'] -= 0.1
            if effect['life'] <= 0:
                self.attack_effects.remove(effect)
        # Damage flash
        if self.player['damage_flash'] > 0: self.player['damage_flash'] -= 1
        for e in self.enemies:
            if e['damage_flash'] > 0: e['damage_flash'] -= 1
        if self.boss['damage_flash'] > 0: self.boss['damage_flash'] -= 1

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Create a dummy state to test observation
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Dungeon Crawler")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        movement = 0  # No-op
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
        
        # The game only advances on an action
        # For manual play, we send an action every frame
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        clock.tick(10) # Control manual play speed

    pygame.quit()