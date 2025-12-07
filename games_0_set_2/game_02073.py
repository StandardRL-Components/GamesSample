import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in your last moved direction. "
        "Explore the dungeon, collect gold, and find the exit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro pixel-art dungeon crawler. Navigate a procedurally generated maze, "
        "battle monsters, and gather treasure on your way to the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (70, 70, 90)
    COLOR_FLOOR = (40, 40, 60)
    COLOR_PLAYER = (60, 220, 120)
    COLOR_EXIT = (100, 100, 255)
    COLOR_GOLD = (255, 223, 0)
    
    ENEMY_COLORS = [
        (255, 80, 80),   # Chaser
        (255, 150, 80),  # Random
        (200, 100, 200), # Patroller (Horizontal)
        (200, 200, 100), # Patroller (Vertical)
        (255, 120, 180)  # Strong Chaser
    ]

    # UI
    UI_FONT_SIZE = 24
    UI_COLOR = (240, 240, 240)
    HEALTH_BAR_COLOR = (220, 50, 50)
    HEALTH_BAR_BG_COLOR = (80, 20, 20)
    
    # Grid and World
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_SIZE = 32
    VIEWPORT_W = SCREEN_WIDTH // CELL_SIZE
    VIEWPORT_H = SCREEN_HEIGHT // CELL_SIZE
    WORLD_W, WORLD_H = 41, 31

    # Game Mechanics
    MAX_STEPS = 1000
    PLAYER_BASE_HEALTH = 100
    PLAYER_BASE_ATTACK = 25
    
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
        self.font = pygame.font.Font(None, self.UI_FONT_SIZE)
        self.game_over_font = pygame.font.Font(None, 60)
        
        self.render_mode = render_mode
        self.np_random = None

        # This will be initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.grid = None
        self.player_pos = None
        self.player_health = 0
        self.player_max_health = 0
        self.player_attack = 0
        self.player_last_move_dir = (0, -1) # Default up
        self.exit_pos = None
        self.enemies = []
        self.gold_piles = {}
        self.visual_effects = []
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.level = 1
        self._setup_level()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self._generate_dungeon()
        self._populate_dungeon()
        
        self.player_max_health = self.PLAYER_BASE_HEALTH
        self.player_health = self.player_max_health
        self.player_attack = self.PLAYER_BASE_ATTACK
        
        self.visual_effects = []

    def _generate_dungeon(self):
        self.grid = np.ones((self.WORLD_W, self.WORLD_H), dtype=np.uint8) # 1 = wall
        
        # Randomized DFS for maze generation
        stack = []
        start_x, start_y = (self.np_random.integers(1, self.WORLD_W//2) * 2, 
                            self.np_random.integers(1, self.WORLD_H//2) * 2)
        self.grid[start_x, start_y] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.WORLD_W - 1 and 0 < ny < self.WORLD_H - 1 and self.grid[nx, ny] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                self.grid[nx, ny] = 0
                self.grid[x + (nx - x) // 2, y + (ny - y) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _populate_dungeon(self):
        floor_tiles = np.argwhere(self.grid == 0).tolist()
        self.np_random.shuffle(floor_tiles)

        # Place player and find furthest point for exit
        self.player_pos = floor_tiles.pop(0)
        
        # BFS to find the furthest tile from the player
        q = deque([(self.player_pos, 0)])
        visited = {tuple(self.player_pos)}
        farthest_pos, max_dist = self.player_pos, 0
        while q:
            pos, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_pos = pos
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if self.grid[nx, ny] == 0 and tuple((nx, ny)) not in visited:
                    visited.add(tuple((nx, ny)))
                    q.append(((nx, ny), dist + 1))
        
        self.exit_pos = farthest_pos
        if tuple(self.exit_pos) in [tuple(t) for t in floor_tiles]:
            floor_tiles.remove(list(self.exit_pos))

        # Place gold
        self.gold_piles = {}
        num_gold = self.np_random.integers(15, 25)
        for _ in range(num_gold):
            if not floor_tiles: break
            pos = tuple(floor_tiles.pop(0))
            self.gold_piles[pos] = self.np_random.integers(10, 51)
        
        # Place enemies
        self.enemies = []
        num_enemies = self.np_random.integers(5, 10)
        enemy_difficulty_mod = 1 + (self.level - 1) * 0.1
        
        for i in range(num_enemies):
            if not floor_tiles: break
            pos = floor_tiles.pop(0)
            enemy_type = self.np_random.integers(len(self.ENEMY_COLORS))
            
            base_health, base_damage, behavior = self._get_enemy_stats(enemy_type)
            
            self.enemies.append({
                "pos": pos,
                "type": enemy_type,
                "health": int(base_health * enemy_difficulty_mod),
                "max_health": int(base_health * enemy_difficulty_mod),
                "damage": int(base_damage * enemy_difficulty_mod),
                "behavior": behavior,
                "patrol_dir": 1 if self.np_random.random() > 0.5 else -1,
                "flash": 0,
            })
    
    def _get_enemy_stats(self, enemy_type):
        if enemy_type == 0: return 20, 5, "chase"
        if enemy_type == 1: return 30, 8, "random"
        if enemy_type == 2: return 25, 7, "patrol_h"
        if enemy_type == 3: return 25, 7, "patrol_v"
        if enemy_type == 4: return 40, 12, "chase"
        return 20, 5, "random" # Default

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = 0
        self.steps += 1
        
        # --- Player Action ---
        movement = action[0]
        space_held = action[1] == 1
        
        action_taken = False
        if space_held:
            # Attack
            action_taken = True
            target_x = self.player_pos[0] + self.player_last_move_dir[0]
            target_y = self.player_pos[1] + self.player_last_move_dir[1]
            
            # Sound placeholder: player_attack_sound()
            self._add_visual_effect('slash', self.player_pos, dir=self.player_last_move_dir)

            for enemy in self.enemies:
                if tuple(enemy["pos"]) == (target_x, target_y):
                    enemy["health"] -= self.player_attack
                    enemy["flash"] = 3 # Flash for 3 frames
                    # Sound placeholder: enemy_hit_sound()
                    if enemy["health"] <= 0:
                        reward += 1.0 # Kill reward
                        self.score += 25 # Bonus score for kill
                        self._add_visual_effect('death', enemy["pos"])
                        self.enemies.remove(enemy)
                        # Sound placeholder: enemy_death_sound()
                    break
        elif movement != 0:
            # Move
            action_taken = True
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            if dx != 0 or dy != 0:
                self.player_last_move_dir = (dx, dy)
            
            nx, ny = self.player_pos[0] + dx, self.player_pos[1] + dy
            
            is_wall = self.grid[nx, ny] == 1
            is_enemy = any(tuple(e["pos"]) == (nx, ny) for e in self.enemies)

            if not is_wall and not is_enemy:
                self.player_pos = [nx, ny]
                # Sound placeholder: player_move_step()

        # --- Post-Action State Update ---
        if action_taken:
            # Gold collection
            player_pos_tuple = tuple(self.player_pos)
            if player_pos_tuple in self.gold_piles:
                gold_amount = self.gold_piles.pop(player_pos_tuple)
                self.score += gold_amount
                reward += gold_amount * 0.1
                self._add_visual_effect('text_popup', self.player_pos, text=f"+{gold_amount}", color=self.COLOR_GOLD)
                # Sound placeholder: gold_pickup_sound()

            # --- Enemy Turn ---
            for enemy in self.enemies:
                px, py = self.player_pos
                ex, ey = enemy["pos"]
                
                # Attack if adjacent
                if abs(px - ex) + abs(py - ey) == 1:
                    damage = enemy["damage"]
                    self.player_health -= damage
                    reward -= damage * 0.1
                    self._add_visual_effect('flash_player', self.player_pos)
                    # Sound placeholder: player_hit_sound()
                else: # Move
                    self._move_enemy(enemy)
        
        self._update_effects()

        # --- Termination Check ---
        terminated = False
        if tuple(self.player_pos) == tuple(self.exit_pos):
            reward += 100
            self.game_over = True
            terminated = True
            self._add_visual_effect('win_text', (self.VIEWPORT_W // 2, self.VIEWPORT_H // 2), text="VICTORY!")
        elif self.player_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            self._add_visual_effect('lose_text', (self.VIEWPORT_W // 2, self.VIEWPORT_H // 2), text="GAME OVER")
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_enemy(self, enemy):
        ex, ey = enemy["pos"]
        moves = []
        
        if enemy["behavior"] == "chase":
            px, py = self.player_pos
            dist_to_player = abs(px - ex) + abs(py - ey)
            if dist_to_player < 8:
                # Simple A* style move
                if px < ex: moves.append((-1, 0))
                if px > ex: moves.append((1, 0))
                if py < ey: moves.append((0, -1))
                if py > ey: moves.append((0, 1))
        elif enemy["behavior"] == "patrol_h":
            moves.append((enemy["patrol_dir"], 0))
        elif enemy["behavior"] == "patrol_v":
            moves.append((0, enemy["patrol_dir"]))
        
        # Default to random move if no specific move is chosen
        if not moves:
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.np_random.shuffle(moves)

        for dx, dy in moves:
            nx, ny = ex + dx, ey + dy
            if self.grid[nx, ny] == 0 and not any(tuple(e["pos"]) == (nx, ny) for e in self.enemies) and tuple(self.player_pos) != (nx, ny):
                enemy["pos"] = [nx, ny]
                return
        
        # If patrol is blocked, reverse direction
        if "patrol" in enemy["behavior"]:
            enemy["patrol_dir"] *= -1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def _render_game(self):
        # This check prevents rendering before reset() is called
        if self.player_pos is None:
            return

        # Camera centered on player, clamped to world bounds
        cam_x = max(0, min(self.player_pos[0] - self.VIEWPORT_W // 2, self.WORLD_W - self.VIEWPORT_W))
        cam_y = max(0, min(self.player_pos[1] - self.VIEWPORT_H // 2, self.WORLD_H - self.VIEWPORT_H))

        for y in range(self.VIEWPORT_H):
            for x in range(self.VIEWPORT_W):
                world_x, world_y = cam_x + x, cam_y + y
                screen_x, screen_y = x * self.CELL_SIZE, y * self.CELL_SIZE
                
                tile = self.grid[world_x, world_y]
                color = self.COLOR_WALL if tile == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE))
                
                # Add a subtle pattern to floor tiles
                if tile == 0 and (world_x + world_y) % 2 == 0:
                    pygame.draw.rect(self.screen, (45, 45, 65), (screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE))

        # Draw elements
        for pos, amount in self.gold_piles.items():
            if cam_x <= pos[0] < cam_x + self.VIEWPORT_W and cam_y <= pos[1] < cam_y + self.VIEWPORT_H:
                screen_x = (pos[0] - cam_x) * self.CELL_SIZE
                screen_y = (pos[1] - cam_y) * self.CELL_SIZE
                size = int(self.CELL_SIZE * (0.4 + 0.4 * (amount / 50)))
                offset = (self.CELL_SIZE - size) // 2
                pygame.draw.rect(self.screen, self.COLOR_GOLD, (screen_x + offset, screen_y + offset, size, size), border_radius=3)

        if self.exit_pos and cam_x <= self.exit_pos[0] < cam_x + self.VIEWPORT_W and cam_y <= self.exit_pos[1] < cam_y + self.VIEWPORT_H:
            screen_x = (self.exit_pos[0] - cam_x) * self.CELL_SIZE
            screen_y = (self.exit_pos[1] - cam_y) * self.CELL_SIZE
            pygame.draw.rect(self.screen, self.COLOR_EXIT, (screen_x + 4, screen_y + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8), border_radius=5)

        for enemy in self.enemies:
            if cam_x <= enemy["pos"][0] < cam_x + self.VIEWPORT_W and cam_y <= enemy["pos"][1] < cam_y + self.VIEWPORT_H:
                screen_x = (enemy["pos"][0] - cam_x) * self.CELL_SIZE
                screen_y = (enemy["pos"][1] - cam_y) * self.CELL_SIZE
                color = self.ENEMY_COLORS[enemy["type"]]
                if enemy["flash"] > 0:
                    color = (255, 255, 255)
                    enemy["flash"] -= 1
                pygame.draw.rect(self.screen, color, (screen_x + 4, screen_y + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8))
                
                # Health bar above enemy
                health_ratio = enemy["health"] / enemy["max_health"]
                bar_w = int((self.CELL_SIZE - 8) * health_ratio)
                pygame.draw.rect(self.screen, self.HEALTH_BAR_BG_COLOR, (screen_x + 4, screen_y, self.CELL_SIZE - 8, 3))
                pygame.draw.rect(self.screen, self.HEALTH_BAR_COLOR, (screen_x + 4, screen_y, bar_w, 3))


        if cam_x <= self.player_pos[0] < cam_x + self.VIEWPORT_W and cam_y <= self.player_pos[1] < cam_y + self.VIEWPORT_H:
            screen_x = (self.player_pos[0] - cam_x) * self.CELL_SIZE
            screen_y = (self.player_pos[1] - cam_y) * self.CELL_SIZE
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (screen_x + 2, screen_y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4), border_radius=4)
            # "Eye" to show direction
            eye_x = screen_x + self.CELL_SIZE // 2 + self.player_last_move_dir[0] * 8
            eye_y = screen_y + self.CELL_SIZE // 2 + self.player_last_move_dir[1] * 8
            pygame.draw.circle(self.screen, (0,0,0), (eye_x, eye_y), 3)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.player_max_health) if self.player_max_health > 0 else 0
        pygame.draw.rect(self.screen, self.HEALTH_BAR_BG_COLOR, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.HEALTH_BAR_COLOR, (10, 10, 200 * health_ratio, 20))
        health_text = self.font.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.UI_COLOR)
        self.screen.blit(health_text, (15, 12))
        
        # Gold Count
        gold_text = self.font.render(f"GOLD: {self.score}", True, self.COLOR_GOLD)
        text_rect = gold_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(gold_text, text_rect)

    def _add_visual_effect(self, type, pos, **kwargs):
        effect = {"type": type, "pos": list(pos), "lifetime": 1, **kwargs}
        if type == 'death':
            effect['particles'] = []
            for _ in range(20):
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.random() * 3 + 1
                effect['particles'].append({
                    'offset': [0, 0],
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'lifetime': self.np_random.integers(10, 20),
                    'color': self.ENEMY_COLORS[self.np_random.integers(len(self.ENEMY_COLORS))]
                })
            effect['lifetime'] = 20
        elif type == 'slash':
            effect['lifetime'] = 5
            effect['dir'] = kwargs.get('dir', (0, -1))
        elif type == 'flash_player':
            effect['lifetime'] = 3
        elif type == 'text_popup':
            effect['lifetime'] = 30
            effect['y_offset'] = 0
        elif 'text' in type:
            effect['lifetime'] = 1 # Persistent until reset
        
        self.visual_effects.append(effect)

    def _update_effects(self):
        for effect in self.visual_effects[:]:
            effect['lifetime'] -= 1
            if effect['type'] == 'death':
                for p in effect['particles']:
                    p['offset'][0] += p['vel'][0]
                    p['offset'][1] += p['vel'][1]
                    p['vel'][1] += 0.2 # gravity
                    p['lifetime'] -= 1
            elif effect['type'] == 'text_popup':
                effect['y_offset'] -= 1
            
            if effect['lifetime'] <= 0 and 'text' not in effect['type']:
                self.visual_effects.remove(effect)

    def _render_effects(self):
        if self.player_pos is None:
            return
            
        cam_x = max(0, min(self.player_pos[0] - self.VIEWPORT_W // 2, self.WORLD_W - self.VIEWPORT_W))
        cam_y = max(0, min(self.player_pos[1] - self.VIEWPORT_H // 2, self.WORLD_H - self.VIEWPORT_H))

        for effect in self.visual_effects:
            screen_x = (effect['pos'][0] - cam_x) * self.CELL_SIZE
            screen_y = (effect['pos'][1] - cam_y) * self.CELL_SIZE
            
            if effect['type'] == 'death':
                for p in effect['particles']:
                    if p['lifetime'] > 0:
                        px = screen_x + self.CELL_SIZE // 2 + int(p['offset'][0])
                        py = screen_y + self.CELL_SIZE // 2 + int(p['offset'][1])
                        alpha = int(255 * (p['lifetime'] / 20))
                        try:
                            pygame.gfxdraw.filled_circle(self.screen, px, py, 3, p['color'] + (alpha,))
                        except TypeError: # Handle color not having alpha
                             pygame.gfxdraw.filled_circle(self.screen, px, py, 3, tuple(list(p['color']) + [alpha]))
            elif effect['type'] == 'slash':
                center_x = screen_x + self.CELL_SIZE // 2
                center_y = screen_y + self.CELL_SIZE // 2
                end_x = center_x + effect['dir'][0] * self.CELL_SIZE * 0.8
                end_y = center_y + effect['dir'][1] * self.CELL_SIZE * 0.8
                alpha = int(255 * (effect['lifetime'] / 5))
                # Create a temporary surface for alpha drawing
                line_surf = self.screen.copy()
                pygame.draw.line(line_surf, (255, 255, 255), (center_x, center_y), (end_x, end_y), 3)
                line_surf.set_alpha(alpha)
                self.screen.blit(line_surf, (0,0))
            elif effect['type'] == 'flash_player':
                alpha = int(150 * (effect['lifetime'] / 3))
                s = pygame.Surface((self.CELL_SIZE-4, self.CELL_SIZE-4), pygame.SRCALPHA)
                s.fill((255, 80, 80, alpha))
                self.screen.blit(s, (screen_x + 2, screen_y + 2))
            elif effect['type'] == 'text_popup':
                alpha = int(255 * (effect['lifetime'] / 30))
                text_surf = self.font.render(effect['text'], True, effect['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=(screen_x + self.CELL_SIZE // 2, screen_y + effect['y_offset']))
                self.screen.blit(text_surf, text_rect)
            elif 'text' in effect['type']:
                text_surf = self.game_over_font.render(effect['text'], True, self.UI_COLOR)
                text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
                # Create a semi-transparent background
                bg_surf = pygame.Surface(text_rect.inflate(20, 20).size, pygame.SRCALPHA)
                bg_surf.fill((0, 0, 0, 180))
                self.screen.blit(bg_surf, text_rect.inflate(20, 20).topleft)
                self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")