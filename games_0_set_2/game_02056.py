
# Generated: 2025-08-27T19:07:38.256262
# Source Brief: brief_02056.md
# Brief Index: 2056

        
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
        "Controls: Arrow keys to move. Space to attack in your facing direction. "
        "Stand still (no movement action) to defend, reducing incoming damage."
    )

    game_description = (
        "Explore a procedurally generated dungeon, battling enemies and collecting "
        "gold to defeat the final boss. A top-down, turn-based, pixel-art RPG."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 32
    MAX_STEPS = 1000
    MAX_LEVELS = 3

    # --- Colors ---
    COLOR_BG = (18, 18, 22)
    COLOR_WALL = (50, 50, 60)
    COLOR_FLOOR = (30, 30, 35)
    COLOR_STAIRS = (100, 80, 50)
    
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_GLOW = (50, 200, 50, 50)
    
    COLOR_ENEMY_CHASER = (200, 50, 50)
    COLOR_ENEMY_RANDOM = (200, 100, 50)
    COLOR_ENEMY_RANGED = (200, 50, 200)
    COLOR_BOSS = (255, 0, 100)
    
    COLOR_GOLD = (255, 215, 0)
    COLOR_WHITE = (240, 240, 240)
    COLOR_HEALTH_GREEN = (0, 200, 0)
    COLOR_HEALTH_RED = (200, 0, 0)
    COLOR_UI_BG = (10, 10, 10, 200)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.player = None
        self.enemies = []
        self.gold_coins = []
        self.particles = []
        self.stairs_pos = None
        self.dungeon = None
        
        self.current_level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.current_level = 1
        
        player_max_hp = 100
        self.player = {
            "x": 0, "y": 0, "hp": player_max_hp, "max_hp": player_max_hp,
            "is_defending": False, "last_direction": (0, -1), "damage": 25,
        }
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.enemies.clear()
        self.gold_coins.clear()
        self.particles.clear()
        
        self.dungeon, floor_tiles = self._generate_dungeon(50, 50, 250)
        random.shuffle(floor_tiles)

        self.player["x"], self.player["y"] = floor_tiles.pop()
        
        if self.current_level < self.MAX_LEVELS:
            self.stairs_pos = floor_tiles.pop()
        else:
            self.stairs_pos = None

        is_boss_level = (self.current_level == self.MAX_LEVELS)
        num_enemies = 1 if is_boss_level else 5
        difficulty_mod = 1 + (self.current_level - 1) * 0.1

        for _ in range(num_enemies):
            if not floor_tiles: break
            pos = floor_tiles.pop()
            if is_boss_level:
                hp = int(200 * difficulty_mod)
                self.enemies.append({
                    "x": pos[0], "y": pos[1], "hp": hp, "max_hp": hp,
                    "damage": int(20 * difficulty_mod), "type": "boss",
                })
            else:
                enemy_type = random.choice(["chaser", "random", "ranged"])
                base_hp, base_damage = {"chaser": (20, 10), "random": (30, 5), "ranged": (15, 8)}[enemy_type]
                hp = int(base_hp * difficulty_mod)
                self.enemies.append({
                    "x": pos[0], "y": pos[1], "hp": hp, "max_hp": hp,
                    "damage": int(base_damage * difficulty_mod), "type": enemy_type,
                })
        
        for _ in range(10):
            if not floor_tiles: break
            self.gold_coins.append({"x": floor_tiles.pop()[0], "y": floor_tiles.pop()[1]})

    def _generate_dungeon(self, width, height, num_tiles):
        grid = [[1] * height for _ in range(width)]
        x, y = width // 2, height // 2
        grid[x][y] = 0
        
        floor_tiles = []
        for _ in range(num_tiles * 2): # More iterations to create complex paths
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                x, y = nx, ny
                if grid[x][y] == 1:
                    grid[x][y] = 0
        
        for r in range(width):
            for c in range(height):
                if grid[r][c] == 0:
                    floor_tiles.append((r, c))
                    
        return grid, floor_tiles

    def step(self, action):
        if self.game_over or self.victory:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Turn ---
        self.player["is_defending"] = (movement == 0 and not space_held)
        
        if space_held:
            # sfx: player_attack
            reward += self._player_attack(movement)
        elif movement != 0:
            reward += self._player_move(movement)
            
        # --- Enemy Turn ---
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy["hp"] <= 0:
                enemies_to_remove.append(enemy)
                continue
            self._enemy_ai(enemy)
        
        for enemy in enemies_to_remove:
            self.enemies.remove(enemy)

        # --- Update State & Particles ---
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.player["hp"] <= 0:
            # sfx: player_death
            self.game_over = True
            terminated = True
            reward -= 100.0
        elif self.victory:
            # sfx: victory_fanfare
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _player_attack(self, movement_action):
        reward = 0
        attack_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement_action, self.player["last_direction"])
        self.player["last_direction"] = attack_dir
        
        target_x, target_y = self.player["x"] + attack_dir[0], self.player["y"] + attack_dir[1]
        
        self._create_particle(target_x, target_y, self.COLOR_WHITE, 1, 0, 5, 'slash')

        for enemy in self.enemies:
            if enemy["x"] == target_x and enemy["y"] == target_y:
                damage = self.player["damage"]
                enemy["hp"] -= damage
                # sfx: enemy_hit
                self._create_particle(enemy["x"], enemy["y"], self.COLOR_ENEMY_CHASER, 5, 3, 15)
                self._create_damage_text(enemy["x"], enemy["y"], damage)
                
                if enemy["hp"] <= 0:
                    # sfx: enemy_die
                    if enemy["type"] == "boss":
                        reward += 100.0
                        self.victory = True
                    else:
                        reward += 1.0
        return reward

    def _player_move(self, movement_action):
        reward = 0
        move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement_action]
        self.player["last_direction"] = move_dir
        
        nx, ny = self.player["x"] + move_dir[0], self.player["y"] + move_dir[1]

        if self.dungeon[nx][ny] == 0: # Is a floor tile
            self.player["x"], self.player["y"] = nx, ny
            # sfx: player_step
        
        # Check for gold
        for coin in self.gold_coins[:]:
            if coin["x"] == self.player["x"] and coin["y"] == self.player["y"]:
                self.gold_coins.remove(coin)
                self.score += 1
                reward += 0.1
                # sfx: collect_gold
                self._create_particle(self.player["x"], self.player["y"], self.COLOR_GOLD, 8, 2, 10)
                break
        
        # Check for stairs
        if self.stairs_pos and self.player["x"] == self.stairs_pos[0] and self.player["y"] == self.stairs_pos[1]:
            self.current_level += 1
            # sfx: level_up
            self._setup_level()
        
        return reward

    def _enemy_ai(self, enemy):
        dist_to_player = math.hypot(enemy["x"] - self.player["x"], enemy["y"] - self.player["y"])

        # Attack if adjacent
        if dist_to_player < 1.5:
            damage = enemy["damage"]
            if self.player["is_defending"]:
                damage = max(1, damage // 2)
            self.player["hp"] -= damage
            # sfx: player_hit
            self._create_particle(self.player["x"], self.player["y"], self.COLOR_ENEMY_CHASER, 5, 3, 15)
            self._create_damage_text(self.player["x"], self.player["y"], damage, self.COLOR_ENEMY_CHASER)
            return

        # Movement AI
        if (enemy["type"] == "chaser" or enemy["type"] == "boss") and dist_to_player < 8:
            dx = np.sign(self.player["x"] - enemy["x"])
            dy = np.sign(self.player["y"] - enemy["y"])
            if abs(self.player["x"] - enemy["x"]) > abs(self.player["y"] - enemy["y"]):
                nx, ny = int(enemy["x"] + dx), enemy["y"]
                if self.dungeon[nx][ny] == 0: enemy["x"], enemy["y"] = nx, ny
            else:
                nx, ny = enemy["x"], int(enemy["y"] + dy)
                if self.dungeon[nx][ny] == 0: enemy["x"], enemy["y"] = nx, ny
        elif enemy["type"] == "random":
            dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
            nx, ny = enemy["x"] + dx, enemy["y"] + dy
            if self.dungeon[nx][ny] == 0: enemy["x"], enemy["y"] = nx, ny

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        cam_x = self.player["x"] * self.TILE_SIZE - self.SCREEN_WIDTH / 2
        cam_y = self.player["y"] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2

        start_col = max(0, int(cam_x / self.TILE_SIZE))
        end_col = min(len(self.dungeon), int((cam_x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 1)
        start_row = max(0, int(cam_y / self.TILE_SIZE))
        end_row = min(len(self.dungeon[0]), int((cam_y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 1)

        for x in range(start_col, end_col):
            for y in range(start_row, end_row):
                screen_x, screen_y = int(x * self.TILE_SIZE - cam_x), int(y * self.TILE_SIZE - cam_y)
                color = self.COLOR_WALL if self.dungeon[x][y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE))
        
        if self.stairs_pos:
            sx, sy = int(self.stairs_pos[0] * self.TILE_SIZE - cam_x), int(self.stairs_pos[1] * self.TILE_SIZE - cam_y)
            pygame.draw.rect(self.screen, self.COLOR_STAIRS, (sx, sy, self.TILE_SIZE, self.TILE_SIZE))

        for coin in self.gold_coins:
            cx, cy = int(coin["x"] * self.TILE_SIZE - cam_x), int(coin["y"] * self.TILE_SIZE - cam_y)
            pygame.draw.rect(self.screen, self.COLOR_GOLD, (cx + self.TILE_SIZE//3, cy + self.TILE_SIZE//3, self.TILE_SIZE//3, self.TILE_SIZE//3))

        for p in self.particles:
            px, py = int(p["x"] * self.TILE_SIZE - cam_x + self.TILE_SIZE//2), int(p["y"] * self.TILE_SIZE - cam_y + self.TILE_SIZE//2)
            if p['type'] == 'slash':
                dir_x, dir_y = self.player["last_direction"]
                len_slash = self.TILE_SIZE * 0.8 * (p['life'] / p['max_life'])
                start_pos = (px - dir_x * len_slash / 2, py - dir_y * len_slash / 2)
                end_pos = (px + dir_x * len_slash / 2, py + dir_y * len_slash / 2)
                pygame.draw.line(self.screen, p['color'], start_pos, end_pos, 4)
            elif p['type'] == 'text':
                alpha = int(255 * (p['life'] / p['max_life']))
                text_surf = self.font_small.render(p['text'], True, p['color'])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, (px, py - (p['max_life'] - p['life'])))
            else:
                radius = int(self.TILE_SIZE * 0.2 * (p['life'] / p['max_life']))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(px + p['vx']), int(py + p['vy']), radius, p['color'])

        for enemy in self.enemies:
            ex, ey = int(enemy["x"] * self.TILE_SIZE - cam_x), int(enemy["y"] * self.TILE_SIZE - cam_y)
            e_color = self.COLOR_ENEMY_CHASER
            if enemy['type'] == 'random': e_color = self.COLOR_ENEMY_RANDOM
            elif enemy['type'] == 'ranged': e_color = self.COLOR_ENEMY_RANGED
            elif enemy['type'] == 'boss': e_color = self.COLOR_BOSS
            
            size = self.TILE_SIZE * 1.5 if enemy['type'] == 'boss' else self.TILE_SIZE * 0.8
            offset = (self.TILE_SIZE - size) / 2
            pygame.draw.rect(self.screen, e_color, (ex + offset, ey + offset, size, size))
            self._draw_health_bar(ex, ey - 10, self.TILE_SIZE, 5, enemy["hp"], enemy["max_hp"])

        player_screen_x, player_screen_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        glow_radius = int(self.TILE_SIZE * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, player_screen_x, player_screen_y, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_screen_x - self.TILE_SIZE//3, player_screen_y - self.TILE_SIZE//3, self.TILE_SIZE*2//3, self.TILE_SIZE*2//3))
        
        # Facing direction indicator
        dir_x, dir_y = self.player["last_direction"]
        eye_x = player_screen_x + dir_x * self.TILE_SIZE // 4
        eye_y = player_screen_y + dir_y * self.TILE_SIZE // 4
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (eye_x-2, eye_y-2, 4, 4))
        
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        ui_surface = pygame.Surface((200, 100), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        
        hp_text = self.font_small.render(f"HP: {self.player['hp']}/{self.player['max_hp']}", True, self.COLOR_WHITE)
        gold_text = self.font_small.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        level_text = self.font_small.render(f"Level: {self.current_level}/{self.MAX_LEVELS}", True, self.COLOR_WHITE)
        
        ui_surface.blit(hp_text, (10, 10))
        ui_surface.blit(gold_text, (10, 35))
        ui_surface.blit(level_text, (10, 60))
        
        self.screen.blit(ui_surface, (10, 10))
        
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY_CHASER)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)
        elif self.victory:
            text = self.font_large.render("VICTORY!", True, self.COLOR_GOLD)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _draw_health_bar(self, x, y, width, height, hp, max_hp):
        if hp < max_hp:
            ratio = max(0, hp / max_hp)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (x, y, width, height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (x, y, int(width * ratio), height))

    def _create_particle(self, x, y, color, count, speed, life, p_type='burst'):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = random.uniform(0, speed)
            self.particles.append({
                "x": x, "y": y, "vx": math.cos(angle) * vel, "vy": math.sin(angle) * vel,
                "life": life, "max_life": life, "color": color, "type": p_type
            })

    def _create_damage_text(self, x, y, text, color=COLOR_WHITE):
        self.particles.append({
            "x": x, "y": y, "vx": 0, "vy": 0, "life": 30, "max_life": 30,
            "color": color, "type": "text", "text": str(text)
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['type'] != 'text' and p['type'] != 'slash':
                p['x'] += p['vx'] / self.TILE_SIZE
                p['y'] += p['vy'] / self.TILE_SIZE
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level, "player_hp": self.player["hp"]}

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
        assert not trunc
        assert isinstance(info, dict)
        
        # Test brief-specific constraints
        assert self.player['hp'] <= self.player['max_hp']
        assert self.score >= 0
        assert 1 <= self.current_level <= self.MAX_LEVELS

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # Default no-op
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                
                # For turn-based, we step on any key press
                if not done:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the environment to a Pygame window
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Upscale for better viewing if desired
        display_surface = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        display_surface.blit(render_surface, (0, 0))
        
        pygame.display.flip()
        
    env.close()