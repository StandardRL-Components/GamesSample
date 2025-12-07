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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Space to attack the nearest enemy. "
        "Reach the stairs to advance to the next level."
    )

    # User-facing game description
    game_description = (
        "A grid-based dungeon crawler. Battle enemies, gain experience, and "
        "descend through three levels to defeat the final boss."
    )

    # Frames wait for user input
    auto_advance = False

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    TILE_SIZE = 32  # 640/20=32, 400/12.5 -> use 384 height, 16px padding

    # Colors
    COLOR_BG = (10, 10, 15)
    COLOR_FLOOR = (50, 45, 40)
    COLOR_FLOOR_BORDER = (40, 35, 30)
    COLOR_WALL = (80, 70, 60)
    COLOR_WALL_BORDER = (70, 60, 50)
    COLOR_PLAYER = (60, 180, 75)
    COLOR_PLAYER_BORDER = (40, 120, 55)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_ENEMY_BORDER = (150, 15, 50)
    COLOR_BOSS = (255, 100, 150)
    COLOR_BOSS_BORDER = (180, 50, 90)
    COLOR_STAIRS = (255, 225, 25)
    COLOR_STAIRS_BORDER = (200, 175, 20)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HP_BAR_BG = (70, 20, 20)
    COLOR_HP_BAR_FILL = (200, 40, 40)
    COLOR_XP_BAR_BG = (20, 20, 70)
    COLOR_XP_BAR_FILL = (40, 40, 200)

    # Player Stats
    PLAYER_MAX_HP = 100
    PLAYER_ATTACK_DMG = 20

    # Enemy Stats
    BASE_ENEMY_HP = 50
    BASE_ENEMY_DMG = 10

    # Boss Stats
    BOSS_HP = 200
    BOSS_DMG = 25

    # Game Rules
    MAX_LEVEL = 3
    MAX_STEPS = 1000

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.render_mode = render_mode
        self.grid = []
        self.player = {}
        self.enemies = []
        self.boss = None
        self.stairs_pos = None
        self.particles = []
        self.damage_popups = []

        # The reset call here is important to initialize the game state
        # before validation.
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.level = 1

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Reset level-specific state
        self.particles = []
        self.damage_popups = []

        # 1. Fill grid with walls
        self.grid = [[1 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]

        # 2. Carve out a random path
        start_x, start_y = self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 1)
        self.grid[start_y][start_x] = 0

        path_length = self.GRID_WIDTH * self.GRID_HEIGHT // 2
        px, py = start_x, start_y
        for _ in range(path_length):
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.np_random.shuffle(dirs)
            moved = False
            for dx, dy in dirs:
                nx, ny = px + dx * 2, py + dy * 2
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and self.grid[ny][nx] == 1:
                    self.grid[py + dy][px + dx] = 0
                    self.grid[ny][nx] = 0
                    px, py = nx, ny
                    moved = True
                    break
            if not moved:  # If stuck, jump to a random floor tile
                floor_tiles = [(x, y) for y, row in enumerate(self.grid) for x, tile in enumerate(row) if tile == 0]
                if floor_tiles:
                    px, py = self.np_random.choice(floor_tiles)

        # 3. Place player
        floor_tiles = [(x, y) for y, row in enumerate(self.grid) for x, tile in enumerate(row) if tile == 0]
        if not floor_tiles:  # Failsafe
            self._generate_level() # Recurse if map generation fails
            return

        player_pos = floor_tiles[self.np_random.integers(len(floor_tiles))]
        self.player = {
            "pos": player_pos,
            "hp": self.PLAYER_MAX_HP,
            "max_hp": self.PLAYER_MAX_HP,
            "xp": 0,
            "xp_to_level": 100,  # Not used for leveling up, just a visual
            "damage": self.PLAYER_ATTACK_DMG,
            "flash_timer": 0
        }
        floor_tiles.remove(player_pos)

        # 4. Place stairs or boss
        if self.level < self.MAX_LEVEL:
            # Place stairs far from player
            if floor_tiles:
                distances = [math.hypot(x - player_pos[0], y - player_pos[1]) for x, y in floor_tiles]
                self.stairs_pos = floor_tiles[np.argmax(distances)]
                floor_tiles.remove(self.stairs_pos)
            else:
                self.stairs_pos = None
            self.boss = None
        else:  # Level 3: Place boss
            self.stairs_pos = None
            boss_hp = self.BOSS_HP * (1 + 0.2 * (self.level - 1))
            boss_dmg = self.BOSS_DMG * (1 + 0.2 * (self.level - 1))

            # Find a large open area for the boss
            boss_candidates = [
                (x, y) for x, y in floor_tiles
                if math.hypot(x - player_pos[0], y - player_pos[1]) > 5
            ]
            if not boss_candidates:
                boss_candidates = floor_tiles

            if boss_candidates:
                boss_pos = boss_candidates[self.np_random.integers(len(boss_candidates))]
                floor_tiles.remove(boss_pos)
                self.boss = {
                    "pos": boss_pos,
                    "hp": boss_hp,
                    "max_hp": boss_hp,
                    "damage": boss_dmg,
                    "flash_timer": 0
                }
            else:
                self.boss = None

        # 5. Place enemies
        self.enemies = []
        num_enemies = self.np_random.integers(3, 6)
        enemy_hp = self.BASE_ENEMY_HP * (1 + 0.2 * (self.level - 1))
        enemy_dmg = self.BASE_ENEMY_DMG * (1 + 0.2 * (self.level - 1))

        for _ in range(num_enemies):
            if not floor_tiles: break
            # Place enemies away from player start
            valid_spawns = [(x, y) for x, y in floor_tiles if math.hypot(x - player_pos[0], y - player_pos[1]) > 3]
            if not valid_spawns: continue

            pos = valid_spawns[self.np_random.integers(len(valid_spawns))]
            floor_tiles.remove(pos)
            self.enemies.append({
                "pos": pos,
                "hp": enemy_hp,
                "max_hp": enemy_hp,
                "damage": enemy_dmg,
                "flash_timer": 0,
                "patrol_dir": random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            })

    def _next_level(self):
        self.level += 1
        self.score += 25  # Reward for completing a level
        self._generate_level()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for each step to encourage efficiency

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        player_acted = False

        # --- Player Turn ---
        # Action: Attack
        if space_held:
            player_acted = True
            all_targets = self.enemies + ([self.boss] if self.boss else [])
            if all_targets:
                # Find nearest target
                player_x, player_y = self.player["pos"]
                targets_by_dist = sorted(
                    all_targets,
                    key=lambda t: math.hypot(t["pos"][0] - player_x, t["pos"][1] - player_y)
                )
                nearest_target = targets_by_dist[0]
                dist = math.hypot(nearest_target["pos"][0] - player_x, nearest_target["pos"][1] - player_y)

                # Attack if in range (adjacent)
                if dist < 1.5:
                    damage = self.player["damage"]
                    nearest_target["hp"] -= damage
                    nearest_target["flash_timer"] = 3
                    self._create_damage_popup(damage, nearest_target["pos"])
                    self._create_attack_particles(self.player["pos"], nearest_target["pos"], self.COLOR_PLAYER)
                else:
                    reward -= 0.5  # Penalty for attacking nothing
            else:
                reward -= 0.5  # Penalty for attacking nothing

        # Action: Movement
        elif movement > 0:
            player_acted = True
            px, py = self.player["pos"]
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            nx, ny = px + dx, py + dy

            # Check if player is adjacent to an enemy
            is_adjacent_to_enemy = False
            for enemy in self.enemies + ([self.boss] if self.boss else []):
                if math.hypot(enemy["pos"][0] - px, enemy["pos"][1] - py) < 1.5:
                    is_adjacent_to_enemy = True
                    break
            if is_adjacent_to_enemy:
                reward -= 0.2  # Penalty for moving away when attack is possible

            # Check for valid move
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny][nx] == 0:
                is_occupied = any(e["pos"] == (nx, ny) for e in self.enemies)
                if self.boss and self.boss["pos"] == (nx, ny):
                    is_occupied = True

                if not is_occupied:
                    self.player["pos"] = (nx, ny)

        if player_acted:
            # --- Enemy Turn ---
            all_combatants = self.enemies + ([self.boss] if self.boss else [])
            for entity in all_combatants:
                if entity["hp"] <= 0: continue

                ex, ey = entity["pos"]
                px, py = self.player["pos"]

                # Attack if adjacent
                if math.hypot(ex - px, ey - py) < 1.5:
                    damage = entity["damage"]
                    self.player["hp"] -= damage
                    self.player["flash_timer"] = 3
                    self._create_damage_popup(damage, self.player["pos"])
                    self._create_attack_particles(entity["pos"], self.player["pos"], self.COLOR_ENEMY)
                else:  # Move
                    # Boss moves towards player
                    if entity.get("max_hp") >= self.BOSS_HP:
                        if self.np_random.random() > 0.3:  # Boss doesn't always move
                            best_dir = (0, 0)
                            min_dist = math.hypot(ex - px, ey - py)
                            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nx, ny = ex + dx, ey + dy
                                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny][nx] == 0:
                                    if not any(e["pos"] == (nx, ny) for e in all_combatants if e is not entity):
                                        dist = math.hypot(nx - px, ny - py)
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_dir = (dx, dy)
                            if best_dir != (0, 0):
                                entity["pos"] = (ex + best_dir[0], ey + best_dir[1])

                    # Regular enemies patrol
                    else:
                        dx, dy = entity["patrol_dir"]
                        nx, ny = ex + dx, ey + dy
                        occupied_by_other = any(e["pos"] == (nx, ny) for e in all_combatants if e is not entity) or self.player["pos"] == (nx, ny)
                        if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny][nx] == 0 and not occupied_by_other):
                            # Reverse direction on collision
                            entity["patrol_dir"] = (-dx, -dy)
                        else:
                            entity["pos"] = (nx, ny)

        # --- State Updates ---
        # Remove dead enemies
        defeated_enemies = [e for e in self.enemies if e["hp"] <= 0]
        for e in defeated_enemies:
            reward += 10
            self.score += 10
            self.player["xp"] += 25
            self.enemies.remove(e)

        # Check for boss defeat
        if self.boss and self.boss["hp"] <= 0:
            reward += 50
            self.score += 50
            if self.level == self.MAX_LEVEL:
                reward += 100  # Final boss victory bonus
                self.score += 100
                self.victory = True
                self.game_over = True
            self.boss = None

        # Check for level transition
        if self.stairs_pos and self.player["pos"] == self.stairs_pos:
            if self.level < self.MAX_LEVEL:
                self._next_level()
                reward += 25
            else:  # Should not happen, but as a failsafe
                self.victory = True
                self.game_over = True

        # Update timers
        if self.player["flash_timer"] > 0: self.player["flash_timer"] -= 1
        for e in self.enemies:
            if e["flash_timer"] > 0: e["flash_timer"] -= 1
        if self.boss and self.boss["flash_timer"] > 0: self.boss["flash_timer"] -= 1

        self.steps += 1

        # Check termination conditions
        if self.player["hp"] <= 0:
            self.game_over = True
            reward = -100  # Large penalty for dying

        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "hp": self.player["hp"], "xp": self.player["xp"]}

    def _render_game(self):
        # Calculate camera offset to center player
        cam_x = self.player["pos"][0] * self.TILE_SIZE - self.SCREEN_WIDTH / 2 + self.TILE_SIZE / 2
        cam_y = self.player["pos"][1] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2 + self.TILE_SIZE / 2

        # Draw grid
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                screen_x, screen_y = int(x * self.TILE_SIZE - cam_x), int(y * self.TILE_SIZE - cam_y)
                rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
                border_rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE - 1, self.TILE_SIZE - 1)

                if tile == 1:  # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_BORDER, border_rect)
                else:  # Floor
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR_BORDER, border_rect)

        # Draw stairs
        if self.stairs_pos:
            sx, sy = self.stairs_pos
            screen_x, screen_y = int(sx * self.TILE_SIZE - cam_x), int(sy * self.TILE_SIZE - cam_y)
            rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_STAIRS, rect)
            pygame.draw.rect(self.screen, self.COLOR_STAIRS_BORDER, pygame.Rect(screen_x, screen_y, self.TILE_SIZE - 1, self.TILE_SIZE - 1))
            for i in range(4):
                pygame.draw.line(self.screen, (0, 0, 0), (screen_x + i * 8, screen_y), (screen_x, screen_y + i * 8), 2)
                pygame.draw.line(self.screen, (0, 0, 0), (screen_x + self.TILE_SIZE, screen_y + self.TILE_SIZE - i * 8), (screen_x + self.TILE_SIZE - i * 8, screen_y + self.TILE_SIZE), 2)

        # Draw entities (enemies, boss, player)
        all_entities = self.enemies + ([self.boss] if self.boss else []) + [self.player]
        for entity in all_entities:
            is_player = "xp" in entity
            is_boss = "max_hp" in entity and entity["max_hp"] >= self.BOSS_HP

            color = self.COLOR_PLAYER if is_player else (self.COLOR_BOSS if is_boss else self.COLOR_ENEMY)
            border_color = self.COLOR_PLAYER_BORDER if is_player else (self.COLOR_BOSS_BORDER if is_boss else self.COLOR_ENEMY_BORDER)
            size_mod = 1.2 if is_boss else 1.0

            ex, ey = entity["pos"]
            screen_x = int(ex * self.TILE_SIZE - cam_x)
            screen_y = int(ey * self.TILE_SIZE - cam_y)

            if entity.get("flash_timer", 0) > 0:
                color = (255, 255, 255)
                border_color = (200, 200, 200)

            size = int(self.TILE_SIZE * 0.7 * size_mod)
            offset = (self.TILE_SIZE - size) // 2
            rect = pygame.Rect(screen_x + offset, screen_y + offset, size, size)
            border_rect = pygame.Rect(screen_x + offset, screen_y + offset, size - 1, size - 1)
            pygame.draw.rect(self.screen, border_color, rect)
            pygame.draw.rect(self.screen, color, border_rect.inflate(-4, -4))

            # Draw health bar
            if "max_hp" in entity and entity["hp"] > 0:
                hp_pct = max(0, entity["hp"] / entity["max_hp"])
                bar_width = int(self.TILE_SIZE * 0.8)
                bar_height = 5
                bar_x = screen_x + (self.TILE_SIZE - bar_width) // 2
                bar_y = screen_y - bar_height - 2
                pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.COLOR_HP_BAR_FILL, (bar_x, bar_y, int(bar_width * hp_pct), bar_height))

        # Draw particles and damage popups
        self._update_and_draw_particles(cam_x, cam_y)
        self._update_and_draw_damage_popups(cam_x, cam_y)

    def _render_ui(self):
        # Level and Score
        level_text = self.font_small.render(f"Level: {self.level}/{self.MAX_LEVEL}", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            message = "VICTORY!" if self.victory else "GAME OVER"
            msg_text = self.font_large.render(message, True, self.COLOR_STAIRS if self.victory else self.COLOR_ENEMY)
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

    def _create_attack_particles(self, start_pos, end_pos, color):
        for _ in range(10):
            start_x = (start_pos[0] + 0.5) * self.TILE_SIZE
            start_y = (start_pos[1] + 0.5) * self.TILE_SIZE
            end_x = (end_pos[0] + 0.5) * self.TILE_SIZE
            end_y = (end_pos[1] + 0.5) * self.TILE_SIZE

            vec_x, vec_y = end_x - start_x, end_y - start_y
            dist = math.hypot(vec_x, vec_y)
            if dist == 0: continue

            vel_x = (vec_x / dist) * self.np_random.uniform(5, 10) + self.np_random.uniform(-2, 2)
            vel_y = (vec_y / dist) * self.np_random.uniform(5, 10) + self.np_random.uniform(-2, 2)

            self.particles.append({
                "x": start_x, "y": start_y,
                "vx": vel_x, "vy": vel_y,
                "lifespan": 10, "color": color
            })

    def _create_damage_popup(self, damage, pos):
        x = (pos[0] + 0.5) * self.TILE_SIZE
        y = pos[1] * self.TILE_SIZE
        self.damage_popups.append({
            "x": x, "y": y, "text": str(int(damage)),
            "lifespan": 20, "vy": -1
        })

    def _update_and_draw_particles(self, cam_x, cam_y):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
            else:
                size = int(p["lifespan"] / 2)
                px, py = int(p["x"] - cam_x), int(p["y"] - cam_y)
                pygame.draw.rect(self.screen, p["color"], (px, py, size, size))

    def _update_and_draw_damage_popups(self, cam_x, cam_y):
        for pop in self.damage_popups[:]:
            pop["y"] += pop["vy"]
            pop["lifespan"] -= 1
            if pop["lifespan"] <= 0:
                self.damage_popups.remove(pop)
            else:
                alpha = int(255 * (pop["lifespan"] / 20))
                text_surf = self.font_small.render(pop["text"], True, self.COLOR_ENEMY)
                text_surf.set_alpha(alpha)
                px, py = int(pop["x"] - cam_x - text_surf.get_width() / 2), int(pop["y"] - cam_y)
                self.screen.blit(text_surf, (px, py))

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
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
        assert trunc is False
        assert isinstance(info, dict)

        # Test game logic assertions
        self.reset()
        assert self.player["hp"] <= self.PLAYER_MAX_HP, "Player HP should not exceed max"
        assert self.level <= self.MAX_LEVEL

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    terminated = False
    total_reward = 0

    # Game loop
    running = True
    while running:
        action = [0, 0, 0]  # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()

            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4

            # Space
            if keys[pygame.K_SPACE]:
                action[1] = 1

            # Shift (unused in this game)
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need a small delay to make it playable
        pygame.time.wait(100)  # 10 FPS for manual play

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)  # Pause for 3 seconds on game over
            obs, info = env.reset()
            terminated = False
            total_reward = 0

    pygame.quit()