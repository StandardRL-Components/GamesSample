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
        "Controls: Arrow keys to move. Space to attack. No movement to defend. Shift also defends."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based dungeon crawler. Defeat enemies, gain experience, and descend 10 levels to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 32
    MAX_LEVEL = 10
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 15, 25)
    COLOR_FLOOR = (60, 40, 50)
    COLOR_WALL = (40, 30, 40)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_GLOW = (100, 255, 100)
    COLOR_EXIT = (200, 50, 255)
    COLOR_EXIT_GLOW = (255, 150, 255)

    ENEMY_COLORS = [
        ((200, 50, 50), (255, 100, 100)),  # Red
        ((50, 100, 200), (100, 150, 255)),  # Blue
        ((200, 200, 50), (255, 255, 100)),  # Yellow
        ((150, 50, 200), (200, 100, 255)),  # Purple
        ((255, 120, 50), (255, 180, 100)),  # Orange
    ]

    # --- UI Colors ---
    COLOR_HP_BAR = (220, 40, 40)
    COLOR_HP_BAR_BG = (80, 20, 20)
    COLOR_XP_BAR = (40, 100, 220)
    COLOR_XP_BAR_BG = (20, 40, 80)
    COLOR_TEXT = (230, 230, 230)
    COLOR_DEFEND_ICON = (100, 150, 255)

    class FloatingText:
        def __init__(self, text, pos, color, lifetime=60, speed=0.5):
            self.text = text
            self.x, self.y = pos
            self.color = color
            self.lifetime = lifetime
            self.initial_lifetime = lifetime
            self.speed = speed

        def update(self):
            self.y -= self.speed
            self.lifetime -= 1

        def draw(self, surface, font, camera_offset):
            if self.lifetime > 0:
                alpha = int(255 * (self.lifetime / self.initial_lifetime))
                alpha = max(0, min(255, alpha))

                text_surf = font.render(self.text, True, self.color)
                text_surf.set_alpha(alpha)

                draw_pos = (
                    int(self.x - camera_offset[0]),
                    int(self.y - camera_offset[1])
                )
                surface.blit(text_surf, draw_pos)

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.dungeon_level = 1
        self.player = None
        self.enemies = []
        self.dungeon = {}
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.floating_texts = []
        self.np_random = np.random.default_rng()

        self.validate_implementation()

    def _generate_dungeon(self, width, height):
        dungeon = {}
        for x in range(width):
            for y in range(height):
                dungeon[(x, y)] = 'wall'

        stack = deque()
        start_x, start_y = (
            self.np_random.integers(1, width // 2) * 2,
            self.np_random.integers(1, height // 2) * 2,
        )
        dungeon[(start_x, start_y)] = 'floor'
        stack.append((start_x, start_y))

        last_pos = (start_x, start_y)

        while stack:
            x, y = stack[-1]
            last_pos = (x, y)
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and dungeon.get((nx, ny)) == 'wall':
                    neighbors.append((nx, ny))

            if neighbors:
                nx_idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[nx_idx]
                dungeon[(nx, ny)] = 'floor'
                dungeon[(x + (nx - x) // 2, y + (ny - y) // 2)] = 'floor'
                stack.append((nx, ny))
            else:
                stack.pop()

        return dungeon, (start_x, start_y), last_pos

    def _generate_level(self):
        dungeon_width, dungeon_height = 50, 50
        self.dungeon, player_start, self.exit_pos = self._generate_dungeon(dungeon_width, dungeon_height)

        if self.dungeon_level == 1 or self.player is None:
            self.player = {
                "x": player_start[0], "y": player_start[1],
                "hp": 100, "max_hp": 100,
                "xp": 0, "level": 1,
                "defending": False
            }
        else:
            self.player["x"], self.player["y"] = player_start

        self.enemies.clear()
        num_enemies = 5 + self.dungeon_level * 2
        floor_tiles = [pos for pos, tile in self.dungeon.items() if
                       tile == 'floor' and pos != player_start and pos != self.exit_pos]

        if len(floor_tiles) < num_enemies:
            num_enemies = len(floor_tiles)

        if num_enemies > 0:
            enemy_indices = self.np_random.choice(len(floor_tiles), num_enemies, replace=False)
            enemy_positions = [floor_tiles[i] for i in enemy_indices]

            for pos in enemy_positions:
                enemy_type = self.np_random.integers(0, len(self.ENEMY_COLORS))
                enemy_level = self.dungeon_level
                max_hp = 20 + enemy_level * 5
                self.enemies.append({
                    "x": pos[0], "y": pos[1],
                    "hp": max_hp, "max_hp": max_hp,
                    "type": enemy_type,
                    "level": enemy_level,
                    "defending": False,
                    "damage": 5 + enemy_level
                })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.dungeon_level = 1
        self.floating_texts.clear()
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty per step to encourage efficiency
        self.player["defending"] = False
        for enemy in self.enemies:
            enemy["defending"] = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Player Turn ---
        player_action_taken = False
        if space_held:  # Attack
            player_action_taken = True
            attacked = False
            for enemy in self.enemies:
                if abs(enemy["x"] - self.player["x"]) <= 1 and abs(enemy["y"] - self.player["y"]) <= 1:
                    if enemy["x"] != self.player["x"] or enemy["y"] != self.player["y"]:
                        attacked = True
                        damage = 10 + self.player["level"] * 2
                        if enemy["defending"]: damage //= 2
                        enemy["hp"] -= damage

                        text_pos = (enemy["x"] * self.TILE_SIZE + self.TILE_SIZE // 2, enemy["y"] * self.TILE_SIZE)
                        self.floating_texts.append(self.FloatingText(str(damage), text_pos, (255, 200, 50)))

                        if enemy["hp"] <= 0:
                            reward += 10.0
                            xp_gain = 10 + enemy["type"] * 5
                            self.player["xp"] += xp_gain

                            text_pos = (self.player["x"] * self.TILE_SIZE + self.TILE_SIZE // 2,
                                        self.player["y"] * self.TILE_SIZE - 20)
                            self.floating_texts.append(self.FloatingText(f"+{xp_gain} XP", text_pos, (100, 150, 255)))

                            # Level up check
                            if self.player["xp"] >= 100:
                                self.player["xp"] -= 100
                                self.player["level"] += 1
                                self.player["max_hp"] += 20
                                self.player["hp"] = self.player["max_hp"]  # Full heal on level up
                                self.floating_texts.append(self.FloatingText("LEVEL UP!", (
                                self.player["x"] * self.TILE_SIZE, self.player["y"] * self.TILE_SIZE - 40),
                                                                             (255, 255, 100)))

            self.enemies = [e for e in self.enemies if e["hp"] > 0]

        elif movement in [1, 2, 3, 4]:  # Move
            player_action_taken = True
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            nx, ny = self.player["x"] + dx, self.player["y"] + dy

            is_occupied = any(e["x"] == nx and e["y"] == ny for e in self.enemies)
            if self.dungeon.get((nx, ny)) == 'floor' and not is_occupied:
                self.player["x"], self.player["y"] = nx, ny

        else:  # Defend (movement == 0) or shift_held
            player_action_taken = True
            self.player["defending"] = True

        # --- Enemy Turn ---
        if player_action_taken:
            for enemy in self.enemies:
                # Simple AI: 50% chance to attack if adjacent, 25% defend, 25% idle
                if abs(enemy["x"] - self.player["x"]) <= 1 and abs(enemy["y"] - self.player["y"]) <= 1:
                    ai_action = self.np_random.random()
                    if ai_action < 0.5:  # Attack
                        damage = enemy["damage"]
                        if self.player["defending"]: damage //= 2
                        self.player["hp"] -= damage

                        text_pos = (self.player["x"] * self.TILE_SIZE + self.TILE_SIZE // 2,
                                    self.player["y"] * self.TILE_SIZE)
                        self.floating_texts.append(self.FloatingText(str(damage), text_pos, (255, 80, 50)))
                    elif ai_action < 0.75:  # Defend
                        enemy["defending"] = True
                # Else: do nothing (too far away)

        # --- Update floating texts ---
        self.floating_texts = [ft for ft in self.floating_texts if ft.lifetime > 0]
        for ft in self.floating_texts:
            ft.update()

        # --- Check Game State ---
        terminated = False
        if self.player["hp"] <= 0:
            self.player["hp"] = 0
            reward = -10.0
            terminated = True
            self.game_over = True

        if (self.player["x"], self.player["y"]) == self.exit_pos:
            if self.dungeon_level >= self.MAX_LEVEL:
                reward += 100.0
                terminated = True
                self.game_over = True
            else:
                reward += 20.0
                self.dungeon_level += 1
                self._generate_level()
                self.floating_texts.append(self.FloatingText(f"Level {self.dungeon_level}", (
                self.player["x"] * self.TILE_SIZE, self.player["y"] * self.TILE_SIZE - 40), (255, 255, 100),
                                                              lifetime=90))

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        # player can be None during initialization before the first reset
        if self.player is None:
            return {
                "score": self.score,
                "steps": self.steps,
                "level": self.dungeon_level,
                "player_hp": 0,
                "player_xp": 0,
                "player_level": 0,
            }
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.dungeon_level,
            "player_hp": self.player["hp"],
            "player_xp": self.player["xp"],
            "player_level": self.player["level"],
        }

    def _render_game(self):
        if self.player is None: return

        # Calculate camera offset to center player
        cam_x = self.player["x"] * self.TILE_SIZE - self.SCREEN_WIDTH / 2 + self.TILE_SIZE / 2
        cam_y = self.player["y"] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2 + self.TILE_SIZE / 2
        camera_offset = (cam_x, cam_y)

        # Get visible tile range
        start_x = max(0, int(cam_x / self.TILE_SIZE))
        end_x = min(50, int((cam_x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 2)
        start_y = max(0, int(cam_y / self.TILE_SIZE))
        end_y = min(50, int((cam_y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 2)

        # Render dungeon
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                tile = self.dungeon.get((x, y), 'wall')
                color = self.COLOR_WALL if tile == 'wall' else self.COLOR_FLOOR
                rect = pygame.Rect(x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y, self.TILE_SIZE,
                                   self.TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        # Render exit
        if self.exit_pos:
            if self.dungeon_level == self.MAX_LEVEL:
                exit_screen_x = self.exit_pos[0] * self.TILE_SIZE - cam_x
                exit_screen_y = self.exit_pos[1] * self.TILE_SIZE - cam_y
                pulse = int((math.sin(self.steps * 0.1) + 1) / 2 * 5)
                pygame.gfxdraw.filled_circle(self.screen, int(exit_screen_x + self.TILE_SIZE // 2),
                                             int(exit_screen_y + self.TILE_SIZE // 2), self.TILE_SIZE // 2 + pulse,
                                             self.COLOR_EXIT_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, int(exit_screen_x + self.TILE_SIZE // 2),
                                             int(exit_screen_y + self.TILE_SIZE // 2), self.TILE_SIZE // 2,
                                             self.COLOR_EXIT)
            else:
                exit_screen_x = self.exit_pos[0] * self.TILE_SIZE - cam_x
                exit_screen_y = self.exit_pos[1] * self.TILE_SIZE - cam_y
                pygame.draw.rect(self.screen, self.COLOR_EXIT,
                                 (exit_screen_x, exit_screen_y, self.TILE_SIZE, self.TILE_SIZE))

        # Render enemies
        for enemy in self.enemies:
            enemy_screen_x = enemy["x"] * self.TILE_SIZE - cam_x
            enemy_screen_y = enemy["y"] * self.TILE_SIZE - cam_y
            color, glow_color = self.ENEMY_COLORS[enemy["type"]]

            rect = pygame.Rect(enemy_screen_x + 4, enemy_screen_y + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            pygame.draw.rect(self.screen, glow_color, rect.inflate(4, 4), border_radius=4)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

            # Health bar
            hp_ratio = enemy["hp"] / enemy["max_hp"]
            bar_w = self.TILE_SIZE - 8
            pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG, (enemy_screen_x + 4, enemy_screen_y - 8, bar_w, 5))
            pygame.draw.rect(self.screen, self.COLOR_HP_BAR,
                             (enemy_screen_x + 4, enemy_screen_y - 8, bar_w * hp_ratio, 5))

            if enemy["defending"]:
                shield_pos = (int(enemy_screen_x + self.TILE_SIZE / 2), int(enemy_screen_y + self.TILE_SIZE / 2))
                pygame.draw.circle(self.screen, self.COLOR_DEFEND_ICON, shield_pos, self.TILE_SIZE // 2, 2)

        # Render player
        player_screen_x = self.SCREEN_WIDTH / 2 - self.TILE_SIZE / 2
        player_screen_y = self.SCREEN_HEIGHT / 2 - self.TILE_SIZE / 2
        player_rect = pygame.Rect(player_screen_x + 4, player_screen_y + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, player_rect.inflate(6, 6), border_radius=6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=6)

        if self.player["defending"]:
            shield_pos = (int(player_screen_x + self.TILE_SIZE / 2), int(player_screen_y + self.TILE_SIZE / 2))
            pygame.draw.circle(self.screen, self.COLOR_DEFEND_ICON, shield_pos, self.TILE_SIZE // 2, 3)

        # Render floating texts
        for ft in self.floating_texts:
            ft.draw(self.screen, self.font_medium, camera_offset)

    def _render_ui(self):
        if self.player is None: return

        # Player HP Bar
        hp_ratio = self.player["hp"] / self.player["max_hp"]
        bar_w = 200
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG, (10, 10, bar_w, 20), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR, (10, 10, bar_w * hp_ratio, 20), border_radius=4)
        hp_text = self.font_small.render(f'HP: {self.player["hp"]}/{self.player["max_hp"]}', True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (15, 12))

        # Player XP Bar
        xp_ratio = self.player["xp"] / 100
        pygame.draw.rect(self.screen, self.COLOR_XP_BAR_BG, (10, 35, bar_w, 15), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_XP_BAR, (10, 35, bar_w * xp_ratio, 15), border_radius=4)
        xp_text = self.font_small.render(f'XP', True, self.COLOR_TEXT)
        self.screen.blit(xp_text, (15, 36))

        # Player Level
        level_text = self.font_medium.render(f'LVL: {self.player["level"]}', True, self.COLOR_TEXT)
        self.screen.blit(level_text, (220, 10))

        # Dungeon Level
        dungeon_level_text = self.font_medium.render(f'Dungeon Level: {self.dungeon_level}', True, self.COLOR_TEXT)
        text_rect = dungeon_level_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(dungeon_level_text, text_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.player["hp"] <= 0:
                end_text = self.font_large.render("YOU DIED", True, self.COLOR_HP_BAR)
            elif self.dungeon_level >= self.MAX_LEVEL and (self.player["x"], self.player["y"]) == self.exit_pos:
                end_text = self.font_large.render("VICTORY!", True, self.COLOR_EXIT_GLOW)
            else: # Truncated
                end_text = self.font_large.render("TIME UP", True, self.COLOR_TEXT)


            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test reset - This MUST come first to initialize the environment state
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space - Now that the env is reset, this will work
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")