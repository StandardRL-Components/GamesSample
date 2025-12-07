import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    user_guide = (
        "Controls: Arrow keys to move. Space to attack. Shift for a powerful special attack (has a cooldown)."
    )

    game_description = (
        "Explore a procedural dungeon, fight monsters, and collect loot. Defeat the boss on level 3 to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_SIZE = 32
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (40, 40, 60)
    COLOR_FLOOR = (80, 80, 100)
    COLOR_PLAYER = (50, 200, 255)
    COLOR_PLAYER_GLOW = (50, 200, 255, 50)
    COLOR_EXIT = (200, 0, 255)
    COLOR_EXIT_GLOW = (200, 0, 255, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR = (40, 200, 60)
    COLOR_HEALTH_BAR_BG = (200, 40, 40)
    COLOR_XP_BAR = (60, 120, 255)
    COLOR_XP_BAR_BG = (40, 40, 80)
    COLOR_COOLDOWN_BAR = (255, 165, 0)
    COLOR_COOLDOWN_BAR_BG = (80, 60, 20)

    ENEMY_COLORS = [
        (255, 80, 80),  # Goblin (Red)
        (80, 255, 80),  # Slime (Green)
        (255, 255, 80),  # Skeleton (Yellow)
        (255, 80, 255),  # Bat (Purple)
        (80, 255, 255),  # Ghost (Cyan)
    ]
    LOOT_COLORS = {
        "potion": (100, 255, 100),
        "xp": (100, 100, 255),
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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        self.game_state = "PLAYING"
        # self.reset() is called at the end of init to ensure all attributes are set up

        self.validate_implementation_before_reset()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = "PLAYING"
        self.level = 1

        self.player = Player(self.np_random)
        self._generate_level()

        self.particles = []
        self.floating_texts = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Player Turn ---
        action_taken = self._handle_player_action(action)
        if action_taken:
            reward += action_taken.get("reward", 0)
            if "xp" in action_taken:
                xp_gain = action_taken["xp"]
                leveled_up, level_up_reward = self.player.gain_xp(xp_gain)
                if leveled_up:
                    reward += level_up_reward
                    self.floating_texts.append(
                        FloatingText(self.player.px, self.player.py, "LEVEL UP!", (255, 223, 0), 60))

        # --- Enemy Turn ---
        if action_taken:
            self._handle_enemy_actions()

        # --- State Updates ---
        self.player.update()
        self._update_particles()
        self._update_floating_texts()

        if self.player.is_on_exit(self.exit_pos) and not self.enemies:
            if self.level < 3:
                self.level += 1
                self._generate_level()
                self.floating_texts.append(
                    FloatingText(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, f"LEVEL {self.level}", (255, 255, 255),
                                 90))
            else:  # Should not happen as boss is on level 3
                pass

        # --- Termination Check ---
        terminated = False
        if self.player.health <= 0:
            terminated = True
            self.game_over = True
            self.game_state = "GAME OVER"
            reward = -100
        elif self.boss and self.boss.health <= 0:
            terminated = True
            self.game_over = True
            self.game_state = "VICTORY"
            reward += 100  # Final boss defeat reward
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_state = "TIME UP"

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        action_taken = None

        target_enemy = self._get_adjacent_enemy()

        if shift_pressed and self.player.special_cooldown == 0 and target_enemy:
            # Special Attack
            damage = self.player.attack_power * 2
            target_enemy.take_damage(damage)
            self.player.special_cooldown = self.player.max_special_cooldown
            self.floating_texts.append(
                FloatingText(target_enemy.px, target_enemy.py, str(damage), (255, 165, 0), 40, dy=-2))
            self._create_particles(target_enemy.px, target_enemy.py, (255, 165, 0), 20, 8)
            # sfx: special_attack
            action_taken = {"type": "special_attack"}
        elif space_pressed and target_enemy:
            # Normal Attack
            damage = self.player.attack_power
            target_enemy.take_damage(damage)
            self.floating_texts.append(
                FloatingText(target_enemy.px, target_enemy.py, str(damage), (255, 255, 255), 30, dy=-1.5))
            self._create_particles(target_enemy.px, target_enemy.py, (200, 200, 200), 10, 4)
            # sfx: normal_attack
            action_taken = {"type": "attack"}
        elif movement > 0:
            # Movement
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            nx, ny = self.player.x + dx, self.player.y + dy
            if self._is_walkable(nx, ny):
                self.player.move(nx, ny)
                action_taken = {"type": "move"}
                # sfx: step
        else:  # No-op
            action_taken = {"type": "wait"}

        # Post-action processing (loot, enemy death)
        if action_taken:
            # Check for loot
            loot_reward = self._check_loot_pickup()
            if loot_reward:
                action_taken["reward"] = action_taken.get("reward", 0) + loot_reward

            # Check for enemy death
            xp_gain = 0
            for enemy in self.enemies[:]:
                if enemy.health <= 0:
                    xp_gain += enemy.xp_value
                    self.enemies.remove(enemy)
                    if enemy == self.boss:
                        self.boss = None
                        action_taken["reward"] = action_taken.get("reward", 0) + 10  # Boss defeat reward
                    else:
                        action_taken["reward"] = action_taken.get("reward", 0) + 0.1  # Enemy defeat reward
                    # sfx: enemy_die

            if xp_gain > 0:
                action_taken["xp"] = xp_gain

        return action_taken

    def _handle_enemy_actions(self):
        for enemy in self.enemies:
            is_adjacent = abs(self.player.x - enemy.x) + abs(self.player.y - enemy.y) == 1
            if is_adjacent:
                damage = enemy.get_attack_damage()
                if damage > 0:
                    self.player.take_damage(damage)
                    self.floating_texts.append(
                        FloatingText(self.player.px, self.player.py, str(damage), (255, 80, 80), 30, dy=-1.5))
                    self._create_particles(self.player.px, self.player.py, (255, 80, 80), 10, 4)
                    # sfx: player_hurt
                else:  # Boss is defending
                    self.floating_texts.append(FloatingText(enemy.px, enemy.py, "DEFEND", (150, 150, 255), 30))
                enemy.update_attack_pattern()

    def _check_loot_pickup(self):
        for item in self.loot_items[:]:
            if self.player.x == item.x and self.player.y == item.y:
                reward = 0.5
                if item.type == "potion":
                    healed = self.player.heal(item.value)
                    self.floating_texts.append(
                        FloatingText(self.player.px, self.player.py, f"+{healed} HP", (80, 255, 80), 30))
                    # sfx: potion_pickup
                elif item.type == "xp":
                    leveled_up, level_up_reward = self.player.gain_xp(item.value)
                    self.floating_texts.append(
                        FloatingText(self.player.px, self.player.py, f"+{item.value} XP", (80, 80, 255), 30))
                    if leveled_up:
                        reward += level_up_reward
                        self.floating_texts.append(
                            FloatingText(self.player.px, self.player.py, "LEVEL UP!", (255, 223, 0), 60))
                    # sfx: xp_pickup
                self.loot_items.remove(item)
                return reward
        return 0

    def _get_adjacent_enemy(self):
        for enemy in self.enemies:
            if abs(self.player.x - enemy.x) + abs(self.player.y - enemy.y) == 1:
                return enemy
        return None

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)  # 0=wall, 1=floor
        self.enemies = []
        self.loot_items = []
        self.boss = None

        if self.level == 3:
            # Boss level: large room
            self.grid[2:-2, 2:-2] = 1
            start_x, start_y = 3, self.GRID_HEIGHT // 2
            self.exit_pos = None  # No exit on boss level

            # Place Boss
            boss_x, boss_y = self.GRID_WIDTH - 4, self.GRID_HEIGHT // 2
            self.boss = Boss(boss_x, boss_y, self.np_random)
            self.enemies.append(self.boss)
        else:
            # Procedural level
            px, py = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
            self.grid[py, px] = 1
            num_tiles = (self.GRID_WIDTH * self.GRID_HEIGHT) // 3
            for _ in range(num_tiles * 2):
                nx, ny = px, py
                d = self.np_random.integers(4)
                if d == 0:
                    nx += 1  # Right
                elif d == 1:
                    nx -= 1  # Left
                elif d == 2:
                    ny += 1  # Down
                else:
                    ny -= 1  # Up

                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1:
                    px, py = nx, ny
                    self.grid[py, px] = 1

            floor_tiles = list(zip(*np.where(self.grid == 1)))
            start_pos = random.choice(floor_tiles)
            start_y, start_x = start_pos[0], start_pos[1]

            # Place Exit
            possible_exits = [pos for pos in floor_tiles if
                              abs(pos[1] - start_x) + abs(pos[0] - start_y) > (self.GRID_WIDTH + self.GRID_HEIGHT) / 4]
            self.exit_pos = random.choice(possible_exits) if possible_exits else random.choice(floor_tiles)

            # Place Enemies
            num_enemies = 3 + self.level
            for _ in range(num_enemies):
                pos = random.choice(floor_tiles)
                if pos != (start_y, start_x) and pos != self.exit_pos and not any(
                        e.x == pos[1] and e.y == pos[0] for e in self.enemies):
                    enemy_type_idx = self.np_random.integers(len(self.ENEMY_COLORS))
                    self.enemies.append(Enemy(pos[1], pos[0], enemy_type_idx, self.level, self.np_random))

            # Place Loot
            num_loot = 2
            for _ in range(num_loot):
                pos = random.choice(floor_tiles)
                if pos != (start_y, start_x) and pos != self.exit_pos and not any(
                        e.x == pos[1] and e.y == pos[0] for e in self.enemies) and not any(
                        l.x == pos[1] and l.y == pos[0] for l in self.loot_items):
                    loot_type = "potion" if self.np_random.random() > 0.5 else "xp"
                    self.loot_items.append(Loot(pos[1], pos[0], loot_type))

        self.player.set_pos(start_x, start_y)

    def _is_walkable(self, x, y):
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        if self.grid[y, x] == 0:
            return False
        if any(e.x == x and e.y == y for e in self.enemies):
            return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_state != "PLAYING":
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Center the grid view
        cam_x = self.SCREEN_WIDTH // 2 - self.GRID_WIDTH * self.TILE_SIZE // 2
        cam_y = self.SCREEN_HEIGHT // 2 - self.GRID_HEIGHT * self.TILE_SIZE // 2 - 20  # Shift up for UI

        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (cam_x + x * self.TILE_SIZE, cam_y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_FLOOR if self.grid[y, x] == 1 else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        if self.exit_pos:
            ex, ey = self.exit_pos
            px = int(cam_x + ex * self.TILE_SIZE + self.TILE_SIZE // 2)
            py = int(cam_y + ey * self.TILE_SIZE + self.TILE_SIZE // 2)
            radius = self.TILE_SIZE // 3
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_EXIT)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_EXIT)
            glow_radius = int(radius * (0.8 + 0.2 * math.sin(self.steps * 0.2)))
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_EXIT_GLOW)

        # Draw loot
        for item in self.loot_items:
            px = int(cam_x + item.x * self.TILE_SIZE + self.TILE_SIZE // 2)
            py = int(cam_y + item.y * self.TILE_SIZE + self.TILE_SIZE // 2)
            color = self.LOOT_COLORS[item.type]
            size = self.TILE_SIZE // 4
            pygame.draw.rect(self.screen, color, (px - size, py - size, size * 2, size * 2))

        # Draw enemies
        for enemy in self.enemies:
            enemy.update_anim(self.steps)
            px = int(cam_x + enemy.x * self.TILE_SIZE + self.TILE_SIZE // 2)
            py = int(cam_y + enemy.y * self.TILE_SIZE + self.TILE_SIZE // 2 + enemy.anim_offset)
            size = self.TILE_SIZE // 3 if not isinstance(enemy, Boss) else self.TILE_SIZE // 2
            pygame.draw.rect(self.screen, enemy.color, (px - size, py - size, size * 2, size * 2))
            self._draw_health_bar(px, py - size - 8, enemy.health, enemy.max_health, 24)

        # Draw player
        self.player.update_anim(self.steps)
        px = int(cam_x + self.player.x * self.TILE_SIZE + self.TILE_SIZE // 2)
        py = int(cam_y + self.player.y * self.TILE_SIZE + self.TILE_SIZE // 2 + self.player.anim_offset)
        self.player.px, self.player.py = px, py
        size = self.TILE_SIZE // 3
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, px, py, int(size * 1.5), self.COLOR_PLAYER_GLOW)
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px - size, py - size, size * 2, size * 2))

        # Draw particles and floating texts
        for p in self.particles:
            p.draw(self.screen)
        for t in self.floating_texts:
            t.draw(self.screen, self.font_small)

    def _render_ui(self):
        # Player Health Bar
        self._draw_ui_bar(10, 10, 200, 20, self.player.health, self.player.max_health, self.COLOR_HEALTH_BAR,
                          self.COLOR_HEALTH_BAR_BG, f"HP: {self.player.health}/{self.player.max_health}")

        # Player XP Bar
        self._draw_ui_bar(10, 35, 200, 15, self.player.xp, self.player.xp_to_next_level, self.COLOR_XP_BAR,
                          self.COLOR_XP_BAR_BG, f"LVL: {self.player.level}")

        # Special Attack Cooldown
        self._draw_ui_bar(220, 10, 150, 20, self.player.max_special_cooldown - self.player.special_cooldown,
                          self.player.max_special_cooldown, self.COLOR_COOLDOWN_BAR, self.COLOR_COOLDOWN_BAR_BG,
                          "Special")

        # Score and Level
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        level_text = self.font_medium.render(f"Dungeon Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 35))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        if self.game_state == "VICTORY":
            text = "VICTORY"
            color = (255, 223, 0)
        else:
            text = "GAME OVER"
            color = (200, 40, 40)

        title_surf = self.font_large.render(text, True, color)
        title_rect = title_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(title_surf, title_rect)

    def _draw_health_bar(self, x, y, current, max_val, width):
        if max_val <= 0: return
        ratio = max(0, min(1, current / max_val))
        bg_rect = pygame.Rect(int(x - width // 2), int(y), width, 5)
        fg_rect = pygame.Rect(int(x - width // 2), int(y), int(width * ratio), 5)
        pygame.draw.rect(self.screen, (80, 0, 0), bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fg_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, 1)

    def _draw_ui_bar(self, x, y, w, h, current, max_val, fg_color, bg_color, label=""):
        if max_val <= 0: return
        ratio = max(0, min(1, current / max_val))
        bg_rect = pygame.Rect(x, y, w, h)
        fg_rect = pygame.Rect(x, y, int(w * ratio), h)
        pygame.draw.rect(self.screen, bg_color, bg_rect)
        pygame.draw.rect(self.screen, fg_color, fg_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 1)

        if label:
            text_surf = self.font_small.render(label, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=bg_rect.center)
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, x, y, color, count, speed):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, speed, self.np_random))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _update_floating_texts(self):
        self.floating_texts = [t for t in self.floating_texts if t.update()]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_level": self.player.level,
            "player_health": self.player.health,
            "dungeon_level": self.level,
        }

    def close(self):
        pygame.quit()

    def validate_implementation_before_reset(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        assert self.observation_space.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert self.observation_space.dtype == np.uint8

    def validate_implementation_after_reset(self):
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")


class Player:
    def __init__(self, np_random):
        self.np_random = np_random
        self.x, self.y = 0, 0
        self.px, self.py = 0, 0  # pixel coords for effects
        self.max_health = 20
        self.health = self.max_health
        self.level = 1
        self.xp = 0
        self.xp_to_next_level = 100
        self.attack_power = 2
        self.special_cooldown = 0
        self.max_special_cooldown = 3  # 3 turns
        self.anim_offset = 0

    def set_pos(self, x, y):
        self.x, self.y = x, y

    def move(self, x, y):
        self.x, self.y = x, y

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)

    def heal(self, amount):
        healed_amount = min(amount, self.max_health - self.health)
        self.health += healed_amount
        return healed_amount

    def gain_xp(self, amount):
        self.xp += amount
        leveled_up = False
        reward = 0
        while self.xp >= self.xp_to_next_level:
            leveled_up = True
            reward += 5  # Level up reward
            self.xp -= self.xp_to_next_level
            self.level += 1
            self.max_health += 5
            self.health = self.max_health
            self.attack_power += 1
            self.xp_to_next_level = int(self.xp_to_next_level * 1.5)
        return leveled_up, reward

    def update(self):
        if self.special_cooldown > 0:
            self.special_cooldown -= 1

    def update_anim(self, steps):
        self.anim_offset = math.sin(steps * 0.2) * 2

    def is_on_exit(self, exit_pos):
        return exit_pos and self.x == exit_pos[1] and self.y == exit_pos[0]


class Enemy:
    def __init__(self, x, y, type_idx, level, np_random):
        self.x, self.y = x, y
        self.px, self.py = 0, 0
        self.np_random = np_random
        self.color = GameEnv.ENEMY_COLORS[type_idx]
        self.max_health = (1 + type_idx) + (level - 1)
        self.health = self.max_health
        self.attack_power = 1 + (type_idx // 2) + (level - 1)
        self.xp_value = self.max_health * 5 + self.attack_power * 10
        self.anim_offset = 0

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)

    def get_attack_damage(self):
        return self.attack_power

    def update_attack_pattern(self):
        pass  # Standard enemies have no complex pattern

    def update_anim(self, steps):
        self.anim_offset = math.sin(steps * 0.2 + self.x) * 2


class Boss(Enemy):
    def __init__(self, x, y, np_random):
        super().__init__(x, y, 0, 3, np_random)
        self.max_health = 50
        self.health = self.max_health
        self.color = (150, 50, 200)
        self.xp_value = 500
        self.attack_pattern = ["strong", "weak", "defend"]
        self.pattern_index = 0
        self.is_defending = False

    def get_attack_damage(self):
        move = self.attack_pattern[self.pattern_index]
        if move == "strong":
            return 5
        elif move == "weak":
            return 2
        else:  # defend
            return 0

    def update_attack_pattern(self):
        self.pattern_index = (self.pattern_index + 1) % len(self.attack_pattern)
        self.is_defending = self.attack_pattern[self.pattern_index] == "defend"

    def take_damage(self, amount):
        if self.is_defending:
            amount //= 2
        super().take_damage(amount)


class Loot:
    def __init__(self, x, y, type):
        self.x, self.y = x, y
        self.type = type
        self.value = 10 if type == "potion" else 50


class Particle:
    def __init__(self, x, y, color, speed, np_random):
        self.x, self.y = x, y
        self.color = color
        angle = np_random.uniform(0, 2 * math.pi)
        s = np_random.uniform(speed * 0.5, speed)
        self.vx = math.cos(angle) * s
        self.vy = math.sin(angle) * s
        self.lifespan = np_random.integers(10, 20)
        self.size = np_random.integers(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.vx *= 0.8
        self.vy *= 0.8
        return self.lifespan > 0

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)


class FloatingText:
    def __init__(self, x, y, text, color, lifespan, dy=-1):
        self.x, self.y = x, y
        self.text = text
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.dy = dy

    def update(self):
        self.y += self.dy
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, screen, font):
        alpha = int(255 * (self.lifespan / self.initial_lifespan))
        text_surf = font.render(self.text, True, self.color)
        text_surf.set_alpha(alpha)
        text_rect = text_surf.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(text_surf, text_rect)


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's not part of the Gymnasium interface but is useful for testing
    import time

    # Un-dummy the video driver to see the game window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    # We can now run the full validation
    env.validate_implementation_after_reset()
    obs, info = env.reset()

    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    terminated = False
    total_reward = 0

    # Game loop for human play
    print(GameEnv.user_guide)
    while not terminated:
        action = [0, 0, 0]  # Default action: no-op

        # Event handling for keyboard input
        events = pygame.event.get()
        if not events:
            # If no new events, just refresh the screen
            obs = env._get_observation()
        else:
            for event in events:
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        action[2] = 1
                    else:
                        # Skip step if key is not a game action
                        continue

                    obs, reward, terminated, _, info = env.step(action)
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

        if terminated:
            # Get final observation before quitting
            obs = env._get_observation()

        # Draw the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over!")
            time.sleep(3)  # Show final screen for a moment

    env.close()