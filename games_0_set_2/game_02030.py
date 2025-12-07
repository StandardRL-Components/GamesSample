
# Generated: 2025-08-27T19:03:14.011832
# Source Brief: brief_02030.md
# Brief Index: 2030

        
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
        "Controls: Arrow keys to move. Space to attack in the direction you are facing. Survive the waves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Command a robot in a grid-based arena to strategically defeat waves of enemies in this turn-based tactics game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.TILE_SIZE = 40
        self.MAX_STEPS = 1000
        self.NUM_ENEMIES_PER_WAVE = 7
        self.PLAYER_MAX_HEALTH = 10

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (60, 120, 255)
        self.COLOR_PLAYER_ACCENT = (150, 200, 255)
        self.COLOR_ENEMY = (255, 60, 60)
        self.COLOR_ENEMY_ACCENT = (255, 150, 150)
        self.COLOR_ATTACK = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_GOOD = (0, 200, 100)
        self.COLOR_HEALTH_BAD = (80, 0, 20)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_facing_direction = (0, -1)  # (dx, dy) -> up
        self.enemies = []
        self.enemy_max_health = 3
        self.attack_effects = []  # List of {'pos': [x,y]}
        self.rng = np.random.default_rng()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_direction = (0, -1)  # Start facing up
        self.attack_effects = []

        # Place player
        self.player_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]

        # Spawn enemies
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        self.enemies = []
        self.enemy_max_health = 2 + self.wave
        occupied_tiles = [tuple(self.player_pos)]
        for _ in range(self.NUM_ENEMIES_PER_WAVE):
            while True:
                pos = [
                    self.rng.integers(0, self.GRID_COLS),
                    self.rng.integers(0, self.GRID_ROWS),
                ]
                if tuple(pos) not in occupied_tiles:
                    self.enemies.append(
                        {'pos': pos, 'health': self.enemy_max_health}
                    )
                    occupied_tiles.append(tuple(pos))
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Cost of existing per turn
        self.attack_effects = []  # Clear effects from previous turn

        # --- 1. Player Action ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Handle movement and update facing direction
        dx, dy = 0, 0
        if movement == 1: dx, dy = 0, -1  # Up
        elif movement == 2: dx, dy = 0, 1   # Down
        elif movement == 3: dx, dy = -1, 0  # Left
        elif movement == 4: dx, dy = 1, 0   # Right

        if dx != 0 or dy != 0:
            self.player_facing_direction = (dx, dy)
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if 0 <= new_pos[0] < self.GRID_COLS and 0 <= new_pos[1] < self.GRID_ROWS:
                self.player_pos = new_pos

        # Handle attack
        if space_held:
            # sfx: player_attack_sound()
            attack_pos = [
                self.player_pos[0] + self.player_facing_direction[0],
                self.player_pos[1] + self.player_facing_direction[1],
            ]
            self.attack_effects.append({'pos': attack_pos})
            for enemy in self.enemies:
                if enemy['pos'] == attack_pos:
                    enemy['health'] -= 1
                    reward += 1.0
                    # sfx: enemy_hit_sound()
                    if enemy['health'] <= 0:
                        self.score += 10
                        # sfx: enemy_destroy_sound()
            self.enemies = [e for e in self.enemies if e['health'] > 0]

        # --- 2. Enemy Turn ---
        all_entity_pos = {tuple(e['pos']) for e in self.enemies}
        all_entity_pos.add(tuple(self.player_pos))

        for enemy in self.enemies:
            px, py = self.player_pos
            ex, ey = enemy['pos']
            if abs(px - ex) + abs(py - ey) == 1:
                self.player_health -= 1
                reward -= 1.0
                # sfx: player_damage_sound()
            else:
                possible_moves = []
                for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_pos = [ex + edx, ey + edy]
                    if (0 <= new_pos[0] < self.GRID_COLS and
                        0 <= new_pos[1] < self.GRID_ROWS and
                        tuple(new_pos) not in all_entity_pos):
                        possible_moves.append(new_pos)
                if possible_moves:
                    # sfx: enemy_move_sound()
                    old_pos = tuple(enemy['pos'])
                    enemy['pos'] = self.rng.choice(possible_moves).tolist()
                    all_entity_pos.remove(old_pos)
                    all_entity_pos.add(tuple(enemy['pos']))

        # --- 3. Check Game State & Termination ---
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            self.score -= 100

        if not self.enemies:
            reward += 100
            self.score += self.wave * 100
            # Terminate episode on wave clear for better RL training signal
            terminated = True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies),
        }

    def _world_to_screen(self, pos):
        return (
            pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2,
            pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        )

    def _draw_health_bar(self, surface, center_x, top_y, width, height, current_hp, max_hp, good_color, bad_color):
        current_hp = max(0, current_hp)
        fill_ratio = current_hp / max_hp
        bg_rect = pygame.Rect(center_x - width // 2, top_y, width, height)
        pygame.draw.rect(surface, bad_color, bg_rect)
        fg_rect = pygame.Rect(center_x - width // 2, top_y, int(width * fill_ratio), height)
        pygame.draw.rect(surface, good_color, fg_rect)
        pygame.draw.rect(surface, (200, 200, 200), bg_rect, 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        for effect in self.attack_effects:
            sx, sy = self._world_to_screen(effect['pos'])
            size = self.TILE_SIZE
            pygame.draw.rect(self.screen, self.COLOR_ATTACK, (sx - size//2, sy - size//2, size, size))

        for enemy in self.enemies:
            sx, sy = self._world_to_screen(enemy['pos'])
            size = int(self.TILE_SIZE * 0.8)
            rect = pygame.Rect(sx - size // 2, sy - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_ACCENT, rect, 3)
            self._draw_health_bar(
                self.screen, sx, sy - size // 2 - 10, size, 5,
                enemy['health'], self.enemy_max_health,
                self.COLOR_HEALTH_GOOD, self.COLOR_HEALTH_BAD
            )

        px, py = self._world_to_screen(self.player_pos)
        size = int(self.TILE_SIZE * 0.85)
        rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, rect, 3)
        self._draw_health_bar(
            self.screen, px, py - size // 2 - 10, size, 5,
            self.player_health, self.PLAYER_MAX_HEALTH,
            self.COLOR_HEALTH_GOOD, self.COLOR_HEALTH_BAD
        )
        f_dx, f_dy = self.player_facing_direction
        p1 = (px + f_dx * size * 0.4, py + f_dy * size * 0.4)
        p2 = (px + f_dx * size * 0.2 - f_dy * size * 0.2, py + f_dy * size * 0.2 + f_dx * size * 0.2)
        p3 = (px + f_dx * size * 0.2 + f_dy * size * 0.2, py + f_dy * size * 0.2 - f_dx * size * 0.2)
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER_ACCENT, [p1, p2, p3])

        wave_text = self.font_ui.render(f"Wave: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))
        health_text = self.font_ui.render(f"Health: {self.player_health}/{self.PLAYER_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 10))
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH//2 - score_text.get_width()//2, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            game_over_text = self.font_game_over.render("GAME OVER", True, self.COLOR_ENEMY)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()