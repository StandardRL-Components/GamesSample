import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:40:19.781238
# Source Brief: brief_02293.md
# Brief Index: 2293
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A turn-based strategy game Gymnasium environment where the player builds Roman
    infrastructure to conquer the Italian peninsula.

    The agent controls a cursor on a grid-based map of Italy and can choose to
    build military or civilian infrastructure on owned or neutral territories.
    The goal is to achieve 80% territory control before running out of resources
    or reaching the maximum number of turns.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Build Military (0=released, 1=held)
    - actions[2]: Build Civilian (0=released, 1=held)

    Observation Space: Box(shape=(400, 640, 3), dtype=uint8)
    - A rendered RGB image of the game state.

    Reward Structure:
    - +100 for winning (80% territory).
    - -100 for losing (running out of a resource).
    - +1 for conquering a new territory.
    - +0.1 for each unit of resource gained.
    - -0.01 for each unit of resource spent.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A turn-based strategy game where you build Roman infrastructure to conquer the Italian peninsula."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to build military units and shift to build civilian structures."
    )
    auto_advance = False

    # Constants
    GRID_W, GRID_H = 32, 20
    CELL_SIZE = 20
    SCREEN_W, SCREEN_H = GRID_W * CELL_SIZE, GRID_H * CELL_SIZE # 640x400

    # Colors
    COLOR_BG = (15, 23, 42) # Dark Slate Blue (Water)
    COLOR_NEUTRAL = (100, 116, 139) # Slate Gray
    COLOR_PLAYER = (34, 197, 94) # Green
    COLOR_ENEMY = (239, 68, 68) # Red
    COLOR_CURSOR = (250, 204, 21) # Amber
    COLOR_TEXT = (241, 245, 249) # Slate 100

    # Game Parameters
    INITIAL_RESOURCES = {"gold": 100, "wood": 100, "stone": 100}
    MILITARY_COST = {"gold": 10, "wood": 10, "stone": 20}
    CIVILIAN_COST = {"gold": 20, "wood": 10, "stone": 10}
    CIVILIAN_YIELD = {"gold": 1}
    MAX_STEPS = 5000
    WIN_PERCENTAGE = 0.8

    ITALY_MAP_MASK = [
        "00000000000001110000000000000000", "00000000000011111000000000000000",
        "00000000000111111100000000000000", "00000000001111111110000000000000",
        "00000000011111111110000000000000", "00000000011111111111000000000000",
        "00000000001111111111100000000000", "00000000000111111111110000000000",
        "00000000000001111111111000000000", "00000000000000111111111000000000",
        "00000000000000011111111000000000", "00000000000000001111110000000000",
        "00000000000000000111111000000000", "00000000000000000011111100000000",
        "00000000000000000011111110000000", "00000000000000000001111111000000",
        "00000000000000000000111111100000", "00000000000000000000011111111000",
        "00000000000000000000001110111100", "00000000000000000000000000111000"
    ]
    OWNER_NEUTRAL, OWNER_PLAYER, OWNER_ENEMY, OWNER_WATER = 0, 1, 2, 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 16, bold=True)

        self.grid_ownership = None
        self.grid_military = None
        self.grid_civilian = None
        self.total_land_cells = 0
        self.resources = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.resources = self.INITIAL_RESOURCES.copy()
        self.particles = []

        self._generate_map()
        self._initialize_factions()

        return self._get_observation(), self._get_info()

    def _generate_map(self):
        self.grid_ownership = np.full((self.GRID_W, self.GRID_H), self.OWNER_WATER, dtype=np.int8)
        self.grid_military = np.zeros((self.GRID_W, self.GRID_H), dtype=np.int8)
        self.grid_civilian = np.zeros((self.GRID_W, self.GRID_H), dtype=np.int8)
        self.total_land_cells = 0

        for y, row in enumerate(self.ITALY_MAP_MASK):
            for x, char in enumerate(row):
                if char == '1':
                    self.grid_ownership[x, y] = self.OWNER_NEUTRAL
                    self.total_land_cells += 1

    def _initialize_factions(self):
        rome_pos = (12, 12)
        self.grid_ownership[rome_pos] = self.OWNER_PLAYER
        self.grid_civilian[rome_pos] = 1
        self.cursor_pos = list(rome_pos)

        initial_enemy_territories = 0
        max_enemy_territories = int(self.total_land_cells * 0.2)

        land_cells = np.argwhere(self.grid_ownership == self.OWNER_NEUTRAL)
        self.np_random.shuffle(land_cells)

        for x, y in land_cells:
            if initial_enemy_territories < max_enemy_territories:
                 self.grid_ownership[x, y] = self.OWNER_ENEMY
                 initial_enemy_territories += 1
            else:
                break


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_movement(movement)

        if space_held:
            reward += self._try_build("military")
        elif shift_held:
            reward += self._try_build("civilian")

        resource_gain = self._collect_resources()
        reward += resource_gain * 0.1

        self._enemy_turn()

        self.steps += 1

        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        x, y = self.cursor_pos
        if movement == 1 and y > 0: y -= 1
        elif movement == 2 and y < self.GRID_H - 1: y += 1
        elif movement == 3 and x > 0: x -= 1
        elif movement == 4 and x < self.GRID_W - 1: x += 1

        if self.grid_ownership[x, y] != self.OWNER_WATER:
            self.cursor_pos = [x, y]

    def _try_build(self, building_type):
        x, y = self.cursor_pos
        owner = self.grid_ownership[x, y]

        if owner not in [self.OWNER_PLAYER, self.OWNER_NEUTRAL]: return 0

        cost = self.MILITARY_COST if building_type == "military" else self.CIVILIAN_COST
        if not all(self.resources[res] >= cost[res] for res in cost): return 0

        reward = 0
        for res, amount in cost.items():
            self.resources[res] -= amount
            reward -= amount * 0.01

        if building_type == "military": self.grid_military[x, y] += 1
        else: self.grid_civilian[x, y] += 1

        if owner == self.OWNER_NEUTRAL:
            self.grid_ownership[x, y] = self.OWNER_PLAYER
            reward += 1.0 # #SFX: TerritoryClaim.wav

        if building_type == "military":
            reward += self._check_conquest(x, y)

        self._create_particles(x, y, self.COLOR_CURSOR)
        # #SFX: Build.wav
        return reward

    def _check_conquest(self, x, y):
        conquest_reward = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid_ownership[nx, ny] == self.OWNER_ENEMY:
                if self.grid_military[x, y] > self.grid_military[nx, ny]:
                    self.grid_ownership[nx, ny] = self.OWNER_PLAYER
                    conquest_reward += 1.0 # #SFX: Conquest.wav
        return conquest_reward

    def _collect_resources(self):
        player_mask = self.grid_ownership == self.OWNER_PLAYER
        total_civilian_level = np.sum(self.grid_civilian[player_mask])
        gained_gold = total_civilian_level * self.CIVILIAN_YIELD["gold"]
        self.resources["gold"] += gained_gold
        return gained_gold

    def _enemy_turn(self):
        expansion_count = 1 + self.steps // 100

        for _ in range(expansion_count):
            enemy_territories = np.argwhere(self.grid_ownership == self.OWNER_ENEMY)
            if len(enemy_territories) == 0: continue

            possible_expansions = set()
            for x, y in enemy_territories:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid_ownership[nx, ny] == self.OWNER_NEUTRAL:
                        possible_expansions.add((nx, ny))

            if possible_expansions:
                expansion_list = list(possible_expansions)
                ex, ey = expansion_list[self.np_random.integers(len(expansion_list))]
                self.grid_ownership[ex, ey] = self.OWNER_ENEMY
                self.grid_military[ex, ey] = 1 # #SFX: EnemyClaim.wav

    def _check_termination(self):
        player_territories = np.sum(self.grid_ownership == self.OWNER_PLAYER)
        if player_territories / self.total_land_cells >= self.WIN_PERCENTAGE:
            return True, 100.0 # #SFX: Victory.wav

        if any(res_count <= 0 for res_count in self.resources.values()):
            return True, -100.0 # #SFX: Defeat.wav

        if self.steps >= self.MAX_STEPS:
            return True, 0.0

        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        player_cells = np.sum(self.grid_ownership == self.OWNER_PLAYER)
        enemy_cells = np.sum(self.grid_ownership == self.OWNER_ENEMY)
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "player_territory_percentage": player_cells / self.total_land_cells if self.total_land_cells > 0 else 0,
            "enemy_territory_percentage": enemy_cells / self.total_land_cells if self.total_land_cells > 0 else 0,
        }

    def _render_game(self):
        colors = {
            self.OWNER_NEUTRAL: self.COLOR_NEUTRAL, self.OWNER_PLAYER: self.COLOR_PLAYER,
            self.OWNER_ENEMY: self.COLOR_ENEMY, self.OWNER_WATER: self.COLOR_BG
        }
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                owner = self.grid_ownership[x, y]
                color = colors[owner]
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)

                if owner in [self.OWNER_PLAYER, self.OWNER_ENEMY]:
                    mil_level = min(self.grid_military[x, y], 5)
                    if mil_level > 0:
                        mil_color = tuple(c * 0.6 for c in color)
                        size = int(self.CELL_SIZE * 0.2 * mil_level)
                        m_rect = pygame.Rect(rect.centerx - size//2, rect.centery - size//2, size, size)
                        pygame.draw.rect(self.screen, mil_color, m_rect, border_radius=2)

                    civ_level = min(self.grid_civilian[x, y], 5)
                    if civ_level > 0:
                        civ_color = tuple(min(255, c * 1.4) for c in color)
                        radius = int(self.CELL_SIZE * 0.1 * civ_level)
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, radius, civ_color)

        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pulse = (math.sin(self.steps * 0.4) + 1) / 2
        width = int(1 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width, border_radius=3)

    def _render_ui(self):
        bar_rect = pygame.Rect(0, 0, self.SCREEN_W, 30)
        pygame.draw.rect(self.screen, (0,0,0,150), bar_rect)

        res_text = (f"Gold: {self.resources['gold']:<4} Wood: {self.resources['wood']:<4} "
                    f"Stone: {self.resources['stone']:<4} | Turn: {self.steps}/{self.MAX_STEPS}")
        self._draw_text(res_text, (10, 7), self.font_ui, self.COLOR_TEXT)

        bar_rect_bottom = pygame.Rect(0, self.SCREEN_H - 25, self.SCREEN_W, 25)
        pygame.draw.rect(self.screen, (0,0,0,150), bar_rect_bottom)

        info = self._get_info()
        player_perc = info["player_territory_percentage"]
        enemy_perc = info["enemy_territory_percentage"]

        player_width = int(self.SCREEN_W * player_perc)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (0, self.SCREEN_H - 25, player_width, 25))

        enemy_width = int(self.SCREEN_W * enemy_perc)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (self.SCREEN_W - enemy_width, self.SCREEN_H - 25, enemy_width, 25))

        win_line_x = int(self.SCREEN_W * self.WIN_PERCENTAGE)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (win_line_x, self.SCREEN_H-25), (win_line_x, self.SCREEN_H), 2)

        self._draw_text(f"Territory: {player_perc:.1%}", (10, self.SCREEN_H-22), self.font_ui, self.COLOR_TEXT)

    def _draw_text(self, text, pos, font, color):
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def _create_particles(self, grid_x, grid_y, color):
        cx, cy = (grid_x + 0.5) * self.CELL_SIZE, (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append([cx, cy, math.cos(angle) * speed, math.sin(angle) * speed, self.np_random.integers(15, 31), color])

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2]; p[1] += p[3]; p[4] -= 1
            if p[4] > 0:
                alpha = max(0, 255 * (p[4] / 30))
                radius = int(max(0, p[4] / 6))
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, (*p[5], alpha))
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    pygame.display.set_caption("Roman Conquest")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    key_to_action = {
        pygame.K_UP: [1, 0, 0], pygame.K_DOWN: [2, 0, 0],
        pygame.K_LEFT: [3, 0, 0], pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0], pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RETURN: [0, 0, 0] # No-op action
    }
    
    print(f"\n--- {GameEnv.game_description} ---")
    print(f"--- {GameEnv.user_guide} ---")
    print("--- Q to quit. ---")

    while not terminated and not truncated:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                terminated = True
            if event.type == pygame.KEYDOWN and event.key in key_to_action:
                action = key_to_action[event.key]
        
        # Always step, even with no action from the user, to keep the game "turn-based"
        # The default action is no-op [0,0,0]
        if action is None:
            action = [0,0,0]

        obs, reward, terminated, truncated, info = env.step(action)
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Player Territory: {info['player_territory_percentage']:.1%}")
        if terminated or truncated:
            print(f"--- GAME OVER --- Final Info: {info}")

        # Render the latest observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(env._get_observation(), (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Run at a slower pace for turn-based game

    env.close()