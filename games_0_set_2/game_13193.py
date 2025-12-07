import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


# Set SDL to dummy mode for headless operation, which is required by the testing environment.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    A side-scrolling survival game where the agent must gather resources,
    craft items, and build an escape raft while evading hostile wildlife.
    The game is turn-based, with each call to step() representing one action or turn.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A side-scrolling survival game. Gather resources, craft items, and build a raft to "
        "escape while evading hostile wildlife."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press Shift to craft items and Space to place traps."
    )
    auto_advance = False

    # --- CONSTANTS ---
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 16
    WORLD_WIDTH_TILES = 200
    WORLD_HEIGHT_TILES = SCREEN_HEIGHT // TILE_SIZE  # 25

    # Game Parameters
    MAX_STEPS = 5000
    STARTING_HEALTH = 100
    RAFT_PARTS_GOAL = 5
    DIFFICULTY_INTERVAL = 500

    # Player
    PLAYER_COLOR_BODY = (50, 180, 255) # Bright Cyan
    PLAYER_COLOR_OUTLINE = (255, 255, 255) # White

    # Colors
    COLOR_BG = (20, 30, 40) # Dark Blue-Gray
    COLOR_UI_BG = (10, 20, 30, 200)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (100, 20, 20)
    COLOR_HEALTH_BAR_FG = (20, 200, 20)
    COLOR_GROUND = (90, 60, 40) # Brown
    COLOR_GRASS = (60, 140, 70) # Green
    COLOR_TREE_TRUNK = (80, 50, 30)
    COLOR_TREE_LEAVES = (40, 100, 50)
    COLOR_ROCK = (100, 100, 110)
    COLOR_WATER = (40, 80, 150)
    COLOR_RAFT = (150, 110, 80)
    COLOR_TRAP = (120, 80, 50)
    COLOR_ENEMY = (220, 50, 50) # Bright Red
    COLOR_ENEMY_OUTLINE = (255, 150, 150)
    COLOR_VICTORY = (255, 215, 0) # Gold
    COLOR_DEFEAT = (180, 0, 0)

    # Tile IDs
    T_SKY = 0
    T_GROUND = 1
    T_GRASS = 2
    T_TREE = 3
    T_ROCK = 4
    T_WATER = 5
    T_RAFT = 6

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
        self._init_pixel_font()

        # Game state variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.camera_x = 0
        self.world_map = None
        self.resources = {}
        self.traps = []
        self.enemies = []
        self.raft_progress = 0
        self.enemy_speed = 0.0
        self.enemy_spawn_chance = 0.0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_action_feedback = ""
        self.feedback_timer = 0
        
        self.recipes = {
            'raft_segment': {'wood': 20, 'stone': 10},
            'trap': {'wood': 5, 'stone': 2}
        }
        self.craft_order = ['raft_segment', 'trap']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.STARTING_HEALTH
        self.resources = {'wood': 10, 'stone': 5, 'food': 5}
        self.traps = []
        self.enemies = []
        self.raft_progress = 0
        self.enemy_speed = 1.0
        self.enemy_spawn_chance = 0.02
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_action_feedback = ""
        self.feedback_timer = 0

        self._procedurally_generate_world()
        
        start_x = 5
        start_y = np.where(self.world_map[:, start_x] > self.T_SKY)[0][0] - 1
        self.player_pos = [start_x, start_y]

        self._spawn_initial_enemies()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, self.steps >= self.MAX_STEPS, self._get_info()

        self.steps += 1
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_taken_this_turn = False

        # --- Handle Player Actions (Prioritized) ---
        if shift_held and not self.prev_shift_held:
            crafted_item = self._attempt_craft()
            if crafted_item:
                action_taken_this_turn = True
                reward += 10.0 if crafted_item == 'raft_segment' else 1.0
        
        if not action_taken_this_turn and space_held and not self.prev_space_held:
            if self._attempt_place_trap():
                action_taken_this_turn = True
                reward += 1.0

        if not action_taken_this_turn and movement != 0:
            self._move_player(movement)

        # --- Update World State ---
        resource_reward, food_collected = self._update_traps()
        reward += resource_reward
        if food_collected > 0:
            self.player_health = min(self.STARTING_HEALTH, self.player_health + food_collected * 5)

        damage_reward = self._update_enemies()
        reward += damage_reward

        self._update_difficulty()
        
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.last_action_feedback = ""

        # --- Check for Termination/Truncation ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100.0
            self._set_feedback("YOU DIED", 1000)
        elif self.raft_progress >= self.RAFT_PARTS_GOAL:
            terminated = True
            self.game_over = True
            reward += 100.0
            self._set_feedback("YOU ESCAPED!", 1000)
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            if not terminated:
                reward -= 50.0
                self._set_feedback("NIGHT FELL...", 1000)

        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Transpose from (width, height, 3) to (height, width, 3)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "raft_progress": self.raft_progress,
            "resources": self.resources.copy(),
        }
        
    def _set_feedback(self, text, duration):
        self.last_action_feedback = text
        self.feedback_timer = duration

    def _procedurally_generate_world(self):
        self.world_map = np.full((self.WORLD_HEIGHT_TILES, self.WORLD_WIDTH_TILES), self.T_SKY, dtype=int)
        
        ground_y = self.WORLD_HEIGHT_TILES - 5
        for x in range(self.WORLD_WIDTH_TILES):
            self.world_map[ground_y:, x] = self.T_GROUND
            self.world_map[ground_y, x] = self.T_GRASS
            
            r = self.np_random.random()
            if x > 2:
                if r < 0.2:
                    ground_y = np.clip(ground_y - 1, 5, self.WORLD_HEIGHT_TILES - 3)
                elif r > 0.8:
                    ground_y = np.clip(ground_y + 1, 5, self.WORLD_HEIGHT_TILES - 3)
        
        for x in range(10, self.WORLD_WIDTH_TILES - 10):
            if self.world_map[self.WORLD_HEIGHT_TILES-1, x] != self.T_WATER:
                r = self.np_random.random()
                if r < 0.1:
                    tree_y = np.where(self.world_map[:, x] > self.T_SKY)[0][0]
                    if tree_y > 3:
                        self.world_map[tree_y-3:tree_y, x] = self.T_TREE
                elif r < 0.15:
                    rock_y = np.where(self.world_map[:, x] > self.T_SKY)[0][0]
                    self.world_map[rock_y-1, x] = self.T_ROCK
                elif r < 0.18:
                    pool_y = np.where(self.world_map[:, x] > self.T_SKY)[0][0] + 1
                    for i in range(x, min(x + self.np_random.integers(3, 7), self.WORLD_WIDTH_TILES)):
                        self.world_map[pool_y:, i] = self.T_WATER
                        self.world_map[pool_y-1, i] = self.T_GROUND

        raft_x = self.WORLD_WIDTH_TILES - 5
        raft_y = np.where(self.world_map[:, raft_x] > self.T_SKY)[0][0]
        self.world_map[raft_y-1:raft_y+1, raft_x] = self.T_RAFT

    def _spawn_initial_enemies(self):
        for _ in range(3):
            self._spawn_enemy()
            
    def _spawn_enemy(self):
        side = self.np_random.choice([-1, 1])
        if side == -1:
            spawn_x = self.player_pos[0] - self.SCREEN_WIDTH // self.TILE_SIZE // 2 - 5
        else:
            spawn_x = self.player_pos[0] + self.SCREEN_WIDTH // self.TILE_SIZE // 2 + 5
        
        spawn_x = np.clip(spawn_x, 0, self.WORLD_WIDTH_TILES - 1)
        
        ground_indices = np.where(self.world_map[:, spawn_x] > self.T_SKY)[0]
        if len(ground_indices) > 0:
            spawn_y = ground_indices[0] - 1
            patrol_range = self.np_random.integers(5, 15)
            self.enemies.append({
                'pos': [spawn_x, spawn_y],
                'dir': self.np_random.choice([-1, 1]),
                'patrol_start': spawn_x,
                'patrol_range': patrol_range
            })

    def _attempt_craft(self):
        for item_name in self.craft_order:
            recipe = self.recipes[item_name]
            can_craft = all(self.resources[res] >= cost for res, cost in recipe.items())
            if can_craft:
                for res, cost in recipe.items():
                    self.resources[res] -= cost
                if item_name == 'raft_segment':
                    self.raft_progress += 1
                    self._set_feedback("RAFT SEGMENT BUILT!", 60)
                return item_name
        self._set_feedback("NOT ENOUGH RESOURCES", 60)
        return None

    def _attempt_place_trap(self):
        cost = self.recipes['trap']
        if all(self.resources[res] >= num for res, num in cost.items()):
            px, py = self.player_pos
            tile_below = self.world_map[py + 1, px]
            is_on_trap = any(t['pos'] == [px, py] for t in self.traps)
            
            if tile_below in [self.T_GRASS, self.T_GROUND] and not is_on_trap:
                for res, num in cost.items():
                    self.resources[res] -= num
                self.traps.append({'pos': [px, py], 'timer': 0})
                self._set_feedback("TRAP PLACED", 60)
                return True
            else:
                self._set_feedback("CANNOT PLACE TRAP HERE", 60)
                return False
        else:
            self._set_feedback("NEED WOOD/STONE FOR TRAP", 60)
            return False

    def _move_player(self, movement):
        px, py = self.player_pos
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up (Jump)
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        next_x, next_y = px + dx, py + dy
        
        if not (0 <= next_x < self.WORLD_WIDTH_TILES and 0 <= next_y < self.WORLD_HEIGHT_TILES):
            return

        target_tile = self.world_map[next_y, next_x]
        is_grounded = self.world_map[py + 1, px] not in [self.T_SKY, self.T_WATER]

        if target_tile in [self.T_SKY, self.T_RAFT]:
            if dy == 1 and not is_grounded: return
            self.player_pos = [next_x, next_y]
            while self.world_map[self.player_pos[1] + 1, self.player_pos[0]] == self.T_SKY:
                self.player_pos[1] += 1
        elif dy == -1 and is_grounded:
             self.player_pos[1] = py - 1

    def _update_traps(self):
        reward = 0.0
        food_collected = 0
        for trap in self.traps:
            trap['timer'] += 1
            if trap['timer'] > 10:
                trap['timer'] = 0
                if self.np_random.random() < 0.5: self.resources['wood'] += 1; reward += 0.1
                if self.np_random.random() < 0.3: self.resources['stone'] += 1; reward += 0.1
                if self.np_random.random() < 0.2: self.resources['food'] += 1; food_collected += 1; reward += 0.1
        return reward, food_collected

    def _update_enemies(self):
        reward = 0.0
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 1, 1)
        
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            next_ex = int(ex + enemy['dir'] * self.enemy_speed)
            
            if not (0 <= next_ex < self.WORLD_WIDTH_TILES) or \
               self.world_map[ey, next_ex] != self.T_SKY or \
               self.world_map[ey+1, next_ex] in [self.T_SKY, self.T_WATER] or \
               abs(next_ex - enemy['patrol_start']) > enemy['patrol_range']:
                enemy['dir'] *= -1
            else:
                enemy['pos'][0] = next_ex

            while enemy['pos'][1] + 1 < self.WORLD_HEIGHT_TILES and self.world_map[enemy['pos'][1] + 1, enemy['pos'][0]] == self.T_SKY:
                enemy['pos'][1] += 1

            if player_rect.colliderect(pygame.Rect(enemy['pos'][0], enemy['pos'][1], 1, 1)):
                self.player_health -= 10
                reward -= 5.0
                self._set_feedback("HURT BY WILDLIFE!", 30)
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.enemy_speed = min(3.0, self.enemy_speed + 0.1)
            self.enemy_spawn_chance = min(0.1, self.enemy_spawn_chance * 1.1)
        
        if self.np_random.random() < self.enemy_spawn_chance:
            self._spawn_enemy()
            
        self.enemies = [e for e in self.enemies if abs(e['pos'][0] - self.player_pos[0]) < self.SCREEN_WIDTH // self.TILE_SIZE * 2]

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        
        self.camera_x = max(0, min(self.player_pos[0] * self.TILE_SIZE - self.SCREEN_WIDTH // 2,
                                     self.WORLD_WIDTH_TILES * self.TILE_SIZE - self.SCREEN_WIDTH))
        camera_tile_x = self.camera_x // self.TILE_SIZE
        
        start_x = max(0, camera_tile_x)
        end_x = min(self.WORLD_WIDTH_TILES, start_x + (self.SCREEN_WIDTH // self.TILE_SIZE) + 2)
        
        for y in range(self.WORLD_HEIGHT_TILES):
            for x in range(start_x, end_x):
                tile_id = self.world_map[y, x]
                if tile_id == self.T_SKY: continue
                
                screen_x = int(x * self.TILE_SIZE - self.camera_x)
                screen_y = int(y * self.TILE_SIZE)
                rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
                
                color_map = {
                    self.T_GRASS: self.COLOR_GRASS, self.T_GROUND: self.COLOR_GROUND,
                    self.T_ROCK: self.COLOR_ROCK, self.T_WATER: self.COLOR_WATER,
                    self.T_RAFT: self.COLOR_RAFT, self.T_TREE: self.COLOR_TREE_TRUNK
                }
                if tile_id in color_map:
                    pygame.draw.rect(self.screen, color_map[tile_id], rect)
                if tile_id == self.T_TREE:
                    leaves_rect = rect.inflate(self.TILE_SIZE, self.TILE_SIZE).move(0, -self.TILE_SIZE)
                    pygame.draw.ellipse(self.screen, self.COLOR_TREE_LEAVES, leaves_rect)
        
        for trap in self.traps:
            screen_x = int(trap['pos'][0] * self.TILE_SIZE - self.camera_x)
            screen_y = int(trap['pos'][1] * self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TRAP, (screen_x + 4, screen_y + 12, self.TILE_SIZE - 8, 4))

        for enemy in self.enemies:
            screen_x = int(enemy['pos'][0] * self.TILE_SIZE - self.camera_x)
            screen_y = int(enemy['pos'][1] * self.TILE_SIZE)
            rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_OUTLINE, rect, 1)

        screen_x = int(self.player_pos[0] * self.TILE_SIZE - self.camera_x)
        screen_y = int(self.player_pos[1] * self.TILE_SIZE)
        player_rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.PLAYER_COLOR_BODY, player_rect.inflate(-4, -2))
        pygame.draw.rect(self.screen, self.PLAYER_COLOR_OUTLINE, player_rect.inflate(-4, -2), 2)

    def _render_ui(self):
        s = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0,0))
        
        health_pct = max(0, self.player_health / self.STARTING_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 150, 20))
        if health_pct > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, 150 * health_pct, 20))
        self._draw_pixel_text("HP", 15, 15, self.COLOR_UI_TEXT)
        
        res_text = f"W:{self.resources['wood']} S:{self.resources['stone']} F:{self.resources['food']}"
        self._draw_pixel_text(res_text, 170, 15, self.COLOR_UI_TEXT)
        
        raft_text = f"RAFT: {self.raft_progress}/{self.RAFT_PARTS_GOAL}"
        self._draw_pixel_text(raft_text, self.SCREEN_WIDTH - 150, 15, self.COLOR_UI_TEXT)

        if self.feedback_timer > 0:
            color = self.COLOR_ENEMY if "HURT" in self.last_action_feedback else self.COLOR_UI_TEXT
            self._draw_pixel_text(self.last_action_feedback, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30, color, center=True)

        if self.game_over:
            if self.raft_progress >= self.RAFT_PARTS_GOAL:
                msg, color = "VICTORY!", self.COLOR_VICTORY
            else:
                msg, color = "GAME OVER", self.COLOR_DEFEAT
            self._draw_pixel_text(msg, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, color, scale=4, center=True)

    def _init_pixel_font(self):
        # A minimal pixel font for UI text rendering
        self.font_map = {'A':[[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],'B':[[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,1],[1,1,1,0]],'C':[[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,1,1]],'D':[[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]],'E':[[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,1,1,1]],'F':[[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0]],'G':[[0,1,1,1],[1,0,0,0],[1,0,1,1],[1,0,0,1],[0,1,1,0]],'H':[[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],'I':[[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]],'K':[[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],'L':[[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]],'M':[[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]],'N':[[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1]],'O':[[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],'P':[[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,0],[1,0,0,0]],'R':[[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,1,0],[1,0,0,1]],'S':[[0,1,1,1],[1,0,0,0],[0,1,1,0],[0,0,0,1],[1,1,1,0]],'T':[[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],'U':[[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],'V':[[1,0,0,1],[1,0,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,0]],'W':[[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,1,0,1,1],[1,0,0,0,1]],'Y':[[1,0,1],[1,0,1],[0,1,0],[0,1,0],[0,1,0]],'0':[[0,1,1,0],[1,0,0,1],[1,0,1,1],[1,1,0,1],[0,1,1,0]],'1':[[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],'2':[[0,1,1,0],[1,0,0,1],[0,0,1,0],[0,1,0,0],[1,1,1,1]],'3':[[1,1,1,0],[0,0,0,1],[0,1,1,0],[0,0,0,1],[1,1,1,0]],'4':[[1,0,1,0],[1,0,1,0],[1,1,1,1],[0,0,1,0],[0,0,1,0]],'5':[[1,1,1,1],[1,0,0,0],[1,1,1,0],[0,0,0,1],[1,1,1,0]],'6':[[0,1,1,0],[1,0,0,0],[1,1,1,0],[1,0,0,1],[0,1,1,0]],'7':[[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,0,0]],'8':[[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0]],'9':[[0,1,1,0],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,1,1,0]],'!':[[1],[1],[1],[],[1]],':':[[0],[1],[0],[1],[0]],' ': [[0,0],[0,0]]}

    def _draw_pixel_text(self, text, x, y, color, scale=2, center=False):
        text = text.upper()
        if center:
            text_width = sum((len(self.font_map.get(char, [[0]])[0]) + 1) * scale for char in text)
            x -= text_width // 2

        for char in text:
            if char in self.font_map:
                matrix = self.font_map[char]
                char_width = 0
                for row_idx, row in enumerate(matrix):
                    char_width = max(char_width, len(row))
                    for col_idx, pixel in enumerate(row):
                        if pixel:
                            pygame.draw.rect(self.screen, color, (x + col_idx * scale, y + row_idx * scale, scale, scale))
                x += (char_width + 1) * scale

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows manual play and is not run by the test suite.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Survival Side-Scroller")
    clock = pygame.time.Clock()
    
    running = True
    total_score = 0
    
    while running:
        movement = 0 # None
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_score = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Score: {total_score:.2f}, Info: {info}")
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_score = 0
                        waiting_for_reset = False
                clock.tick(30)

        clock.tick(10)
        
    env.close()