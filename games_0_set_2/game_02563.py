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
    """
    A turn-based, grid-based dungeon crawler RPG where the player battles enemies
    and collects treasure to level up. The goal is to reach level 5.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in the direction you last moved. "
        "Turns are processed after each action."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a dungeon, fight monsters, and collect treasure. Gain experience to level up, "
        "increasing your power. Reach level 5 to win!"
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GAME_WIDTH, self.GAME_HEIGHT = 400, 400
        self.GRID_SIZE = 20
        self.TILE_SIZE = self.GAME_WIDTH // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_LEVEL = 5

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_UI_BG = (35, 35, 50)
        self.COLOR_WALL = (70, 70, 90)
        self.COLOR_FLOOR = (50, 45, 40)
        self.COLOR_HERO = (60, 180, 220)
        self.COLOR_GOBLIN = (80, 160, 80)
        self.COLOR_BAT = (160, 80, 80)
        self.COLOR_TREASURE = (255, 200, 0)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_RED = (220, 40, 40)
        self.COLOR_GREEN = (40, 220, 40)
        self.COLOR_XP = (200, 180, 50)
        self.COLOR_GOLD = (255, 215, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_s = pygame.font.SysFont("Consolas", 14)
            self.font_m = pygame.font.SysFont("Consolas", 16, bold=True)
            self.font_l = pygame.font.SysFont("Consolas", 20, bold=True)
        except pygame.error:
            self.font_s = pygame.font.SysFont(None, 18)
            self.font_m = pygame.font.SysFont(None, 22)
            self.font_l = pygame.font.SysFont(None, 28)

        # --- Game State Initialization ---
        self.grid = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.hero_pos = (0, 0)
        self.hero_health = 0
        self.hero_max_health = 0
        self.hero_level = 0
        self.hero_exp = 0
        self.hero_exp_to_level = 0
        self.hero_damage = 0
        self.hero_gold = 0
        self.last_move_dir = (0, 1) # Start facing down
        self.enemies = []
        self.treasure_chests = []
        self.effects = []
        self.log = deque(maxlen=8)
        self.steps = 0
        self.score = 0
        self.game_over = False

        # self.reset() is called here to set up the initial state
        # The initial call to reset() will use the default seed if none is provided.
        # A seed will be provided by the environment wrapper later.
        # We don't need to call validate_implementation() here as it's for debug.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.log.clear()

        # Generate dungeon
        self._generate_dungeon()
        
        # Get valid spawn points
        floor_tiles = np.argwhere(self.grid == 0).tolist()
        self.np_random.shuffle(floor_tiles)

        # Place Hero
        self.hero_pos = tuple(floor_tiles.pop())
        self.hero_level = 1
        self.hero_max_health = 10
        self.hero_health = self.hero_max_health
        self.hero_exp = 0
        self.hero_exp_to_level = 100
        self.hero_damage = 2
        self.hero_gold = 0
        self.last_move_dir = (0, 1)

        # Base enemy stats
        self.base_enemy_health = 5
        self.base_enemy_damage = 1

        # Place Enemies
        self.enemies = []
        num_enemies = 20
        for _ in range(min(num_enemies, len(floor_tiles))):
            pos = tuple(floor_tiles.pop())
            enemy_type = self.np_random.choice(['goblin', 'bat'])
            self.enemies.append({
                'pos': pos,
                'type': enemy_type,
                'health': self.base_enemy_health,
                'max_health': self.base_enemy_health,
                'damage': self.base_enemy_damage
            })

        # Place Treasure
        self.treasure_chests = []
        num_chests = self.np_random.integers(5, 11)
        for _ in range(min(num_chests, len(floor_tiles))):
            self.treasure_chests.append({'pos': tuple(floor_tiles.pop())})
            
        self.effects = []
        self._add_log("Welcome to the dungeon!", self.COLOR_WHITE)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.effects = [e for e in self.effects if e['life'] > 1]
        for effect in self.effects:
            effect['life'] -= 1

        # --- 1. Player Action Phase ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_taken = False

        if space_held: # Attack takes priority
            action_taken = True
            target_pos = (self.hero_pos[0] + self.last_move_dir[0], self.hero_pos[1] + self.last_move_dir[1])
            enemy_hit = None
            for enemy in self.enemies:
                if enemy['pos'] == target_pos:
                    enemy_hit = enemy
                    break
            
            if enemy_hit:
                # sfx: sword_hit.wav
                damage = self.hero_damage
                enemy_hit['health'] -= damage
                reward += 1.0 # Reward for damaging
                self._add_effect('damage_text', enemy_hit['pos'], str(damage), self.COLOR_WHITE, 3)
                self._add_effect('hit_spark', enemy_hit['pos'], None, self.COLOR_WHITE, 2)
                self._add_log(f"You hit the {enemy_hit['type']} for {damage} damage.", self.COLOR_WHITE)
                
                if enemy_hit['health'] <= 0:
                    # sfx: enemy_die.wav
                    reward += 5.0 # Reward for defeating
                    exp_gain = 25
                    self.hero_exp += exp_gain
                    self.enemies.remove(enemy_hit)
                    self._add_log(f"You defeated the {enemy_hit['type']}! +{exp_gain} EXP.", self.COLOR_XP)
                    self._check_level_up()
            else:
                # sfx: sword_miss.wav
                self._add_log("You swing at the empty air.", self.COLOR_WHITE)

        elif movement != 0: # Movement
            action_taken = True
            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            dx, dy = dirs[movement]
            self.last_move_dir = (dx, dy)
            
            # Reward for moving towards/away from nearest enemy
            dist_before = self._get_dist_to_nearest_enemy()
            
            new_pos = (self.hero_pos[0] + dx, self.hero_pos[1] + dy)
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE and self.grid[new_pos] == 0:
                is_occupied_by_enemy = any(e['pos'] == new_pos for e in self.enemies)
                if not is_occupied_by_enemy:
                    self.hero_pos = new_pos
                    # sfx: footstep.wav
                    
                    # Check for treasure
                    chest_found = None
                    for chest in self.treasure_chests:
                        if chest['pos'] == self.hero_pos:
                            chest_found = chest
                            break
                    if chest_found:
                        # sfx: treasure.wav
                        reward += 2.0
                        exp_gain = 50
                        gold_gain = self.np_random.integers(10, 21)
                        self.hero_exp += exp_gain
                        self.hero_gold += gold_gain
                        self.treasure_chests.remove(chest_found)
                        self._add_log(f"Found a chest! +{exp_gain} EXP, +{gold_gain} Gold.", self.COLOR_GOLD)
                        self._add_effect('sparkle', self.hero_pos, None, self.COLOR_GOLD, 4)
                        self._check_level_up()
            
            dist_after = self._get_dist_to_nearest_enemy()
            if dist_after < dist_before:
                reward += 0.1
            elif dist_after > dist_before:
                reward -= 0.1

        if not action_taken: # No-op / Wait
            self._add_log("You wait a moment.", (150, 150, 150))
            reward += 0.0

        # --- 2. Enemy Action Phase ---
        if not self.game_over:
            for enemy in self.enemies:
                dist_to_hero = abs(enemy['pos'][0] - self.hero_pos[0]) + abs(enemy['pos'][1] - self.hero_pos[1])
                
                if dist_to_hero == 1: # Attack if adjacent
                    # sfx: enemy_attack.wav
                    self.hero_health -= enemy['damage']
                    self._add_effect('damage_text', self.hero_pos, str(enemy['damage']), self.COLOR_RED, 3)
                    self._add_effect('hit_spark', self.hero_pos, None, self.COLOR_RED, 2)
                    self._add_log(f"The {enemy['type']} hits you for {enemy['damage']} damage!", self.COLOR_RED)

                elif dist_to_hero <= 5: # Move towards hero if in range
                    ex, ey = enemy['pos']
                    hx, hy = self.hero_pos
                    
                    dx, dy = 0, 0
                    if hx > ex: dx = 1
                    elif hx < ex: dx = -1
                    if hy > ey: dy = 1
                    elif hy < ey: dy = -1

                    # Simple pathing: try x, then y
                    if dx != 0 and dy != 0:
                        if self.np_random.random() < 0.5: dy = 0
                        else: dx = 0
                    
                    next_pos = (ex + dx, ey + dy)
                    if self.grid[next_pos] == 0 and not any(e['pos'] == next_pos for e in self.enemies) and next_pos != self.hero_pos:
                        enemy['pos'] = next_pos

        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and self.hero_health <= 0:
            reward -= 100
            self.score -= 100
        elif terminated and self.hero_level >= self.WIN_LEVEL:
            reward += 100
            self.score += 100
        
        if truncated:
             self._add_log("Time runs out...", (150, 150, 150))
             self.game_over = True


        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _check_level_up(self):
        if self.hero_exp >= self.hero_exp_to_level:
            # sfx: level_up.wav
            self.hero_level += 1
            self.hero_exp -= self.hero_exp_to_level
            self.hero_exp_to_level = int(self.hero_exp_to_level * 1.5)
            self.hero_max_health += 5
            self.hero_health = self.hero_max_health # Full heal
            self.hero_damage += 1
            self._add_log(f"LEVEL UP! You are now level {self.hero_level}!", self.COLOR_GREEN)
            self._add_effect('level_up', self.hero_pos, None, self.COLOR_XP, 5)
            
            # Scale enemies
            self.base_enemy_health = int(self.base_enemy_health * 1.05)
            self.base_enemy_damage = int(self.base_enemy_damage * 1.05) if self.np_random.random() < 0.5 else self.base_enemy_damage

    def _get_dist_to_nearest_enemy(self):
        if not self.enemies:
            return 0
        return min([abs(e['pos'][0] - self.hero_pos[0]) + abs(e['pos'][1] - self.hero_pos[1]) for e in self.enemies])

    def _check_termination(self):
        if self.hero_health <= 0:
            self.game_over = True
            self.hero_health = 0
            self._add_log("You have been defeated.", self.COLOR_RED)
            return True
        if self.hero_level >= self.WIN_LEVEL:
            self.game_over = True
            self._add_log("VICTORY! You reached level 5!", self.COLOR_GREEN)
            return True
        return False
        
    def _generate_dungeon(self):
        self.grid = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        
        def is_valid(x, y):
            return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE

        stack = [(1, 1)]
        self.grid[1, 1] = 0
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and self.grid[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Correctly select a random neighbor
                random_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[random_index]
                
                self.grid[nx, ny] = 0
                self.grid[x + (nx - x) // 2, y + (ny - y) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Carve some loops
        for _ in range(self.GRID_SIZE // 2):
            x, y = self.np_random.integers(1, self.GRID_SIZE-1, size=2)
            if self.grid[x, y] == 1:
                self.grid[x, y] = 0

    def _add_log(self, message, color):
        self.log.appendleft({'text': message, 'color': color})

    def _add_effect(self, type, pos, data, color, life):
        self.effects.append({'type': type, 'pos': pos, 'data': data, 'color': color, 'life': life})

    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        
        # Render game area
        self._render_game()
        
        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw treasure
        for chest in self.treasure_chests:
            x, y = chest['pos']
            rect = (x * self.TILE_SIZE + 2, y * self.TILE_SIZE + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
            pygame.draw.rect(self.screen, self.COLOR_TREASURE, rect)
            pygame.draw.rect(self.screen, (0,0,0), rect, 1)

        # Draw enemies
        for enemy in self.enemies:
            x, y = enemy['pos']
            color = self.COLOR_GOBLIN if enemy['type'] == 'goblin' else self.COLOR_BAT
            rect = (x * self.TILE_SIZE + 4, y * self.TILE_SIZE + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            pygame.draw.rect(self.screen, color, rect)
            
            # Health bar
            hp_ratio = enemy['health'] / enemy['max_health']
            bar_y = y * self.TILE_SIZE
            pygame.draw.rect(self.screen, self.COLOR_RED, (x * self.TILE_SIZE, bar_y, self.TILE_SIZE, 2))
            pygame.draw.rect(self.screen, self.COLOR_GREEN, (x * self.TILE_SIZE, bar_y, int(self.TILE_SIZE * hp_ratio), 2))

        # Draw hero
        hx, hy = self.hero_pos
        rect = (hx * self.TILE_SIZE + 3, hy * self.TILE_SIZE + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6)
        pygame.draw.rect(self.screen, self.COLOR_HERO, rect)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, rect, 1)

        # Draw effects
        for effect in self.effects:
            ex, ey = effect['pos']
            px, py = ex * self.TILE_SIZE + self.TILE_SIZE // 2, ey * self.TILE_SIZE + self.TILE_SIZE // 2
            
            if effect['type'] == 'damage_text':
                alpha = int(255 * (effect['life'] / 3.0))
                text_surf = self.font_m.render(effect['data'], True, effect['color'])
                text_surf.set_alpha(alpha)
                offset = int(10 * (1 - effect['life'] / 3.0))
                self.screen.blit(text_surf, (px - text_surf.get_width() // 2, py - offset - 10))
            
            elif effect['type'] == 'hit_spark':
                alpha = int(255 * (effect['life'] / 2.0))
                color = (*effect['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, px, py, 5, color)
            
            elif effect['type'] == 'sparkle':
                for _ in range(3):
                    angle = self.np_random.random() * 2 * math.pi
                    dist = self.np_random.random() * 10
                    sx, sy = px + math.cos(angle) * dist, py + math.sin(angle) * dist
                    pygame.draw.circle(self.screen, effect['color'], (int(sx), int(sy)), 1)
            
            elif effect['type'] == 'level_up':
                alpha = int(255 * (effect['life'] / 5.0))
                color = (*effect['color'], alpha)
                radius = int(self.TILE_SIZE * (1.5 - effect['life']/5.0))
                pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)
                pygame.gfxdraw.aacircle(self.screen, px, py, radius-2, color)


    def _render_ui(self):
        ui_x = self.GAME_WIDTH
        ui_width = self.WIDTH - self.GAME_WIDTH
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x, 0, ui_width, self.HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_WALL, (ui_x, 0), (ui_x, self.HEIGHT), 2)
        
        y_offset = 15

        def draw_text(text, x, y, font, color=self.COLOR_WHITE):
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x, y))
            return surf.get_height()

        def draw_bar(x, y, width, height, ratio, color, bg_color):
            pygame.draw.rect(self.screen, bg_color, (x, y, width, height))
            pygame.draw.rect(self.screen, color, (x, y, int(width * ratio), height))

        # Title
        y_offset += draw_text("Dungeon Crawler", ui_x + 15, y_offset, self.font_l) + 15

        # Stats
        draw_text(f"Level: {self.hero_level}", ui_x + 15, y_offset, self.font_m)
        y_offset += 25
        
        draw_text(f"HP: {self.hero_health}/{self.hero_max_health}", ui_x + 15, y_offset, self.font_m)
        draw_bar(ui_x + 15, y_offset + 20, ui_width - 30, 10, self.hero_health / self.hero_max_health if self.hero_max_health > 0 else 0, self.COLOR_GREEN, self.COLOR_RED)
        y_offset += 40

        draw_text(f"XP: {self.hero_exp}/{self.hero_exp_to_level}", ui_x + 15, y_offset, self.font_m)
        draw_bar(ui_x + 15, y_offset + 20, ui_width - 30, 10, self.hero_exp / self.hero_exp_to_level if self.hero_exp_to_level > 0 else 0, self.COLOR_XP, (60,60,60))
        y_offset += 40

        draw_text(f"Gold: {self.hero_gold}", ui_x + 15, y_offset, self.font_m, self.COLOR_GOLD)
        y_offset += 25
        draw_text(f"Damage: {self.hero_damage}", ui_x + 15, y_offset, self.font_m)
        y_offset += 40

        # Log
        pygame.draw.line(self.screen, self.COLOR_WALL, (ui_x + 10, y_offset), (self.WIDTH - 10, y_offset), 1)
        y_offset += 10
        for i, log_entry in enumerate(self.log):
            if y_offset + 15 > self.HEIGHT: break
            alpha = 255 - i * 25
            color = (*log_entry['color'], alpha)
            log_surf = self.font_s.render(log_entry['text'], True, log_entry['color'])
            log_surf.set_alpha(alpha)
            self.screen.blit(log_surf, (ui_x + 15, y_offset))
            y_offset += 15

        # Game Over Text
        if self.game_over:
            msg = "VICTORY!" if self.hero_level >= self.WIN_LEVEL else "GAME OVER"
            color = self.COLOR_GREEN if self.hero_level >= self.WIN_LEVEL else self.COLOR_RED
            text_surf = self.font_l.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.GAME_WIDTH // 2, self.GAME_HEIGHT // 2))
            
            s = pygame.Surface((text_rect.width + 20, text_rect.height + 10), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (text_rect.left - 10, text_rect.top - 5))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.hero_level,
            "health": self.hero_health,
            "gold": self.hero_gold,
        }

    def render(self):
        # This method is not strictly required by the new API but is good for compatibility
        return self._get_observation()

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # To run with display, you need to unset the dummy videodriver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # No-op
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        # --- Human Controls ---
        human_action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                human_action_taken = True
                if event.key == pygame.K_UP:
                    action = [1, 0, 0]
                elif event.key == pygame.K_DOWN:
                    action = [2, 0, 0]
                elif event.key == pygame.K_LEFT:
                    action = [3, 0, 0]
                elif event.key == pygame.K_RIGHT:
                    action = [4, 0, 0]
                elif event.key == pygame.K_SPACE:
                    action = [0, 1, 0]
                elif event.key == pygame.K_w: # Wait
                    action = [0, 0, 0]
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset(seed=42)
                    action = [0, 0, 0]
                else:
                    human_action_taken = False

        if human_action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}, Info: {info}")
            action = [0, 0, 0] # Reset action after processing

        # --- Rendering ---
        frame = np.transpose(env.render(), (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(15) # Limit frame rate for human play
        
        if terminated or truncated:
            print("Game Over!")
            pygame.time.wait(2000) # Pause for 2 seconds before closing
            
    env.close()