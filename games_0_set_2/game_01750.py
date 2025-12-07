
# Generated: 2025-08-27T18:09:37.152788
# Source Brief: brief_01750.md
# Brief Index: 1750

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Press space to attack adjacent enemies."
    )

    game_description = (
        "Navigate a procedurally generated grid-based dungeon, battling enemies and collecting gold to reach the exit."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAP_WIDTH, MAP_HEIGHT = 50, 40
    TILE_SIZE = 32
    MAX_STEPS = 1000
    FINAL_FLOOR = 5

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (80, 80, 90)
    COLOR_FLOOR = (60, 40, 30)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150, 60)
    COLOR_ENEMY_GOBLIN = (255, 50, 50)
    COLOR_GOLD = (255, 223, 0)
    COLOR_EXIT = (50, 255, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    COLOR_HIT_EFFECT = (255, 255, 255)
    COLOR_ATTACK_SLASH = (200, 200, 255)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.game_map = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT), dtype=int)
        self.player = {}
        self.enemies = []
        self.gold_piles = []
        self.exit_pos = None
        self.floor_level = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.visual_effects = [] # For hit sparks, etc.
        self.attack_effect = None # For player's attack slash

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.floor_level = 1
        
        self.player = {
            'pos': (0, 0),
            'health': 20,
            'max_health': 20,
            'attack_power': 3,
            'last_hit_timer': 0
        }
        
        self._generate_floor()

        return self._get_observation(), self._get_info()

    def _generate_floor(self):
        # Generate a new floor layout until it's valid
        while True:
            self._create_map_layout()
            
            floor_tiles = np.argwhere(self.game_map == 0)
            if len(floor_tiles) < 50: continue # Map too small, retry

            # Place player
            self.player['pos'] = tuple(random.choice(floor_tiles))
            
            # Place exit far from player
            max_dist = 0
            for _ in range(100): # Try 100 times to find a far exit
                potential_exit = tuple(random.choice(floor_tiles))
                dist = math.dist(self.player['pos'], potential_exit)
                if dist > max_dist:
                    max_dist = dist
                    self.exit_pos = potential_exit
            if self.exit_pos is None: continue

            # Check connectivity
            if not self._is_path_possible(self.player['pos'], self.exit_pos):
                continue
            
            # If all checks pass, place other entities
            self._place_entities(floor_tiles)
            break

    def _create_map_layout(self):
        # Cellular automata-like generation
        self.game_map = np.random.choice([0, 1], size=(self.MAP_WIDTH, self.MAP_HEIGHT), p=[0.55, 0.45])
        for _ in range(4): # Smoothing iterations
            new_map = np.copy(self.game_map)
            for x in range(1, self.MAP_WIDTH - 1):
                for y in range(1, self.MAP_HEIGHT - 1):
                    wall_neighbors = np.sum(self.game_map[x-1:x+2, y-1:y+2]) - self.game_map[x,y]
                    if wall_neighbors > 4:
                        new_map[x,y] = 1
                    elif wall_neighbors < 4:
                        new_map[x,y] = 0
            self.game_map = new_map

    def _is_path_possible(self, start, end):
        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == end:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.MAP_WIDTH and 0 <= ny < self.MAP_HEIGHT and \
                   self.game_map[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _place_entities(self, floor_tiles):
        occupied = {self.player['pos'], self.exit_pos}
        
        # Place enemies
        self.enemies = []
        num_enemies = min(10, self.floor_level + random.randint(0, 2))
        for _ in range(num_enemies):
            pos = tuple(random.choice(floor_tiles))
            if pos not in occupied and math.dist(pos, self.player['pos']) > 5:
                enemy_health = 1 + (self.floor_level // 2) + random.randint(0, 1)
                enemy_damage = 1 + (self.floor_level // 3)
                self.enemies.append({
                    'pos': pos, 
                    'health': enemy_health, 
                    'max_health': enemy_health,
                    'damage': enemy_damage,
                    'last_hit_timer': 0
                })
                occupied.add(pos)
        
        # Place gold
        self.gold_piles = []
        num_gold = random.randint(3, 8)
        for _ in range(num_gold):
            pos = tuple(random.choice(floor_tiles))
            if pos not in occupied:
                self.gold_piles.append(pos)
                occupied.add(pos)

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        
        # Clear one-frame visual effects from previous step
        self.visual_effects.clear()
        self.attack_effect = None
        if self.player['last_hit_timer'] > 0: self.player['last_hit_timer'] -= 1
        for enemy in self.enemies:
            if enemy['last_hit_timer'] > 0: enemy['last_hit_timer'] -= 1

        # Unpack action
        movement_action = action[0]
        attack_action = action[1] == 1

        # --- Player Turn ---
        player_turn_taken = False
        dist_before = math.dist(self.player['pos'], self.exit_pos)

        # 1. Attack action
        if attack_action:
            attacked_enemy = None
            for enemy in self.enemies:
                if math.dist(self.player['pos'], enemy['pos']) < 1.5: # Adjacent
                    # sfx: player_attack.wav
                    enemy['health'] -= self.player['attack_power']
                    enemy['last_hit_timer'] = 2 # Flash for 2 frames (if auto-advancing) or 1 turn
                    self.visual_effects.append({'type': 'hit', 'pos': enemy['pos'], 'timer': 1})
                    self.attack_effect = {'start': self.player['pos'], 'end': enemy['pos'], 'timer': 1}
                    attacked_enemy = enemy
                    player_turn_taken = True
                    break # Attack only one enemy
            
            if attacked_enemy and attacked_enemy['health'] <= 0:
                # sfx: enemy_die.wav
                self.enemies.remove(attacked_enemy)
                reward += 1.0
                self.score += 100

        # 2. Movement action (if no attack was made)
        if not player_turn_taken and movement_action != 0:
            px, py = self.player['pos']
            dx, dy = 0, 0
            if movement_action == 1: dy = -1 # Up
            elif movement_action == 2: dy = 1 # Down
            elif movement_action == 3: dx = -1 # Left
            elif movement_action == 4: dx = 1 # Right
            
            new_pos = (px + dx, py + dy)
            if 0 <= new_pos[0] < self.MAP_WIDTH and 0 <= new_pos[1] < self.MAP_HEIGHT and \
               self.game_map[new_pos[0], new_pos[1]] == 0 and \
               new_pos not in [e['pos'] for e in self.enemies]:
                self.player['pos'] = new_pos
                player_turn_taken = True

        # 3. Wait action (if no move or attack)
        if not player_turn_taken:
            # Player waits a turn
            pass
        
        # Check for reward from movement
        dist_after = math.dist(self.player['pos'], self.exit_pos)
        if dist_after < dist_before:
            reward += 0.01
        elif dist_after > dist_before:
            reward -= 0.01

        # Check for gold collection
        if self.player['pos'] in self.gold_piles:
            # sfx: gold_pickup.wav
            self.gold_piles.remove(self.player['pos'])
            reward += 0.5
            self.score += 50
            self.visual_effects.append({'type': 'gold', 'pos': self.player['pos'], 'timer': 1})

        # Check for reaching exit
        if self.player['pos'] == self.exit_pos:
            if self.floor_level == self.FINAL_FLOOR:
                # sfx: victory.wav
                reward += 100.0
                self.score += 1000 * self.floor_level
                terminated = True
            else:
                # sfx: next_level.wav
                reward += 10.0
                self.score += 200 * self.floor_level
                self.floor_level += 1
                self._generate_floor()

        # --- Enemy Turn ---
        if not terminated:
            for enemy in self.enemies:
                if math.dist(self.player['pos'], enemy['pos']) < 1.5:
                    # sfx: player_hit.wav
                    self.player['health'] -= enemy['damage']
                    self.player['last_hit_timer'] = 2
                    self.visual_effects.append({'type': 'hit', 'pos': self.player['pos'], 'timer': 1})
                else: # Move towards player
                    ex, ey = enemy['pos']
                    px, py = self.player['pos']
                    # Simple greedy move
                    best_move = (ex, ey)
                    min_dist = math.dist((ex, ey), (px, py))
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = ex + dx, ey + dy
                        new_pos = (nx, ny)
                        dist = math.dist(new_pos, (px, py))
                        if 0 <= nx < self.MAP_WIDTH and 0 <= ny < self.MAP_HEIGHT and \
                           self.game_map[nx, ny] == 0 and \
                           new_pos not in [e['pos'] for e in self.enemies] and \
                           new_pos != self.player['pos'] and dist < min_dist:
                            min_dist = dist
                            best_move = new_pos
                    enemy['pos'] = best_move

        # --- Check Termination Conditions ---
        if self.player['health'] <= 0:
            # sfx: game_over.wav
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera offset to center player
        cam_x = self.player['pos'][0] * self.TILE_SIZE - self.SCREEN_WIDTH / 2
        cam_y = self.player['pos'][1] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2

        # Visible tile range
        start_x = max(0, int(cam_x / self.TILE_SIZE))
        end_x = min(self.MAP_WIDTH, int((cam_x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 1)
        start_y = max(0, int(cam_y / self.TILE_SIZE))
        end_y = min(self.MAP_HEIGHT, int((cam_y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 1)

        # Draw map
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                screen_x, screen_y = int(x * self.TILE_SIZE - cam_x), int(y * self.TILE_SIZE - cam_y)
                tile_rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.game_map[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, tile_rect)

        # Draw exit
        ex, ey = self.exit_pos
        screen_x, screen_y = int(ex * self.TILE_SIZE - cam_x), int(ey * self.TILE_SIZE - cam_y)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE))
        
        # Draw gold
        for gx, gy in self.gold_piles:
            screen_x, screen_y = int(gx * self.TILE_SIZE - cam_x), int(gy * self.TILE_SIZE - cam_y)
            center = (screen_x + self.TILE_SIZE // 2, screen_y + self.TILE_SIZE // 2)
            radius = int(self.TILE_SIZE * 0.3 + math.sin(self.steps * 0.5) * 2)
            pygame.draw.circle(self.screen, self.COLOR_GOLD, center, max(0, radius))

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            screen_x, screen_y = int(ex * self.TILE_SIZE - cam_x), int(ey * self.TILE_SIZE - cam_y)
            color = self.COLOR_HIT_EFFECT if enemy['last_hit_timer'] > 0 else self.COLOR_ENEMY_GOBLIN
            pygame.draw.rect(self.screen, color, (screen_x + 4, screen_y + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))

        # Draw player
        px, py = self.player['pos']
        screen_x, screen_y = int(px * self.TILE_SIZE - cam_x), int(py * self.TILE_SIZE - cam_y)
        center = (screen_x + self.TILE_SIZE // 2, screen_y + self.TILE_SIZE // 2)
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.TILE_SIZE // 2 + 2, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.TILE_SIZE // 2 + 2, self.COLOR_PLAYER_GLOW)
        
        color = self.COLOR_HIT_EFFECT if self.player['last_hit_timer'] > 0 else self.COLOR_PLAYER
        pygame.draw.rect(self.screen, color, (screen_x + 2, screen_y + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4))
        
        # Draw attack slash
        if self.attack_effect:
            start_scr_x = int(self.attack_effect['start'][0] * self.TILE_SIZE - cam_x + self.TILE_SIZE // 2)
            start_scr_y = int(self.attack_effect['start'][1] * self.TILE_SIZE - cam_y + self.TILE_SIZE // 2)
            end_scr_x = int(self.attack_effect['end'][0] * self.TILE_SIZE - cam_x + self.TILE_SIZE // 2)
            end_scr_y = int(self.attack_effect['end'][1] * self.TILE_SIZE - cam_y + self.TILE_SIZE // 2)
            pygame.draw.line(self.screen, self.COLOR_ATTACK_SLASH, (start_scr_x, start_scr_y), (end_scr_x, end_scr_y), 3)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player['health'] / self.player['max_health'])
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 20, 200, 25))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (20, 20, 200 * health_ratio, 25))
        health_text = self.font_small.render(f"HP: {self.player['health']}/{self.player['max_health']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (25, 23))

        # Gold
        gold_text = self.font_small.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (20, 55))

        # Floor Level
        floor_text = self.font_large.render(f"Floor: {self.floor_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(floor_text, (self.SCREEN_WIDTH - floor_text.get_width() - 20, 20))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY_GOBLIN)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player['health'],
            "floor": self.floor_level,
            "player_pos": self.player['pos'],
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Event Handling ---
        action = [0, 0, 0] # Default: no-op
        turn_taken = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # Only register an action if the game is not over
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                        turn_taken = True
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                        turn_taken = True
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                        turn_taken = True
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                        turn_taken = True
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                        turn_taken = True
                    elif event.key == pygame.K_r: # Reset button
                        obs, info = env.reset()
                        terminated = False
                        turn_taken = False # Don't step after reset
                        print("--- Game Reset ---")
        
        # --- Game Step ---
        if turn_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Player HP: {info['player_health']}, Floor: {info['floor']}")
            if terminated:
                print("--- Episode Finished ---")
                print(f"Final Score: {info['score']}")

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    env.close()