
# Generated: 2025-08-28T02:07:59.035387
# Source Brief: brief_01601.md
# Brief Index: 1601

        
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
        "Controls: Arrow keys to move. Hold Space to attack in your last moved direction. "
        "Reach the blue exit with 50+ gold to win."
    )

    game_description = (
        "A retro pixel-art dungeon crawler. Navigate a maze, fight enemies, and collect gold. "
        "Your goal is to find the exit, but you'll need enough treasure to escape."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.TILE_SIZE = 32
        self.DUNGEON_WIDTH, self.DUNGEON_HEIGHT = 41, 41  # Odd numbers for maze generation
        self.MAX_STEPS = 1000
        self.WIN_GOLD_TARGET = 50
        self.CHASE_RADIUS = 8

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (70, 70, 80)
        self.COLOR_WALL_BORDER = (50, 50, 60)
        self.COLOR_FLOOR = (100, 80, 60)
        self.COLOR_FLOOR_ACCENT = (90, 70, 50)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_PLAYER_EYE = (255, 255, 255)
        self.COLOR_EXIT = (50, 100, 255)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_GOLD_SPARKLE = (255, 255, 150)
        self.COLOR_ENEMY_RANDOM = (200, 50, 50)
        self.COLOR_ENEMY_PATROL = (200, 100, 50)
        self.COLOR_ENEMY_CHASE = (220, 0, 100)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_UI_BACK = (40, 40, 55, 200)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 28)
        self.msg_font = pygame.font.Font(None, 48)

        # --- Game State (Persistent) ---
        self.successful_exits = 0

        # --- Game State (Per-Episode) ---
        self.dungeon = None
        self.player_pos = None
        self.player_health = None
        self.player_gold = None
        self.last_move_dir = None
        self.enemies = None
        self.gold_pieces = None
        self.exit_pos = None
        self.steps = None
        self.game_over = None
        self.last_dist_to_exit = None
        self.particles = None
        self.win_message = ""

        # Initial reset to populate state variables
        self.reset()
        
        # self.validate_implementation() # For testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.player_health = 100
        self.player_gold = 0
        self.last_move_dir = np.array([1, 0])  # Start facing right
        self.particles = []

        self._generate_dungeon()

        self.last_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.particles.clear()
        reward = 0

        # 1. Process Player Action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        moved = False
        if movement != 0:
            move_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}
            move_dir = np.array(move_map[movement])
            self.last_move_dir = move_dir
            
            target_pos = self.player_pos + move_dir
            if self._is_walkable(target_pos[0], target_pos[1]):
                self.player_pos = target_pos
                moved = True
        
        if space_held:
            # sfx: player_attack_swing
            attack_pos = self.player_pos + self.last_move_dir
            self._add_particles(self._grid_to_pixel(attack_pos) + self.TILE_SIZE // 2, 5, self.COLOR_PLAYER_EYE, 1)
            
            enemy_to_remove = None
            for enemy in self.enemies:
                if np.array_equal(enemy['pos'], attack_pos):
                    enemy['health'] -= 25 # Player deals 25 damage
                    if enemy['health'] <= 0:
                        # sfx: enemy_die
                        reward += 1.0
                        self.score += 100
                        enemy_to_remove = enemy
                        self._add_particles(self._grid_to_pixel(enemy['pos']) + self.TILE_SIZE // 2, 20, enemy['color'], 3)
                    else:
                        # sfx: enemy_hit
                        self._add_particles(self._grid_to_pixel(enemy['pos']) + self.TILE_SIZE // 2, 10, (255,255,255), 2)
                    break
            if enemy_to_remove:
                self.enemies.remove(enemy_to_remove)

        # 2. Process Environment Interactions
        if moved:
            # Gold pickup
            gold_to_remove = None
            for gold_pos in self.gold_pieces:
                if np.array_equal(self.player_pos, gold_pos):
                    # sfx: collect_gold
                    self.player_gold += 10
                    reward += 1.0
                    self.score += 50
                    gold_to_remove = gold_pos
                    break
            if gold_to_remove is not None:
                self.gold_pieces = [gp for gp in self.gold_pieces if not np.array_equal(gp, gold_to_remove)]

        # 3. Process Enemy Turns
        for enemy in self.enemies:
            if self._manhattan_distance(self.player_pos, enemy['pos']) == 1:
                # sfx: player_hit
                self.player_health -= 10
                self._add_particles(self._grid_to_pixel(self.player_pos) + self.TILE_SIZE // 2, 15, self.COLOR_ENEMY_RANDOM, 3)
            else:
                self._move_enemy(enemy)

        # 4. Calculate Rewards & Check Termination
        new_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        reward += (self.last_dist_to_exit - new_dist) * 0.1
        self.last_dist_to_exit = new_dist

        terminated = False
        if self.player_health <= 0:
            # sfx: game_over_lose
            self.player_health = 0
            terminated = True
            reward = -100
            self.game_over = True
            self.win_message = "YOU DIED"
        elif np.array_equal(self.player_pos, self.exit_pos):
            terminated = True
            self.game_over = True
            if self.player_gold >= self.WIN_GOLD_TARGET:
                # sfx: game_over_win
                reward = 100
                self.score += 1000
                self.successful_exits += 1
                self.win_message = "YOU ESCAPED!"
            else:
                reward = -50
                self.win_message = "NEED MORE GOLD!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME'S UP!"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_dungeon(self):
        self.dungeon = np.ones((self.DUNGEON_WIDTH, self.DUNGEON_HEIGHT), dtype=np.int8) # 1 = wall
        
        stack = deque()
        start_x, start_y = (self.np_random.integers(1, self.DUNGEON_WIDTH // 2) * 2 - 1,
                            self.np_random.integers(1, self.DUNGEON_HEIGHT // 2) * 2 - 1)
        
        self.dungeon[start_x, start_y] = 0 # 0 = floor
        stack.append((start_x, start_y))
        
        furthest_cell = ((start_x, start_y), 0)

        while stack:
            x, y = stack[-1]
            
            # Update furthest cell
            dist = abs(x - start_x) + abs(y - start_y)
            if dist > furthest_cell[1]:
                furthest_cell = ((x, y), dist)

            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.DUNGEON_WIDTH and 0 <= ny < self.DUNGEON_HEIGHT and self.dungeon[nx, ny] == 1:
                    neighbors.append((nx, ny, dx // 2, dy // 2))
            
            if neighbors:
                nx, ny, hx, hy = self.np_random.choice(neighbors, 1)[0]
                self.dungeon[nx, ny] = 0
                self.dungeon[x + hx, y + hy] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Place player, exit, items
        self.player_pos = np.array([start_x, start_y])
        self.exit_pos = np.array(furthest_cell[0])
        
        floor_tiles = np.argwhere(self.dungeon == 0)
        self.np_random.shuffle(floor_tiles)
        
        num_gold = 15
        num_enemies = 10
        
        valid_spawns = [tuple(tile) for tile in floor_tiles 
                        if self._manhattan_distance(tile, self.player_pos) > 5 and
                           self._manhattan_distance(tile, self.exit_pos) > 5]
        
        self.np_random.shuffle(valid_spawns)

        self.gold_pieces = [np.array(pos) for pos in valid_spawns[:num_gold]]
        
        enemy_health = 10 + 5 * self.successful_exits
        self.enemies = []
        for pos in valid_spawns[num_gold:num_gold + num_enemies]:
            enemy_type = self.np_random.integers(0, 3)
            color = [self.COLOR_ENEMY_RANDOM, self.COLOR_ENEMY_PATROL, self.COLOR_ENEMY_CHASE][enemy_type]
            self.enemies.append({
                'pos': np.array(pos),
                'type': enemy_type,
                'health': enemy_health,
                'max_health': enemy_health,
                'color': color,
                'patrol_dir': self.np_random.choice([[1,0], [-1,0], [0,1], [0,-1]])
            })

    def _move_enemy(self, enemy):
        # Type 0: Random
        if enemy['type'] == 0:
            moves = [[0,1], [0,-1], [1,0], [-1,0]]
            self.np_random.shuffle(moves)
            for move in moves:
                target = enemy['pos'] + move
                if self._is_walkable(target[0], target[1]) and not self._is_occupied_by_enemy(target):
                    enemy['pos'] = target
                    break
        # Type 1: Patrol
        elif enemy['type'] == 1:
            target = enemy['pos'] + enemy['patrol_dir']
            if self._is_walkable(target[0], target[1]) and not self._is_occupied_by_enemy(target):
                enemy['pos'] = target
            else:
                enemy['patrol_dir'] = -enemy['patrol_dir']
        # Type 2: Chase
        elif enemy['type'] == 2:
            if self._manhattan_distance(self.player_pos, enemy['pos']) <= self.CHASE_RADIUS:
                dx, dy = self.player_pos - enemy['pos']
                move = np.array([np.sign(dx), np.sign(dy)])
                if move[0] != 0 and move[1] != 0: # Prefer cardinal moves
                    if self.np_random.random() > 0.5: move[1] = 0
                    else: move[0] = 0
                
                target = enemy['pos'] + move
                if self._is_walkable(target[0], target[1]) and not self._is_occupied_by_enemy(target):
                    enemy['pos'] = target
                else: # Try other direction if blocked
                    move = np.flip(move)
                    target = enemy['pos'] + move
                    if self._is_walkable(target[0], target[1]) and not self._is_occupied_by_enemy(target):
                        enemy['pos'] = target

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_offset = self._grid_to_pixel(self.player_pos) - np.array([self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2])

        # Determine visible grid range
        start_x = max(0, cam_offset[0] // self.TILE_SIZE)
        end_x = min(self.DUNGEON_WIDTH, (cam_offset[0] + self.SCREEN_WIDTH) // self.TILE_SIZE + 2)
        start_y = max(0, cam_offset[1] // self.TILE_SIZE)
        end_y = min(self.DUNGEON_HEIGHT, (cam_offset[1] + self.SCREEN_HEIGHT) // self.TILE_SIZE + 2)

        # Draw dungeon
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                screen_pos = np.array([x, y]) * self.TILE_SIZE - cam_offset
                rect = pygame.Rect(int(screen_pos[0]), int(screen_pos[1]), self.TILE_SIZE, self.TILE_SIZE)
                if self.dungeon[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_BORDER, rect, 1)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    if (x + y) % 2 == 0:
                         pygame.draw.rect(self.screen, self.COLOR_FLOOR_ACCENT, rect.inflate(-self.TILE_SIZE*0.8, -self.TILE_SIZE*0.8))

        # Draw exit
        screen_pos = self._grid_to_pixel(self.exit_pos) - cam_offset
        rect = pygame.Rect(int(screen_pos[0]), int(screen_pos[1]), self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)
        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.TILE_SIZE // 4, (*self.COLOR_EXIT, 100))
        
        # Draw gold
        for gold_pos in self.gold_pieces:
            screen_pos = self._grid_to_pixel(gold_pos) - cam_offset
            center = (int(screen_pos[0] + self.TILE_SIZE // 2), int(screen_pos[1] + self.TILE_SIZE // 2))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.TILE_SIZE // 4, self.COLOR_GOLD)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.TILE_SIZE // 4, self.COLOR_GOLD)
            pygame.gfxdraw.pixel(self.screen, center[0] + 2, center[1] - 2, self.COLOR_GOLD_SPARKLE)

        # Draw enemies
        for enemy in self.enemies:
            screen_pos = self._grid_to_pixel(enemy['pos']) - cam_offset
            center = (int(screen_pos[0] + self.TILE_SIZE // 2), int(screen_pos[1] + self.TILE_SIZE // 2))
            size = self.TILE_SIZE * 0.7
            rect = pygame.Rect(center[0] - size // 2, center[1] - size // 2, size, size)
            
            if enemy['type'] == 0: # Circle
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(size/2), enemy['color'])
            elif enemy['type'] == 1: # Square
                pygame.draw.rect(self.screen, enemy['color'], rect)
            elif enemy['type'] == 2: # Triangle
                points = [(center[0], center[1] - size//2), (center[0] - size//2, center[1] + size//2), (center[0] + size//2, center[1] + size//2)]
                pygame.gfxdraw.filled_polygon(self.screen, points, enemy['color'])
            
            # Health bar
            if enemy['health'] < enemy['max_health']:
                bar_w = self.TILE_SIZE * 0.8
                bar_h = 4
                health_pct = enemy['health'] / enemy['max_health']
                pygame.draw.rect(self.screen, (50,0,0), (center[0] - bar_w/2, rect.top - bar_h - 2, bar_w, bar_h))
                pygame.draw.rect(self.screen, (255,0,0), (center[0] - bar_w/2, rect.top - bar_h - 2, bar_w * health_pct, bar_h))


        # Draw player
        player_screen_pos = np.array([self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2])
        size = self.TILE_SIZE * 0.8
        rect = pygame.Rect(int(player_screen_pos[0] - size/2), int(player_screen_pos[1] - size/2), int(size), int(size))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=3)
        
        # Eyes indicating direction
        eye_offset_x = self.last_move_dir[0] * size * 0.25
        eye_offset_y = self.last_move_dir[1] * size * 0.25
        eye_pos1 = (rect.centerx + eye_offset_x - size*0.15*self.last_move_dir[1], rect.centery + eye_offset_y - size*0.15*self.last_move_dir[0])
        eye_pos2 = (rect.centerx + eye_offset_x + size*0.15*self.last_move_dir[1], rect.centery + eye_offset_y + size*0.15*self.last_move_dir[0])
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, eye_pos1, 2)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, eye_pos2, 2)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'] - cam_offset, p['size'])

    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((200, 70), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BACK)
        self.screen.blit(ui_panel, (10, 10))

        # Health
        health_text = self.ui_font.render(f"HP: {self.player_health}/100", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (20, 20))
        
        # Gold
        gold_text = self.ui_font.render(f"GOLD: {self.player_gold}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gold_text, (20, 50))

        # Game Over Message
        if self.game_over and self.win_message:
            msg_surf = self.msg_font.render(self.win_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "gold": self.player_gold,
            "pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def _is_walkable(self, x, y):
        if not (0 <= x < self.DUNGEON_WIDTH and 0 <= y < self.DUNGEON_HEIGHT):
            return False
        return self.dungeon[x, y] == 0

    def _is_occupied_by_enemy(self, pos):
        for enemy in self.enemies:
            if np.array_equal(enemy['pos'], pos):
                return True
        return False

    def _manhattan_distance(self, pos1, pos2):
        return np.sum(np.abs(np.array(pos1) - np.array(pos2)))

    def _grid_to_pixel(self, grid_pos):
        return np.array(grid_pos) * self.TILE_SIZE

    def _add_particles(self, pos, count, color, max_size):
        for _ in range(count):
            self.particles.append({
                'pos': pos + self.np_random.standard_normal(2) * max_size * 2,
                'color': color,
                'size': self.np_random.integers(1, max_size + 1)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires a window, so we'll re-init pygame for display
    pygame.display.init()
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)

    while True:
        # Get action from keyboard
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        # In a manual play loop, we only step when an action is taken.
        if any(keys):
            if terminated:
                print("Game over. Resetting.")
                obs, info = env.reset()
                terminated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            
            # Draw the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Since auto_advance is False, we need a delay to make it playable
            pygame.time.wait(100)