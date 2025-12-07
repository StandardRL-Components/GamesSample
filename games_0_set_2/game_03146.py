
# Generated: 2025-08-27T22:30:19.778095
# Source Brief: brief_03146.md
# Brief Index: 3146

        
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
        "Controls: Arrow keys to move. Hold Space to attack in your facing direction. "
        "Hold Shift to open chests in your facing direction."
    )

    game_description = (
        "Explore a procedurally generated dungeon, battling enemies and collecting "
        "treasure to find the exit. Your turn, then the enemies' turn."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.TILE_SIZE = 36
        self.MAX_STEPS = 2000
        self.PLAYER_MAX_HEALTH = 10.0
        self.ENEMY_MAX_HEALTH = 2.0
        self.ENEMY_RESPAWN_TURNS = 20
        self.CHEST_RESPAWN_TURNS = 30

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_FLOOR = (60, 45, 40)
        self.COLOR_WALL = (110, 110, 120)
        self.COLOR_PLAYER = (50, 220, 100)
        self.COLOR_EXIT = (150, 50, 255)
        self.COLOR_CHEST = (255, 200, 0)
        self.COLOR_ENEMY_TYPES = [
            (220, 50, 50),   # 0: Random Mover
            (230, 100, 50),  # 1: Horizontal Chaser
            (240, 50, 100),  # 2: Vertical Chaser
            (250, 120, 120), # 3: Diagonal Chaser
            (200, 80, 80),   # 4: Stationary Attacker
        ]
        self.COLOR_HEALTH_FG = (40, 200, 40)
        self.COLOR_HEALTH_BG = (180, 40, 40)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_GREY = (180, 180, 180)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # --- Grid and Camera ---
        self.grid_width = self.GRID_SIZE * self.TILE_SIZE
        self.grid_height = self.GRID_SIZE * self.TILE_SIZE
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_height) // 2

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.player_pos = None
        self.player_health = None
        self.player_facing_dir = None
        self.exit_pos = None
        self.enemies = None
        self.chests = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = None
        self.enemy_respawn_timer = None
        self.chest_respawn_timer = None
        self.base_enemy_damage = None
        self.last_dist_to_exit = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_dir = (0, 1)  # Down
        self.particles = []
        self.enemy_respawn_timer = self.ENEMY_RESPAWN_TURNS
        self.chest_respawn_timer = self.CHEST_RESPAWN_TURNS
        self.base_enemy_damage = 0.5

        self._generate_dungeon()
        
        self.last_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

        return self._get_observation(), self._get_info()
    
    def _generate_dungeon(self):
        self.grid = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=int) # 1 = wall
        
        # Recursive backtracking for maze generation
        stack = deque()
        start_x, start_y = self.np_random.integers(0, self.GRID_SIZE, size=2)
        self.grid[start_y, start_x] = 0
        stack.append((start_x, start_y))
        
        floor_tiles = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                wall_x, wall_y = x + (nx - x) // 2, y + (ny - y) // 2
                self.grid[ny, nx] = 0
                self.grid[wall_y, wall_x] = 0
                floor_tiles.extend([(nx, ny), (wall_x, wall_y)])
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Place objects
        self.np_random.shuffle(floor_tiles)
        
        self.player_pos = list(floor_tiles.pop())
        self.exit_pos = list(floor_tiles.pop())

        num_enemies = self.np_random.integers(3, 6)
        num_chests = self.np_random.integers(2, 5)
        
        self.enemies = []
        for _ in range(num_enemies):
            if not floor_tiles: break
            pos = list(floor_tiles.pop())
            self.enemies.append({
                "pos": pos,
                "health": self.ENEMY_MAX_HEALTH,
                "type": self.np_random.integers(0, len(self.COLOR_ENEMY_TYPES))
            })

        self.chests = []
        for _ in range(num_chests):
            if not floor_tiles: break
            pos = list(floor_tiles.pop())
            self.chests.append(pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        
        # --- Player Turn ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Movement
        moved = False
        if movement != 0:
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_map[movement]
            self.player_facing_dir = (dx, dy)
            
            next_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if self._is_valid_and_empty(next_pos):
                self.player_pos = next_pos
                moved = True

        # 2. Interaction (Shift)
        if shift_held:
            target_pos = [self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1]]
            chest_idx = self._find_chest_at(target_pos)
            if chest_idx is not None:
                self.chests.pop(chest_idx)
                self.score += 10
                reward += 2
                self._spawn_particles(target_pos, 20, self.COLOR_CHEST, 1.5)
                # SFX: Chest open sound

        # 3. Attack (Space)
        if space_held:
            target_pos = [self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1]]
            enemy_idx = self._find_enemy_at(target_pos)
            self._spawn_attack_slash(target_pos)
            # SFX: Sword swing
            if enemy_idx is not None:
                self.enemies[enemy_idx]["health"] -= 1
                self._spawn_particles(target_pos, 10, self.COLOR_WHITE, 1.0)
                if self.enemies[enemy_idx]["health"] <= 0:
                    self.enemies.pop(enemy_idx)
                    self.score += 25
                    reward += 5
                    # SFX: Enemy defeated
                else:
                    # SFX: Enemy hit
                    pass

        # --- Enemy Turn ---
        current_enemy_damage = self.base_enemy_damage + (self.steps // 200) * 0.1
        enemies_to_process = self.enemies[:]
        for enemy in enemies_to_process:
            if enemy not in self.enemies: continue # Skip if defeated this turn
            
            # Simple AI based on type
            ex, ey = enemy["pos"]
            px, py = self.player_pos
            
            # Attack if adjacent
            if self._manhattan_distance(enemy["pos"], self.player_pos) == 1:
                self.player_health -= current_enemy_damage
                self._spawn_particles(self.player_pos, 15, self.COLOR_ENEMY_TYPES[0], 2.0)
                # SFX: Player takes damage
                continue

            # Movement AI
            next_enemy_pos = list(enemy["pos"])
            enemy_type = enemy["type"]
            
            if enemy_type == 0: # Random
                dx, dy = self.np_random.choice([-1, 1], size=2)
                if self.np_random.random() < 0.5: next_enemy_pos[0] += dx
                else: next_enemy_pos[1] += dy
            elif enemy_type == 1 and ey == py: # Horizontal
                next_enemy_pos[0] += np.sign(px - ex)
            elif enemy_type == 2 and ex == px: # Vertical
                next_enemy_pos[1] += np.sign(py - ey)
            elif enemy_type == 3: # Diagonal
                next_enemy_pos[0] += np.sign(px - ex)
                next_enemy_pos[1] += np.sign(py - ey)
            elif enemy_type == 4: # Stationary attacker (attacks in + shape)
                 if (abs(px - ex) <= 1 and py == ey) or (abs(py - ey) <= 1 and px == ex):
                    self.player_health -= current_enemy_damage
                    self._spawn_particles(self.player_pos, 15, self.COLOR_ENEMY_TYPES[0], 2.0)
                    # SFX: Player takes damage
            
            if self._is_valid_and_empty(next_enemy_pos, check_player=True):
                enemy["pos"] = next_enemy_pos

        # --- State Updates ---
        self.steps += 1
        
        # Respawn logic
        self.enemy_respawn_timer -= 1
        self.chest_respawn_timer -= 1
        if self.enemy_respawn_timer <= 0:
            self._respawn_entities('enemy')
            self.enemy_respawn_timer = self.ENEMY_RESPAWN_TURNS
        if self.chest_respawn_timer <= 0:
            self._respawn_entities('chest')
            self.chest_respawn_timer = self.CHEST_RESPAWN_TURNS

        # Movement reward
        new_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        if new_dist_to_exit < self.last_dist_to_exit:
            reward += 0.1
        elif new_dist_to_exit > self.last_dist_to_exit:
            reward -= 0.2
        self.last_dist_to_exit = new_dist_to_exit
        
        # --- Termination Check ---
        if self.player_health <= 0:
            reward -= 50
            terminated = True
            self.game_over = True
        elif self.player_pos == self.exit_pos:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = self._get_screen_rect(x, y)
                color = self.COLOR_WALL if self.grid[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self._get_screen_rect(*self.exit_pos), border_radius=5)
        
        # Draw chests
        for pos in self.chests:
            pygame.draw.rect(self.screen, self.COLOR_CHEST, self._get_screen_rect(*pos), border_radius=3)
        
        # Draw enemies
        for enemy in self.enemies:
            rect = self._get_screen_rect(*enemy["pos"])
            color = self.COLOR_ENEMY_TYPES[enemy["type"]]
            pygame.draw.rect(self.screen, color, rect)
            self._render_health_bar(rect.topleft, enemy["health"], self.ENEMY_MAX_HEALTH)
        
        # Draw player
        player_rect = self._get_screen_rect(*self.player_pos)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        self._render_health_bar(player_rect.topleft, self.player_health, self.PLAYER_MAX_HEALTH)
        
        # Draw facing indicator
        center_x = player_rect.centerx
        center_y = player_rect.centery
        indicator_end_x = center_x + self.player_facing_dir[0] * self.TILE_SIZE * 0.4
        indicator_end_y = center_y + self.player_facing_dir[1] * self.TILE_SIZE * 0.4
        pygame.draw.line(self.screen, self.COLOR_WHITE, (center_x, center_y), (indicator_end_x, indicator_end_y), 3)

        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Health
        health_text = self.font_large.render(f"Health: {max(0, math.ceil(self.player_health))}", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"Turn: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_GREY)
        self.screen.blit(steps_text, (10, self.HEIGHT - steps_text.get_height() - 10))

    def _render_health_bar(self, top_left, current_hp, max_hp):
        x, y = top_left
        bar_width = self.TILE_SIZE
        bar_height = 5
        hp_ratio = max(0, current_hp) / max_hp
        
        bg_rect = pygame.Rect(x, y - bar_height - 2, bar_width, bar_height)
        fg_rect = pygame.Rect(x, y - bar_height - 2, bar_width * hp_ratio, bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect)

    def _get_screen_rect(self, x, y):
        return pygame.Rect(
            self.grid_offset_x + x * self.TILE_SIZE,
            self.grid_offset_y + y * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE,
        )

    def _get_grid_pos_from_screen(self, screen_pos):
        return (
            (screen_pos[0] - self.grid_offset_x) // self.TILE_SIZE,
            (screen_pos[1] - self.grid_offset_y) // self.TILE_SIZE
        )

    def _is_valid_and_empty(self, pos, check_player=False):
        x, y = pos
        if not (0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE):
            return False
        if self.grid[y, x] == 1:
            return False
        if self._find_enemy_at(pos) is not None:
            return False
        if check_player and self.player_pos == pos:
            return False
        return True

    def _find_enemy_at(self, pos):
        for i, enemy in enumerate(self.enemies):
            if enemy["pos"] == pos:
                return i
        return None

    def _find_chest_at(self, pos):
        for i, chest in enumerate(self.chests):
            if chest == pos:
                return i
        return None

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _spawn_particles(self, grid_pos, count, color, speed_mult):
        rect = self._get_screen_rect(*grid_pos)
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = (self.np_random.random() * 2 + 1) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(rect.center),
                "vel": vel,
                "lifetime": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.random() * 2 + 2
            })
    
    def _spawn_attack_slash(self, grid_pos):
        rect = self._get_screen_rect(*grid_pos)
        center = list(rect.center)
        for i in range(15):
            self.particles.append({
                "pos": [center[0] - self.player_facing_dir[0] * 15 + self.np_random.uniform(-5, 5),
                        center[1] - self.player_facing_dir[1] * 15 + self.np_random.uniform(-5, 5)],
                "vel": [self.player_facing_dir[0] * 2, self.player_facing_dir[1] * 2],
                "lifetime": 8,
                "color": self.COLOR_WHITE,
                "radius": 3 - i * 0.1
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            p["radius"] *= 0.95
            if p["radius"] > 1:
                 pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), p["color"])
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _respawn_entities(self, entity_type):
        floor_tiles = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y,x] == 0:
                    pos = [x,y]
                    if self._is_valid_and_empty(pos, check_player=True) and self._find_chest_at(pos) is None:
                        floor_tiles.append(pos)
        
        if not floor_tiles: return

        pos_to_spawn = list(self.np_random.choice(floor_tiles, axis=0))

        if entity_type == 'enemy' and len(self.enemies) < 5:
            self.enemies.append({
                "pos": pos_to_spawn,
                "health": self.ENEMY_MAX_HEALTH,
                "type": self.np_random.integers(0, len(self.COLOR_ENEMY_TYPES))
            })
        elif entity_type == 'chest' and len(self.chests) < 4:
            self.chests.append(pos_to_spawn)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test assertions
        assert 0 <= self.player_pos[0] < self.GRID_SIZE and 0 <= self.player_pos[1] < self.GRID_SIZE
        assert self.player_health <= self.PLAYER_MAX_HEALTH
        assert self.score >= 0

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping from keyboard ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Event handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Any keydown triggers a step in turn-based mode
                action_taken = True
        
        # For turn-based, only step on an action
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

            if terminated:
                print("Game Over!")
                print(f"Final Score: {info['score']}, Final Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    pygame.quit()