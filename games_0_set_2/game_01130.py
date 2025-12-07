
# Generated: 2025-08-27T16:08:30.911201
# Source Brief: brief_01130.md
# Brief Index: 1130

        
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
        "Controls: Arrow keys to move. Space to attack in the last moved direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated dungeon, battling enemies to reach the golden exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False # Turn-based roguelike

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (70, 70, 80)
        self.COLOR_FLOOR = (40, 40, 50)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_ACCENT = (150, 220, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_ACCENT = (255, 150, 150)
        self.COLOR_EXIT = (255, 215, 0)
        self.COLOR_EXIT_ACCENT = (255, 255, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_HIGH = (0, 200, 0)
        self.COLOR_HEALTH_MID = (255, 255, 0)
        self.COLOR_HEALTH_LOW = (200, 0, 0)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Game state variables (initialized in reset)
        self.dungeon = None
        self.player_pos = None
        self.player_health = None
        self.max_player_health = None
        self.player_last_move_dir = None
        self.enemies = None
        self.exit_pos = None
        self.steps = None
        self.kill_count = None
        self.level = None
        self.game_over = None
        self.visual_effects = None
        
        self.reset()
        
        self.validate_implementation()

    def _generate_dungeon(self):
        dungeon = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        path_len = (self.GRID_WIDTH * self.GRID_HEIGHT) // 4
        
        # Ensure start is on a floor tile
        start_x, start_y = self.np_random.integers(1, self.GRID_WIDTH - 2), self.np_random.integers(1, self.GRID_HEIGHT - 2)
        x, y = start_x, start_y
        dungeon[x, y] = 0
        
        for _ in range(path_len):
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            # Move two steps to create thicker corridors
            for _ in range(2):
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.GRID_WIDTH - 1 and 1 <= ny < self.GRID_HEIGHT - 1:
                    x, y = nx, ny
                    dungeon[x, y] = 0
        
        self.dungeon = dungeon
        self.player_pos = (start_x, start_y)
        self.exit_pos = (x, y)

        if self.player_pos == self.exit_pos:
            if dungeon[x-1, y] == 0: self.exit_pos = (x-1, y)
            elif dungeon[x+1, y] == 0: self.exit_pos = (x+1, y)
            else: self.exit_pos = (x, y-1)

    def _spawn_enemies(self):
        self.enemies = []
        num_enemies = 10
        for _ in range(num_enemies):
            for _ in range(100): # Failsafe attempts
                x = self.np_random.integers(0, self.GRID_WIDTH)
                y = self.np_random.integers(0, self.GRID_HEIGHT)
                pos = (x, y)
                is_wall = self.dungeon[x, y] == 1
                is_player = pos == self.player_pos
                is_exit = pos == self.exit_pos
                is_other_enemy = any(e['pos'] == pos for e in self.enemies)
                if not is_wall and not is_player and not is_exit and not is_other_enemy:
                    self.enemies.append({'pos': pos, 'health': 1})
                    break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.kill_count = 0
        self.level = 1
        self.game_over = False
        self.max_player_health = 5
        self.player_health = self.max_player_health
        self.player_last_move_dir = (0, 1) # Facing down
        self.visual_effects = []

        self._generate_dungeon()
        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Survival reward
        
        movement, space_held, _ = action
        player_action_taken = movement != 0 or space_held == 1
        
        # 1. Handle Player Movement
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
        if movement in move_map:
            dx, dy = move_map[movement]
            self.player_last_move_dir = (dx, dy)
            px, py = self.player_pos
            nx, ny = px + dx, py + dy
            
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.dungeon[nx, ny] == 0:
                self.player_pos = (nx, ny)
        
        # 2. Handle Player Attack
        if space_held:
            # sfx: player_attack_swing.wav
            atk_dx, atk_dy = self.player_last_move_dir
            px, py = self.player_pos
            target_pos = (px + atk_dx, py + atk_dy)
            
            self._add_visual_effect(target_pos, 'attack_swing', 1)
            enemy_hit = next((enemy for enemy in self.enemies if enemy['pos'] == target_pos), None)
            
            if enemy_hit:
                # sfx: enemy_hit_and_die.wav
                self.enemies.remove(enemy_hit)
                reward += 1.0
                self.kill_count += 1
                self._add_visual_effect(target_pos, 'death_particles', 1)
        
        # 3. Handle Enemy Turn (if player acted)
        if player_action_taken:
            for enemy in self.enemies:
                ex, ey = enemy['pos']
                px, py = self.player_pos
                
                dist_to_player = abs(ex - px) + abs(ey - py)

                if dist_to_player == 1: # Attack if adjacent
                    # sfx: player_damage.wav
                    self.player_health -= 1
                    self._add_visual_effect(self.player_pos, 'damage_flash', 1)
                elif 1 < dist_to_player <= 3: # Chase player
                    # Simple greedy chase
                    if px > ex and self.dungeon[ex + 1, ey] == 0 and not self._is_occupied((ex + 1, ey), include_player=False):
                        enemy['pos'] = (ex + 1, ey)
                    elif px < ex and self.dungeon[ex - 1, ey] == 0 and not self._is_occupied((ex - 1, ey), include_player=False):
                        enemy['pos'] = (ex - 1, ey)
                    elif py > ey and self.dungeon[ex, ey + 1] == 0 and not self._is_occupied((ex, ey + 1), include_player=False):
                        enemy['pos'] = (ex, ey + 1)
                    elif py < ey and self.dungeon[ex, ey - 1] == 0 and not self._is_occupied((ex, ey - 1), include_player=False):
                        enemy['pos'] = (ex, ey - 1)
                else: # Patrol randomly
                    dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.dungeon[nx,ny] == 0 and not self._is_occupied((nx, ny)):
                        enemy['pos'] = (nx, ny)
        
        self.steps += 1
        
        # 4. Check Termination Conditions
        terminated = False
        if self.player_health <= 0:
            # sfx: game_over.wav
            reward = -100.0
            terminated = True
            self.game_over = True
        elif self.player_pos == self.exit_pos:
            # sfx: level_complete.wav
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_occupied(self, pos, include_player=True):
        if include_player and pos == self.player_pos:
            return True
        return any(enemy['pos'] == pos for enemy in self.enemies)

    def _add_visual_effect(self, pos, effect_type, duration):
        self.visual_effects.append({'pos': pos, 'type': effect_type, 'duration': duration})

    def _update_and_draw_effects(self):
        self.visual_effects = [e for e in self.visual_effects if e['duration'] > 0]
        for effect in self.visual_effects:
            effect['duration'] -= 1
            gx, gy = effect['pos']
            px, py = gx * self.TILE_SIZE, gy * self.TILE_SIZE
            
            if effect['type'] == 'attack_swing':
                pygame.draw.rect(self.screen, (255, 255, 255), (px, py, self.TILE_SIZE, self.TILE_SIZE), 3)
            elif effect['type'] == 'damage_flash':
                s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                s.fill((255, 0, 0, 128))
                self.screen.blit(s, (px, py))
            elif effect['type'] == 'death_particles':
                for _ in range(5):
                    rx = px + self.np_random.integers(0, self.TILE_SIZE)
                    ry = py + self.np_random.integers(0, self.TILE_SIZE)
                    pygame.draw.circle(self.screen, self.COLOR_ENEMY_ACCENT, (rx, ry), self.np_random.integers(1, 4))

    def _render_game(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color = self.COLOR_WALL if self.dungeon[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        ex, ey = self.exit_pos
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_EXIT_ACCENT, (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), 2)

        for enemy in self.enemies:
            ex, ey = enemy['pos']
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_ACCENT, (ex * self.TILE_SIZE + 2, ey * self.TILE_SIZE + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4), 2)
            
        px, py = self.player_pos
        player_rect = (px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, player_rect, 2)
        
        p_center_x = px * self.TILE_SIZE + self.TILE_SIZE // 2
        p_center_y = py * self.TILE_SIZE + self.TILE_SIZE // 2
        f_dx, f_dy = self.player_last_move_dir
        indicator_x = p_center_x + f_dx * (self.TILE_SIZE // 3)
        indicator_y = p_center_y + f_dy * (self.TILE_SIZE // 3)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_ACCENT, (p_center_x, p_center_y), (indicator_x, indicator_y), 3)

        self._update_and_draw_effects()

    def _render_ui(self):
        health_ratio = max(0, self.player_health / self.max_player_health)
        health_bar_width, health_bar_height, health_bar_x, health_bar_y = 200, 20, 10, 10
        current_health_width = int(health_bar_width * health_ratio)
        health_color = self.COLOR_HEALTH_LOW if health_ratio <= 0.3 else (self.COLOR_HEALTH_MID if health_ratio <= 0.6 else self.COLOR_HEALTH_HIGH)

        pygame.draw.rect(self.screen, (50, 50, 50), (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, health_color, (health_bar_x, health_bar_y, current_health_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), 2)
        
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.max_player_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (health_bar_x + health_bar_width + 10, health_bar_y))

        level_text = self.font_small.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"Kills: {self.kill_count}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 35))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU DIED" if self.player_health <= 0 else "VICTORY!"
            msg_color = self.COLOR_ENEMY if self.player_health <= 0 else self.COLOR_EXIT
            msg_render = self.font_large.render(msg, True, msg_color)
            msg_rect = msg_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_render, msg_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.kill_count,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Roguelike Dungeon")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    key_action = -1
                    if event.key == pygame.K_UP: key_action = 1
                    elif event.key == pygame.K_DOWN: key_action = 2
                    elif event.key == pygame.K_LEFT: key_action = 3
                    elif event.key == pygame.K_RIGHT: key_action = 4
                    elif event.key == pygame.K_SPACE: action[1] = 1
                    
                    if key_action != -1:
                        action[0] = key_action
                    
                    if action != [0, 0, 0]:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()