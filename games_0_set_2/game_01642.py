
# Generated: 2025-08-27T17:48:12.582481
# Source Brief: brief_01642.md
# Brief Index: 1642

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Space to attack adjacent enemies. "
        "Stand still (no-op) to defend, halving incoming damage for one turn."
    )

    game_description = (
        "Explore a procedurally generated dungeon, battling enemies and "
        "collecting experience to defeat the final boss on level 10."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 32
    DUNGEON_WIDTH = 50
    DUNGEON_HEIGHT = 50

    # Colors
    COLOR_BG = (10, 5, 15)
    COLOR_WALL = (40, 40, 50)
    COLOR_FLOOR = (80, 70, 60)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150)
    COLOR_STAIRS = (200, 50, 200)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_BOSS = (255, 100, 100)
    COLOR_UI_BG = (30, 30, 40, 200)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_HP_BAR_BG = (80, 20, 20)
    COLOR_HP_BAR_FILL = (50, 200, 50)
    COLOR_XP_BAR_BG = (20, 20, 80)
    COLOR_XP_BAR_FILL = (50, 100, 250)
    
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
        self.font_s = pygame.font.SysFont("monospace", 15)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.np_random = None
        self.game_over_message = ""
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback to a default or existing generator if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.dungeon_level = 1
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        
        self.player = {
            'x': 0, 'y': 0,
            'hp': 100, 'max_hp': 100,
            'xp': 0, 'xp_to_level': 10,
            'level': 1, 'attack': 10,
            'is_defending': False
        }
        
        self.enemies = []
        self.stairs_pos = None
        self.dungeon_map = []
        self._generate_level()

        self.particles = deque()
        self.floating_texts = deque()
        self.camera_shake = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = -0.01  # Small cost for taking a turn
        action_taken = False
        
        # Reset defense state at the start of the turn
        self.player['is_defending'] = False

        # --- Player Action Phase ---
        if space_held: # Attack action
            action_taken = True
            attack_reward, hit_something = self._handle_player_attack()
            reward += attack_reward
            if hit_something:
                # sound_placeholder: play_sword_swing()
                self.particles.extend(self._create_hit_particles(self.player['x'], self.player['y']))
        elif movement != 0: # Movement action
            action_taken = True
            move_reward = self._handle_player_move(movement)
            reward += move_reward
        else: # Defend action (no movement, no attack)
            action_taken = True
            self.player['is_defending'] = True
            self.floating_texts.append(self._create_floating_text("Defend!", (200, 200, 255), self.player['x'], self.player['y']))

        # --- Enemy Phase ---
        if action_taken:
            self.steps += 1
            enemy_damage_dealt = self._update_enemies()
            if enemy_damage_dealt > 0:
                self.camera_shake = 5 # Add screen shake on taking damage
                # sound_placeholder: play_player_hurt()

        # --- Update game state and check for termination ---
        self._update_effects()
        self.score += reward
        terminated = self._check_termination()
        
        if self.dungeon_level > 10 and not self.game_over: # Boss defeated on level 10
            reward += 100
            self.score += 100
            self.game_over = True
            terminated = True
            self.game_over_message = "VICTORY!"

        if self.player['hp'] <= 0 and not self.game_over:
            reward -= 100
            self.score -= 100
            self.game_over = True
            terminated = True
            self.game_over_message = "YOU DIED"

        if self.steps >= 2000:
             terminated = True
             if not self.game_over:
                 self.game_over_message = "TIME UP"

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
            "player_hp": self.player['hp'],
            "player_level": self.player['level'],
            "dungeon_level": self.dungeon_level
        }

    # --- Helper Methods: Game Logic ---

    def _generate_level(self):
        # 1. Fill map with walls
        self.dungeon_map = [[1 for _ in range(self.DUNGEON_WIDTH)] for _ in range(self.DUNGEON_HEIGHT)]
        
        # 2. Carve out rooms using random walker
        walker_pos = [self.DUNGEON_WIDTH // 2, self.DUNGEON_HEIGHT // 2]
        num_floors = 0
        floor_tiles = []
        
        for _ in range(2500):
            if self.dungeon_map[walker_pos[1]][walker_pos[0]] == 1:
                self.dungeon_map[walker_pos[1]][walker_pos[0]] = 0
                num_floors += 1
                floor_tiles.append(tuple(walker_pos))
            
            # Move walker
            direction = self.np_random.integers(0, 4)
            if direction == 0 and walker_pos[1] > 1: walker_pos[1] -= 1 # Up
            elif direction == 1 and walker_pos[1] < self.DUNGEON_HEIGHT - 2: walker_pos[1] += 1 # Down
            elif direction == 2 and walker_pos[0] > 1: walker_pos[0] -= 1 # Left
            elif direction == 3 and walker_pos[0] < self.DUNGEON_WIDTH - 2: walker_pos[0] += 1 # Right

            if self.np_random.random() < 0.2: # Occasionally jump to a random floor tile
                if floor_tiles:
                    walker_pos = list(self.np_random.choice(floor_tiles))

        # 3. Place player
        player_idx = self.np_random.integers(0, len(floor_tiles))
        self.player['x'], self.player['y'] = floor_tiles.pop(player_idx)

        # 4. Place stairs (if not final level)
        if self.dungeon_level < 10:
            best_stair_pos = None
            max_dist = -1
            for pos in floor_tiles:
                dist = abs(pos[0] - self.player['x']) + abs(pos[1] - self.player['y'])
                if dist > max_dist:
                    max_dist = dist
                    best_stair_pos = pos
            self.stairs_pos = best_stair_pos
            if self.stairs_pos in floor_tiles:
                floor_tiles.remove(self.stairs_pos)

        # 5. Place enemies
        self.enemies.clear()
        num_enemies = min(len(floor_tiles) -1, 3 + self.dungeon_level)
        
        if self.dungeon_level == 10: # Boss level
            boss_health = 200 * (1 + (self.dungeon_level - 1) * 0.2)
            boss_attack = 25 * (1 + (self.dungeon_level - 1) * 0.2)
            boss_pos = self.stairs_pos if self.stairs_pos else floor_tiles.pop(self.np_random.integers(0, len(floor_tiles)))
            self.enemies.append({
                'x': boss_pos[0], 'y': boss_pos[1],
                'hp': int(boss_health), 'max_hp': int(boss_health),
                'attack': int(boss_attack), 'is_boss': True
            })
        else:
            for _ in range(num_enemies):
                if not floor_tiles: break
                idx = self.np_random.integers(0, len(floor_tiles))
                pos = floor_tiles.pop(idx)
                enemy_health = 20 * (1 + (self.dungeon_level - 1) * 0.1)
                enemy_attack = 5 * (1 + (self.dungeon_level - 1) * 0.1)
                self.enemies.append({
                    'x': pos[0], 'y': pos[1],
                    'hp': int(enemy_health), 'max_hp': int(enemy_health),
                    'attack': int(enemy_attack), 'is_boss': False
                })

    def _handle_player_attack(self):
        reward = 0
        hit_something = False
        for enemy in self.enemies:
            if abs(self.player['x'] - enemy['x']) <= 1 and abs(self.player['y'] - enemy['y']) <= 1:
                hit_something = True
                damage = self.player['attack']
                enemy['hp'] -= damage
                self.floating_texts.append(self._create_floating_text(f"-{damage}", (255, 150, 50), enemy['x'], enemy['y']))
                
                if enemy['hp'] <= 0:
                    reward += 5 if enemy['is_boss'] else 1 # Defeat reward
                    xp_gain = 25 if enemy['is_boss'] else 5
                    self.player['xp'] += xp_gain
                    self.floating_texts.append(self._create_floating_text(f"+{xp_gain} XP", (100, 150, 255), self.player['x'], self.player['y']))
                    self._check_player_levelup()
                    if enemy['is_boss']:
                        # sound_placeholder: play_boss_die()
                        self.dungeon_level += 1 # Mark victory
                    else:
                        # sound_placeholder: play_enemy_die()
                        pass
        
        # Remove dead enemies
        self.enemies = [e for e in self.enemies if e['hp'] > 0]
        return reward, hit_something

    def _handle_player_move(self, movement):
        reward = 0
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        target_x, target_y = self.player['x'] + dx, self.player['y'] + dy
        
        if self.dungeon_map[target_y][target_x] == 0: # Is floor
            is_occupied = any(e['x'] == target_x and e['y'] == target_y for e in self.enemies)
            if not is_occupied:
                self.player['x'], self.player['y'] = target_x, target_y
                # sound_placeholder: play_footstep()
                if self.stairs_pos and (target_x, target_y) == self.stairs_pos:
                    # sound_placeholder: play_level_up_ fanfare()
                    self.dungeon_level += 1
                    self.player['hp'] = min(self.player['max_hp'], self.player['hp'] + 20) # Heal on level change
                    self._generate_level()
                    reward += 10
        return reward

    def _update_enemies(self):
        total_damage = 0
        for enemy in self.enemies:
            dist_x = self.player['x'] - enemy['x']
            dist_y = self.player['y'] - enemy['y']
            
            if abs(dist_x) <= 1 and abs(dist_y) <= 1: # Attack if adjacent
                damage = enemy['attack']
                if self.player['is_defending']:
                    damage = max(1, damage // 2)
                self.player['hp'] -= damage
                total_damage += damage
                self.floating_texts.append(self._create_floating_text(f"-{damage}", (255, 50, 50), self.player['x'], self.player['y']))
            else: # Move towards player
                move_x, move_y = 0, 0
                if abs(dist_x) > abs(dist_y):
                    move_x = np.sign(dist_x)
                else:
                    move_y = np.sign(dist_y)
                
                # Check if random move is needed
                if move_x == 0 and move_y == 0:
                    r_dir = self.np_random.integers(0, 4)
                    if r_dir == 0: move_x = 1
                    elif r_dir == 1: move_x = -1
                    elif r_dir == 2: move_y = 1
                    else: move_y = -1

                target_x, target_y = enemy['x'] + move_x, enemy['y'] + move_y
                if self.dungeon_map[target_y][target_x] == 0:
                    is_occupied = any(e['x'] == target_x and e['y'] == target_y for e in self.enemies)
                    is_player = self.player['x'] == target_x and self.player['y'] == target_y
                    if not is_occupied and not is_player:
                        enemy['x'], enemy['y'] = target_x, target_y
        return total_damage

    def _check_player_levelup(self):
        if self.player['xp'] >= self.player['xp_to_level']:
            # sound_placeholder: play_player_levelup()
            self.player['level'] += 1
            self.player['xp'] -= self.player['xp_to_level']
            self.player['xp_to_level'] = int(self.player['xp_to_level'] * 1.5)
            self.player['max_hp'] += 10
            self.player['attack'] += 2
            self.player['hp'] = self.player['max_hp'] # Full heal on level up
            self.floating_texts.append(self._create_floating_text("LEVEL UP!", (255, 255, 100), self.player['x'], self.player['y'], 90))

    def _check_termination(self):
        if self.player['hp'] <= 0: return True
        if self.dungeon_level > 10: return True # Victory condition
        if self.steps >= 2000: return True
        return False

    # --- Helper Methods: Effects ---
    
    def _create_floating_text(self, text, color, tile_x, tile_y, duration=45):
        return {'text': text, 'color': color, 'x': tile_x, 'y': tile_y, 'timer': duration}

    def _create_hit_particles(self, tile_x, tile_y):
        particles = []
        for _ in range(10):
            particles.append({
                'x': tile_x * self.TILE_SIZE + self.TILE_SIZE / 2,
                'y': tile_y * self.TILE_SIZE + self.TILE_SIZE / 2,
                'vx': (self.np_random.random() - 0.5) * 4,
                'vy': (self.np_random.random() - 0.5) * 4,
                'timer': self.np_random.integers(15, 30),
                'color': self.np_random.choice([(255, 255, 255), (255, 200, 100)])
            })
        return particles
    
    def _update_effects(self):
        # Particles
        for p in list(self.particles):
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['timer'] -= 1
            if p['timer'] <= 0: self.particles.remove(p)
        # Floating Text
        for ft in list(self.floating_texts):
            ft['y'] -= 0.01
            ft['timer'] -= 1
            if ft['timer'] <= 0: self.floating_texts.remove(ft)
        # Camera Shake
        if self.camera_shake > 0: self.camera_shake -= 1

    # --- Helper Methods: Rendering ---

    def _render_game(self):
        # Camera centered on player
        cam_x = self.player['x'] * self.TILE_SIZE - self.SCREEN_WIDTH // 2
        cam_y = self.player['y'] * self.TILE_SIZE - self.SCREEN_HEIGHT // 2
        
        if self.camera_shake > 0:
            cam_x += self.np_random.integers(-self.camera_shake, self.camera_shake + 1)
            cam_y += self.np_random.integers(-self.camera_shake, self.camera_shake + 1)

        # Draw dungeon
        start_col = max(0, cam_x // self.TILE_SIZE)
        end_col = min(self.DUNGEON_WIDTH, (cam_x + self.SCREEN_WIDTH) // self.TILE_SIZE + 1)
        start_row = max(0, cam_y // self.TILE_SIZE)
        end_row = min(self.DUNGEON_HEIGHT, (cam_y + self.SCREEN_HEIGHT) // self.TILE_SIZE + 1)

        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                color = self.COLOR_WALL if self.dungeon_map[y][x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y, self.TILE_SIZE, self.TILE_SIZE))

        # Draw stairs
        if self.stairs_pos:
            sx, sy = self.stairs_pos
            pygame.draw.rect(self.screen, self.COLOR_STAIRS, (sx * self.TILE_SIZE - cam_x, sy * self.TILE_SIZE - cam_y, self.TILE_SIZE, self.TILE_SIZE))

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy['x'] * self.TILE_SIZE - cam_x, enemy['y'] * self.TILE_SIZE - cam_y
            size = self.TILE_SIZE * 1.5 if enemy['is_boss'] else self.TILE_SIZE * 0.8
            offset = (self.TILE_SIZE - size) / 2
            color = self.COLOR_BOSS if enemy['is_boss'] else self.COLOR_ENEMY
            pygame.draw.rect(self.screen, color, (ex + offset, ey + offset, size, size))
            self._draw_health_bar(enemy['hp'], enemy['max_hp'], ex, ey - 10, self.TILE_SIZE, 5)

        # Draw player
        bob = math.sin(pygame.time.get_ticks() * 0.005) * 2
        px, py = self.player['x'] * self.TILE_SIZE - cam_x, self.player['y'] * self.TILE_SIZE - cam_y + bob
        size = self.TILE_SIZE * 0.9
        offset = (self.TILE_SIZE - size) / 2
        
        # Glow effect
        glow_radius = int(self.TILE_SIZE * 0.8)
        glow_center = (int(px + self.TILE_SIZE/2), int(py + self.TILE_SIZE/2))
        for i in range(glow_radius, 0, -2):
            alpha = 60 * (1 - i / glow_radius)
            pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], i, (*self.COLOR_PLAYER_GLOW, int(alpha)))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px + offset, py + offset, size, size))
        self._draw_health_bar(self.player['hp'], self.player['max_hp'], px, py - 10, self.TILE_SIZE, 5)

        # Draw particles and floating text
        for p in self.particles:
            alpha = 255 * (p['timer'] / 30)
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (p['x'] - cam_x, p['y'] - cam_y))

        for ft in self.floating_texts:
            alpha = int(255 * (ft['timer'] / 45))
            text_surf = self.font_m.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            pos_x = ft['x'] * self.TILE_SIZE + self.TILE_SIZE/2 - text_surf.get_width()/2 - cam_x
            pos_y = ft['y'] * self.TILE_SIZE - self.TILE_SIZE/2 - text_surf.get_height()/2 - cam_y
            self.screen.blit(text_surf, (pos_x, pos_y))

    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Health
        self._draw_text(f"HP: {max(0, self.player['hp'])} / {self.player['max_hp']}", 20, 10, self.font_m)
        self._draw_bar(self.player['hp'], self.player['max_hp'], 130, 10, 150, 20, self.COLOR_HP_BAR_FILL, self.COLOR_HP_BAR_BG)
        
        # XP
        self._draw_text(f"LV: {self.player['level']}", 20, 35, self.font_m)
        self._draw_bar(self.player['xp'], self.player['xp_to_level'], 130, 35, 150, 10, self.COLOR_XP_BAR_FILL, self.COLOR_XP_BAR_BG)
        
        # Dungeon Level
        self._draw_text(f"Dungeon Level: {self.dungeon_level}", self.SCREEN_WIDTH - 200, 10, self.font_m)
        # Score
        self._draw_text(f"Score: {int(self.score)}", self.SCREEN_WIDTH - 200, 35, self.font_s)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._draw_text(self.game_over_message, self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 - 20, self.font_l, centered=True)

    def _draw_text(self, text, x, y, font, color=COLOR_UI_TEXT, centered=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if centered:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surface, text_rect)

    def _draw_health_bar(self, current, maximum, x, y, width, height):
        if maximum <= 0: return
        ratio = max(0, current / maximum)
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG, (x, y, width, height))
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR_FILL, (x, y, int(width * ratio), height))

    def _draw_bar(self, current, maximum, x, y, width, height, fill_color, bg_color):
        if maximum <= 0: return
        ratio = max(0, min(1, current / maximum))
        pygame.draw.rect(self.screen, bg_color, (x, y, width, height))
        pygame.draw.rect(self.screen, fill_color, (x, y, int(width * ratio), height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (x, y, width, height), 1)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame to run headlessly
    env = GameEnv(render_mode="rgb_array")
    
    # To run with manual controls:
    # 1. Comment out the os.environ line above.
    # 2. In __init__, change self.screen to:
    #    self.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # 3. Uncomment the code below.
    
    # env.close()
    # env = GameEnv(render_mode="human")
    # obs, info = env.reset()
    # done = False
    # total_reward = 0
    
    # # --- Pygame display setup for human play ---
    # pygame.display.set_caption("Dungeon Crawler")
    # screen = pygame.display.get_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    # clock = pygame.time.Clock()
    
    # while not done:
    #     movement, space, shift = 0, 0, 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     if keys[pygame.K_SPACE]: space = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     total_reward += reward

    #     # Render the observation to the display window
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     clock.tick(10) # Control the speed of the game for human play

    # print(f"Game Over! Total Reward: {total_reward}, Final Info: {info}")
    # env.close()