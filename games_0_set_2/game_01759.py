
# Generated: 2025-08-28T02:37:38.621332
# Source Brief: brief_01759.md
# Brief Index: 1759

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to attack the nearest enemy. "
        "Hold Shift to attack the enemy with the lowest health."
    )

    # User-facing game description
    game_description = (
        "Control a robot in a grid-based arena, battling increasingly "
        "difficult waves of enemies to survive. A tactical, turn-based shooter."
    )

    # Frames advance on action receipt (turn-based game)
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 10
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_WIDTH
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.GRID_HEIGHT

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_DMG = (255, 255, 255)
        self.COLOR_ENEMY_WAVE_1 = (255, 50, 50)
        self.COLOR_ENEMY_WAVE_2 = (255, 100, 50)
        self.COLOR_ENEMY_WAVE_3 = (255, 150, 50)
        self.ENEMY_COLORS = [self.COLOR_ENEMY_WAVE_1, self.COLOR_ENEMY_WAVE_2, self.COLOR_ENEMY_WAVE_3]
        self.COLOR_ATTACK = (100, 200, 255)
        self.COLOR_HEALTH_BG = (70, 20, 20)
        self.COLOR_HEALTH_FG = (50, 200, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_POPUP = (255, 200, 0)
        self.COLOR_GAMEOVER = (200, 30, 30)
        self.COLOR_WIN = (50, 255, 150)

        # Game Parameters
        self.MAX_STEPS = 1000
        self.PLAYER_MAX_HEALTH = 50
        self.PLAYER_DAMAGE = 10
        self.WAVE_DEFINITIONS = [
            {'count': 3, 'health': 10, 'damage': 2},
            {'count': 4, 'health': 15, 'damage': 3},
            {'count': 5, 'health': 20, 'damage': 4},
        ]
        self.TOTAL_WAVES = len(self.WAVE_DEFINITIONS)

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 32, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 64, bold=True)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = (0, 0)
        self.player_health = 0
        self.player_hit_timer = 0
        self.current_wave = 0
        self.enemies = []
        self.effects = [] # For attack lines, damage popups, etc.

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_hit_timer = 0
        
        self.enemies = []
        self.effects = []
        self.current_wave = 0
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        self.player_hit_timer = max(0, self.player_hit_timer - 1)

        # --- Player's Turn ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_taken = False

        # Prioritize attack over movement
        if shift_held:
            target = self._find_lowest_health_enemy()
            if target:
                reward += self._player_attack(target)
                action_taken = True
        elif space_held:
            target = self._find_nearest_enemy()
            if target:
                reward += self._player_attack(target)
                action_taken = True
        
        if not action_taken and movement > 0:
            self._player_move(movement)

        # --- Enemies' Turn ---
        damage_taken_this_turn = 0
        occupied_positions = {tuple(e['pos']) for e in self.enemies}
        
        for enemy in self.enemies:
            # Enemy AI: Move towards player if not adjacent
            px, py = self.player_pos
            ex, ey = enemy['pos']
            
            if abs(px - ex) + abs(py - ey) > 1: # Not adjacent
                new_ex, new_ey = ex, ey
                if px != ex and py != ey: # Diagonal, move randomly in one axis
                    if self.np_random.random() > 0.5:
                        new_ex += np.sign(px - ex)
                    else:
                        new_ey += np.sign(py - ey)
                elif px != ex: # Horizontal
                    new_ex += np.sign(px - ex)
                elif py != ey: # Vertical
                    new_ey += np.sign(py - ey)

                # Check for collisions with other enemies before moving
                if (new_ex, new_ey) not in occupied_positions:
                    occupied_positions.remove(tuple(enemy['pos']))
                    enemy['pos'] = (new_ex, new_ey)
                    occupied_positions.add(tuple(enemy['pos']))

            # Enemy Attack: if adjacent to player after moving
            if abs(self.player_pos[0] - enemy['pos'][0]) + abs(self.player_pos[1] - enemy['pos'][1]) == 1:
                damage = self.WAVE_DEFINITIONS[self.current_wave - 1]['damage']
                self.player_health -= damage
                damage_taken_this_turn += damage
                self.player_hit_timer = 2 # Flash for 2 frames
                self._add_effect('popup', self.player_pos, f"-{damage}", self.COLOR_POPUP, 20)

        if damage_taken_this_turn > 0:
            reward -= 0.2 * damage_taken_this_turn

        # --- Update Game State ---
        self._update_effects()
        if not self.enemies and self.current_wave <= self.TOTAL_WAVES:
            if self.current_wave < self.TOTAL_WAVES:
                self._start_next_wave()
            else: # All waves defeated
                self.win = True

        # Check for termination conditions
        terminated = False
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
        elif self.win:
            self.game_over = True
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _player_attack(self, target):
        reward = 0
        damage = self.PLAYER_DAMAGE
        target['health'] -= damage
        target['hit_timer'] = 2 # Flash for 2 frames
        reward += 0.1 * damage

        self._add_effect('line', self.player_pos, target['pos'], self.COLOR_ATTACK, 10)
        self._add_effect('popup', target['pos'], f"-{damage}", self.COLOR_POPUP, 20)

        if target['health'] <= 0:
            reward += 1
            self.enemies.remove(target)
        
        return reward

    def _player_move(self, movement_action):
        px, py = self.player_pos
        if movement_action == 1: py -= 1  # Up
        elif movement_action == 2: py += 1  # Down
        elif movement_action == 3: px -= 1  # Left
        elif movement_action == 4: px += 1  # Right
        
        # Boundary and collision checks
        if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
            is_occupied = any(e['pos'] == (px, py) for e in self.enemies)
            if not is_occupied:
                self.player_pos = (px, py)

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            return

        wave_data = self.WAVE_DEFINITIONS[self.current_wave - 1]
        occupied_positions = {self.player_pos}
        
        for i in range(wave_data['count']):
            while True:
                x = self.np_random.integers(0, self.GRID_WIDTH)
                y = self.np_random.integers(0, self.GRID_HEIGHT)
                if (x, y) not in occupied_positions:
                    occupied_positions.add((x, y))
                    self.enemies.append({
                        'pos': (x, y),
                        'health': wave_data['health'],
                        'max_health': wave_data['health'],
                        'color': self.ENEMY_COLORS[self.current_wave - 1],
                        'hit_timer': 0,
                        'id': f"e_{self.current_wave}_{i}"
                    })
                    break

    def _find_nearest_enemy(self):
        if not self.enemies: return None
        px, py = self.player_pos
        return min(self.enemies, key=lambda e: abs(e['pos'][0] - px) + abs(e['pos'][1] - py))

    def _find_lowest_health_enemy(self):
        if not self.enemies: return None
        return min(self.enemies, key=lambda e: e['health'])

    def _add_effect(self, type, pos, data, color, lifetime):
        self.effects.append({'type': type, 'pos': pos, 'data': data, 'color': color, 'lifetime': lifetime})

    def _update_effects(self):
        self.effects = [e for e in self.effects if e['lifetime'] > 0]
        for effect in self.effects:
            effect['lifetime'] -= 1

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
            "player_health": self.player_health,
            "wave": self.current_wave,
            "enemies_left": len(self.enemies),
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)

        # Draw enemies
        for enemy in self.enemies:
            self._draw_entity(enemy['pos'], enemy['color'], enemy['health'], enemy['max_health'], enemy.get('hit_timer', 0) > 0)
            enemy['hit_timer'] = max(0, enemy['hit_timer'] - 1)

        # Draw player
        self._draw_entity(self.player_pos, self.COLOR_PLAYER, self.player_health, self.PLAYER_MAX_HEALTH, self.player_hit_timer > 0)

        # Draw effects
        for effect in self.effects:
            if effect['type'] == 'line':
                start_pixel = self._grid_to_pixel_center(effect['pos'])
                end_pixel = self._grid_to_pixel_center(effect['data'])
                pygame.draw.line(self.screen, effect['color'], start_pixel, end_pixel, 3)
            elif effect['type'] == 'popup':
                fade = effect['lifetime'] / 20.0
                offset_y = (20 - effect['lifetime'])
                px, py = self._grid_to_pixel_center(effect['pos'])
                text_surf = self.font_small.render(effect['data'], True, effect['color'])
                text_surf.set_alpha(int(255 * fade))
                text_rect = text_surf.get_rect(center=(px, py - 25 - offset_y))
                self.screen.blit(text_surf, text_rect)

    def _draw_entity(self, pos, color, health, max_health, is_hit):
        px, py = self._grid_to_pixel_center(pos)
        size = int(self.CELL_WIDTH * 0.6)
        rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
        
        # Draw hit flash
        if is_hit:
            flash_rect = rect.inflate(8, 8)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_DMG, flash_rect, border_radius=4)

        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Health bar
        bar_width = self.CELL_WIDTH * 0.8
        bar_height = 8
        bar_x = px - bar_width // 2
        bar_y = py - size // 2 - bar_height - 5
        
        health_ratio = max(0, health / max_health)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_width * health_ratio, bar_height), border_radius=2)

    def _grid_to_pixel_center(self, grid_pos):
        x, y = grid_pos
        px = int((x + 0.5) * self.CELL_WIDTH)
        py = int((y + 0.5) * self.CELL_HEIGHT)
        return px, py

    def _render_ui(self):
        # Wave display
        wave_text = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}"
        wave_surf = self.font_medium.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (10, 10))

        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg_text = "VICTORY"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "GAME OVER"
                msg_color = self.COLOR_GAMEOVER
            
            msg_surf = self.font_large.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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


if __name__ == "__main__":
    # This block allows you to run the environment directly for testing
    # and visualization.
    
    # To prevent the Pygame window from opening, you can run in "headless" mode
    # by setting the render_mode to "rgb_array". For interactive play,
    # we'll create a window.
    
    env = GameEnv()
    
    # --- Interactive Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for interactive display
    pygame.display.set_caption("Grid Robot Arena")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            
            # Map keys to actions
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since this is turn-based, we wait for key input. A small delay prevents
        # the loop from running too fast and using 100% CPU.
        clock.tick(10) # Limit to 10 actions per second for human playability
        
    env.close()