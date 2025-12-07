
# Generated: 2025-08-28T04:56:25.216684
# Source Brief: brief_05414.md
# Brief Index: 5414

        
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

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your square. "
        "Collect the color that matches your own."
    )

    game_description = (
        "Chromatic Chaos: Navigate a shifting grid, collecting matching colors "
        "to conquer the chaos before time runs out. Hitting a mismatched color "
        "costs you precious time."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER_BORDER = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLLECTIBLE_COLORS = [
            (255, 87, 87), (255, 170, 87), (255, 255, 87), (170, 255, 87),
            (87, 255, 87), (87, 255, 170), (87, 255, 255), (87, 170, 255),
            (87, 87, 255), (170, 87, 255), (255, 87, 255), (255, 87, 170)
        ]
        self.NUM_COLORS = len(self.COLLECTIBLE_COLORS)
        
        # Game Constants
        self.MAX_STEPS = 60 * 30  # 60 seconds at 30 FPS
        self.SHIFT_INTERVAL = 5 # Shift cells every 5 frames

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.player = {}
        self.cells = []
        self.collected_mask = []
        self.particles = []
        self.cell_shift_speed = 0.0

        self.reset()
        self.validate_implementation()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        self.collected_mask = [False] * self.NUM_COLORS
        self.particles = []
        self.cell_shift_speed = 0.05

        all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_pos)

        self.cells = []
        for i in range(self.NUM_COLORS):
            pos = all_pos.pop()
            self.cells.append({
                'color_index': i,
                'gx': pos[0], 'gy': pos[1],
                'screen_x': pos[0] * self.CELL_SIZE, 'screen_y': pos[1] * self.CELL_SIZE,
                'target_x': pos[0] * self.CELL_SIZE, 'target_y': pos[1] * self.CELL_SIZE,
                'lerp': 1.0,
            })
        
        player_pos = all_pos.pop()
        self.player = {
            'gx': player_pos[0],
            'gy': player_pos[1],
            'color_index': 0
        }
        self._update_player_color()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        prev_player_pos = (self.player['gx'], self.player['gy'])
        
        movement = action[0]
        
        self.steps += 1
        self.timer -= 1
        
        interaction = self._update_player(movement)
        self._update_grid()
        self._update_particles()

        if self.steps > 0 and self.steps % 100 == 0:
            self.cell_shift_speed = min(0.5, self.cell_shift_speed + 0.01)

        terminated = self._check_termination()
        reward = self._calculate_reward(prev_player_pos, interaction, terminated)
        self.score += reward

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        interaction = {'type': 'none'}
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx == 0 and dy == 0:
            return interaction

        target_gx = self.player['gx'] + dx
        target_gy = self.player['gy'] + dy

        if not (0 <= target_gx < self.GRID_WIDTH and 0 <= target_gy < self.GRID_HEIGHT):
            return interaction

        self.player['gx'] = target_gx
        self.player['gy'] = target_gy

        for cell in self.cells:
            if cell['gx'] == self.player['gx'] and cell['gy'] == self.player['gy']:
                if cell['color_index'] == self.player['color_index']:
                    # --- Correct Collection ---
                    # sfx: positive chime
                    self.collected_mask[cell['color_index']] = True
                    self._create_particles(self.player['gx'], self.player['gy'], self.COLLECTIBLE_COLORS[cell['color_index']], 30)
                    self.cells.remove(cell)
                    self._update_player_color()
                    interaction = {'type': 'collect', 'risky': self._is_risky_move(self.player['gx'], self.player['gy'])}
                else:
                    # --- Mismatched Collection ---
                    # sfx: error buzz
                    self.timer = max(0, self.timer - 150) # 5 second penalty
                    self._create_particles(self.player['gx'], self.player['gy'], (255, 0, 0), 20, "shockwave")
                    interaction = {'type': 'mismatch'}
                break
        
        return interaction

    def _update_grid(self):
        # Update lerping
        for cell in self.cells:
            if cell['lerp'] < 1.0:
                cell['lerp'] = min(1.0, cell['lerp'] + self.cell_shift_speed)
                cell['screen_x'] = cell['screen_x'] * (1 - cell['lerp']) + cell['target_x'] * cell['lerp']
                cell['screen_y'] = cell['screen_y'] * (1 - cell['lerp']) + cell['target_y'] * cell['lerp']

        # Trigger new shift
        if self.steps % self.SHIFT_INTERVAL == 0:
            # sfx: subtle whoosh
            occupied_pos = {(cell['gx'], cell['gy']) for cell in self.cells}
            occupied_pos.add((self.player['gx'], self.player['gy']))
            
            all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
            empty_pos = [p for p in all_pos if p not in occupied_pos]
            self.np_random.shuffle(empty_pos)

            for cell in self.cells:
                if empty_pos:
                    new_pos = empty_pos.pop()
                    cell['gx'], cell['gy'] = new_pos
                    cell['target_x'] = new_pos[0] * self.CELL_SIZE
                    cell['target_y'] = new_pos[1] * self.CELL_SIZE
                    cell['lerp'] = 0.0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['type'] == "spark":
                p['vx'] *= 0.95
                p['vy'] *= 0.95

    def _update_player_color(self):
        if all(self.collected_mask):
            return
        try:
            next_color_index = self.collected_mask.index(False)
            self.player['color_index'] = next_color_index
        except ValueError:
            pass # All collected

    def _calculate_reward(self, prev_player_pos, interaction, terminated):
        reward = 0.0
        
        # Event-based rewards
        if interaction['type'] == 'collect':
            reward += 5.0
            if interaction['risky']:
                reward += 1.0
        elif interaction['type'] == 'mismatch':
            reward -= 2.0
            
        # Continuous reward for moving towards target
        if not all(self.collected_mask):
            target_cell = self._find_cell_by_color(self.player['color_index'])
            if target_cell:
                target_pos = (target_cell['gx'], target_cell['gy'])
                dist_before = abs(prev_player_pos[0] - target_pos[0]) + abs(prev_player_pos[1] - target_pos[1])
                dist_after = abs(self.player['gx'] - target_pos[0]) + abs(self.player['gy'] - target_pos[1])
                
                if dist_after < dist_before:
                    reward += 0.1
                elif dist_after > dist_before:
                    reward -= 0.1
        
        # Terminal rewards
        if terminated:
            if all(self.collected_mask):
                reward += 100.0  # Win
            elif self.timer <= 0:
                reward -= 50.0  # Lose by time out

        return reward

    def _check_termination(self):
        return self.timer <= 0 or all(self.collected_mask)

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
            "timer": self.timer,
            "colors_collected": sum(self.collected_mask)
        }

    def _render_game(self):
        # Grid lines
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Cells
        for cell in self.cells:
            color = self.COLLECTIBLE_COLORS[cell['color_index']]
            rect = pygame.Rect(
                int(cell['screen_x']), int(cell['screen_y']), 
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Player
        player_color = self.COLLECTIBLE_COLORS[self.player['color_index']]
        player_rect = pygame.Rect(
            self.player['gx'] * self.CELL_SIZE, self.player['gy'] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, player_color, player_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_BORDER, player_rect, 3, border_radius=8)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['x']) - p['size'], int(p['y']) - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Timer bar
        timer_ratio = self.timer / self.MAX_STEPS
        bar_width = 200
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        if timer_ratio > 0.5:
            timer_color = (100, 220, 100)
        elif timer_ratio > 0.2:
            timer_color = (220, 220, 100)
        else:
            timer_color = (220, 100, 100)

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height), border_radius=4)

        # Collected colors HUD
        hud_y = self.HEIGHT - 25
        hud_x_start = (self.WIDTH - (self.NUM_COLORS * 20)) / 2
        for i in range(self.NUM_COLORS):
            color = self.COLLECTIBLE_COLORS[i] if self.collected_mask[i] else self.COLOR_GRID
            rect = (hud_x_start + i * 20, hud_y, 15, 15)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            if i == self.player['color_index'] and not all(self.collected_mask):
                 pygame.draw.rect(self.screen, self.COLOR_PLAYER_BORDER, rect, 2, border_radius=3)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if all(self.collected_mask):
                msg = "YOU WIN!"
                msg_color = (150, 255, 150)
            else:
                msg = "TIME'S UP!"
                msg_color = (255, 150, 150)
            
            text_surf = self.font_game_over.render(msg, True, msg_color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, gx, gy, color, count, p_type="spark"):
        center_x = gx * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = gy * self.CELL_SIZE + self.CELL_SIZE / 2
        
        for _ in range(count):
            if p_type == "spark":
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                life = self.np_random.integers(15, 30)
                self.particles.append({
                    'x': center_x, 'y': center_y,
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'life': life, 'max_life': life,
                    'color': color, 'size': self.np_random.integers(2, 5), 'type': p_type
                })
            elif p_type == "shockwave":
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = 4
                life = 15
                self.particles.append({
                    'x': center_x, 'y': center_y,
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'life': life, 'max_life': life,
                    'color': color, 'size': 10, 'type': p_type
                })

    def _find_cell_by_color(self, color_index):
        for cell in self.cells:
            if cell['color_index'] == color_index:
                return cell
        return None

    def _is_risky_move(self, gx, gy):
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            check_gx, check_gy = gx + dx, gy + dy
            for cell in self.cells:
                if cell['gx'] == check_gx and cell['gy'] == check_gy:
                    return True # A colored cell is adjacent
        return False

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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for human play
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Chromatic Chaos")
    
    done = False
    total_score = 0
    
    # Mapping from Pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        movement_action = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, action_val in key_to_action.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize first key found (e.g., up over down)

        # Construct the MultiDiscrete action
        # space and shift are not used in this game
        action = [movement_action, 0, 0] 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # Blit the environment's screen to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before closing
            pygame.time.wait(3000)
            done = True

    env.close()
    pygame.quit()