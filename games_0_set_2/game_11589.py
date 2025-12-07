import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:16:40.078005
# Source Brief: brief_01589.md
# Brief Index: 1589
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A strategic puzzle-fighter where you match colored tiles to summon fighters that attack an opponent. "
        "Create powerful chain reactions to overwhelm the enemy and claim victory."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to select a tile, then select an "
        "adjacent tile to swap. Hold 'shift' while swapping to trigger a powerful chain reaction."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_BG = (25, 30, 50)
    COLOR_GRID_LINE = (40, 50, 75)
    
    BASE_TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 150, 255),  # Blue
        (80, 255, 150),  # Green
        (255, 150, 80),  # Orange
        (200, 80, 255),  # Purple
        (255, 255, 80),  # Yellow
    ]
    COLOR_TEXT = (220, 220, 240)
    COLOR_PLAYER_HP = (80, 255, 150)
    COLOR_OPPONENT_HP = (255, 80, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    
    # Game Board
    GRID_COLS, GRID_ROWS = 12, 7
    TILE_SIZE = 32
    TILE_MARGIN = 4
    GRID_WIDTH = GRID_COLS * (TILE_SIZE + TILE_MARGIN) - TILE_MARGIN
    GRID_HEIGHT = GRID_ROWS * (TILE_SIZE + TILE_MARGIN) - TILE_MARGIN
    GRID_X = (WIDTH - GRID_WIDTH) // 2
    GRID_Y = (HEIGHT - GRID_HEIGHT) // 2 + 20

    # Player/Opponent
    PLAYER_BASE_Y = HEIGHT - 40
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Persistent State (between episodes) ---
        self.total_wins = 0
        self.unlocked_color_count = 3
        
        # --- Episode State (reset every episode) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_hp = 0
        self.player_hp_max = 100
        self.opponent_hp = 0
        self.opponent_hp_max = 100

        self.board = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.tile_animations = {} # (r, c) -> {type: 'shrink'/'grow', progress: 0.0}

        self.cursor_pos = [0, 0]
        self.cursor_render_pos = [0.0, 0.0]
        self.selected_tile = None
        
        self.fighters = []
        self.player_attacks = []
        self.opponent_attacks = []
        self.particles = []
        self.impact_effects = []
        
        self.opponent_attack_timer = 0
        self.step_reward = 0.0

        # --- Initialize and Validate ---
        # self.reset() # Removed to follow Gymnasium API standard (reset is called by user)
        # self.validate_implementation() # Removed as it's for dev, not for final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_hp = self.player_hp_max
        self.opponent_hp_max = int(100 * (1.05 ** self.total_wins))
        self.opponent_hp = self.opponent_hp_max
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.cursor_render_pos = [
            self.GRID_X + self.cursor_pos[0] * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2,
            self.GRID_Y + self.cursor_pos[1] * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2
        ]
        self.selected_tile = None
        
        self.fighters.clear()
        self.player_attacks.clear()
        self.opponent_attacks.clear()
        self.particles.clear()
        self.impact_effects.clear()
        self.tile_animations.clear()
        
        self.opponent_attack_timer = self.np_random.integers(60, 120)
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0.0
        self.steps += 1

        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_pressed, shift_held)
        self._update_game_logic()
        
        reward = self.step_reward
        terminated = self._check_termination()
        
        if terminated:
            if self.opponent_hp <= 0:
                reward += 100 # Win bonus
                self.score += 100
                self.total_wins += 1
                if self.total_wins > 0 and self.total_wins % 10 == 0 and self.unlocked_color_count < len(self.BASE_TILE_COLORS):
                    self.unlocked_color_count += 1
                    reward += 1 # Unlock bonus
            elif self.player_hp <= 0:
                reward -= 100 # Loss penalty
                self.score -= 100

        self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_held):
        # --- Cursor Movement ---
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1: self.cursor_pos[0] += 1
        
        # --- Tile Selection ---
        if space_pressed:
            # sound placeholder: # sfx_select_tile
            r, c = self.cursor_pos[1], self.cursor_pos[0]
            if self.board[r][c] == 0: # Cannot select empty tile
                self.selected_tile = None
                return

            if self.selected_tile is None:
                self.selected_tile = (r, c)
            elif self.selected_tile == (r, c): # Deselect
                self.selected_tile = None
            else:
                self._try_match((r, c), self.selected_tile, shift_held)
                self.selected_tile = None

    def _try_match(self, pos1, pos2, is_chain_reaction):
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Must be adjacent and same color
        is_adjacent = abs(r1 - r2) + abs(c1 - c2) == 1
        is_same_color = self.board[r1][c1] == self.board[r2][c2] and self.board[r1][c1] > 0
        
        if not (is_adjacent and is_same_color):
            # sound placeholder: # sfx_match_fail
            return

        # --- Process Match ---
        color_idx = self.board[r1][c1]
        
        if is_chain_reaction:
            # sound placeholder: # sfx_chain_reaction
            to_visit = [(r1, c1), (r2, c2)]
            visited = set(to_visit)
            matched_tiles = []
            while to_visit:
                r, c = to_visit.pop(0)
                matched_tiles.append((r, c))
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                       (nr, nc) not in visited and self.board[nr][nc] == color_idx:
                        visited.add((nr, nc))
                        to_visit.append((nr, nc))
        else:
            # sound placeholder: # sfx_match_success
            matched_tiles = [(r1, c1), (r2, c2)]
        
        if not matched_tiles: return

        self.step_reward += 0.1 * len(matched_tiles)

        # Calculate spawn position and power
        avg_r, avg_c = 0, 0
        for r, c in matched_tiles:
            self.board[r][c] = 0
            self.tile_animations[(r, c)] = {'type': 'shrink', 'progress': 0.0}
            avg_r += r
            avg_c += c
            self._create_particles(self._get_tile_center(r, c), self.BASE_TILE_COLORS[color_idx-1], 5)
        
        avg_r /= len(matched_tiles)
        avg_c /= len(matched_tiles)
        spawn_pos = self._get_tile_center(avg_r, avg_c)
        power = len(matched_tiles)
        
        self._spawn_fighter(spawn_pos, power, color_idx)

    def _spawn_fighter(self, pos, power, color_idx):
        fighter = {
            'pos': list(pos),
            'render_pos': list(pos),
            'power': power,
            'color_idx': color_idx,
            'attack_cooldown': self.np_random.integers(30, 60),
            'pulse': self.np_random.random() * math.pi * 2,
            'life': 200 + 50 * power # Fighters fade over time
        }
        self.fighters.append(fighter)

    def _update_game_logic(self):
        # --- Update Animations ---
        for key, anim in list(self.tile_animations.items()):
            anim['progress'] += 0.1
            if anim['progress'] >= 1.0:
                del self.tile_animations[key]
                if anim['type'] == 'shrink':
                    self._refill_board()
        
        # --- Interpolate cursor ---
        target_x = self.GRID_X + self.cursor_pos[0] * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2
        target_y = self.GRID_Y + self.cursor_pos[1] * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2
        self.cursor_render_pos[0] += (target_x - self.cursor_render_pos[0]) * 0.5
        self.cursor_render_pos[1] += (target_y - self.cursor_render_pos[1]) * 0.5

        # --- Update Fighters ---
        for f in self.fighters:
            f['attack_cooldown'] -= 1
            f['pulse'] += 0.1
            f['life'] -= 1
            if f['attack_cooldown'] <= 0:
                # sound placeholder: # sfx_fighter_attack
                self._create_player_attack(f)
                f['attack_cooldown'] = 100 - min(50, f['power'] * 5) # Faster attacks for powerful fighters
        self.fighters = [f for f in self.fighters if f['life'] > 0]
        
        # --- Update Player Attacks ---
        for attack in list(self.player_attacks):
            attack['pos'][0] += attack['vel'][0]
            attack['pos'][1] += attack['vel'][1]
            # Check for hit with opponent
            if attack['pos'][1] < 40: # Opponent area
                self.step_reward += 0.01 * attack['power']
                self.opponent_hp = max(0, self.opponent_hp - attack['power'])
                self._create_impact_effect(attack['pos'], self.BASE_TILE_COLORS[attack['color_idx']-1], attack['power'])
                # sound placeholder: # sfx_opponent_hit
                self.player_attacks.remove(attack)

        # --- Update Opponent ---
        self.opponent_attack_timer -= 1
        if self.opponent_attack_timer <= 0:
            # sound placeholder: # sfx_opponent_attack
            self._create_opponent_attack()
            self.opponent_attack_timer = self.np_random.integers(120, 200) - int(self.total_wins * 2)

        # --- Update Opponent Attacks ---
        for attack in list(self.opponent_attacks):
            attack['pos'][0] += attack['vel'][0]
            attack['pos'][1] += attack['vel'][1]
            if attack['pos'][1] > self.PLAYER_BASE_Y:
                self.player_hp = max(0, self.player_hp - attack['power'])
                self._create_impact_effect(attack['pos'], (200,200,200), attack['power'])
                # sound placeholder: # sfx_player_hit
                self.opponent_attacks.remove(attack)
            # Remove if off-screen
            elif attack['pos'][1] > self.HEIGHT:
                self.opponent_attacks.remove(attack)

        # --- Update Effects ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            
        self.impact_effects = [e for e in self.impact_effects if e['progress'] < 1.0]
        for e in self.impact_effects:
            e['progress'] += 0.05
    
    def _create_player_attack(self, fighter):
        target_pos = [self.WIDTH / 2, 20]
        direction = math.atan2(target_pos[1] - fighter['pos'][1], target_pos[0] - fighter['pos'][0])
        speed = 5
        attack = {
            'pos': list(fighter['pos']),
            'vel': [math.cos(direction) * speed, math.sin(direction) * speed],
            'power': fighter['power'],
            'color_idx': fighter['color_idx']
        }
        self.player_attacks.append(attack)

    def _create_opponent_attack(self):
        start_pos = [self.WIDTH / 2 + self.np_random.uniform(-100, 100), 20]
        target_pos = [self.np_random.uniform(self.WIDTH*0.2, self.WIDTH*0.8), self.PLAYER_BASE_Y]
        direction = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
        speed = 3
        power = 5 + int(self.total_wins * 0.5)
        attack = {
            'pos': start_pos,
            'vel': [math.cos(direction) * speed, math.sin(direction) * speed],
            'power': power
        }
        self.opponent_attacks.append(attack)

    def _generate_board(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self.board[r][c] = self.np_random.integers(1, self.unlocked_color_count + 1)
                self.tile_animations[(r,c)] = {'type': 'grow', 'progress': self.np_random.random() * 0.5}

    def _refill_board(self):
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.board[r][c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.board[r + empty_count][c] = self.board[r][c]
                    self.board[r][c] = 0
            
            for r in range(empty_count):
                self.board[r][c] = self.np_random.integers(1, self.unlocked_color_count + 1)
                self.tile_animations[(r,c)] = {'type': 'grow', 'progress': 0.0}

    def _check_termination(self):
        if self.player_hp <= 0 or self.opponent_hp <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT), border_radius=10)

        # Tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.board[r][c]
                if color_idx > 0:
                    x = self.GRID_X + c * (self.TILE_SIZE + self.TILE_MARGIN)
                    y = self.GRID_Y + r * (self.TILE_SIZE + self.TILE_MARGIN)
                    
                    size = self.TILE_SIZE
                    anim = self.tile_animations.get((r,c))
                    if anim:
                        if anim['type'] == 'grow':
                            size = self.TILE_SIZE * anim['progress']
                        elif anim['type'] == 'shrink':
                            size = self.TILE_SIZE * (1.0 - anim['progress'])
                    
                    offset = (self.TILE_SIZE - size) / 2
                    rect = pygame.Rect(x + offset, y + offset, size, size)
                    
                    color = self.BASE_TILE_COLORS[color_idx-1]
                    pygame.draw.rect(self.screen, color, rect, border_radius=5)
                    
                    if self.selected_tile == (r, c):
                        pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3, border_radius=5)
        
        # Fighters
        for f in self.fighters:
            size = 8 + f['power'] + math.sin(f['pulse']) * 2
            color = self.BASE_TILE_COLORS[f['color_idx']-1]
            points = [
                (f['pos'][0], f['pos'][1] - size),
                (f['pos'][0] - size * 0.866, f['pos'][1] + size * 0.5),
                (f['pos'][0] + size * 0.866, f['pos'][1] + size * 0.5),
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)

        # Attacks
        for a in self.player_attacks:
            color = self.BASE_TILE_COLORS[a['color_idx']-1]
            pygame.draw.circle(self.screen, color, (int(a['pos'][0]), int(a['pos'][1])), 4 + a['power'] // 2)
        for a in self.opponent_attacks:
            pygame.draw.circle(self.screen, (200,200,200), (int(a['pos'][0]), int(a['pos'][1])), 4)
            
        # Effects
        for p in self.particles:
            size = p['life'] / p['max_life'] * p['size']
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0] - size/2, p['pos'][1] - size/2, size, size))
        for e in self.impact_effects:
            radius = e['progress'] * e['max_radius']
            alpha = int(255 * (1.0 - e['progress']))
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(radius), (*e['color'], alpha))

        # Cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (self.cursor_render_pos[0] - self.TILE_SIZE/2-2, self.cursor_render_pos[1] - self.TILE_SIZE/2-2, self.TILE_SIZE+4, self.TILE_SIZE+4), 2, border_radius=7)

    def _render_ui(self):
        # --- HP Bars ---
        # Opponent HP
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (10, 10, self.WIDTH - 20, 25), border_radius=5)
        hp_ratio = self.opponent_hp / self.opponent_hp_max if self.opponent_hp_max > 0 else 0
        pygame.draw.rect(self.screen, self.COLOR_OPPONENT_HP, (12, 12, (self.WIDTH - 24) * hp_ratio, 21), border_radius=4)
        hp_text = self.font_small.render(f"OPPONENT: {int(self.opponent_hp)}/{int(self.opponent_hp_max)}", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (self.WIDTH/2 - hp_text.get_width()/2, 14))

        # Player HP
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (10, self.HEIGHT - 35, self.WIDTH - 20, 25), border_radius=5)
        hp_ratio = self.player_hp / self.player_hp_max if self.player_hp_max > 0 else 0
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HP, (12, self.HEIGHT - 33, (self.WIDTH - 24) * hp_ratio, 21), border_radius=4)
        hp_text = self.font_small.render(f"PLAYER: {int(self.player_hp)}/{int(self.player_hp_max)}", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (self.WIDTH/2 - hp_text.get_width()/2, self.HEIGHT - 31))

        # --- Score & Wins ---
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 45))
        wins_text = self.font_small.render(f"WINS: {self.total_wins}", True, self.COLOR_TEXT)
        self.screen.blit(wins_text, (self.WIDTH - wins_text.get_width() - 15, 45))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_hp": self.player_hp,
            "opponent_hp": self.opponent_hp,
            "total_wins": self.total_wins
        }
        
    def _get_tile_center(self, r, c):
        x = self.GRID_X + c * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2
        y = self.GRID_Y + r * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2
        return x, y
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _create_impact_effect(self, pos, color, power):
        self.impact_effects.append({
            'pos': pos,
            'color': color,
            'max_radius': 20 + power * 5,
            'progress': 0.0
        })
        self._create_particles(pos, color, 5 + power)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for manual testing and will not be run by the evaluation system.
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Gem Fighter")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    action = [0, 0, 0] # No-op, no space, no shift
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Actions
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated or truncated:
            print(f"--- Episode Over ---")
            print(f"Final Score: {info['score']:.2f}, Total Wins: {info['total_wins']}")
            obs, info = env.reset()

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()