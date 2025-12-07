import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade-style, grid-based tower defense game where the player strategically
    places words to create attack patterns and fend off waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press SHIFT to cycle through "
        "your available words. Press SPACE to place the selected word at the cursor's "
        "position to attack incoming enemies."
    )

    game_description = (
        "Defend your tower from encroaching enemies by placing words on the grid. "
        "Each word unleashes a unique attack pattern. Survive 10 waves to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        try:
            self.font_s = pygame.font.Font(pygame.font.get_default_font(), 14)
            self.font_m = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_l = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_xl = pygame.font.Font(pygame.font.get_default_font(), 32)
        except IOError:
            self.font_s = pygame.font.SysFont("Arial", 14)
            self.font_m = pygame.font.SysFont("Arial", 18)
            self.font_l = pygame.font.SysFont("Arial", 24)
            self.font_xl = pygame.font.SysFont("Arial", 32)

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PATH = (30, 40, 50)
        self.COLOR_TOWER = (0, 150, 200)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_STRONG = (255, 100, 100)
        self.COLOR_WORD_TILE = (200, 200, 200)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_ATTACK = (255, 200, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (50, 200, 50)
        self.COLOR_HEALTH_RED = (200, 50, 50)
        self.COLOR_SCORE = (255, 220, 0)

        # --- Game Layout ---
        self.GRID_COLS, self.GRID_ROWS = 12, 10
        self.CELL_SIZE = 32
        self.GRID_X_OFFSET = 30
        self.GRID_Y_OFFSET = (self.screen_height - self.GRID_ROWS * self.CELL_SIZE) // 2
        self.UI_X_OFFSET = self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE + 30

        # --- Word & Attack Data ---
        self.WORD_BANK = {
            "HIT": {"damage": 15, "pattern": [(0, 0), (1, 0), (2, 0)]},
            "ZAP": {"damage": 15, "pattern": [(0, 0), (0, 1), (0, 2)]},
            "RAY": {"damage": 10, "pattern": [(0, 0), (1, 1), (2, 2)]},
            "BLAST": {"damage": 25, "pattern": [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]},
            "NOVA": {"damage": 20, "pattern": [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1] if not (x == 0 and y == 0)]},
            "WALL": {"damage": 5, "pattern": [(-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]}
        }
        self.WORD_LIFETIME = 3

        # --- Enemy Data ---
        self.ENEMY_PATH = [(self.GRID_COLS, 5), (self.GRID_COLS - 2, 5), (self.GRID_COLS - 2, 2), (2, 2), (2, 8), (-1, 8)]

        # --- Game State ---
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            # Use a dedicated numpy random generator
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.max_steps = 1000

        self.tower_health = 100
        self.max_tower_health = 100
        self.current_wave = 1
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.available_words = self._get_new_words(4)
        self.selected_word_index = 0
        
        self.enemies = []
        self.placed_words = []
        self.particles = []
        self.text_popups = []
        self.screen_flash = 0
        
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Update Effects ---
        self._update_particles()
        self._update_text_popups()
        if self.screen_flash > 0:
            self.screen_flash -= 1

        # --- Player Action Phase ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_press, shift_press)

        if space_press:
            place_reward = self._place_word()
            reward += place_reward

        # --- Enemy & World Phase ---
        for word in self.placed_words:
            word['lifetime'] -= 1
        self.placed_words = [w for w in self.placed_words if w['lifetime'] > 0]

        tower_damage_reward = self._update_enemies()
        reward += tower_damage_reward

        # --- Progression Phase ---
        wave_completion_reward = self._check_wave_completion()
        reward += wave_completion_reward

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.tower_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100  # Penalty for losing
        elif self.victory:
            self.game_over = True
            terminated = True
            # Final wave bonus is already given
        elif self.steps >= self.max_steps:
            truncated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_press, shift_press):
        # Cycle selected word with Shift
        if shift_press:
            self.selected_word_index = (self.selected_word_index + 1) % len(self.available_words)
            # sfx: UI_cycle.wav

        # Move cursor with arrow keys
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
    
    def _place_word(self):
        word_str = self.available_words[self.selected_word_index]
        word_data = self.WORD_BANK[word_str]
        cx, cy = self.cursor_pos
        
        # Check if placement is valid (no overlap)
        grid_positions = {(w['pos'][0], w['pos'][1]) for w in self.placed_words}
        if (cx, cy) in grid_positions:
            # sfx: error.wav
            return 0 # Cannot place on an existing word

        # Place the word
        # sfx: word_place.wav
        self.placed_words.append({
            "word": word_str,
            "pos": [cx, cy],
            "lifetime": self.WORD_LIFETIME + 1, # +1 to account for this turn
            "attack_effect_timer": 15 # frames
        })
        self.available_words.pop(self.selected_word_index)
        self.available_words.extend(self._get_new_words(1))
        self.selected_word_index = min(self.selected_word_index, len(self.available_words) - 1)
        
        # Apply damage
        total_damage = 0
        kills = 0
        for offset_x, offset_y in word_data["pattern"]:
            attack_pos = (cx + offset_x, cy + offset_y)
            if not (0 <= attack_pos[0] < self.GRID_COLS and 0 <= attack_pos[1] < self.GRID_ROWS):
                continue

            for enemy in self.enemies:
                if tuple(enemy['grid_pos']) == attack_pos:
                    damage = word_data['damage']
                    enemy['health'] -= damage
                    total_damage += damage
                    # sfx: enemy_hit.wav
                    self._create_particles(self._grid_to_pixel(attack_pos), 10, self.COLOR_ATTACK)
                    self._create_text_popup(f"-{damage}", self._grid_to_pixel(attack_pos), self.COLOR_ENEMY)
        
        # Remove killed enemies
        surviving_enemies = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                surviving_enemies.append(enemy)
            else:
                kills += 1
                self.score += 10 # Bonus for kill
                # sfx: enemy_explode.wav
                self._create_particles(self._grid_to_pixel(enemy['grid_pos']), 30, self.COLOR_ENEMY)
        self.enemies = surviving_enemies
        
        reward = (total_damage * 0.1) + (kills * 5.0)
        self.score += int(total_damage)
        return reward

    def _update_enemies(self):
        tower_damage = 0
        for enemy in self.enemies:
            # Move towards next waypoint
            target_waypoint = self.ENEMY_PATH[enemy['waypoint_idx']]
            current_pos = np.array(enemy['grid_pos'], dtype=float)
            target_pos = np.array(target_waypoint, dtype=float)
            
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance < enemy['speed']:
                enemy['grid_pos'] = list(target_waypoint)
                enemy['waypoint_idx'] += 1
            else:
                move_vec = (direction / distance) * enemy['speed']
                new_pos = current_pos + move_vec
                enemy['grid_pos'] = list(new_pos)
            
            # Check if reached the tower
            if enemy['waypoint_idx'] >= len(self.ENEMY_PATH):
                tower_damage += enemy['damage']
                enemy['health'] = 0 # Mark for removal
                # sfx: tower_hit.wav
                self.screen_flash = 10
        
        self.enemies = [e for e in self.enemies if e['health'] > 0]
        self.tower_health = max(0, self.tower_health - tower_damage)
        assert self.tower_health <= self.max_tower_health

        return -tower_damage * 0.1

    def _check_wave_completion(self):
        if not self.enemies and not self.game_over:
            if self.current_wave >= 10:
                self.victory = True
                return 100 # Victory bonus
            
            self.current_wave += 1
            assert self.current_wave <= 10
            self._spawn_wave()
            self.tower_health = min(self.max_tower_health, self.tower_health + 10) # Heal between waves
            # sfx: wave_complete.wav
            return 100 # Wave survival bonus
        return 0

    def _spawn_wave(self):
        num_enemies = 2 + self.current_wave
        for i in range(num_enemies):
            health_multiplier = 1 + (self.current_wave - 1) * 0.1
            base_health = 30
            is_strong = random.random() < 0.2 * self.current_wave

            enemy = {
                "grid_pos": [self.ENEMY_PATH[0][0] + i * 0.5, self.ENEMY_PATH[0][1]],
                "waypoint_idx": 1,
                "max_health": (base_health * 2 if is_strong else base_health) * health_multiplier,
                "health": (base_health * 2 if is_strong else base_health) * health_multiplier,
                "speed": 0.05 + self.current_wave * 0.005,
                "damage": 15 if is_strong else 10,
                "is_strong": is_strong,
            }
            assert enemy['health'] <= enemy['max_health']
            self.enemies.append(enemy)

    def _get_new_words(self, count):
        return [random.choice(list(self.WORD_BANK.keys())) for _ in range(count)]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.screen_flash > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            alpha = int(100 * (self.screen_flash / 10.0))
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "tower_health": self.tower_health,
            "enemies_remaining": len(self.enemies),
        }

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return int(x), int(y)

    def _render_game(self):
        # Draw path
        for i in range(len(self.ENEMY_PATH) - 1):
            p1 = self._grid_to_pixel(self.ENEMY_PATH[i])
            p2 = self._grid_to_pixel(self.ENEMY_PATH[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.CELL_SIZE)

        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.CELL_SIZE))

        # Draw tower (at the end of the path)
        tx, ty = self._grid_to_pixel((-1, self.ENEMY_PATH[-1][1]))
        pygame.gfxdraw.box(self.screen, pygame.Rect(tx - self.CELL_SIZE//2, ty - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE), (*self.COLOR_TOWER, 150))
        pygame.gfxdraw.rectangle(self.screen, pygame.Rect(tx - self.CELL_SIZE//2, ty - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE), self.COLOR_TOWER)

        # Draw placed words and attack effects
        for word in self.placed_words:
            px, py = self._grid_to_pixel(word['pos'])
            rect = pygame.Rect(px - self.CELL_SIZE//2, py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WORD_TILE, rect, border_radius=3)
            word_text = self.font_s.render(word['word'], True, self.COLOR_BG)
            self.screen.blit(word_text, word_text.get_rect(center=rect.center))
            
            if word['attack_effect_timer'] > 0:
                word['attack_effect_timer'] -= 1
                word_data = self.WORD_BANK[word['word']]
                alpha = int(200 * (word['attack_effect_timer'] / 15.0))
                for offset in word_data['pattern']:
                    ax, ay = word['pos'][0] + offset[0], word['pos'][1] + offset[1]
                    if 0 <= ax < self.GRID_COLS and 0 <= ay < self.GRID_ROWS:
                        apx, apy = self._grid_to_pixel((ax, ay))
                        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                        s.fill((*self.COLOR_ATTACK, alpha))
                        self.screen.blit(s, (apx - self.CELL_SIZE//2, apy - self.CELL_SIZE//2))

        # Draw enemies
        for enemy in self.enemies:
            px, py = self._grid_to_pixel(enemy['grid_pos'])
            radius = self.CELL_SIZE // 3
            color = self.COLOR_ENEMY_STRONG if enemy['is_strong'] else self.COLOR_ENEMY
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
            # Health bar
            bar_w = self.CELL_SIZE * 0.8
            bar_h = 5
            health_pct = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (px - bar_w/2, py - radius - 10, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (px - bar_w/2, py - radius - 10, bar_w * health_pct, bar_h))

        # Draw cursor
        cx, cy = self._grid_to_pixel(self.cursor_pos)
        cursor_rect = pygame.Rect(cx - self.CELL_SIZE//2, cy - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

        # Draw particles and text popups
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'])
        for t in self.text_popups:
            text_surf = self.font_m.render(t['text'], True, t['color'])
            text_surf.set_alpha(t['alpha'])
            self.screen.blit(text_surf, t['pos'])

    def _render_ui(self):
        # Tower Health
        self._render_text("TOWER HEALTH", self.font_m, (self.UI_X_OFFSET, 30))
        health_pct = self.tower_health / self.max_tower_health
        bar_w, bar_h = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (self.UI_X_OFFSET, 55, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (self.UI_X_OFFSET, 55, bar_w * health_pct, bar_h))
        self._render_text(f"{int(self.tower_health)}/{self.max_tower_health}", self.font_s, (self.UI_X_OFFSET + bar_w + 10, 58))

        # Score and Wave
        self._render_text(f"SCORE: {self.score}", self.font_l, (self.UI_X_OFFSET, 100), color=self.COLOR_SCORE)
        self._render_text(f"WAVE: {self.current_wave}/10", self.font_l, (self.UI_X_OFFSET, 135))

        # Available Words
        self._render_text("AVAILABLE WORDS", self.font_m, (self.UI_X_OFFSET, 200))
        for i, word in enumerate(self.available_words):
            y_pos = 230 + i * 25
            color = self.COLOR_CURSOR if i == self.selected_word_index else self.COLOR_UI_TEXT
            prefix = "> " if i == self.selected_word_index else "  "
            self._render_text(f"{prefix}{word}", self.font_m, (self.UI_X_OFFSET, y_pos), color=color)

        # Game Over / Victory Message
        if self.game_over:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_HEALTH_GREEN if self.victory else self.COLOR_ENEMY
            self._render_text(msg, self.font_xl, (self.screen_width/2, self.screen_height/2 - 20), color, center=True)
            self._render_text(f"Final Score: {self.score}", self.font_l, (self.screen_width/2, self.screen_height/2 + 20), self.COLOR_SCORE, center=True)
    
    def _render_text(self, text, font, pos, color=None, center=False):
        if color is None: color = self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        if center:
            pos = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, pos)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [random.uniform(-2, 2), random.uniform(-2, 2)],
                'size': random.randint(2, 5),
                'lifetime': random.randint(10, 20),
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['size'] = max(0, p['size'] - 0.2)
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _create_text_popup(self, text, pos, color):
        self.text_popups.append({
            'text': text,
            'pos': [pos[0], pos[1] - 10],
            'vel_y': -0.5,
            'lifetime': 30,
            'color': color,
            'alpha': 255
        })

    def _update_text_popups(self):
        for t in self.text_popups:
            t['pos'][1] += t['vel_y']
            t['lifetime'] -= 1
            t['alpha'] = max(0, 255 * (t['lifetime'] / 30.0))
        self.text_popups = [t for t in self.text_popups if t['lifetime'] > 0]
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # To run and play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Word Tower Defense")
    
    running = True
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print(env.user_guide)
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_press = 1 if keys[pygame.K_SPACE] else 0
        shift_press = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_press, shift_press])

        # --- Step the Environment ---
        # For this turn-based game, we only step on an action
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game Over. Final Score: {info['score']}")
                # Wait for a moment before auto-resetting
                pygame.time.wait(3000)
                obs, info = env.reset()
        else:
            # If no action, just get the current observation
            obs = env._get_observation()
        
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    pygame.quit()