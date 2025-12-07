
# Generated: 2025-08-28T03:40:16.561525
# Source Brief: brief_05000.md
# Brief Index: 5000

        
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
        "Controls: Use arrow keys to move the selector. Press space to flip a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced Memory Match game. Find all matching pairs against the clock for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 6, 4
        self.TOTAL_CARDS = self.GRID_COLS * self.GRID_ROWS
        self.TOTAL_PAIRS = self.TOTAL_CARDS // 2
        
        # Visuals
        self.CARD_W, self.CARD_H = 80, 80
        self.CARD_ROUNDING = 0.2
        self.MARGIN_X = (self.WIDTH - self.GRID_COLS * self.CARD_W) / (self.GRID_COLS + 1)
        self.MARGIN_Y = (self.HEIGHT - self.GRID_ROWS * self.CARD_H) / (self.GRID_ROWS + 1)
        
        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_CARD_BACK = (60, 80, 100)
        self.COLOR_CARD_FACE = (210, 220, 230)
        self.COLOR_CARD_MATCHED = (40, 55, 70)
        self.COLOR_CURSOR = (255, 180, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_MATCH = (0, 255, 128)
        self.COLOR_MISMATCH = (255, 80, 80)
        
        self.SYMBOL_SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star', 'hexagon']
        self.SYMBOL_COLORS = [
            (255, 87, 34), (33, 150, 243), (76, 175, 80), (255, 235, 59),
            (156, 39, 176), (0, 188, 212), (255, 193, 7), (233, 30, 99),
            (139, 195, 74), (103, 58, 183), (255, 152, 0), (63, 81, 181)
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_cards = []
        self.mismatch_timer = 0
        self.mismatch_pair = []
        self.pairs_found = 0
        self.particles = []
        self.last_space_press = False
        self.last_move_action = 0
        self.move_cooldown = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 60.0
        self.cursor_pos = [0, 0]
        self.selected_cards = []
        self.mismatch_timer = 0
        self.mismatch_pair = []
        self.pairs_found = 0
        self.particles = []
        self.last_space_press = False
        self.last_move_action = 0
        self.move_cooldown = 0

        # Create and shuffle cards
        symbols = list(range(self.TOTAL_PAIRS)) * 2
        self.np_random.shuffle(symbols)
        
        self.grid = []
        for r in range(self.GRID_ROWS):
            row_list = []
            for c in range(self.GRID_COLS):
                card = {
                    "symbol_id": symbols.pop(),
                    "revealed": False,
                    "matched": False,
                    "anim_scale": 1.0, # 1.0 = face down, -1.0 = face up
                    "anim_speed": 0.2
                }
                row_list.append(card)
            self.grid.append(row_list)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- UPDATE TIMERS AND COOLDOWNS ---
        self.time_remaining -= 1.0 / 30.0
        self.steps += 1
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                for r, c in self.mismatch_pair:
                    self.grid[r][c]['revealed'] = False
                self.mismatch_pair = []
                self.selected_cards = []

        # --- HANDLE PLAYER INPUT ---
        # Movement
        if movement != 0 and self.move_cooldown == 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
            self.move_cooldown = 5 # 5-frame cooldown

        # Card selection
        space_pressed = space_held and not self.last_space_press
        if space_pressed and len(self.selected_cards) < 2 and self.mismatch_timer == 0:
            c, r = self.cursor_pos
            card = self.grid[r][c]
            if not card['revealed'] and not card['matched']:
                # sfx: card_flip.wav
                card['revealed'] = True
                self.selected_cards.append((r, c))
                reward += 0.1
            else:
                reward -= 0.02

        self.last_space_press = space_held

        # --- PROCESS GAME LOGIC ---
        if len(self.selected_cards) == 2:
            r1, c1 = self.selected_cards[0]
            r2, c2 = self.selected_cards[1]
            card1 = self.grid[r1][c1]
            card2 = self.grid[r2][c2]

            if card1['symbol_id'] == card2['symbol_id']:
                # Match
                # sfx: match_success.wav
                card1['matched'] = True
                card2['matched'] = True
                self.pairs_found += 1
                self.selected_cards = []
                reward += 5
                self.score += 100
                self._spawn_particles(c1, r1, self.COLOR_MATCH)
                self._spawn_particles(c2, r2, self.COLOR_MATCH)
            else:
                # Mismatch
                # sfx: mismatch_fail.wav
                self.mismatch_timer = 30 # 1 second
                self.mismatch_pair = [self.selected_cards[0], self.selected_cards[1]]
                reward -= 1
                self.score -= 10
        
        self._update_particles()
        self._update_card_animations()

        # --- CHECK TERMINATION CONDITIONS ---
        if self.pairs_found == self.TOTAL_PAIRS:
            # sfx: win_jingle.wav
            reward += 50
            self.score += self.time_remaining * 10
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0 or self.steps >= 1800: # 60s * 30fps
            # sfx: lose_sound.wav
            reward -= 50
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
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
            "time_remaining": self.time_remaining,
            "pairs_found": self.pairs_found,
        }

    def _render_game(self):
        # Draw cards
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self._draw_card(c, r)

        # Draw cursor
        self._draw_cursor()

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer
        time_str = f"{max(0, self.time_remaining):.1f}"
        time_color = self.COLOR_TEXT if self.time_remaining > 10 else self.COLOR_MISMATCH
        time_text = self.font_medium.render(time_str, True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(time_text, time_rect)
        
        # Pairs found
        pairs_text = self.font_small.render(f"{self.pairs_found} / {self.TOTAL_PAIRS} Pairs", True, self.COLOR_TEXT)
        pairs_rect = pairs_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(pairs_text, pairs_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.pairs_found == self.TOTAL_PAIRS:
                msg = "YOU WIN!"
                color = self.COLOR_MATCH
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_MISMATCH
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _update_card_animations(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                target_scale = -1.0 if card['revealed'] else 1.0
                if card['anim_scale'] != target_scale:
                    if card['anim_scale'] < target_scale:
                        card['anim_scale'] = min(target_scale, card['anim_scale'] + card['anim_speed'])
                    else:
                        card['anim_scale'] = max(target_scale, card['anim_scale'] - card['anim_speed'])

    def _draw_card(self, c, r):
        card = self.grid[r][c]
        card_x = self.MARGIN_X + c * (self.CARD_W + self.MARGIN_X)
        card_y = self.MARGIN_Y + r * (self.CARD_H + self.MARGIN_Y)
        
        # Animation scale
        scale = card['anim_scale']
        display_width = self.CARD_W * abs(math.cos(scale * math.pi / 2))
        
        card_rect = pygame.Rect(
            card_x + (self.CARD_W - display_width) / 2,
            card_y,
            display_width,
            self.CARD_H
        )

        if card['matched']:
            pygame.gfxdraw.box(self.screen, card_rect, (*self.COLOR_CARD_MATCHED, 100))
            return

        is_face_up = scale < 0
        if is_face_up: # Draw face
            pygame.draw.rect(self.screen, self.COLOR_CARD_FACE, card_rect, border_radius=int(self.CARD_W * self.CARD_ROUNDING))
            if display_width > 10: # Only draw symbol if card is somewhat open
                self._draw_symbol(card['symbol_id'], card_rect.center, self.CARD_W * 0.35)
        else: # Draw back
            pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, card_rect, border_radius=int(self.CARD_W * self.CARD_ROUNDING))

    def _draw_symbol(self, symbol_id, center, size):
        shape = self.SYMBOL_SHAPES[symbol_id % len(self.SYMBOL_SHAPES)]
        color = self.SYMBOL_COLORS[symbol_id % len(self.SYMBOL_COLORS)]
        x, y = center
        
        points = []
        if shape == 'circle':
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(size), color)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), color)
        elif shape == 'square':
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect)
        elif shape == 'triangle':
            points = [(x, y - size), (x - size, y + size * 0.7), (x + size, y + size * 0.7)]
        elif shape == 'diamond':
            points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        elif shape == 'star':
            points = []
            for i in range(10):
                angle = i * math.pi / 5
                r = size if i % 2 == 0 else size * 0.5
                points.append((x + r * math.sin(angle), y - r * math.cos(angle)))
        elif shape == 'hexagon':
            points = [(x + size * math.cos(a), y + size * math.sin(a)) for a in np.linspace(0, 2*math.pi, 7)[:-1]]

        if points:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_cursor(self):
        c, r = self.cursor_pos
        x = self.MARGIN_X + c * (self.CARD_W + self.MARGIN_X) - 5
        y = self.MARGIN_Y + r * (self.CARD_H + self.MARGIN_Y) - 5
        w = self.CARD_W + 10
        h = self.CARD_H + 10
        
        alpha = 150 + 100 * math.sin(self.steps * 0.2)
        color = (*self.COLOR_CURSOR, int(alpha))
        
        rect = pygame.Rect(x, y, w, h)
        
        # Use gfxdraw for anti-aliased rounded rectangle
        pygame.gfxdraw.box(self.screen, rect.inflate(4, 4), (*color[:3], int(alpha*0.3)))
        pygame.gfxdraw.box(self.screen, rect.inflate(2, 2), (*color[:3], int(alpha*0.6)))
        pygame.gfxdraw.rectangle(self.screen, rect, color)


    def _spawn_particles(self, c, r, color):
        x = self.MARGIN_X + c * (self.CARD_W + self.MARGIN_X) + self.CARD_W / 2
        y = self.MARGIN_Y + r * (self.CARD_H + self.MARGIN_Y) + self.CARD_H / 2
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'size': random.uniform(2, 5),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            p['size'] *= 0.97
            if p['life'] <= 0 or p['size'] < 0.5:
                self.particles.remove(p)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a dummy window to display the game
    pygame.display.set_caption("Memory Match")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action.fill(0) # Reset actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render ---
        # The observation is already the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
        
        env.clock.tick(30) # Maintain 30 FPS
        
    env.close()