import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:50:55.092135
# Source Brief: brief_01988.md
# Brief Index: 1988
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A fast-paced match-3 puzzle game with a seasonal festival theme. "
        "Match tiles to gain 'Favor', craft powerful items, and survive through rounds before the timer runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Hold space and press an arrow key to swap tiles. "
        "Hold shift to aim a card, press space to craft it, or tap shift to cycle cards."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    BOARD_ROWS, BOARD_COLS = 8, 8
    TILE_SIZE = 40
    NUM_TILE_TYPES = 4  # Spring, Summer, Autumn, Winter
    BOARD_X = (WIDTH - BOARD_COLS * TILE_SIZE) // 2
    BOARD_Y = (HEIGHT - BOARD_ROWS * TILE_SIZE) // 2 + 20

    MAX_STEPS_PER_EPISODE = 1200
    STEPS_PER_ROUND = 300

    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_GRID = (30, 45, 65)
    COLOR_UI_BG = (25, 40, 60)
    COLOR_TEXT = (220, 230, 255)
    COLOR_TEXT_ACCENT = (255, 215, 100)
    COLOR_CURSOR = (255, 255, 255)
    TILE_COLORS = [
        (100, 220, 120),  # 0: Spring Green
        (230, 80, 80),    # 1: Summer Red
        (240, 160, 50),   # 2: Autumn Orange
        (80, 150, 230),   # 3: Winter Blue
    ]
    
    # Game Feel
    ANIMATION_SPEED = 0.2
    PARTICLE_LIFESPAN = 20
    FLOAT_TEXT_LIFESPAN = 45

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # Game state not reset every episode
        self._initialize_cards()

        # Initialize state variables (will be properly set in reset)
        self.board = None
        self.visual_board = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.favor = 0
        self.current_round = 0
        self.round_timer = 0
        self.game_over = False
        self.selected_card_index = 0
        self.prev_shift_held = False
        self.crafted_this_round = False
        self.board_locked_timer = 0
        self.particles = []
        self.floating_texts = []

        self.reset()

    def _initialize_cards(self):
        self.cards = [
            {"name": "Celebration Firework", "cost": 20, "effect": "craft", "desc": "+5 Score"},
            {"name": "Bountiful Harvest", "cost": 40, "effect": "clear_type", "desc": "Clear all Spring tiles"},
            {"name": "Grand Finale", "cost": 80, "effect": "clear_3x3", "desc": "Clear 3x3 area"},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.favor = 10
        self.current_round = 1
        self.round_timer = self.STEPS_PER_ROUND
        self.game_over = False
        self.crafted_this_round = False
        
        self.cursor_pos = [self.BOARD_COLS // 2, self.BOARD_ROWS // 2]
        self.selected_card_index = 0
        self.prev_shift_held = False
        
        self.board_locked_timer = 0
        self.particles = []
        self.floating_texts = []

        self._create_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.round_timer -= 1
        self.board_locked_timer = max(0, self.board_locked_timer - 1)
        
        reward = 0
        is_board_idle = self.board_locked_timer == 0

        # --- Action Processing ---
        if is_board_idle:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            shift_pressed = shift_held and not self.prev_shift_held

            # 1. Cycle Cards (Shift press)
            if shift_pressed:
                self.selected_card_index = (self.selected_card_index + 1) % len(self.cards)
                # sfx: card_cycle.wav

            # 2. Craft Item (Space + Shift)
            if space_held and shift_held:
                card = self.cards[self.selected_card_index]
                if self.favor >= card['cost']:
                    self.favor -= card['cost']
                    self.crafted_this_round = True
                    reward += 0.5
                    # sfx: craft_success.wav
                    self._apply_card_effect(card)
                    self.board_locked_timer = 30 # Lock board for effect animation
                else:
                    # sfx: craft_fail.wav
                    pass

            # 3. Swap Tiles (Space + Movement, no Shift)
            elif space_held and not shift_held and movement != 0:
                reward += self._attempt_swap(movement)

            # 4. Move Cursor (Movement only)
            elif movement != 0:
                self._move_cursor(movement)
            
            self.prev_shift_held = shift_held
        
        # --- Game Logic Update ---
        self._update_animations()

        if is_board_idle:
            match_reward = self._resolve_board()
            if match_reward > 0:
                reward += match_reward
                self.board_locked_timer = 20 # small delay to appreciate match

        # --- Round and Termination Check ---
        terminated = False
        truncated = False
        if self.round_timer <= 0:
            if self.crafted_this_round:
                self.current_round += 1
                self.round_timer = self.STEPS_PER_ROUND
                self.crafted_this_round = False
                reward += 2.0 # Round survival bonus
                self._create_floating_text(f"Round {self.current_round}", (self.WIDTH//2, self.HEIGHT//2), self.COLOR_TEXT_ACCENT, size='large')
            else:
                terminated = True
                self.game_over = True
                reward = -50.0 # Failure penalty
                self._create_floating_text("TIME UP!", (self.WIDTH//2, self.HEIGHT//2), (255, 50, 50), size='large')

        if self.steps >= self.MAX_STEPS_PER_EPISODE:
            terminated = True
            self.game_over = True
            if self.score > 0: # Check if player actually played
                reward = 50.0 # Success bonus
                self._create_floating_text("FESTIVAL COMPLETE!", (self.WIDTH//2, self.HEIGHT//2), self.COLOR_TEXT_ACCENT, size='large')

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_cursor(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        self.cursor_pos = [np.clip(x, 0, self.BOARD_COLS - 1), np.clip(y, 0, self.BOARD_ROWS - 1)]

    def _attempt_swap(self, movement):
        x1, y1 = self.cursor_pos
        x2, y2 = x1, y1
        if movement == 1: y2 -= 1
        elif movement == 2: y2 += 1
        elif movement == 3: x2 -= 1
        elif movement == 4: x2 += 1

        if 0 <= x2 < self.BOARD_COLS and 0 <= y2 < self.BOARD_ROWS:
            # Perform swap
            self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
            self.visual_board[y1][x1]['pos'], self.visual_board[y2][x2]['pos'] = self.visual_board[y2][x2]['pos'], self.visual_board[y1][x1]['pos']
            
            # Check for matches
            matches1 = self._find_matches_at(x1, y1)
            matches2 = self._find_matches_at(x2, y2)
            
            if not matches1 and not matches2:
                # Invalid swap, swap back
                self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
                self.visual_board[y1][x1]['pos'], self.visual_board[y2][x2]['pos'] = self.visual_board[y2][x2]['pos'], self.visual_board[y1][x1]['pos']
                # sfx: swap_fail.wav
                return -0.01 # Small penalty for invalid move
            else:
                # Valid swap
                self.board_locked_timer = 10 # Lock for swap animation
                # sfx: swap_success.wav
                return 0
        return 0

    def _apply_card_effect(self, card):
        effect = card['effect']
        if effect == 'craft':
            self.score += 5
            self._create_floating_text("+5 Score!", self._get_pixel_pos(self.cursor_pos[0], self.cursor_pos[1]), self.COLOR_TEXT_ACCENT)
            self._create_particles(self._get_pixel_pos(self.cursor_pos[0], self.cursor_pos[1]), (255, 215, 0), 30)
        elif effect == 'clear_type':
            tiles_to_clear = np.where(self.board == 0) # Clear Spring tiles
            if tiles_to_clear[0].size > 0:
                self._handle_matches(list(zip(tiles_to_clear[1], tiles_to_clear[0])))
        elif effect == 'clear_3x3':
            cx, cy = self.cursor_pos
            tiles_to_clear = []
            for y in range(cy - 1, cy + 2):
                for x in range(cx - 1, cx + 2):
                    if 0 <= x < self.BOARD_COLS and 0 <= y < self.BOARD_ROWS:
                        tiles_to_clear.append((x, y))
            if tiles_to_clear:
                self._handle_matches(tiles_to_clear)

    def _resolve_board(self):
        total_reward = 0
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            num_matched_tiles = len(matches)
            total_reward += 0.1 * num_matched_tiles
            if num_matched_tiles >= 5:
                total_reward += 1.0 # Bonus for large match

            self._handle_matches(matches)
            self._apply_gravity()
            self._fill_top_rows()
        
        if total_reward > 0 and self.board_locked_timer == 0:
             self._check_for_valid_moves_and_reshuffle()
        
        return total_reward

    def _handle_matches(self, matches):
        # sfx: match.wav
        favor_gain = len(matches)
        self.favor += favor_gain
        self._create_floating_text(f"+{favor_gain} Favor", self._get_pixel_pos(matches[0][0], matches[0][1]), (200, 200, 255))

        for x, y in matches:
            if self.board[y, x] != -1:
                self._create_particles(self._get_pixel_pos(x, y), self.TILE_COLORS[self.board[y, x]], 10)
                self.board[y, x] = -1 # Mark for removal
    
    def _apply_gravity(self):
        for x in range(self.BOARD_COLS):
            empty_row = self.BOARD_ROWS - 1
            for y in range(self.BOARD_ROWS - 1, -1, -1):
                if self.board[y, x] != -1:
                    if y != empty_row:
                        self.board[empty_row, x] = self.board[y, x]
                        self.board[y, x] = -1
                        # Update visual board for animation
                        self.visual_board[y][x]['type'] = self.board[empty_row, x]
                        self.visual_board[empty_row][x], self.visual_board[y][x] = self.visual_board[y][x], self.visual_board[empty_row][x]
                    empty_row -= 1

    def _fill_top_rows(self):
        for y in range(self.BOARD_ROWS):
            for x in range(self.BOARD_COLS):
                if self.board[y, x] == -1:
                    self.board[y, x] = self.np_random.integers(0, self.NUM_TILE_TYPES)
                    # Reset visual tile
                    self.visual_board[y][x]['type'] = self.board[y, x]
                    self.visual_board[y][x]['pos'] = [self.BOARD_X + x * self.TILE_SIZE, self.BOARD_Y + y * self.TILE_SIZE - self.BOARD_ROWS * self.TILE_SIZE]
                    self.visual_board[y][x]['scale'] = 1.0

    def _find_matches_at(self, x, y):
        if not (0 <= x < self.BOARD_COLS and 0 <= y < self.BOARD_ROWS):
            return []
        
        tile_type = self.board[y, x]
        if tile_type == -1: return []

        # Horizontal
        h_match = [(x, y)]
        for i in range(x - 1, -1, -1):
            if self.board[y, i] == tile_type: h_match.append((i, y))
            else: break
        for i in range(x + 1, self.BOARD_COLS):
            if self.board[y, i] == tile_type: h_match.append((i, y))
            else: break
        
        # Vertical
        v_match = [(x, y)]
        for i in range(y - 1, -1, -1):
            if self.board[i, x] == tile_type: v_match.append((x, i))
            else: break
        for i in range(y + 1, self.BOARD_ROWS):
            if self.board[i, x] == tile_type: v_match.append((x, i))
            else: break
            
        matches = set()
        if len(h_match) >= 3: matches.update(h_match)
        if len(v_match) >= 3: matches.update(v_match)
        return list(matches)

    def _find_all_matches(self):
        all_matches = set()
        for y in range(self.BOARD_ROWS):
            for x in range(self.BOARD_COLS):
                matches = self._find_matches_at(x,y)
                if matches:
                    all_matches.update(matches)
        return list(all_matches)

    def _create_board(self):
        self.board = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.BOARD_ROWS, self.BOARD_COLS))
        # Ensure no initial matches
        while self._find_all_matches():
            matches = self._find_all_matches()
            for x, y in matches:
                self.board[y, x] = self.np_random.integers(0, self.NUM_TILE_TYPES)
        
        self.visual_board = [[{'type': self.board[y, x], 
                               'pos': [self.BOARD_X + x * self.TILE_SIZE, self.BOARD_Y + y * self.TILE_SIZE],
                               'scale': 1.0} 
                              for x in range(self.BOARD_COLS)] for y in range(self.BOARD_ROWS)]
        self._check_for_valid_moves_and_reshuffle()

    def _check_for_valid_moves_and_reshuffle(self):
        if self._has_possible_moves():
            return
        
        # No moves, reshuffle
        flat_board = self.board.flatten().tolist()
        self.np_random.shuffle(flat_board)
        self.board = np.array(flat_board).reshape((self.BOARD_ROWS, self.BOARD_COLS))
        
        # Reset visual board and resolve any new matches
        for y in range(self.BOARD_ROWS):
            for x in range(self.BOARD_COLS):
                self.visual_board[y][x]['type'] = self.board[y,x]
        
        self._resolve_board()
        self._check_for_valid_moves_and_reshuffle() # Recurse until a valid state is found

    def _has_possible_moves(self):
        for y in range(self.BOARD_ROWS):
            for x in range(self.BOARD_COLS):
                # Check swap right
                if x < self.BOARD_COLS - 1:
                    self.board[y, x], self.board[y, x+1] = self.board[y, x+1], self.board[y, x]
                    if self._find_matches_at(x, y) or self._find_matches_at(x+1, y):
                        self.board[y, x], self.board[y, x+1] = self.board[y, x+1], self.board[y, x]
                        return True
                    self.board[y, x], self.board[y, x+1] = self.board[y, x+1], self.board[y, x]
                # Check swap down
                if y < self.BOARD_ROWS - 1:
                    self.board[y, x], self.board[y+1, x] = self.board[y+1, x], self.board[y, x]
                    if self._find_matches_at(x, y) or self._find_matches_at(x, y+1):
                        self.board[y, x], self.board[y+1, x] = self.board[y+1, x], self.board[y, x]
                        return True
                    self.board[y, x], self.board[y+1, x] = self.board[y+1, x], self.board[y, x]
        return False

    def _update_animations(self):
        # Tile fall animation
        for y in range(self.BOARD_ROWS):
            for x in range(self.BOARD_COLS):
                v_tile = self.visual_board[y][x]
                target_pos = [self.BOARD_X + x * self.TILE_SIZE, self.BOARD_Y + y * self.TILE_SIZE]
                v_tile['pos'][0] += (target_pos[0] - v_tile['pos'][0]) * self.ANIMATION_SPEED
                v_tile['pos'][1] += (target_pos[1] - v_tile['pos'][1]) * self.ANIMATION_SPEED
                if self.board[y, x] == -1:
                    v_tile['scale'] = max(0, v_tile['scale'] - 0.1)
                else:
                    v_tile['scale'] = min(1, v_tile['scale'] + 0.1)
        
        # Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)
        
        # Floating texts
        for ft in self.floating_texts[:]:
            ft['pos'][1] -= ft['vel']
            ft['life'] -= 1
            if ft['life'] <= 0: self.floating_texts.remove(ft)

    def _get_pixel_pos(self, x, y):
        return (self.BOARD_X + x * self.TILE_SIZE + self.TILE_SIZE // 2, 
                self.BOARD_Y + y * self.TILE_SIZE + self.TILE_SIZE // 2)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': self.np_random.integers(self.PARTICLE_LIFESPAN // 2, self.PARTICLE_LIFESPAN)
            })
    
    def _create_floating_text(self, text, pos, color, size='medium'):
        font = self.font_medium if size == 'medium' else self.font_large
        self.floating_texts.append({
            'text': text,
            'pos': list(pos),
            'color': color,
            'life': self.FLOAT_TEXT_LIFESPAN,
            'vel': 1.0,
            'font': font
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.BOARD_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X, self.BOARD_Y + r * self.TILE_SIZE), 
                             (self.BOARD_X + self.BOARD_COLS * self.TILE_SIZE, self.BOARD_Y + r * self.TILE_SIZE))
        for c in range(self.BOARD_COLS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X + c * self.TILE_SIZE, self.BOARD_Y),
                             (self.BOARD_X + c * self.TILE_SIZE, self.BOARD_Y + self.BOARD_ROWS * self.TILE_SIZE))

        # Draw tiles
        for y in range(self.BOARD_ROWS):
            for x in range(self.BOARD_COLS):
                v_tile = self.visual_board[y][x]
                if v_tile['scale'] > 0:
                    color = self.TILE_COLORS[v_tile['type']]
                    size = int(self.TILE_SIZE * 0.8 * v_tile['scale'])
                    pos_x = int(v_tile['pos'][0] + self.TILE_SIZE // 2)
                    pos_y = int(v_tile['pos'][1] + self.TILE_SIZE // 2)
                    
                    rect = pygame.Rect(pos_x - size // 2, pos_y - size // 2, size, size)
                    pygame.draw.rect(self.screen, color, rect, border_radius=5)

        # Draw cursor
        cursor_pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 4
        cx, cy = self.cursor_pos
        crect = pygame.Rect(self.BOARD_X + cx * self.TILE_SIZE - 2 + cursor_pulse/2, 
                            self.BOARD_Y + cy * self.TILE_SIZE - 2 + cursor_pulse/2,
                            self.TILE_SIZE + 4 - cursor_pulse, self.TILE_SIZE + 4 - cursor_pulse)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, crect, 2, border_radius=7)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / self.PARTICLE_LIFESPAN))
            # Create a temporary surface to draw the circle with alpha
            temp_surf = pygame.Surface((int(p['life'] / self.PARTICLE_LIFESPAN * 10), int(p['life'] / self.PARTICLE_LIFESPAN * 10)), pygame.SRCALPHA)
            size = int(p['life'] / self.PARTICLE_LIFESPAN * 5)
            pygame.draw.circle(temp_surf, p['color'] + (alpha,), (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

        # Draw floating texts
        for ft in self.floating_texts:
            alpha = int(255 * (ft['life'] / self.FLOAT_TEXT_LIFESPAN))
            text_surf = ft['font'].render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, text_surf.get_rect(center=(int(ft['pos'][0]), int(ft['pos'][1]))))

    def _render_ui(self):
        # Top UI bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.WIDTH, 50))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 50), (self.WIDTH, 50))
        
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 15))
        
        favor_text = self.font_medium.render(f"Favor: {self.favor}", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(favor_text, (180, 15))

        round_text = self.font_medium.render(f"Round: {self.current_round}", True, self.COLOR_TEXT)
        self.screen.blit(round_text, (self.WIDTH - 280, 15))

        # Timer bar
        timer_ratio = self.round_timer / self.STEPS_PER_ROUND
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.WIDTH - 150, 15, 140, 20))
        pygame.draw.rect(self.screen, self.COLOR_TEXT_ACCENT, (self.WIDTH - 150, 15, 140 * timer_ratio, 20))

        # Bottom Card UI
        card_y = self.HEIGHT - 40
        card = self.cards[self.selected_card_index]
        card_text = f"Card: {card['name']} (Cost: {card['cost']}) - {card['desc']}"
        card_render = self.font_small.render(card_text, True, self.COLOR_TEXT)
        self.screen.blit(card_render, card_render.get_rect(centerx=self.WIDTH/2, y=card_y))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "favor": self.favor,
            "round": self.current_round,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block is for human play and is not part of the environment's core API
    # It will not be executed by the validation tests.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for human play
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Festival of Favor")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = False
        shift_held = False

        # Event handling first
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        # Key state processing
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Optionally reset here or just stop
            running = False 

        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()
    pygame.quit()