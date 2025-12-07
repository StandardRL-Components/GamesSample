import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:17:35.363462
# Source Brief: brief_00330.md
# Brief Index: 330
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Archive Defense
    Match enchanted books to build magical defenses against monstrous pages.
    Strategically rewind time to shrink past threats and maintain archive integrity.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Cursor Movement & Swap Attempt (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Board Shuffle (0=released, 1=pressed) - Reshuffles the board if no matches are possible.
    - actions[2]: Time Rewind (0=released, 1=pressed) - Shrinks all on-screen enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match enchanted books to build magical defenses against monstrous pages. "
        "Strategically rewind time to shrink past threats and maintain archive integrity."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor and swap books. "
        "Press space to shuffle the board. Press shift to rewind time."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GAME_WIDTH, GAME_HEIGHT = 400, 400
    UI_WIDTH = SCREEN_WIDTH - GAME_WIDTH
    
    GRID_COLS, GRID_ROWS = 8, 8
    CELL_SIZE = GAME_WIDTH // GRID_COLS
    
    NUM_BOOK_TYPES = 5
    MATCH_MIN_LENGTH = 3
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_BG_SHELF = (30, 20, 45)
    COLOR_UI_BG = (25, 20, 40)
    COLOR_UI_FRAME = (60, 50, 80)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (10, 5, 15)
    COLOR_CURSOR = (255, 255, 100)
    
    BOOK_COLORS = [
        (50, 150, 255),  # Sapphire Blue
        (255, 80, 80),   # Ruby Red
        (80, 255, 120),  # Emerald Green
        (255, 200, 50),  # Topaz Yellow
        (200, 100, 255), # Amethyst Purple
    ]
    
    ENEMY_COLOR = (220, 40, 40)
    INTEGRITY_ORB_COLOR = (0, 191, 255)
    REWIND_EFFECT_COLOR = (170, 0, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_l = pygame.font.Font(None, 36)
        self.font_m = pygame.font.Font(None, 24)
        self.font_s = pygame.font.Font(None, 18)

        # Game state variables
        self.board = None
        self.cursor_pos = None
        self.enemies = None
        self.particles = None
        self.archive_integrity = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.rewind_cooldown = None
        self.shuffle_cooldown = None
        self.enemy_spawn_rate = None
        self.enemy_base_speed = None
        self.last_action_feedback = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.archive_integrity = 100.0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.enemies = []
        self.particles = []
        
        self.rewind_cooldown = 0
        self.shuffle_cooldown = 0
        
        self.enemy_spawn_rate = 0.02
        self.enemy_base_speed = 0.2
        
        self._create_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        self.last_action_feedback = {}

        # --- 1. Handle Actions ---
        move_action = action[0]
        shuffle_action = action[1] == 1
        rewind_action = action[2] == 1
        
        # Cooldowns
        self.rewind_cooldown = max(0, self.rewind_cooldown - 1)
        self.shuffle_cooldown = max(0, self.shuffle_cooldown - 1)

        # Action: Time Rewind
        if rewind_action and self.rewind_cooldown == 0:
            reward += self._rewind_time()
            self.last_action_feedback['rewind'] = 'success'
        elif rewind_action:
            self.last_action_feedback['rewind'] = 'cooldown'

        # Action: Board Shuffle
        if shuffle_action and self.shuffle_cooldown == 0:
            if not self._find_possible_matches():
                self._shuffle_board()
                reward -= 2.0 # Small penalty for needing a shuffle
                self.shuffle_cooldown = 50
                self.last_action_feedback['shuffle'] = 'success'
            else:
                self.last_action_feedback['shuffle'] = 'matches_exist'
        elif shuffle_action:
            self.last_action_feedback['shuffle'] = 'cooldown'

        # Action: Cursor Movement / Swap
        if move_action != 0:
            reward += self._handle_swap(move_action)

        # --- 2. Update Game Logic ---
        self._update_enemies()
        reward += self._check_enemy_breach()
        self._spawn_enemies()
        self._update_particles()
        self._update_difficulty()

        # --- 3. Check Termination ---
        terminated = self.archive_integrity <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # This environment does not truncate based on time limit in the same way, MAX_STEPS is a termination condition
        if self.steps >= self.MAX_STEPS:
            terminated = True

        if self.archive_integrity <= 0:
            terminated = True
            
        if terminated:
            self.game_over = True
            if self.archive_integrity <= 0:
                reward -= 100.0 # Loss
                self.last_action_feedback['game_over'] = 'integrity_zero'
            else:
                # If terminated by steps, it's a win
                reward += 100.0 # Win
                self.last_action_feedback['game_over'] = 'victory'
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_swap(self, move_action):
        # Map action to delta
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = deltas[move_action]
        
        r1, c1 = self.cursor_pos
        r2, c2 = r1 + dr, c1 + dc

        # Update cursor position regardless of swap validity
        self.cursor_pos = [max(0, min(self.GRID_ROWS - 1, r1 + dr)), max(0, min(self.GRID_COLS - 1, c1 + dc))]
        
        # New cursor position for the swap check
        nr, nc = self.cursor_pos

        # Check if swap is within bounds (r1,c1 to nr,nc)
        if not (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS):
             self.last_action_feedback['swap'] = 'out_of_bounds'
             return 0

        # Perform swap
        self._swap_books((r1, c1), (nr, nc))
        
        # Check for matches
        all_matches = self._find_all_matches()
        
        if all_matches:
            # Successful match
            # SFX: MatchSuccess.wav
            reward = self._resolve_matches(all_matches)
            self.last_action_feedback['swap'] = 'match_found'
            return reward
        else:
            # No match, swap back
            self._swap_books((r1, c1), (nr, nc))
            self.last_action_feedback['swap'] = 'no_match'
            return -0.01 # Small penalty for invalid move

    def _rewind_time(self):
        # SFX: TimeRewind.wav
        self.rewind_cooldown = 100
        enemies_on_screen = len(self.enemies) > 0
        
        for enemy in self.enemies:
            enemy['size'] *= 0.6 # Shrink
        
        self.enemies = [e for e in self.enemies if e['size'] > 2]
        
        # Add visual effect
        for _ in range(100):
            self._create_particle(
                (self.np_random.uniform(0, self.GAME_WIDTH), self.np_random.uniform(0, self.GAME_HEIGHT)),
                self.REWIND_EFFECT_COLOR,
                lifespan=20,
                size=self.np_random.uniform(2, 5),
                velocity=(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            )

        if enemies_on_screen and not self.enemies:
            return 5.0 # Bonus for clearing the screen
        return 0.1 # Small reward for using the ability

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['y'] += self.enemy_base_speed * (enemy['size'] / 15.0) # Larger enemies are faster
            enemy['anim_offset'] = (enemy['anim_offset'] + 1) % 10

    def _check_enemy_breach(self):
        breached_reward = 0
        surviving_enemies = []
        for enemy in self.enemies:
            if enemy['y'] + enemy['size'] / 2 > self.GAME_HEIGHT:
                # SFX: Damage.wav
                damage = (enemy['size'] / 20.0) * 10
                self.archive_integrity = max(0, self.archive_integrity - damage)
                breached_reward -= 0.1
                
                # Damage particle effect
                for _ in range(20):
                    self._create_particle(
                        (enemy['x'], self.GAME_HEIGHT - 5),
                        self.ENEMY_COLOR,
                        lifespan=30,
                        size=self.np_random.uniform(2, 6),
                        velocity=(self.np_random.uniform(-3, 3), self.np_random.uniform(-5, -1))
                    )
            else:
                surviving_enemies.append(enemy)
        self.enemies = surviving_enemies
        return breached_reward

    def _spawn_enemies(self):
        if self.np_random.random() < self.enemy_spawn_rate:
            # SFX: EnemySpawn.wav
            self.enemies.append({
                'x': self.np_random.uniform(20, self.GAME_WIDTH - 20),
                'y': -10,
                'size': self.np_random.uniform(10, 25),
                'speed': self.enemy_base_speed,
                'anim_offset': self.np_random.integers(0, 10)
            })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 10 == 0:
            self.enemy_spawn_rate = min(0.2, self.enemy_spawn_rate + 0.001)
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_base_speed = min(1.0, self.enemy_base_speed + 0.005)

    def _resolve_matches(self, all_matches):
        reward = 0
        
        # Flatten the set of matched coordinates
        matched_coords = set()
        for match in all_matches:
            for pos in match:
                matched_coords.add(pos)
        
        # Add reward and create particles
        for r, c in matched_coords:
            if self.board[r][c] != -1: # Avoid double counting
                reward += 1.0
                self.score += 1
                book_type = self.board[r][c]
                px, py = c * self.CELL_SIZE + self.CELL_SIZE / 2, r * self.CELL_SIZE + self.CELL_SIZE / 2
                for _ in range(15):
                    self._create_particle((px, py), self.BOOK_COLORS[book_type])
                self.board[r][c] = -1 # Mark for removal
        
        # Gravity and refill
        self._apply_gravity()
        self._refill_board()
        
        # Chain reaction check
        chain_matches = self._find_all_matches()
        if chain_matches:
            # SFX: ChainMatch.wav
            reward += 2.0 # Bonus for chain
            reward += self._resolve_matches(chain_matches)
        
        return reward

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.board[r][c] != -1:
                    if r != empty_row:
                        self.board[empty_row][c] = self.board[r][c]
                        self.board[r][c] = -1
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.board[r][c] == -1:
                    self.board[r][c] = self.np_random.integers(0, self.NUM_BOOK_TYPES)

    def _create_board(self):
        self.board = self.np_random.integers(0, self.NUM_BOOK_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
        while self._find_all_matches():
            self._resolve_matches(self._find_all_matches())

    def _shuffle_board(self):
        # SFX: BoardShuffle.wav
        flat_board = list(self.board.flatten())
        self.np_random.shuffle(flat_board)
        self.board = np.array(flat_board).reshape((self.GRID_ROWS, self.GRID_COLS))
        # Ensure shuffle doesn't create instant matches
        while self._find_all_matches():
            self._resolve_matches(self._find_all_matches())
            
    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - self.MATCH_MIN_LENGTH + 1):
                if self.board[r][c] != -1:
                    book_type = self.board[r][c]
                    if all(self.board[r][c+i] == book_type for i in range(self.MATCH_MIN_LENGTH)):
                        match = tuple(sorted([(r, c+i) for i in range(self.MATCH_MIN_LENGTH)]))
                        matches.add(match)
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - self.MATCH_MIN_LENGTH + 1):
                if self.board[r][c] != -1:
                    book_type = self.board[r][c]
                    if all(self.board[r+i][c] == book_type for i in range(self.MATCH_MIN_LENGTH)):
                        match = tuple(sorted([(r+i, c) for i in range(self.MATCH_MIN_LENGTH)]))
                        matches.add(match)
        return list(matches)

    def _find_possible_matches(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Try swapping right
                if c < self.GRID_COLS - 1:
                    self._swap_books((r, c), (r, c + 1))
                    if self._find_all_matches():
                        self._swap_books((r, c), (r, c + 1)) # Swap back
                        return True
                    self._swap_books((r, c), (r, c + 1)) # Swap back
                # Try swapping down
                if r < self.GRID_ROWS - 1:
                    self._swap_books((r, c), (r + 1, c))
                    if self._find_all_matches():
                        self._swap_books((r, c), (r + 1, c)) # Swap back
                        return True
                    self._swap_books((r, c), (r + 1, c)) # Swap back
        return False

    def _swap_books(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.board[r1][c1], self.board[r2][c2] = self.board[r2][c2], self.board[r1][c1]

    def _create_particle(self, pos, color, lifespan=20, size=3, velocity=None):
        if velocity is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
        else:
            vel = velocity
        self.particles.append({'x': pos[0], 'y': pos[1], 'vx': vel[0], 'vy': vel[1], 'lifespan': lifespan, 'max_life': lifespan, 'color': color, 'size': size})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "archive_integrity": self.archive_integrity, "rewind_cooldown": self.rewind_cooldown, "shuffle_cooldown": self.shuffle_cooldown}

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for i in range(self.GRID_COLS + 1):
            x = i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG_SHELF, (x, 0), (x, self.GAME_HEIGHT), 2)
        
        # Draw Books
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                book_type = self.board[r][c]
                if book_type != -1:
                    self._draw_book(r, c, book_type)
        
        # Draw Cursor
        self._draw_cursor()
        
        # Draw Enemies
        for enemy in self.enemies:
            self._draw_enemy(enemy)
            
        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), (*color, alpha))

        # Draw UI
        self._render_ui()

    def _draw_book(self, r, c, book_type):
        x, y = c * self.CELL_SIZE, r * self.CELL_SIZE
        color = self.BOOK_COLORS[book_type]
        rect = pygame.Rect(x + 4, y + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        
        # Draw symbol
        center_x, center_y = x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2
        symbol_color = (255,255,255)
        if book_type == 0: # Circle
            pygame.draw.circle(self.screen, symbol_color, (center_x, center_y), 10, 2)
        elif book_type == 1: # Triangle
            points = [(center_x, center_y - 8), (center_x - 9, center_y + 6), (center_x + 9, center_y + 6)]
            pygame.draw.polygon(self.screen, symbol_color, points, 2)
        elif book_type == 2: # Square
            pygame.draw.rect(self.screen, symbol_color, (center_x - 8, center_y - 8, 16, 16), 2)
        elif book_type == 3: # Star
            n = 5
            radius1, radius2 = 10, 4
            points = []
            for i in range(2 * n):
                radius = radius1 if i % 2 == 0 else radius2
                angle = i * math.pi / n - math.pi / 2
                points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
            pygame.draw.polygon(self.screen, symbol_color, points, 2)
        elif book_type == 4: # Diamond
            points = [(center_x, center_y-10), (center_x+8, center_y), (center_x, center_y+10), (center_x-8, center_y)]
            pygame.draw.polygon(self.screen, symbol_color, points, 2)

    def _draw_cursor(self):
        r, c = self.cursor_pos
        x, y = c * self.CELL_SIZE, r * self.CELL_SIZE
        
        # Pulsating glow effect
        glow_alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        
        # Draw a slightly larger, semi-transparent rectangle for the glow
        glow_rect = pygame.Rect(x - 2, y - 2, self.CELL_SIZE + 4, self.CELL_SIZE + 4)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_CURSOR, glow_alpha), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)

        # Draw the main cursor outline
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (x, y, self.CELL_SIZE, self.CELL_SIZE), 3, border_radius=6)

    def _draw_enemy(self, enemy):
        x, y, size = enemy['x'], enemy['y'], enemy['size']
        points = []
        num_points = 8
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Jitter radius and angle for crumpled paper effect
            offset = (i + enemy['anim_offset']) % num_points
            radius_jitter = (math.sin(offset * 1.3) * 0.2 + 0.9)
            angle_jitter = math.sin(offset * 2.1) * 0.1
            
            radius = size / 2 * radius_jitter
            px = x + radius * math.cos(angle + angle_jitter)
            py = y + radius * math.sin(angle + angle_jitter)
            points.append((int(px), int(py)))
        
        # Draw outline and fill
        outline_color = tuple(max(0, c-50) for c in self.ENEMY_COLOR)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.ENEMY_COLOR)
        pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _render_ui(self):
        ui_rect = pygame.Rect(self.GAME_WIDTH, 0, self.UI_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_FRAME, (self.GAME_WIDTH, 0), (self.GAME_WIDTH, self.SCREEN_HEIGHT), 2)
        
        def draw_text(text, font, color, pos, shadow=True):
            if shadow:
                text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Title
        draw_text("ARCHIVE DEFENSE", self.font_l, self.COLOR_TEXT, (self.GAME_WIDTH + 15, 20))
        
        # Score
        draw_text(f"SCORE: {self.score}", self.font_m, self.COLOR_TEXT, (self.GAME_WIDTH + 20, 70))
        
        # Steps
        draw_text(f"STEPS: {self.steps}/{self.MAX_STEPS}", self.font_m, self.COLOR_TEXT, (self.GAME_WIDTH + 20, 100))
        
        # Archive Integrity
        draw_text("ARCHIVE INTEGRITY", self.font_m, self.COLOR_TEXT, (self.GAME_WIDTH + 20, 150))
        orb_center = (self.GAME_WIDTH + self.UI_WIDTH // 2, 220)
        orb_radius = 40
        
        # Orb glow
        glow_alpha = 100 + 50 * math.sin(pygame.time.get_ticks() * 0.002)
        pygame.gfxdraw.filled_circle(self.screen, orb_center[0], orb_center[1], orb_radius + 5, (*self.INTEGRITY_ORB_COLOR, glow_alpha))
        
        # Orb background and fill
        pygame.gfxdraw.filled_circle(self.screen, orb_center[0], orb_center[1], orb_radius, (30,30,60))
        fill_height = int(orb_radius * 2 * (self.archive_integrity / 100.0))
        if fill_height > 0:
            fill_rect = pygame.Rect(orb_center[0] - orb_radius, orb_center[1] + orb_radius - fill_height, orb_radius * 2, fill_height)
            pygame.draw.rect(self.screen, self.INTEGRITY_ORB_COLOR, fill_rect, border_bottom_left_radius=orb_radius, border_bottom_right_radius=orb_radius)

        pygame.gfxdraw.aacircle(self.screen, orb_center[0], orb_center[1], orb_radius, self.COLOR_UI_FRAME)
        integrity_text = f"{int(self.archive_integrity)}%"
        text_surf = self.font_m.render(integrity_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (orb_center[0] - text_surf.get_width() // 2, orb_center[1] - text_surf.get_height() // 2))

        # Cooldowns
        draw_text("ABILITIES", self.font_m, self.COLOR_TEXT, (self.GAME_WIDTH + 20, 290))
        
        # Rewind Cooldown
        rewind_color = (0, 255, 0) if self.rewind_cooldown == 0 else (255, 100, 0)
        draw_text(f"Rewind [Shift]: {'READY' if self.rewind_cooldown == 0 else self.rewind_cooldown}", self.font_s, rewind_color, (self.GAME_WIDTH + 25, 320), shadow=False)
        
        # Shuffle Cooldown
        possible_matches = not self._find_possible_matches()
        shuffle_ready = self.shuffle_cooldown == 0 and possible_matches
        shuffle_color = (0, 255, 0) if shuffle_ready else (255, 100, 0)
        shuffle_status = "READY" if shuffle_ready else ("COOLDOWN" if self.shuffle_cooldown > 0 else "MOVES AVAIL")
        draw_text(f"Shuffle [Space]: {shuffle_status}", self.font_s, shuffle_color, (self.GAME_WIDTH + 25, 340), shadow=False)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block is not run during tests, so we can unset the dummy driver for local play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # Use arrow keys for swapping, space for shuffle, left shift for rewind
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Archive Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        move = 0 # no-op
        shuffle = 0
        rewind = 0

        # Process a single key press event to avoid sticky keys
        action_taken_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not action_taken_this_frame:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                if event.key == pygame.K_UP:
                    move = 1
                    action_taken_this_frame = True
                elif event.key == pygame.K_DOWN:
                    move = 2
                    action_taken_this_frame = True
                elif event.key == pygame.K_LEFT:
                    move = 3
                    action_taken_this_frame = True
                elif event.key == pygame.K_RIGHT:
                    move = 4
                    action_taken_this_frame = True
                
                if event.key == pygame.K_SPACE:
                    shuffle = 1
                    action_taken_this_frame = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    rewind = 1
                    action_taken_this_frame = True

        # If no key was pressed, we can still use held keys for abilities
        if not action_taken_this_frame:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                shuffle = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                rewind = 1
        
        action = [move, shuffle, rewind]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Integrity: {info['archive_integrity']:.1f}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(30) # Limit to 30 FPS for playability
        
    env.close()