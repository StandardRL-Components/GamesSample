import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric match-3 puzzle game Gymnasium environment.

    The player controls a cursor on an 8x8 grid of gems. The goal is to
    swap adjacent gems to create lines of 3 or more of the same color.
    Successful matches award points, and the game ends when a target score
    is reached or the player runs out of moves.

    This environment prioritizes visual quality with smooth animations for
    swaps, matches, and gem falling, made possible by a state machine
    that runs on an auto-advancing clock.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to swap the "
        "selected gem with the one in the direction you last moved. "
        "Press Shift to reshuffle the board (costs 1 move)."
    )

    game_description = (
        "A vibrant, isometric match-3 puzzle game. Swap gems to create "
        "matches of three or more, triggering chain reactions to maximize "
        "your score. Reach the target score before you run out of moves!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    BOARD_SIZE = 8
    NUM_GEM_TYPES = 5
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    
    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    
    WIN_SCORE = 1000
    MAX_MOVES = 50
    ANIMATION_SPEED = 0.2 # Higher is faster

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.board_origin = (
            self.SCREEN_WIDTH // 2,
            self.SCREEN_HEIGHT // 2 - (self.BOARD_SIZE * self.TILE_HEIGHT) // 2 + 30
        )
        
        # self.reset() is called by the wrapper or test harness
        # We need to initialize some attributes here to avoid attribute errors before the first reset
        self.board = []
        self.cursor_pos = [0, 0]
        self.game_state = "IDLE"
        self.animations = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.display_score = 0
        self.moves_left = 0
        self.game_over = False
        self.prev_action = [0, 0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.display_score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        
        self.game_state = "IDLE" # IDLE, SWAP, MATCH, FALL
        self.board = self._create_initial_board()
        
        self.cursor_pos = [self.BOARD_SIZE // 2, self.BOARD_SIZE // 2]
        self.last_move_dir = (1, 0) # Default to right
        
        self.prev_action = [0, 0, 0]
        self.animations = []
        self.particles = []
        
        self.turn_reward = 0
        self.steps = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self._update_animations()

        if self.game_state == "IDLE":
            reward = self.turn_reward
            self.turn_reward = 0
            self._handle_input(action)
        
        self.prev_action = action
        
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and self.prev_action[1] == 0
        shift_press = shift_held and self.prev_action[2] == 0

        # --- Cursor Movement ---
        if movement != 0:
            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
            dx, dy = dirs[movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.BOARD_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.BOARD_SIZE
            self.last_move_dir = (dx, dy)
        
        # --- Reshuffle Action ---
        if shift_press and self.moves_left > 0:
            self.moves_left -= 1
            self._reshuffle_board()
            self.turn_reward = -1 # Cost for reshuffling
            # sfx: board_shuffle
            
        # --- Swap Action ---
        if space_press and self.moves_left > 0:
            self._attempt_swap()
            
    def _attempt_swap(self):
        self.moves_left -= 1
        p1 = self.cursor_pos
        p2 = [p1[0] + self.last_move_dir[0], p1[1] + self.last_move_dir[1]]

        if not (0 <= p2[0] < self.BOARD_SIZE and 0 <= p2[1] < self.BOARD_SIZE):
            # Tried to swap off-board, do nothing, don't consume move
            self.moves_left += 1
            return

        # Temporarily swap to check for matches
        self._swap_gems(p1, p2)
        matches = self._find_all_matches()
        
        is_valid_swap = len(matches) > 0
        
        # Animate the swap
        self.animations.append({
            "type": "SWAP", "p1": p1, "p2": p2, "progress": 0.0, "is_valid": is_valid_swap
        })
        self.game_state = "SWAP"
        # sfx: gem_swap
        
        if not is_valid_swap:
            self.turn_reward += -0.1
            self._swap_gems(p1, p2) # Swap back data if invalid

    def _update_animations(self):
        if not self.animations and self.game_state != "IDLE":
             self._on_animation_complete()
             return

        # Process active animations
        for anim in self.animations[:]:
            anim["progress"] += self.ANIMATION_SPEED
            if anim["progress"] >= 1.0:
                self.animations.remove(anim)
        
        # Update particles
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _on_animation_complete(self):
        if self.game_state == "SWAP":
            matches = self._find_all_matches()
            if matches:
                self._process_matches(matches)
                self.game_state = "FALL"
            else:
                self.game_state = "IDLE" # Invalid swap animation finished
        
        elif self.game_state == "FALL":
            self._drop_and_refill_gems()
            new_matches = self._find_all_matches()
            if new_matches:
                self._process_matches(new_matches) # Chain reaction
                self.game_state = "FALL"
            else:
                self.game_state = "IDLE"
                if not self._find_possible_moves():
                    self._reshuffle_board()
                    # sfx: board_reshuffle_auto

    def _process_matches(self, matches):
        # sfx: match_success
        gems_to_remove = set()
        # `matches` is a flat list of unique positions, e.g., [[r1, c1], [r2, c2], ...]
        for pos in matches:
            gems_to_remove.add(tuple(pos))
        
        # Calculate reward and score
        num_cleared = len(gems_to_remove)
        self.turn_reward += num_cleared
        self.score += num_cleared * 10
        
        if num_cleared == 4:
            self.turn_reward += 5
            self.score += 50
        elif num_cleared >= 5:
            self.turn_reward += 10
            self.score += 100

        # Create particles and mark gems for removal
        for r, c in gems_to_remove:
            if self.board[r][c] != -1: # Ensure we don't double-process
                self._create_particles(r, c)
                self.board[r][c] = -1 # Mark as empty

    def _drop_and_refill_gems(self):
        for c in range(self.BOARD_SIZE):
            empty_row = self.BOARD_SIZE - 1
            for r in range(self.BOARD_SIZE - 1, -1, -1):
                if self.board[r][c] != -1:
                    if r != empty_row:
                        self.board[empty_row][c] = self.board[r][c]
                        self.board[r][c] = -1
                    empty_row -= 1
            
            # Refill from top
            for r in range(empty_row, -1, -1):
                self.board[r][c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
        # sfx: gems_fall
    
    def _create_particles(self, r, c):
        screen_x, screen_y = self._iso_to_screen(r, c)
        gem_color = self.GEM_COLORS[self.board[r][c]]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "x": screen_x, "y": screen_y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(15, 30),
                "color": gem_color,
                "radius": self.np_random.uniform(2, 5)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            self.turn_reward += 100
            self.game_over = True
            # sfx: win_game
        elif self.moves_left <= 0 and self.game_state == "IDLE":
            self.turn_reward += -10
            self.game_over = True
            # sfx: lose_game
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_board()
        self._render_gems()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_board(self):
        for r in range(self.BOARD_SIZE + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.BOARD_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
                
        for c in range(self.BOARD_SIZE + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.BOARD_SIZE, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

    def _render_gems(self):
        active_swap = next((anim for anim in self.animations if anim["type"] == "SWAP"), None)
        
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                gem_type = self.board[r][c]
                if gem_type == -1:
                    continue

                pos = (r,c)
                screen_pos = self._iso_to_screen(r, c)

                # Handle swap animation
                if active_swap:
                    p1, p2, progress = active_swap["p1"], active_swap["p2"], active_swap["progress"]
                    if tuple(pos) == tuple(p1):
                        screen_pos = self._interpolate_pos(p1, p2, progress)
                    elif tuple(pos) == tuple(p2):
                        screen_pos = self._interpolate_pos(p2, p1, progress)

                self._draw_gem(screen_pos, self.GEM_COLORS[gem_type])

    def _draw_gem(self, center_pos, color):
        x, y = center_pos
        points = [
            (x, y - self.TILE_HEIGHT),
            (x + self.TILE_WIDTH, y),
            (x, y + self.TILE_HEIGHT),
            (x - self.TILE_WIDTH, y)
        ]
        
        # Use gfxdraw for antialiasing
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Add a subtle highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
        highlight_points = [
            (x, y - self.TILE_HEIGHT),
            (x + self.TILE_WIDTH * 0.4, y - self.TILE_HEIGHT * 0.4),
            (x, y - self.TILE_HEIGHT * 0.2),
            (x - self.TILE_WIDTH * 0.4, y - self.TILE_HEIGHT * 0.4)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, highlight_color)

    def _render_cursor(self):
        if self.game_state == "IDLE":
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            alpha = int(150 + pulse * 105)
            
            x, y = self._iso_to_screen(*self.cursor_pos)
            width = int(self.TILE_WIDTH * (1.1 + pulse * 0.1))
            height = int(self.TILE_HEIGHT * (1.1 + pulse * 0.1))
            
            points = [(x, y - height), (x + width, y), (x, y + height), (x - width, y)]
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.lines(s, self.COLOR_CURSOR + (alpha,), True, points, 3)
            self.screen.blit(s, (0,0))


    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30.0))
            # Create a temporary surface for the particle to handle alpha
            radius = int(p["radius"])
            if radius <= 0: continue
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, radius, radius, radius, p["color"] + (alpha,))
            self.screen.blit(s, (int(p["x"]) - radius, int(p["y"]) - radius))
            
    def _render_ui(self):
        # Animate score display
        self.display_score += (self.score - self.display_score) * 0.1
        
        # Score Text
        score_text = self.font_main.render(f"Score: {round(self.display_score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        # Moves Left
        moves_text = self.font_main.render("Moves", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 15, 15))
        for i in range(self.MAX_MOVES):
            x = self.SCREEN_WIDTH - 15 - i * 11
            y = 55
            color = self.COLOR_UI_TEXT if i < self.moves_left else self.COLOR_GRID
            pygame.draw.circle(self.screen, color, (x, y), 4)
            
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "You Win!" if self.score >= self.WIN_SCORE else "Game Over"
            msg_surf = self.font_main.render(msg, True, self.COLOR_CURSOR)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "moves_left": self.moves_left, "steps": self.steps}

    # --- Helper and Logic Functions ---

    def _iso_to_screen(self, r, c):
        x = self.board_origin[0] + (c - r) * self.TILE_WIDTH / 2
        y = self.board_origin[1] + (c + r) * self.TILE_HEIGHT / 2
        return int(x), int(y)
    
    def _interpolate_pos(self, p1_grid, p2_grid, progress):
        p1_screen = self._iso_to_screen(*p1_grid)
        p2_screen = self._iso_to_screen(*p2_grid)
        return (
            p1_screen[0] + (p2_screen[0] - p1_screen[0]) * progress,
            p1_screen[1] + (p2_screen[1] - p1_screen[1]) * progress
        )
        
    def _swap_gems(self, p1, p2):
        r1, c1 = p1
        r2, c2 = p2
        self.board[r1][c1], self.board[r2][c2] = \
            self.board[r2][c2], self.board[r1][c1]

    def _find_all_matches(self):
        matches = []
        # Use a set to avoid adding the same gem multiple times
        matched_gems = set()

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                gem = self.board[r][c]
                if gem == -1: continue
                
                # Horizontal check
                if c < self.BOARD_SIZE - 2 and self.board[r][c+1] == gem and self.board[r][c+2] == gem:
                    h_match = [(r, c), (r, c+1), (r, c+2)]
                    i = c + 3
                    while i < self.BOARD_SIZE and self.board[r][i] == gem:
                        h_match.append((r,i))
                        i += 1
                    for pos in h_match: matched_gems.add(pos)

                # Vertical check
                if r < self.BOARD_SIZE - 2 and self.board[r+1][c] == gem and self.board[r+2][c] == gem:
                    v_match = [(r, c), (r+1, c), (r+2, c)]
                    i = r + 3
                    while i < self.BOARD_SIZE and self.board[i][c] == gem:
                        v_match.append((i,c))
                        i += 1
                    for pos in v_match: matched_gems.add(pos)
        
        return [list(pos) for pos in matched_gems]


    def _find_possible_moves(self):
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                # Check swap right
                if c < self.BOARD_SIZE - 1:
                    self._swap_gems((r,c), (r,c+1))
                    if self._find_all_matches():
                        self._swap_gems((r,c), (r,c+1))
                        return True
                    self._swap_gems((r,c), (r,c+1))
                # Check swap down
                if r < self.BOARD_SIZE - 1:
                    self._swap_gems((r,c), (r+1,c))
                    if self._find_all_matches():
                        self._swap_gems((r,c), (r+1,c))
                        return True
                    self._swap_gems((r,c), (r+1,c))
        return False
        
    def _create_initial_board(self):
        while True:
            board = [[self.np_random.integers(0, self.NUM_GEM_TYPES) for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
            self.board = board
            if not self._find_all_matches() and self._find_possible_moves():
                return board
                
    def _reshuffle_board(self):
        flat_board = [gem for row in self.board for gem in row if gem != -1]
        self.np_random.shuffle(flat_board)
        
        new_board = []
        idx = 0
        for r in range(self.BOARD_SIZE):
            row = []
            for c in range(self.BOARD_SIZE):
                row.append(flat_board[idx])
                idx += 1
            new_board.append(row)
        
        self.board = new_board
        if not self._find_possible_moves() or self._find_all_matches():
             self._reshuffle_board() # Recurse if shuffle is bad

if __name__ == '__main__':
    # To run and play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Match-3")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    prev_keys = pygame.key.get_pressed()
    
    while running:
        # --- Human Controls ---
        # This logic ensures actions are triggered on key press, not hold
        current_keys = pygame.key.get_pressed()
        
        movement = 0
        if current_keys[pygame.K_UP] and not prev_keys[pygame.K_UP]: movement = 1
        elif current_keys[pygame.K_DOWN] and not prev_keys[pygame.K_DOWN]: movement = 2
        elif current_keys[pygame.K_LEFT] and not prev_keys[pygame.K_LEFT]: movement = 3
        elif current_keys[pygame.K_RIGHT] and not prev_keys[pygame.K_RIGHT]: movement = 4
        
        space_held = current_keys[pygame.K_SPACE]
        shift_held = current_keys[pygame.K_LSHIFT] or current_keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        prev_keys = current_keys

        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it.
        # Pygame uses (width, height), but our obs is (height, width, 3).
        # We need to transpose it back for display.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

    pygame.quit()