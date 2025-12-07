
# Generated: 2025-08-28T00:20:21.356480
# Source Brief: brief_03756.md
# Brief Index: 3756

        
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
        "Controls: Use arrow keys to move the cursor. Hold SPACE and press an arrow key to swap blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A colorful match-3 puzzle game. Swap adjacent blocks to create lines of 3 or more. "
        "Clear the board before you run out of moves or time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 8
    BLOCK_SIZE = 40
    GRID_X = (640 - GRID_WIDTH * BLOCK_SIZE) // 2
    GRID_Y = (400 - GRID_HEIGHT * BLOCK_SIZE) // 2 + 30
    NUM_COLORS = 6
    ANIMATION_SPEED = 0.2
    MAX_STEPS = 3600  # 120 seconds * 30 fps
    STARTING_MOVES = 50
    STARTING_TIME = 120 # seconds

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID_BG = (40, 45, 55)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_PARTICLE = (255, 255, 220)
    BLOCK_COLORS = [
        (220, 50, 50),    # Red
        (50, 220, 50),    # Green
        (50, 100, 220),   # Blue
        (220, 220, 50),   # Yellow
        (150, 50, 220),   # Purple
        (220, 120, 50),   # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.cursor_pos = [0, 0]
        self.game_phase = "IDLE"
        self.animations = []
        self.particles = []
        self.reward_buffer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed the standard random for board generation
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.STARTING_MOVES
        self.time_left = self.STARTING_TIME
        self.game_phase = "IDLE"
        self.animations = []
        self.particles = []
        self.reward_buffer = 0

        self._generate_solvable_board()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Check if any blocks need to fall initially
        if self._apply_gravity():
            self._fill_top_rows()
            self._start_fall_animation()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.time_left = max(0, self.time_left - 1/30.0)
        self.reward_buffer = -0.01 # Small penalty for taking time

        self._handle_input(action)
        self._update_animations()
        self._update_particles()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_phase != "IDLE":
            return

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if space_held and movement in [1, 2, 3, 4]:
            # Attempt a swap
            cx, cy = self.cursor_pos
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            nx, ny = cx + dx, cy + dy

            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                self.moves_left -= 1
                self._start_swap_animation((cx, cy), (nx, ny))
        
        elif not space_held and movement in [1, 2, 3, 4]:
            # Move cursor
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos[0] = max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx))
            self.cursor_pos[1] = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))

    def _calculate_reward(self):
        reward = self.reward_buffer
        self.reward_buffer = 0
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
            
        if np.sum(self.grid) == 0: # Board cleared
            self.reward_buffer += 100
            self.game_over = True
        elif self.moves_left <= 0:
            self.reward_buffer -= 100
            self.game_over = True
        elif self.time_left <= 0:
            self.reward_buffer -= 100
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_background()
        self._render_blocks()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left, "time_left": self.time_left}

    # --- Game Logic ---
    def _generate_solvable_board(self):
        while True:
            self._generate_board()
            if self._find_possible_moves():
                break
    
    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        # Ensure no initial matches
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _find_possible_moves(self):
        moves = []
        temp_grid = self.grid.copy()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Try swapping right
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r,c], temp_grid[r,c+1] = temp_grid[r,c+1], temp_grid[r,c]
                    if self._find_all_matches_on_grid(temp_grid): moves.append(((r,c), (r,c+1)))
                    temp_grid[r,c], temp_grid[r,c+1] = temp_grid[r,c+1], temp_grid[r,c] # Swap back
                # Try swapping down
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r,c], temp_grid[r+1,c] = temp_grid[r+1,c], temp_grid[r,c]
                    if self._find_all_matches_on_grid(temp_grid): moves.append(((r,c), (r+1,c)))
                    temp_grid[r,c], temp_grid[r+1,c] = temp_grid[r+1,c], temp_grid[r,c] # Swap back
        return moves

    def _find_all_matches_on_grid(self, grid):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if grid[r, c] == 0: continue
                if c < self.GRID_WIDTH - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    return True
                if r < self.GRID_HEIGHT - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    return True
        return False

    def _apply_gravity(self):
        grid_changed = False
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                        grid_changed = True
                    empty_row -= 1
        return grid_changed

    def _fill_top_rows(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
    
    def _reshuffle_board(self):
        # Flatten, shuffle, and reshape, then ensure solvability
        self._generate_solvable_board() 
        self._start_fall_animation() # Animate the reshuffle

    # --- Animations ---
    def _start_swap_animation(self, pos1, pos2):
        self.game_phase = "SWAPPING"
        self.animations.append({
            "type": "SWAP",
            "pos1": pos1, "pos2": pos2,
            "progress": 0.0
        })

    def _start_fall_animation(self):
        fall_anims = {}
        temp_grid = self.grid.copy()
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if temp_grid[r, c] != 0:
                    if r != empty_row:
                        fall_anims[(c,r)] = empty_row - r
                        temp_grid[empty_row,c] = temp_grid[r,c]
                        temp_grid[r,c] = 0
                    empty_row -= 1
        
        if fall_anims:
            self.game_phase = "FALLING"
            self.animations.append({
                "type": "FALL",
                "blocks": fall_anims,
                "progress": 0.0
            })
        else: # If nothing to fall, check for cascades or end turn
            self._on_animation_finish({"type": "FALL", "progress": 1.0})


    def _update_animations(self):
        if not self.animations:
            return

        anim = self.animations[0]
        anim["progress"] = min(1.0, anim["progress"] + self.ANIMATION_SPEED)

        if anim["progress"] >= 1.0:
            self.animations.pop(0)
            self._on_animation_finish(anim)

    def _on_animation_finish(self, anim):
        if anim["type"] == "SWAP":
            c1, r1 = anim["pos1"]
            c2, r2 = anim["pos2"]
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            
            matches = self._find_all_matches()
            if matches:
                self.game_phase = "CLEARING"
                num_cleared = len(matches)
                self.reward_buffer += num_cleared
                self.score += num_cleared * 10
                if num_cleared > 3:
                    self.reward_buffer += 5
                    self.score += (num_cleared - 3) * 20
                
                # Sfx: Match found
                for r, c in matches:
                    self.grid[r, c] = 0
                    self._spawn_particles((c, r))
                
                self._start_fall_animation()
            else: # Invalid swap, swap back
                # Sfx: Invalid move
                self.moves_left += 1 # Refund move
                self.game_phase = "REVERSING" # Special state to prevent player input
                self._start_swap_animation(anim["pos2"], anim["pos1"])

        elif anim["type"] == "FALL":
            self._apply_gravity()
            self._fill_top_rows()
            
            matches = self._find_all_matches()
            if matches:
                # Sfx: Cascade match
                self.game_phase = "CLEARING"
                num_cleared = len(matches)
                self.reward_buffer += num_cleared * 1.5 # Cascade bonus
                self.score += num_cleared * 15
                if num_cleared > 3:
                    self.reward_buffer += 10
                    self.score += (num_cleared - 3) * 30

                for r, c in matches:
                    self.grid[r, c] = 0
                    self._spawn_particles((c, r))

                self._start_fall_animation()
            else:
                self.game_phase = "IDLE"
                if not self._find_possible_moves() and np.sum(self.grid) > 0:
                    # Sfx: Reshuffle
                    self._reshuffle_board()
        
        elif anim["type"] == "REVERSING":
             c1, r1 = anim["pos1"]
             c2, r2 = anim["pos2"]
             self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
             self.game_phase = "IDLE"

    # --- Particles ---
    def _spawn_particles(self, pos):
        cx, cy = self._get_pixel_pos(pos[0], pos[1], center=True)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "x": cx, "y": cy,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.uniform(15, 30),
                "size": random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1 # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    # --- Rendering ---
    def _get_pixel_pos(self, c, r, center=False):
        x = self.GRID_X + c * self.BLOCK_SIZE
        y = self.GRID_Y + r * self.BLOCK_SIZE
        if center:
            x += self.BLOCK_SIZE // 2
            y += self.BLOCK_SIZE // 2
        return x, y

    def _render_grid_background(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                x, y = self._get_pixel_pos(c, r)
                pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE))

    def _render_blocks(self):
        rendered_blocks = set()

        # Render animated blocks first
        for anim in self.animations:
            if anim["type"] in ["SWAP", "REVERSING"]:
                p = anim["progress"]
                c1, r1 = anim["pos1"]
                c2, r2 = anim["pos2"]
                x1, y1 = self._get_pixel_pos(c1, r1)
                x2, y2 = self._get_pixel_pos(c2, r2)
                
                curr_x1 = x1 + (x2 - x1) * p
                curr_y1 = y1 + (y2 - y1) * p
                curr_x2 = x2 + (x1 - x2) * p
                curr_y2 = y2 + (y1 - y2) * p
                
                self._draw_block(curr_x1, curr_y1, self.grid[r1, c1])
                self._draw_block(curr_x2, curr_y2, self.grid[r2, c2])
                
                rendered_blocks.add((r1, c1))
                rendered_blocks.add((r2, c2))
            
            elif anim["type"] == "FALL":
                p = anim["progress"]
                for (c, r), dist in anim["blocks"].items():
                    x, y_start = self._get_pixel_pos(c, r)
                    y_end = y_start + dist * self.BLOCK_SIZE
                    curr_y = y_start + (y_end - y_start) * p
                    self._draw_block(x, curr_y, self.grid[r,c])
                    rendered_blocks.add((r,c))

        # Render static blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in rendered_blocks and self.grid[r, c] != 0:
                    x, y = self._get_pixel_pos(c, r)
                    self._draw_block(x, y, self.grid[r, c])

    def _draw_block(self, x, y, color_idx):
        if color_idx == 0: return
        color = self.BLOCK_COLORS[color_idx - 1]
        shadow_color = tuple(max(0, val - 40) for val in color)
        highlight_color = tuple(min(255, val + 40) for val in color)
        
        rect = pygame.Rect(int(x) + 4, int(y) + 4, self.BLOCK_SIZE - 8, self.BLOCK_SIZE - 8)
        pygame.draw.rect(self.screen, shadow_color, rect.move(0, 2), border_radius=8)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, highlight_color, rect.inflate(-6, -6), border_radius=6)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        x, y = self._get_pixel_pos(cx, cy)
        rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * 8.5)))
            color = self.COLOR_PARTICLE + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(p["size"]), color)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.screen.get_width() // 2 - moves_text.get_width() // 2, 5))

        time_str = f"{int(self.time_left // 60):02}:{int(self.time_left % 60):02}"
        time_text = self.font_large.render(f"Time: {time_str}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.screen.get_width() - time_text.get_width() - 10, 5))
        
    def _render_game_over(self):
        s = pygame.Surface((640, 400), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        
        win = np.sum(self.grid) == 0
        text = "BOARD CLEARED!" if win else "GAME OVER"
        color = (100, 255, 100) if win else (255, 100, 100)
        
        game_over_text = self.font_large.render(text, True, color)
        text_rect = game_over_text.get_rect(center=(320, 200))
        s.blit(game_over_text, text_rect)
        self.screen.blit(s, (0, 0))

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Match-3 Gym Environment")
    clock = pygame.time.Clock()

    movement_action = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = False
    shift_held = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        movement_action = 0
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        action = [movement_action, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)


    pygame.quit()