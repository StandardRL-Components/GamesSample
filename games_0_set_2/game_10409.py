import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:21:32.068341
# Source Brief: brief_00409.md
# Brief Index: 409
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
        "Clear the puzzle grid by dropping blocks to create horizontal matches of five or more. "
        "Match blocks by both color and shape, and press space to swap the shape of the falling block."
    )
    user_guide = (
        "Controls: ←→ to move the block, ↓ to speed up its fall, and ↑ to drop it instantly. "
        "Press space to swap the block's shape between a square and a circle."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 12, 10
    BLOCK_SIZE = 32
    GAME_AREA_WIDTH = PLAYFIELD_WIDTH * BLOCK_SIZE
    GAME_AREA_HEIGHT = PLAYFIELD_HEIGHT * BLOCK_SIZE
    GAME_AREA_X_OFFSET = (SCREEN_WIDTH - GAME_AREA_WIDTH) // 2
    GAME_AREA_Y_OFFSET = SCREEN_HEIGHT - GAME_AREA_HEIGHT - 20

    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_TIMER_GREEN = (40, 200, 120)
    COLOR_TIMER_YELLOW = (220, 200, 80)
    COLOR_TIMER_RED = (220, 80, 80)

    BLOCK_COLORS = [
        (255, 90, 90),   # Red
        (90, 255, 90),   # Green
        (90, 150, 255),  # Blue
        (255, 255, 90),  # Yellow
    ]

    FPS = 60
    LOGIC_FPS = 10 # Game logic updates per second
    MAX_TIME_SECONDS = 120
    MAX_STEPS = MAX_TIME_SECONDS * LOGIC_FPS

    FALL_SPEED_NORMAL = 2.0 / LOGIC_FPS # Grid units per second
    FALL_SPEED_FAST = 10.0 / LOGIC_FPS
    SHAPE_SHIFT_INTERVAL = 2 * LOGIC_FPS # 2 seconds

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

        self.grid = None
        self.falling_block = None
        self.block_queue = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_progress = 0.0
        self.prev_space_held = False
        self.total_blocks_to_clear = 0

        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # this is for dev, not for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[None for _ in range(self.PLAYFIELD_WIDTH)] for _ in range(self.PLAYFIELD_HEIGHT)]
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_progress = 0.0
        self.prev_space_held = False

        # Generate a puzzle to solve
        self.total_blocks_to_clear = 40
        self._populate_initial_grid(self.total_blocks_to_clear)

        self._spawn_next_block()

        return self._get_observation(), self._get_info()
    
    def _populate_initial_grid(self, num_blocks):
        for _ in range(num_blocks):
            col = self.np_random.integers(0, self.PLAYFIELD_WIDTH)
            row = self.np_random.integers(self.PLAYFIELD_HEIGHT // 2, self.PLAYFIELD_HEIGHT)
            if self.grid[row][col] is None:
                self.grid[row][col] = self._create_random_block(col, row)

    def _create_random_block(self, grid_x, grid_y):
        color = self.np_random.choice(len(self.BLOCK_COLORS))
        shape = self.np_random.integers(0, 2) # 0: square, 1: circle
        return Block(grid_x, grid_y, color, shape)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        reward = 0
        self.steps += 1

        # --- Action Handling ---
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if self.falling_block:
            # Action: Swap shape
            if space_pressed:
                self.falling_block.shape = 1 - self.falling_block.shape
                # Sound: Shape swap

            # Action: Movement
            moved_sideways = False
            if movement == 3: # Left
                if self._is_valid_position(self.falling_block.grid_x - 1, self.falling_block.grid_y):
                    self.falling_block.grid_x -= 1
                    reward += 0.01
                    moved_sideways = True
                else:
                    reward -= 0.01
            elif movement == 4: # Right
                if self._is_valid_position(self.falling_block.grid_x + 1, self.falling_block.grid_y):
                    self.falling_block.grid_x += 1
                    reward += 0.01
                    moved_sideways = True
                else:
                    reward -= 0.01
            
            if moved_sideways:
                self.falling_block.last_move_time = self.steps

            # Action: Hard Drop
            if movement == 1:
                while not self._check_collision(self.falling_block.grid_x, self.falling_block.grid_y + 1):
                    self.falling_block.grid_y += 1
                self.fall_progress = 1.0 # Force landing
                # Sound: Hard drop

        # --- Game Logic Update ---
        if self.falling_block:
            # Automatic shape shifting
            self.falling_block.shape_shift_timer -= 1
            if self.falling_block.shape_shift_timer <= 0:
                self.falling_block.shape = 1 - self.falling_block.shape
                self.falling_block.shape_shift_timer = self.SHAPE_SHIFT_INTERVAL
                # Sound: Auto-shift

            # Gravity
            fall_speed = self.FALL_SPEED_FAST if movement == 2 else self.FALL_SPEED_NORMAL
            self.fall_progress += fall_speed
            
            if self.fall_progress >= 1.0:
                if not self._check_collision(self.falling_block.grid_x, self.falling_block.grid_y + 1):
                    self.falling_block.grid_y += 1
                    self.fall_progress = 0.0
                else:
                    # Land the block
                    self._land_block()
                    clear_reward, cleared_count = self._resolve_board()
                    reward += clear_reward
                    self.score += cleared_count
                    self.total_blocks_to_clear -= cleared_count

                    if self.total_blocks_to_clear <= 0:
                        self.game_over = True
                        reward += 100 # Win bonus
                    else:
                        self._spawn_next_block()
        
        # --- Termination Check ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward -= 100 # Timeout penalty
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _land_block(self):
        if self.falling_block is None: return
        bx, by = self.falling_block.grid_x, self.falling_block.grid_y
        if 0 <= by < self.PLAYFIELD_HEIGHT:
            self.grid[by][bx] = self.falling_block
            self.falling_block.render_x = self.GAME_AREA_X_OFFSET + bx * self.BLOCK_SIZE
            self.falling_block.render_y = self.GAME_AREA_Y_OFFSET + by * self.BLOCK_SIZE
            self.falling_block = None
            # Sound: Block land
        else: # Topped out
            self.game_over = True
            # Sound: Game over
    
    def _resolve_board(self):
        total_reward = 0
        total_cleared_count = 0
        chain_multiplier = 1

        while True:
            cleared_groups = self._find_clears()
            if not cleared_groups:
                break
            
            # Sound: Line clear
            cleared_this_pass = 0
            for group in cleared_groups:
                line_length = len(group)
                total_reward += (line_length - 4) * chain_multiplier
                cleared_this_pass += line_length

                for block in group:
                    self._create_clear_particles(block)
                    self.grid[block.grid_y][block.grid_x] = None
            
            total_cleared_count += cleared_this_pass
            
            self._apply_grid_gravity()
            chain_multiplier += 1
        
        return total_reward, total_cleared_count

    def _find_clears(self):
        cleared_groups = []
        marked_for_clear = set()

        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                block = self.grid[r][c]
                if block and block not in marked_for_clear:
                    # Horizontal check
                    h_group = [block]
                    # Check right
                    for i in range(c + 1, self.PLAYFIELD_WIDTH):
                        neighbor = self.grid[r][i]
                        if neighbor and neighbor.is_match(block):
                            h_group.append(neighbor)
                        else:
                            break
                    if len(h_group) >= 5:
                        cleared_groups.append(h_group)
                        for b in h_group:
                            marked_for_clear.add(b)
        return cleared_groups

    def _apply_grid_gravity(self):
        for c in range(self.PLAYFIELD_WIDTH):
            empty_row = self.PLAYFIELD_HEIGHT - 1
            for r in range(self.PLAYFIELD_HEIGHT - 1, -1, -1):
                if self.grid[r][c] is not None:
                    block = self.grid[r][c]
                    if r != empty_row:
                        self.grid[empty_row][c] = block
                        self.grid[r][c] = None
                        block.grid_y = empty_row
                    empty_row -= 1

    def _create_clear_particles(self, block):
        # Sound: Particle burst
        cx = self.GAME_AREA_X_OFFSET + block.grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        cy = self.GAME_AREA_Y_OFFSET + block.grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        color = self.BLOCK_COLORS[block.color_idx]
        for _ in range(15):
            self.particles.append(Particle(cx, cy, color, self.np_random))

    def _spawn_next_block(self):
        start_x = self.PLAYFIELD_WIDTH // 2
        start_y = 0
        if self._check_collision(start_x, start_y):
            self.game_over = True
            return

        self.falling_block = self._create_random_block(start_x, start_y)
        self.fall_progress = 0.0

    def _is_valid_position(self, grid_x, grid_y):
        return 0 <= grid_x < self.PLAYFIELD_WIDTH and not self._check_collision(grid_x, grid_y)

    def _check_collision(self, grid_x, grid_y):
        if not (0 <= grid_x < self.PLAYFIELD_WIDTH and 0 <= grid_y < self.PLAYFIELD_HEIGHT):
            return True
        return self.grid[grid_y][grid_x] is not None

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
            "time_left": self.MAX_TIME_SECONDS - (self.steps / self.LOGIC_FPS),
            "blocks_to_clear": self.total_blocks_to_clear,
        }

    def _render_game(self):
        # Draw playfield border and background
        play_area_rect = pygame.Rect(self.GAME_AREA_X_OFFSET, self.GAME_AREA_Y_OFFSET, self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, play_area_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, play_area_rect, 2, 5)

        # Draw grid blocks
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                if self.grid[r][c]:
                    self.grid[r][c].update_render_pos()
                    self.grid[r][c].draw(self.screen, self.BLOCK_COLORS, self.BLOCK_SIZE, self.GAME_AREA_X_OFFSET, self.GAME_AREA_Y_OFFSET)

        # Draw falling block and ghost
        if self.falling_block:
            # Ghost piece
            ghost_y = self.falling_block.grid_y
            while not self._check_collision(self.falling_block.grid_x, ghost_y + 1):
                ghost_y += 1
            
            ghost_block = Block(self.falling_block.grid_x, ghost_y, self.falling_block.color_idx, self.falling_block.shape)
            ghost_block.draw(self.screen, self.BLOCK_COLORS, self.BLOCK_SIZE, self.GAME_AREA_X_OFFSET, self.GAME_AREA_Y_OFFSET, is_ghost=True)

            # Falling block
            self.falling_block.update_render_pos()
            self.falling_block.draw(self.screen, self.BLOCK_COLORS, self.BLOCK_SIZE, self.GAME_AREA_X_OFFSET, self.GAME_AREA_Y_OFFSET)

        # Update and draw particles
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles:
            p.draw(self.screen)
    
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Blocks left
        blocks_text = self.font_small.render(f"GOAL: {max(0, self.total_blocks_to_clear)} blocks", True, self.COLOR_UI_TEXT)
        self.screen.blit(blocks_text, (20, 60))

        # Timer
        time_left = self.MAX_TIME_SECONDS - (self.steps / self.LOGIC_FPS)
        time_percent = max(0, time_left / self.MAX_TIME_SECONDS)
        
        timer_color = self.COLOR_TIMER_GREEN
        if time_percent < 0.5: timer_color = self.COLOR_TIMER_YELLOW
        if time_percent < 0.2: timer_color = self.COLOR_TIMER_RED

        time_text = self.font_main.render(f"TIME: {int(time_left)}", True, timer_color)
        text_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(time_text, text_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = self.total_blocks_to_clear <= 0
            msg = "PUZZLE CLEARED!" if win_condition else "TIME UP!"
            
            end_text = self.font_main.render(msg, True, self.COLOR_TIMER_GREEN if win_condition else self.COLOR_TIMER_RED)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(end_text, end_rect)
            
            score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Block:
    def __init__(self, grid_x, grid_y, color_idx, shape):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.color_idx = color_idx
        self.shape = shape # 0 for square, 1 for circle
        
        self.render_x = GameEnv.GAME_AREA_X_OFFSET + grid_x * GameEnv.BLOCK_SIZE
        self.render_y = GameEnv.GAME_AREA_Y_OFFSET + grid_y * GameEnv.BLOCK_SIZE
        self.squash = 1.0
        self.shape_shift_timer = GameEnv.SHAPE_SHIFT_INTERVAL
        self.last_move_time = 0

    def is_match(self, other_block):
        return self.color_idx == other_block.color_idx and self.shape == other_block.shape

    def update_render_pos(self):
        target_x = GameEnv.GAME_AREA_X_OFFSET + self.grid_x * GameEnv.BLOCK_SIZE
        target_y = GameEnv.GAME_AREA_Y_OFFSET + self.grid_y * GameEnv.BLOCK_SIZE
        self.render_x = self.render_x * 0.6 + target_x * 0.4
        self.render_y = self.render_y * 0.6 + target_y * 0.4

    def draw(self, surface, colors, block_size, offset_x, offset_y, is_ghost=False):
        color = colors[self.color_idx]
        if is_ghost:
            color = (color[0] // 4, color[1] // 4, color[2] // 4)

        rect = pygame.Rect(self.render_x, self.render_y, block_size, block_size)
        center = rect.center
        
        # Glow effect
        glow_size = block_size * 1.4
        glow_color = (*color, 100) if not is_ghost else (*color, 50)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        
        if self.shape == 0: # Square
            pygame.draw.rect(glow_surf, glow_color, (0, 0, glow_size, glow_size), border_radius=int(glow_size*0.2))
        else: # Circle
            pygame.draw.circle(glow_surf, glow_color, (glow_size//2, glow_size//2), glow_size//2)
        surface.blit(glow_surf, (center[0] - glow_size//2, center[1] - glow_size//2), special_flags=pygame.BLEND_RGBA_ADD)

        # Main shape
        if self.shape == 0: # Square
            pygame.gfxdraw.box(surface, rect, (*color, 255 if not is_ghost else 100))
            pygame.gfxdraw.rectangle(surface, rect, (255, 255, 255, 50 if not is_ghost else 20))
        else: # Circle
            cx, cy = int(rect.centerx), int(rect.centery)
            radius = block_size // 2
            pygame.gfxdraw.filled_circle(surface, cx, cy, radius, (*color, 255 if not is_ghost else 100))
            pygame.gfxdraw.aacircle(surface, cx, cy, radius, (255, 255, 255, 50 if not is_ghost else 20))

class Particle:
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = np_random.integers(20, 40)
        self.color = color
        self.size = np_random.uniform(2, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # Gravity
        self.lifespan -= 1
        self.size *= 0.98
        return self.lifespan > 0 and self.size > 0.5

    def draw(self, surface):
        alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
        color = (*self.color, alpha)
        rect = pygame.Rect(self.x - self.size/2, self.y - self.size/2, self.size, self.size)
        pygame.draw.rect(surface, color, rect)

if __name__ == '__main__':
    # This block is for manual play and debugging
    # It will not be executed by the test environment
    real_render = "human" 
    try:
        env = GameEnv(render_mode=real_render)
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Shape Fall")
    except pygame.error:
        # Fallback to rgb_array if display is not available
        print("Human render mode not available, falling back to 'rgb_array'.")
        real_render = "rgb_array"
        env = GameEnv(render_mode=real_render)
        # Create a dummy screen for the main loop to run
        screen = pygame.Surface((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # none, space_up, shift_up

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Action mapping
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Blocks Left: {info['blocks_to_clear']}")

        if real_render == "human":
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(GameEnv.LOGIC_FPS) # Run manual play at logic speed for consistency

    print("Game Over!")
    env.close()