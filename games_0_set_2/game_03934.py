
# Generated: 2025-08-28T00:53:57.509364
# Source Brief: brief_03934.md
# Brief Index: 3934

        
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
        "Controls: Arrow keys to move the cursor. Press space to select a block and start a chain reaction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect adjacent same-colored blocks to clear them from the grid. Create large chains to score more points. Clear 75% of the board before the timer runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    # Set to True for smooth animations and a real-time timer.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 10, 12
    BLOCK_SIZE = 30
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_COLS * BLOCK_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_ROWS * BLOCK_SIZE) // 2 + 20
    FPS = 30
    TIME_LIMIT = 60.0  # seconds
    WIN_PERCENTAGE = 0.75
    MAX_STEPS = FPS * int(TIME_LIMIT) + 100 # Time limit in frames plus a buffer

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Verdana", 48, bold=True)
        
        # Internal state variables
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.timer = 0.0
        self.game_over = False
        self.win_state = False
        self.initial_block_count = 0
        self.reward_buffer = 0.0
        
        self.prev_space_state = 0
        self.cursor_move_cooldown = 0
        
        self.animations = []
        self.particles = []
        self.is_board_settling = False
        
        # Initialize state variables for the first time
        # self.reset() # This is called by the environment runner
        
        # Run self-check
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.timer = self.TIME_LIMIT
        self.reward_buffer = 0.0
        
        self.grid = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1, size=(self.GRID_ROWS, self.GRID_COLS))
        self.initial_block_count = self.GRID_ROWS * self.GRID_COLS
        
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.prev_space_state = 0
        self.cursor_move_cooldown = 0
        
        self.animations.clear()
        self.particles.clear()
        self.is_board_settling = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        self.reward_buffer = 0.0

        if not self.game_over:
            self.timer = max(0, self.timer - 1.0 / self.FPS)
            self._handle_input(action)
        
        self._update_animations()
        self._update_particles()

        terminated = self._check_termination()
        
        # The reward buffer is populated by game events during the step
        reward = self.reward_buffer
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1], action[2]
        
        # --- Cursor Movement ---
        self.cursor_move_cooldown = max(0, self.cursor_move_cooldown - 1)
        if self.cursor_move_cooldown == 0 and not self.is_board_settling:
            moved = False
            if movement == 1: # Up
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                moved = True
            elif movement == 2: # Down
                self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
                moved = True
            elif movement == 3: # Left
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
                moved = True
            elif movement == 4: # Right
                self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
                moved = True
            
            if moved:
                self.cursor_move_cooldown = 4 # 4-frame cooldown

        # --- Block Selection ---
        space_pressed = space_held and not self.prev_space_state
        if space_pressed and not self.is_board_settling:
            self._trigger_chain_reaction()

        self.prev_space_state = space_held

    def _trigger_chain_reaction(self):
        r, c = self.cursor_pos
        color_idx = self.grid[r, c]
        if color_idx == 0: # Empty block
            return

        # Find all connected blocks of the same color (must be 2 or more)
        q = [(r, c)]
        visited = set(q)
        group = []
        while q:
            curr_r, curr_c = q.pop(0)
            group.append((curr_r, curr_c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in visited:
                    if self.grid[nr, nc] == color_idx:
                        visited.add((nr, nc))
                        q.append((nr, nc))

        if len(group) >= 2:
            # Sound effect placeholder: # sfx_match.play()
            self.is_board_settling = True
            
            # Add rewards
            self.reward_buffer += len(group)
            self.score += len(group) * len(group) # Exponential scoring for larger chains
            if len(group) > 5:
                self.reward_buffer += 5

            # Animate and remove blocks
            for block_r, block_c in group:
                self.grid[block_r, block_c] = 0
                self.animations.append({
                    "type": "disappear", "pos": (block_r, block_c),
                    "color": self.BLOCK_COLORS[color_idx - 1], "progress": 0.0
                })
                # Spawn particles
                for _ in range(5):
                    self.particles.append(self._create_particle(block_r, block_c, self.BLOCK_COLORS[color_idx - 1]))
            
            self._apply_gravity()

    def _apply_gravity(self):
        new_grid = np.zeros_like(self.grid)
        max_fall_duration = 0
        for c in range(self.GRID_COLS):
            write_r = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    color_idx = self.grid[r, c]
                    new_grid[write_r, c] = color_idx
                    if write_r != r:
                        duration = 8 + (write_r - r) # Longer fall = longer animation
                        self.animations.append({
                            "type": "fall", "start_pos": (r, c), "end_pos": (write_r, c),
                            "color": self.BLOCK_COLORS[color_idx - 1], "progress": 0.0,
                            "duration": duration
                        })
                        max_fall_duration = max(max_fall_duration, duration)
                    write_r -= 1
        
        self.grid = new_grid
        if max_fall_duration > 0:
            # Sound effect placeholder: # sfx_fall.play()
            pass

    def _check_termination(self):
        if self.game_over:
            return True

        cleared_blocks = self.initial_block_count - np.count_nonzero(self.grid)
        cleared_ratio = cleared_blocks / self.initial_block_count
        
        win_condition = cleared_ratio >= self.WIN_PERCENTAGE
        lose_condition = self.timer <= 0
        
        if win_condition:
            self.game_over = True
            self.win_state = True
            self.reward_buffer += 100
            # Sound effect placeholder: # sfx_win.play()
        elif lose_condition:
            self.game_over = True
            self.win_state = False
            self.reward_buffer -= 100
            # Sound effect placeholder: # sfx_lose.play()
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_state = False # Reaching step limit is a loss
            self.reward_buffer -= 100

        return self.game_over

    def _update_animations(self):
        if not self.animations:
            self.is_board_settling = False
            return

        active_animations = []
        for anim in self.animations:
            if anim["type"] == "disappear":
                anim["progress"] += 1.0 / 5.0 # 5-frame animation
            elif anim["type"] == "fall":
                anim["progress"] += 1.0 / anim["duration"]
            
            if anim["progress"] < 1.0:
                active_animations.append(anim)
        self.animations = active_animations

    def _create_particle(self, r, c, color):
        x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(2, 5)
        return {
            "pos": [x, y], "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
            "color": color, "lifespan": self.np_random.integers(15, 25), "radius": self.np_random.uniform(3, 6)
        }

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid_lines()
        
        # Render static blocks first
        animating_blocks = set()
        for anim in self.animations:
            if anim["type"] == "fall":
                animating_blocks.add(anim["end_pos"])
            elif anim["type"] == "disappear":
                animating_blocks.add(anim["pos"])

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] != 0 and (r, c) not in animating_blocks:
                    self._render_block(r, c, self.BLOCK_COLORS[self.grid[r, c] - 1])
        
        # Render animated elements
        self._render_animations()
        self._render_particles()
        
        # Render cursor on top of blocks
        if not self.is_board_settling:
            self._render_cursor()
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_lines(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_COLS * self.BLOCK_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.BLOCK_SIZE))

    def _render_block(self, r, c, color, size_mod=0, alpha=255):
        x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
        y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
        size = self.BLOCK_SIZE - 2
        
        if size_mod != 0:
            size = max(0, size * size_mod)
            x += (self.BLOCK_SIZE - size) / 2
            y += (self.BLOCK_SIZE - size) / 2

        rect = pygame.Rect(int(x + 1), int(y + 1), int(size), int(size))
        
        # Use a surface for alpha blending
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        
        # Draw a slightly darker version for depth
        dark_color = tuple(max(0, val - 40) for val in color)
        pygame.draw.rect(shape_surf, dark_color, (0, 0, rect.width, rect.height), border_radius=4)
        
        # Draw the main bright part, inset
        pygame.draw.rect(shape_surf, color, (0, 0, rect.width, rect.height-2), border_radius=4)
        
        shape_surf.set_alpha(alpha)
        self.screen.blit(shape_surf, rect.topleft)

    def _render_animations(self):
        for anim in self.animations:
            if anim["type"] == "disappear":
                size_mod = 1.0 - anim["progress"]
                self._render_block(anim["pos"][0], anim["pos"][1], anim["color"], size_mod=size_mod)
            elif anim["type"] == "fall":
                start_y = self.GRID_Y_OFFSET + anim["start_pos"][0] * self.BLOCK_SIZE
                end_y = self.GRID_Y_OFFSET + anim["end_pos"][0] * self.BLOCK_SIZE
                interp_y = start_y + (end_y - start_y) * anim["progress"]
                
                x = self.GRID_X_OFFSET + anim["end_pos"][1] * self.BLOCK_SIZE
                size = self.BLOCK_SIZE - 2
                
                rect = pygame.Rect(int(x + 1), int(interp_y + 1), int(size), int(size))
                
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                dark_color = tuple(max(0, val - 40) for val in anim["color"])
                pygame.draw.rect(shape_surf, dark_color, (0, 0, rect.width, rect.height), border_radius=4)
                pygame.draw.rect(shape_surf, anim["color"], (0, 0, rect.width, rect.height-2), border_radius=4)
                self.screen.blit(shape_surf, rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 25.0))
            radius = int(p["radius"] * (p["lifespan"] / 25.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, (*p["color"], alpha))
                pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, (*p["color"], alpha))

    def _render_cursor(self):
        r, c = self.cursor_pos
        x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
        y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
        
        # Pulsing effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
        alpha = 50 + int(pulse * 50)
        
        cursor_surf = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, (255, 255, 255, alpha), (0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE), border_radius=5)
        pygame.draw.rect(cursor_surf, (255, 255, 255, alpha*2), (2, 2, self.BLOCK_SIZE-4, self.BLOCK_SIZE-4), width=2, border_radius=4)
        self.screen.blit(cursor_surf, (x, y))

    def _render_text(self, text, pos, font, color, shadow_color=None, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        if shadow_color:
            shadow_surf = font.render(text, True, shadow_color)
            shadow_rect = shadow_surf.get_rect()
            shadow_rect.topleft = (text_rect.left + 2, text_rect.top + 2)
            if center:
                shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surf, shadow_rect)
            
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", (15, 15), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Timer
        timer_str = f"TIME: {self.timer:.1f}"
        timer_color = self.COLOR_TEXT if self.timer > 10 else self.COLOR_LOSE
        self._render_text(timer_str, (self.SCREEN_WIDTH - 15, 15), self.font_ui, timer_color, self.COLOR_TEXT_SHADOW, center=(False, False))
        text_width = self.font_ui.size(timer_str)[0]
        self.screen.blit(self.font_ui.render(timer_str, True, timer_color), (self.SCREEN_WIDTH - 15 - text_width, 15))


        # Cleared Percentage
        cleared_blocks = self.initial_block_count - np.count_nonzero(self.grid)
        cleared_ratio = cleared_blocks / self.initial_block_count if self.initial_block_count > 0 else 0
        progress_str = f"Cleared: {cleared_ratio:.0%}"
        self._render_text(progress_str, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state else "TIME'S UP!"
            color = self.COLOR_WIN if self.win_state else self.COLOR_LOSE
            self._render_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_big, color, self.COLOR_TEXT_SHADOW, center=True)

    def _get_info(self):
        cleared_blocks = self.initial_block_count - np.count_nonzero(self.grid)
        cleared_ratio = cleared_blocks / self.initial_block_count if self.initial_block_count > 0 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cleared_ratio": cleared_ratio,
            "is_board_settling": self.is_board_settling
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # A reset is needed to initialize the state for observation
        self.reset()
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set up a window to display the game
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = env.action_space.sample() # Start with a no-op
    action.fill(0)

    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = np.array([movement, space, shift])
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

    env.close()