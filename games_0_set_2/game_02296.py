
# Generated: 2025-08-27T19:55:55.190719
# Source Brief: brief_02296.md
# Brief Index: 2296

        
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
        "Controls: ←→ to move cursor. Press space to shoot the next block into the selected column."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid by shooting blocks to create matches of 3 or more. Trigger chain reactions for big scores! Clear 75% of the blocks before time runs out to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 12
        self.BLOCK_SIZE = 28
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_Y_OFFSET = self.HEIGHT - (self.GRID_HEIGHT * self.BLOCK_SIZE) - 20

        self.MAX_TIME_STEPS = 60 * 30  # 60 seconds at 30fps
        self.WIN_PERCENTAGE = 0.75
        self.MOVE_COOLDOWN_FRAMES = 4
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLORS = [
            (0, 0, 0),  # 0: Empty
            (231, 76, 60),   # 1: Red
            (46, 204, 113),  # 2: Green
            (52, 152, 219),  # 3: Blue
            (241, 196, 15),  # 4: Yellow
            (155, 89, 182),  # 5: Purple
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.last_space_held = False
        self.move_cooldown = 0
        self.initial_block_count = 0
        self.blocks_cleared = 0
        self.next_block_color = 0
        self.particles = []
        
        self.reset()

        # self.validate_implementation() # Uncomment to run self-check
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[0] * self.GRID_HEIGHT for _ in range(self.GRID_WIDTH)]
        self._generate_initial_grid()
        
        self.initial_block_count = sum(1 for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if self.grid[x][y] != 0)
        if self.initial_block_count == 0: self.initial_block_count = 1 # Avoid division by zero
        
        self.blocks_cleared = 0
        self.cursor_pos = self.GRID_WIDTH // 2
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME_STEPS
        self.last_space_held = False
        self.move_cooldown = 0
        self.particles = []
        self._pick_next_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty for time passing

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update game logic
        self._handle_input(movement, space_held)
        reward += self._update_game_state()
        self._update_particles()
        
        self.steps += 1
        self.time_left -= 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self._get_cleared_percentage() >= self.WIN_PERCENTAGE:
                reward += 100 # Win bonus
            else:
                reward += -50 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_initial_grid(self):
        # Fill the bottom 60% of the grid
        for x in range(self.GRID_WIDTH):
            for y in range(int(self.GRID_HEIGHT * 0.6)):
                if self.np_random.random() > 0.1: # 90% chance of a block
                    self.grid[x][y] = self.np_random.integers(1, len(self.COLORS))

    def _pick_next_block(self):
        self.next_block_color = self.np_random.integers(1, len(self.COLORS))

    def _handle_input(self, movement, space_held):
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if self.move_cooldown == 0:
            if movement == 3:  # Left
                self.cursor_pos = max(0, self.cursor_pos - 1)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            elif movement == 4:  # Right
                self.cursor_pos = min(self.GRID_WIDTH - 1, self.cursor_pos + 1)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        # Detect rising edge for shooting
        if space_held and not self.last_space_held:
            self._shoot_block()
        
        self.last_space_held = space_held

    def _shoot_block(self):
        # sound: shoot.wav
        col = self.cursor_pos
        # Check if column is full
        if self.grid[col][-1] != 0:
            return # Shot fails, no penalty

        # Push blocks up
        for y in range(self.GRID_HEIGHT - 1, 0, -1):
            self.grid[col][y] = self.grid[col][y-1]
        
        # Insert new block at the bottom
        self.grid[col][0] = self.next_block_color
        self._pick_next_block()

    def _update_game_state(self):
        total_reward = 0
        chain = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            chain += 1
            num_cleared = len(matches)
            self.blocks_cleared += num_cleared
            
            # sound: clear.wav
            reward = num_cleared  # +1 per block
            if num_cleared > 3:
                reward += 5 # Bonus for clearing more than 3
            
            if chain > 1:
                # sound: chain.wav
                reward *= 1.5 # Chain reaction bonus
            
            total_reward += reward
            self.score += int(reward)

            for x, y in matches:
                self._create_particles(x, y, self.grid[x][y])
                self.grid[x][y] = 0
            
            self._apply_gravity()

        return total_reward

    def _find_matches(self):
        to_clear = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color = self.grid[x][y]
                if color == 0:
                    continue

                # Horizontal match
                h_match = [ (x+i, y) for i in range(3) if x + i < self.GRID_WIDTH and self.grid[x+i][y] == color ]
                if len(h_match) == 3:
                    to_clear.update(h_match)

                # Vertical match
                v_match = [ (x, y+i) for i in range(3) if y + i < self.GRID_HEIGHT and self.grid[x][y+i] == color ]
                if len(v_match) == 3:
                    to_clear.update(v_match)
        return list(to_clear)

    def _apply_gravity(self):
        # sound: fall.wav
        for x in range(self.GRID_WIDTH):
            empty_y = 0
            for y in range(self.GRID_HEIGHT):
                if self.grid[x][y] != 0:
                    if y != empty_y:
                        self.grid[x][empty_y] = self.grid[x][y]
                        self.grid[x][y] = 0
                    empty_y += 1

    def _create_particles(self, grid_x, grid_y, color_index):
        px = self.GRID_X_OFFSET + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.GRID_Y_OFFSET + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.COLORS[color_index]
        for _ in range(10): # 10 particles per block
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            size = self.np_random.integers(3, 6)
            self.particles.append([px, py, vx, vy, lifetime, color, size])

    def _update_particles(self):
        self.particles = [
            [p[0]+p[2], p[1]+p[3], p[2]*0.95, p[3]*0.95, p[4]-1, p[5], p[6]]
            for p in self.particles if p[4] > 0
        ]

    def _check_termination(self):
        win = self._get_cleared_percentage() >= self.WIN_PERCENTAGE
        timeout = self.time_left <= 0
        return win or timeout or self.steps >= 1000

    def _get_cleared_percentage(self):
        return self.blocks_cleared / self.initial_block_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, 
                                self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_index = self.grid[x][y]
                if color_index != 0:
                    color = self.COLORS[color_index]
                    darker_color = tuple(c * 0.7 for c in color)
                    block_rect = pygame.Rect(self.GRID_X_OFFSET + x * self.BLOCK_SIZE,
                                             self.GRID_Y_OFFSET + y * self.BLOCK_SIZE,
                                             self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, darker_color, block_rect)
                    inner_rect = block_rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 30.0))))
            color_with_alpha = p[5] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), int(p[6] * (p[4]/30.0)), color_with_alpha)

        # Draw cursor and next block
        cursor_x = self.GRID_X_OFFSET + self.cursor_pos * self.BLOCK_SIZE
        cursor_y = self.GRID_Y_OFFSET - self.BLOCK_SIZE - 5
        
        # Next block preview
        next_color = self.COLORS[self.next_block_color]
        darker_next = tuple(c * 0.7 for c in next_color)
        preview_rect = pygame.Rect(cursor_x, cursor_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, darker_next, preview_rect)
        inner_preview_rect = preview_rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, next_color, inner_preview_rect, border_radius=3)
        
        # Cursor highlight
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, self.GRID_Y_OFFSET, self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE), 2, border_radius=3)


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Cleared %
        cleared_pct = self._get_cleared_percentage() * 100
        cleared_text = self.font_small.render(f"Cleared: {cleared_pct:.1f}% / {self.WIN_PERCENTAGE*100}%", True, self.COLOR_TEXT)
        self.screen.blit(cleared_text, (20, 50))
        
        # Timer bar
        timer_width = 200
        timer_height = 20
        timer_x = self.WIDTH - timer_width - 20
        timer_y = 20
        
        time_pct = self.time_left / self.MAX_TIME_STEPS
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (timer_x, timer_y, timer_width, timer_height))
        fill_width = int(timer_width * time_pct)
        fill_color = self.COLORS[2] if time_pct > 0.5 else self.COLORS[4] if time_pct > 0.2 else self.COLORS[1]
        pygame.draw.rect(self.screen, fill_color, (timer_x, timer_y, fill_width, timer_height))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_text_str = "YOU WIN!" if self._get_cleared_percentage() >= self.WIN_PERCENTAGE else "TIME'S UP!"
            result_text = self.font_main.render(result_text_str, True, self.COLOR_CURSOR)
            text_rect = result_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(result_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "cleared_pct": self._get_cleared_percentage()
        }

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Action Puzzle Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action Mapping for Human Play ---
        movement = 0 # no-op
        space = 0
        shift = 0 # unused
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
            
        if keys[pygame.K_SPACE]:
            space = 1
            
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")

        clock.tick(30) # Run at 30 FPS
        
    env.close()