import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press space to select a block and match it with adjacent same-colored blocks."
    )
    game_description = (
        "A fast-paced puzzle game. Match 3 or more adjacent colored blocks to score points. Reach the target score before the timer runs out!"
    )

    # Frame advance behavior
    auto_advance = True

    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 10
    GRID_ROWS = 8
    BLOCK_SIZE = 40
    SCORE_TARGET = 1000
    MAX_STEPS = 10000  # Generous step limit
    GAME_DURATION_SECONDS = 60
    FPS = 30

    # UI
    UI_HEIGHT = 60
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_COLS * BLOCK_SIZE) // 2
    GRID_Y_OFFSET = UI_HEIGHT + (SCREEN_HEIGHT - UI_HEIGHT - GRID_ROWS * BLOCK_SIZE) // 2
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 40, 50)
    COLOR_UI_BG = (40, 50, 60)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    BLOCK_COLORS = {
        1: ((227, 52, 47), (252, 128, 125)), # Red
        2: ((64, 191, 64), (139, 245, 139)), # Green
        3: ((59, 130, 246), (147, 197, 253)), # Blue
        4: ((234, 179, 8), (253, 224, 71)),   # Yellow
        5: ((139, 92, 246), (196, 181, 253))  # Purple
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.timer = 0
        self.game_over = False
        self.game_phase = 'IDLE' # IDLE, CLEARING, FALLING
        self.last_space_held = False
        self.move_cooldown = 0
        self.animation_timer = 0
        self.clearing_blocks = []
        self.falling_blocks = []
        self.particles = []
        self.color_probs = None
        self.last_reward = 0

        # This is called here to set up the np_random generator
        # before it's used in _initialize_grid
        self.reset()
        # self.validate_implementation() # Optional: can be removed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS * self.FPS
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.game_phase = 'IDLE'
        self.last_space_held = False
        self.move_cooldown = 0
        self.clearing_blocks = []
        self.falling_blocks = []
        self.particles = []
        self.last_reward = 0
        
        # Difficulty scaling
        self.color_probs = [1.0 / len(self.BLOCK_COLORS)] * len(self.BLOCK_COLORS)

        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # After termination, reset should be called, but we provide a consistent return
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        
        reward = self._update_game_state(action)
        self.last_reward = reward

        terminated = self._check_termination()
        truncated = False # This env doesn't truncate based on steps
        if self.steps >= self.MAX_STEPS:
            terminated = True # Treat step limit as termination

        if terminated:
            if self.score >= self.SCORE_TARGET:
                reward += 100
            elif self.timer <= 0:
                reward += -100 # Timer ran out penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, action):
        reward = 0
        
        # Update animations and particles regardless of phase
        self._update_particles()
        if self.move_cooldown > 0: self.move_cooldown -= 1

        if self.game_phase == 'IDLE':
            reward += self._handle_input(action)
        elif self.game_phase == 'CLEARING':
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self._apply_gravity()
                self.game_phase = 'FALLING'
                self.animation_timer = 15 # Fall duration
        elif self.game_phase == 'FALLING':
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self._spawn_new_blocks()
                if self._check_and_resolve_new_matches():
                    # Chain reaction!
                    self.game_phase = 'CLEARING'
                    self.animation_timer = 10 # Clear duration
                    # For simplicity, chain rewards are not implemented in this version
                else:
                    self.game_phase = 'IDLE'

        # Update difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self._adjust_difficulty()

        return reward

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Movement ---
        if self.move_cooldown == 0:
            moved = False
            if movement == 1 and self.cursor_pos[1] > 0: # Up
                self.cursor_pos[1] -= 1
                moved = True
            elif movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1: # Down
                self.cursor_pos[1] += 1
                moved = True
            elif movement == 3 and self.cursor_pos[0] > 0: # Left
                self.cursor_pos[0] -= 1
                moved = True
            elif movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1: # Right
                self.cursor_pos[0] += 1
                moved = True
            
            if moved:
                self.move_cooldown = 4 # 4 frame delay between moves

        # --- Handle Selection ---
        reward = 0
        if space_held and not self.last_space_held:
            matches = self._find_matches(self.cursor_pos[0], self.cursor_pos[1])
            if len(matches) >= 3:
                self.clearing_blocks = [{'pos': pos, 'color': self.grid[pos[1]][pos[0]]} for pos in matches]
                
                # Calculate score and reward
                num_cleared = len(matches)
                self.score += num_cleared * 10
                reward += num_cleared
                if num_cleared >= 4:
                    self.score += 20
                    reward += 10
                if num_cleared >= 5:
                    self.score += 30
                    reward += 20
                
                for block in self.clearing_blocks:
                    x, y = block['pos']
                    self.grid[y][x] = 0 # Mark as empty
                    self._spawn_particles(x, y, block['color'])

                self.game_phase = 'CLEARING'
                self.animation_timer = 10 # Clear animation duration
            else:
                pass

        self.last_space_held = space_held
        return reward
    
    def _find_matches(self, start_x, start_y):
        if self.grid[start_y][start_x] == 0:
            return []
        
        target_color = self.grid[start_y][start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        matches = [(start_x, start_y)]

        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    if (nx, ny) not in visited and self.grid[ny][nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
                        matches.append((nx, ny))
        return matches

    def _apply_gravity(self):
        self.falling_blocks = []
        for x in range(self.GRID_COLS):
            empty_count = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[y][x] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    color = self.grid[y][x]
                    self.grid[y + empty_count][x] = color
                    self.grid[y][x] = 0
                    self.falling_blocks.append({
                        'start_pos': (x, y),
                        'end_pos': (x, y + empty_count),
                        'color': color
                    })

    def _spawn_new_blocks(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[y][x] == 0:
                    self.grid[y][x] = self._get_random_color()

    def _check_and_resolve_new_matches(self):
        all_matches = set()
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                if (x, y) not in all_matches and self.grid[y][x] != 0:
                    matches = self._find_matches(x, y)
                    if len(matches) >= 3:
                        for pos in matches:
                            all_matches.add(pos)
        
        if all_matches:
            # Use a consistent color for particles from a chain reaction
            first_match_pos = list(all_matches)[0]
            particle_color = self.grid[first_match_pos[1]][first_match_pos[0]]
            
            self.clearing_blocks = [{'pos': pos, 'color': self.grid[pos[1]][pos[0]]} for pos in all_matches]
            for x, y in all_matches:
                self.grid[y][x] = 0
                self._spawn_particles(x, y, particle_color)
            return True
        return False

    def _initialize_grid(self):
        self.grid = [[0] * self.GRID_COLS for _ in range(self.GRID_ROWS)]
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                self.grid[y][x] = self._get_random_color()
        
        # Ensure no initial matches
        while True:
            if not self._check_and_resolve_new_matches():
                break
            self._apply_gravity()
            self._spawn_new_blocks()

    def _get_random_color(self):
        # FIX: Removed the '+ 1' which was causing an out-of-bounds key access.
        # self.BLOCK_COLORS.keys() returns [1, 2, 3, 4, 5], so choice will
        # correctly select one of these valid keys.
        return self.np_random.choice(list(self.BLOCK_COLORS.keys()), p=self.color_probs)

    def _adjust_difficulty(self):
        # Slightly reduce probability of the most common color
        # to make matching harder over time.
        if len(self.color_probs) > 1:
            total_prob = sum(self.color_probs)
            if total_prob > 0.0:
                reduction = 0.001
                # Find index of max probability
                max_prob_idx = self.color_probs.index(max(self.color_probs))
                if self.color_probs[max_prob_idx] > 0.05: # Don't reduce below a threshold
                    self.color_probs[max_prob_idx] -= reduction
                    # Distribute the reduction among other colors
                    for i in range(len(self.color_probs)):
                        if i != max_prob_idx:
                            self.color_probs[i] += reduction / (len(self.color_probs) - 1)


    def _check_termination(self):
        return self.timer <= 0 or self.score >= self.SCORE_TARGET or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer // self.FPS}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_ui()
        self._render_game()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's coordinate system is (width, height), but numpy/gym expects (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_BG, (0, self.UI_HEIGHT-1), (self.SCREEN_WIDTH, self.UI_HEIGHT-1), 2)

        score_text = f"SCORE: {self.score}"
        self._render_text(score_text, self.font_small, self.COLOR_TEXT, (20, 15))

        time_left = max(0, self.timer // self.FPS)
        timer_text = f"TIME: {time_left}"
        timer_color = self.COLOR_TEXT if time_left > 10 else self.BLOCK_COLORS[1][0]
        self._render_text(timer_text, self.font_small, timer_color, (self.SCREEN_WIDTH - 150, 15))

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(
            self.screen, self.COLOR_GRID_BG,
            (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_COLS * self.BLOCK_SIZE, self.GRID_ROWS * self.BLOCK_SIZE)
        )
        
        # Draw blocks
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                color_idx = self.grid[y][x]
                if color_idx > 0:
                    self._draw_block(x, y, color_idx, 1.0)
        
        # Draw falling blocks
        if self.game_phase == 'FALLING':
            progress = 1.0 - (self.animation_timer / 15.0)
            for block in self.falling_blocks:
                start_x, start_y = block['start_pos']
                end_x, end_y = block['end_pos']
                interp_y = start_y + (end_y - start_y) * progress
                self._draw_block(end_x, interp_y, block['color'], 1.0)

        # Draw clearing blocks
        if self.game_phase == 'CLEARING':
            scale = self.animation_timer / 10.0
            for block in self.clearing_blocks:
                self._draw_block(block['pos'][0], block['pos'][1], block['color'], scale)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

        # Draw cursor if idle
        if self.game_phase == 'IDLE':
            self._render_cursor()

    def _draw_block(self, grid_x, grid_y, color_idx, scale):
        main_color, highlight_color = self.BLOCK_COLORS[color_idx]
        
        scaled_size = self.BLOCK_SIZE * scale
        offset = (self.BLOCK_SIZE - scaled_size) / 2
        
        px = self.GRID_X_OFFSET + grid_x * self.BLOCK_SIZE + offset
        py = self.GRID_Y_OFFSET + grid_y * self.BLOCK_SIZE + offset

        rect = pygame.Rect(int(px), int(py), int(scaled_size), int(scaled_size))
        pygame.draw.rect(self.screen, main_color, rect, border_radius=4)
        
        inner_offset = scaled_size * 0.1
        inner_rect = pygame.Rect(
            int(px + inner_offset), int(py + inner_offset),
            int(scaled_size - inner_offset * 2), int(scaled_size - inner_offset * 2)
        )
        pygame.draw.rect(self.screen, highlight_color, inner_rect, border_radius=3)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px = self.GRID_X_OFFSET + cx * self.BLOCK_SIZE
        py = self.GRID_Y_OFFSET + cy * self.BLOCK_SIZE
        
        # Pulsating effect
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        thickness = 2 + int(pulse * 3)
        
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, width=thickness, border_radius=5)

    def _spawn_particles(self, grid_x, grid_y, color_idx):
        px = self.GRID_X_OFFSET + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.GRID_Y_OFFSET + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_idx][0]

        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': random.uniform(2, 5),
                'life': random.randint(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] *= 0.97
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # To run with a display window, comment out the os.environ line at the top
    try:
        del os.environ['SDL_VIDEODRIVER']
    except KeyError:
        pass
    
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Block Matcher")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
    }

    while running:
        # --- Action mapping for human play ---
        movement = 0 # No-op
        if keys_held[pygame.K_UP]: movement = 1
        elif keys_held[pygame.K_DOWN]: movement = 2
        elif keys_held[pygame.K_LEFT]: movement = 3
        elif keys_held[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys_held[pygame.K_SPACE] else 0
        shift = 1 if keys_held[pygame.K_LSHIFT] else 0
        
        action = [movement, space, shift]

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            # Pause briefly on reset
            pygame.time.wait(1000)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()