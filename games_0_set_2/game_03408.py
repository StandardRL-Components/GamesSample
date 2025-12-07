
# Generated: 2025-08-27T23:16:08.685987
# Source Brief: brief_03408.md
# Brief Index: 3408

        
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
        "Controls: Arrow keys to move the cursor. Press Space to select a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect 20 gems before the timer runs out. Avoid the traps, which will cost you precious time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS

    INITIAL_TIMER = 30.0  # seconds
    TRAP_PENALTY = 5.0  # seconds
    GEMS_TO_WIN = 20
    NUM_GEMS = 25
    NUM_TRAPS = 15
    MAX_STEPS = 1800 # 60 seconds at 30 FPS

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (50, 50, 70)
    COLOR_GEM = (0, 255, 128)
    COLOR_TRAP = (50, 50, 50)
    COLOR_TRAP_X = (255, 50, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GEM_PARTICLE = (0, 255, 128)
    COLOR_TRAP_PARTICLE = (255, 80, 80)
    
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables to prevent errors before first reset
        self.grid = []
        self.cursor_pos = [0, 0]
        self.timer = 0.0
        self.gems_collected = 0
        self.game_over = False
        self.win_state = False
        self.steps = 0
        self.particles = []
        self.last_space_held = False
        
        # Initialize state
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.gems_collected = 0
        self.game_over = False
        self.win_state = False
        self.timer = self.INITIAL_TIMER
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.particles = []
        self.last_space_held = False

        # Generate grid
        self.grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        all_coords = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_coords)

        # Place gems (value 1)
        for i in range(self.NUM_GEMS):
            x, y = all_coords[i]
            self.grid[y][x] = 1
        
        # Place traps (value 2)
        for i in range(self.NUM_GEMS, self.NUM_GEMS + self.NUM_TRAPS):
            x, y = all_coords[i]
            self.grid[y][x] = 2
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_action, shift_action = action
        space_held = space_action == 1
        reward = 0
        terminated = False

        if not self.game_over:
            # --- Update Game Logic ---
            self.timer -= 1.0 / 30.0 # Assuming 30 FPS
            self._handle_movement(movement)
            reward = self._handle_selection(space_held)
        
        self._update_particles()
        self.steps += 1

        # --- Check Termination ---
        if not self.game_over:
            if self.gems_collected >= self.GEMS_TO_WIN:
                terminated = True
                self.game_over = True
                self.win_state = True
                reward += 100
                # Sound: GAME_WIN
            elif self.timer <= 0:
                self.timer = 0
                terminated = True
                self.game_over = True
                self.win_state = False
                reward -= 100
                # Sound: GAME_LOSE
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True # End game if it runs too long

        self.last_space_held = space_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

    def _handle_selection(self, space_held):
        reward = 0
        # Trigger on rising edge of space press
        if space_held and not self.last_space_held:
            cx, cy = self.cursor_pos
            cell_content = self.grid[cy][cx]

            if cell_content == 1: # Gem
                self.gems_collected += 1
                self.grid[cy][cx] = 0 # Consume gem
                reward = 1.0
                self._create_particles(cx, cy, self.COLOR_GEM_PARTICLE, 25)
                # Sound: GEM_COLLECT
            elif cell_content == 2: # Trap
                self.timer = max(0, self.timer - self.TRAP_PENALTY)
                self.grid[cy][cx] = 0 # Consume trap
                reward = -1.0
                self._create_particles(cx, cy, self.COLOR_TRAP_PARTICLE, 15)
                # Sound: TRAP_SPRING
            else: # Empty
                # Sound: EMPTY_SELECT
                pass
        return reward

    def _create_particles(self, grid_x, grid_y, color, count):
        center_x = (grid_x + 0.5) * self.CELL_WIDTH
        center_y = (grid_y + 0.5) * self.CELL_HEIGHT
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30) # frames
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw gems and traps
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                cell_content = self.grid[y][x]
                center_x = int((x + 0.5) * self.CELL_WIDTH)
                center_y = int((y + 0.5) * self.CELL_HEIGHT)
                
                if cell_content == 1: # Gem
                    radius = int(self.CELL_WIDTH * 0.3)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GEM)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_GEM)
                elif cell_content == 2: # Trap
                    rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
                    pygame.draw.rect(self.screen, self.COLOR_TRAP, rect.inflate(-8, -8))
                    pygame.draw.line(self.screen, self.COLOR_TRAP_X, rect.topleft, rect.bottomright, 2)
                    pygame.draw.line(self.screen, self.COLOR_TRAP_X, rect.topright, rect.bottomleft, 2)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

        # Draw cursor
        cursor_alpha = 100 + 50 * math.sin(self.steps * 0.2)
        cursor_color = self.COLOR_CURSOR + (cursor_alpha,)
        cx, cy = self.cursor_pos
        rect = pygame.Rect(cx * self.CELL_WIDTH, cy * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, cursor_color, temp_surf.get_rect(), border_radius=4)
        pygame.draw.rect(temp_surf, self.COLOR_CURSOR, temp_surf.get_rect(), 2, border_radius=4)
        self.screen.blit(temp_surf, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"GEMS: {self.gems_collected} / {self.GEMS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        timer_ratio = max(0, self.timer / self.INITIAL_TIMER)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        # Color transition for timer bar
        if timer_ratio > 0.5:
            bar_color = (0, 200, 0)
        elif timer_ratio > 0.2:
            bar_color = (255, 200, 0)
        else:
            bar_color = (200, 0, 0)

        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Game over text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.win_state:
                text = self.font_game_over.render("YOU WIN!", True, self.COLOR_GEM)
            else:
                text = self.font_game_over.render("TIME'S UP!", True, self.COLOR_TRAP_X)
            
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.gems_collected,
            "steps": self.steps,
            "timer": self.timer,
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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map keyboard keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        # --- Create action from keyboard input ---
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, shift_held]

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation to the screen ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        obs_transposed = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_transposed)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Handle Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30) # Control the loop speed

    env.close()