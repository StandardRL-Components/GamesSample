
# Generated: 2025-08-28T04:06:30.350439
# Source Brief: brief_02205.md
# Brief Index: 2205

        
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
        "Controls: Arrow keys to move. Press space to interact with objects on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a haunted grid in Whispering Walls. Solve puzzles to find the key and escape before the timer runs out. Be wary of traps!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_STEPS = 1000
    INITIAL_TIME = 180

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_GRID_BASE = (40, 40, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 50)
    COLOR_PUZZLE = (0, 100, 255)
    COLOR_TRAP = (200, 0, 50)
    COLOR_CLUE = (0, 200, 100)
    COLOR_KEY = (255, 215, 0)
    COLOR_EXIT = (150, 50, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CHECKMARK = (150, 255, 150)
    COLOR_TIMER_WARN = (255, 100, 100)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Grid rendering setup
        self.grid_area_size = 360
        self.cell_size = self.grid_area_size // self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_size) // 2

        # Initialize state variables (will be properly set in reset)
        self.player_pos = None
        self.puzzles = []
        self.traps = []
        self.clues = []
        self.exit_pos = None
        self.key_pos = None
        self.key_possessed = False
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.INITIAL_TIME
        self.key_possessed = False
        self.key_pos = None
        self.last_space_held = False

        # Procedurally generate the level
        all_cells = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_cells)

        self.player_pos = all_cells.pop()
        self.exit_pos = all_cells.pop()
        
        self.puzzles = [{'pos': all_cells.pop(), 'solved': False} for _ in range(5)]
        self.traps = [{'pos': all_cells.pop()} for _ in range(3)]
        self.clues = [{'pos': all_cells.pop()} for _ in range(2)]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.1  # Small penalty for each step to encourage speed
        self.steps += 1
        self.time_remaining -= 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is not used in this game

        # --- Handle Movement ---
        px, py = self.player_pos
        if movement == 1:  # Up
            py -= 1
        elif movement == 2:  # Down
            py += 1
        elif movement == 3:  # Left
            px -= 1
        elif movement == 4:  # Right
            px += 1
        
        # Clamp position to grid boundaries
        px = max(0, min(self.GRID_SIZE - 1, px))
        py = max(0, min(self.GRID_SIZE - 1, py))
        self.player_pos = (px, py)

        # --- Handle Interaction (on space press) ---
        is_space_press = space_held and not self.last_space_held
        if is_space_press:
            # Interact with Puzzles
            for puzzle in self.puzzles:
                if self.player_pos == puzzle['pos'] and not puzzle['solved']:
                    puzzle['solved'] = True
                    reward += 10
                    self.score += 10
                    # sfx: puzzle_solve.wav
                    break
            
            # Interact with Traps
            for trap in self.traps:
                if self.player_pos == trap['pos']:
                    self.time_remaining -= 30
                    reward -= 5
                    self.score -= 5
                    # sfx: trap_trigger.wav
                    break
            
            # Interact with Key
            if self.key_pos and self.player_pos == self.key_pos:
                self.key_possessed = True
                self.key_pos = None # Key is picked up
                # sfx: key_pickup.wav
        
        self.last_space_held = space_held

        # --- Game Logic Updates ---
        # Check if all puzzles are solved to spawn the key
        if not self.key_pos and not self.key_possessed and all(p['solved'] for p in self.puzzles):
            # Find a safe, empty spot to spawn the key
            occupied_cells = (
                [p['pos'] for p in self.puzzles] +
                [t['pos'] for t in self.traps] +
                [c['pos'] for c in self.clues] +
                [self.player_pos, self.exit_pos]
            )
            all_cells = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            empty_cells = [cell for cell in all_cells if cell not in occupied_cells]
            if empty_cells:
                self.key_pos = self.np_random.choice(empty_cells)
            else: # Fallback if no empty cells
                self.key_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
            # sfx: key_spawn.wav

        # --- Termination Check ---
        terminated = False
        if self.time_remaining <= 0:
            terminated = True
            reward -= 100
            self.score -= 100
            # sfx: game_over_timeout.wav
        
        if self.player_pos == self.exit_pos and self.key_possessed:
            terminated = True
            reward += 100
            self.score += 100
            # sfx: win_level.wav
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
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
            "key_possessed": self.key_possessed,
            "puzzles_solved": sum(p['solved'] for p in self.puzzles),
        }

    def _world_to_screen(self, x, y):
        """Converts grid coordinates to screen pixel coordinates."""
        screen_x = self.grid_offset_x + x * self.cell_size
        screen_y = self.grid_offset_y + y * self.cell_size
        return screen_x, screen_y

    def _render_game(self):
        # --- Draw Grid (with flicker) ---
        flicker_intensity = self.np_random.integers(0, 15)
        grid_color = (
            self.COLOR_GRID_BASE[0] + flicker_intensity,
            self.COLOR_GRID_BASE[1] + flicker_intensity,
            self.COLOR_GRID_BASE[2] + flicker_intensity,
        )
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, grid_color, (start_x, self.grid_offset_y), (start_x, self.grid_offset_y + self.grid_area_size))
            # Horizontal lines
            start_y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, grid_color, (self.grid_offset_x, start_y), (self.grid_offset_x + self.grid_area_size, start_y))

        # --- Draw Entities ---
        entity_size = int(self.cell_size * 0.7)
        offset = (self.cell_size - entity_size) // 2
        
        # Exit
        ex, ey = self._world_to_screen(*self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex + offset, ey + offset, entity_size, entity_size))

        # Puzzles
        for puzzle in self.puzzles:
            px, py = self._world_to_screen(*puzzle['pos'])
            pygame.draw.rect(self.screen, self.COLOR_PUZZLE, (px + offset, py + offset, entity_size, entity_size))
            if puzzle['solved']:
                # Draw checkmark
                p1 = (px + self.cell_size * 0.25, py + self.cell_size * 0.5)
                p2 = (px + self.cell_size * 0.45, py + self.cell_size * 0.7)
                p3 = (px + self.cell_size * 0.75, py + self.cell_size * 0.3)
                pygame.draw.lines(self.screen, self.COLOR_CHECKMARK, False, [p1, p2, p3], 3)

        # Traps
        for trap in self.traps:
            tx, ty = self._world_to_screen(*trap['pos'])
            pygame.draw.rect(self.screen, self.COLOR_TRAP, (tx + offset, ty + offset, entity_size, entity_size))
        
        # Clues
        for clue in self.clues:
            cx, cy = self._world_to_screen(*clue['pos'])
            pygame.draw.rect(self.screen, self.COLOR_CLUE, (cx + offset, cy + offset, entity_size, entity_size))
        
        # Key
        if self.key_pos:
            kx, ky = self._world_to_screen(*self.key_pos)
            pygame.draw.rect(self.screen, self.COLOR_KEY, (kx + offset, ky + offset, entity_size, entity_size))
        
        # Player
        plx, ply = self._world_to_screen(*self.player_pos)
        
        # Glow effect
        glow_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.cell_size // 2, self.cell_size // 2), self.cell_size // 2)
        self.screen.blit(glow_surf, (plx, ply))
        
        # Player square
        player_size = int(self.cell_size * 0.8)
        player_offset = (self.cell_size - player_size) // 2
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (plx + player_offset, ply + player_offset, player_size, player_size))


    def _render_ui(self):
        # --- Timer ---
        timer_color = self.COLOR_TEXT
        if self.time_remaining < 60 and self.steps % 2 == 0: # Flash when low
            timer_color = self.COLOR_TIMER_WARN
        
        timer_text = self.font_large.render(f"{self.time_remaining}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # --- Score ---
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topleft=(20, 10))
        self.screen.blit(score_text, score_rect)

        # --- Key Indicator ---
        if self.key_possessed:
            key_indicator_text = self.font_small.render("KEY", True, self.COLOR_KEY)
            key_indicator_rect = key_indicator_text.get_rect(topleft=(20, 35))
            self.screen.blit(key_indicator_text, key_indicator_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[pygame]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Whispering Walls")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      WHISPERING WALLS")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Not used

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
            space_held = 1
        
        # Assemble the action from keyboard input
        action = [movement, space_held, shift_held]
        
        # Only step if an action is taken (since auto_advance is False)
        # For a better human play experience, we step every frame.
        # An RL agent would decide when to step.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(10) # Control human play speed

    env.close()