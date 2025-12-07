import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:34:34.497731
# Source Brief: brief_00491.md
# Brief Index: 491
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a two-robot puzzle game.
    The agent must guide two robots on a 7x7 grid to simultaneously occupy
    five gate locations to activate them.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement Direction (0:none, 1:up, 2:down, 3:left, 4:right)
    - actions[1]: Space Button (0:released, 1:held) - Modifies which robot moves
    - actions[2]: Shift Button (0:released, 1:held) - Modifies which robot moves
    Action Logic:
    - Direction + Space: Move Robot 1 only.
    - Direction + Shift: Move Robot 2 only.
    - Direction + No Modifiers or Both Modifiers: Move both robots.

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game.

    Reward Structure:
    - +100 for activating all 5 gates (terminal reward).
    - +10 for activating a single gate.
    - +0.1 for each robot moving closer to the nearest inactive gate.
    - -0.01 penalty per step to encourage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide two robots on a grid to simultaneously occupy gate locations. "
        "Control them individually or together to solve the puzzle and activate all gates."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Hold Space to move only the blue robot, or Shift to move only the red robot. "
        "Move both robots by using arrow keys alone."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 7
    NUM_GATES = 5
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)

    COLOR_ROBOT1 = (80, 150, 255)
    COLOR_ROBOT1_GLOW = (80, 150, 255, 60)
    COLOR_ROBOT2 = (255, 100, 100)
    COLOR_ROBOT2_GLOW = (255, 100, 100, 60)

    COLOR_GATE_INACTIVE = (100, 100, 110)
    COLOR_GATE_ACTIVE = (80, 220, 120)
    COLOR_GATE_ACTIVE_GLOW = (80, 220, 120, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Grid & Drawing Calculations ---
        self.grid_area_height = self.SCREEN_HEIGHT - 40
        self.cell_size = self.grid_area_height // self.GRID_SIZE
        self.grid_width = self.cell_size * self.GRID_SIZE
        self.grid_start_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_start_y = (self.SCREEN_HEIGHT - self.grid_width) // 2
        
        # --- Game State Variables (initialized in reset) ---
        self.robot1_pos = None
        self.robot2_pos = None
        self.gates = []
        self.gate_activated = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        # self.reset() # Removed to avoid calling reset before rng is seeded
        
        # --- Final Validation ---
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Generate unique gate positions
        possible_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(possible_coords)
        self.gates = [possible_coords.pop() for _ in range(self.NUM_GATES)]
        self.gate_activated = [False] * self.NUM_GATES

        # Place robots at random, non-gate positions
        self.robot1_pos = list(possible_coords.pop())
        self.robot2_pos = list(possible_coords.pop())

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for each step

        # --- Action Processing ---
        movement_id = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        move_map = {
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0),   # Right
            0: (0, 0),   # None
        }
        dx, dy = move_map[movement_id]

        # Determine which robot(s) to move based on modifiers
        # Move R1 if space is held (and not shift), or if neither is held, or if both are held.
        # This simplifies to: move R1 unless ONLY shift is held.
        move_r1 = not (shift_held and not space_held)
        # Move R2 if shift is held (and not space), or if neither is held, or if both are held.
        # This simplifies to: move R2 unless ONLY space is held.
        move_r2 = not (space_held and not shift_held)
        
        # --- Reward Calculation (Proximity) ---
        old_dist_r1 = self._get_min_dist_to_gate(self.robot1_pos)
        old_dist_r2 = self._get_min_dist_to_gate(self.robot2_pos)

        # --- Update Robot Positions ---
        if move_r1 and (dx != 0 or dy != 0):
            # Sfx: Robot 1 move
            self.robot1_pos[0] = np.clip(self.robot1_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.robot1_pos[1] = np.clip(self.robot1_pos[1] + dy, 0, self.GRID_SIZE - 1)

        if move_r2 and (dx != 0 or dy != 0):
            # Sfx: Robot 2 move
            self.robot2_pos[0] = np.clip(self.robot2_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.robot2_pos[1] = np.clip(self.robot2_pos[1] + dy, 0, self.GRID_SIZE - 1)

        new_dist_r1 = self._get_min_dist_to_gate(self.robot1_pos)
        new_dist_r2 = self._get_min_dist_to_gate(self.robot2_pos)
        
        if new_dist_r1 < old_dist_r1:
            reward += 0.1
        if new_dist_r2 < old_dist_r2:
            reward += 0.1

        # --- Gate Activation Check ---
        for i, gate_pos in enumerate(self.gates):
            if not self.gate_activated[i]:
                if self.robot1_pos == list(gate_pos) and self.robot2_pos == list(gate_pos):
                    # Sfx: Gate activation success
                    self.gate_activated[i] = True
                    reward += 10
                    self.score += 10

        # --- Termination Check ---
        terminated = False
        truncated = False
        if all(self.gate_activated):
            # Sfx: Victory fanfare
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # Sfx: Failure sound
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_min_dist_to_gate(self, robot_pos):
        inactive_gates = [self.gates[i] for i, active in enumerate(self.gate_activated) if not active]
        if not inactive_gates:
            return 0
        
        distances = [abs(robot_pos[0] - gx) + abs(robot_pos[1] - gy) for gx, gy in inactive_gates]
        return min(distances) if distances else 0

    def _grid_to_pixel(self, x, y):
        px = self.grid_start_x + x * self.cell_size + self.cell_size // 2
        py = self.grid_start_y + y * self.cell_size + self.cell_size // 2
        return int(px), int(py)

    def _draw_glow(self, surface, center, color, max_radius):
        """Draws a soft, glowing circle."""
        radius = int(max_radius)
        for i in range(radius, 0, -2):
            alpha = int(color[3] * (1 - i / radius) ** 2)
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, i, color[:3] + (alpha,))
            surface.blit(temp_surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.grid_start_x + i * self.cell_size, self.grid_start_y),
                             (self.grid_start_x + i * self.cell_size, self.grid_start_y + self.grid_width), 1)
            # Horizontal lines
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.grid_start_x, self.grid_start_y + i * self.cell_size),
                             (self.grid_start_x + self.grid_width, self.grid_start_y + i * self.cell_size), 1)

        # Draw gates
        gate_radius = int(self.cell_size * 0.35)
        for i, pos in enumerate(self.gates):
            px, py = self._grid_to_pixel(pos[0], pos[1])
            if self.gate_activated[i]:
                self._draw_glow(self.screen, (px, py), self.COLOR_GATE_ACTIVE_GLOW, int(self.cell_size * 0.5))
                pygame.gfxdraw.filled_circle(self.screen, px, py, gate_radius, self.COLOR_GATE_ACTIVE)
                pygame.gfxdraw.aacircle(self.screen, px, py, gate_radius, self.COLOR_GATE_ACTIVE)
            else:
                pygame.gfxdraw.filled_circle(self.screen, px, py, gate_radius, self.COLOR_GATE_INACTIVE)
                pygame.gfxdraw.aacircle(self.screen, px, py, gate_radius, self.COLOR_GATE_INACTIVE)

        # Draw robots
        robot_size = int(self.cell_size * 0.7)
        r1_px, r1_py = self._grid_to_pixel(self.robot1_pos[0], self.robot1_pos[1])
        r2_px, r2_py = self._grid_to_pixel(self.robot2_pos[0], self.robot2_pos[1])

        # Robot 1
        self._draw_glow(self.screen, (r1_px, r1_py), self.COLOR_ROBOT1_GLOW, int(self.cell_size*0.45))
        pygame.draw.rect(self.screen, self.COLOR_ROBOT1, (r1_px - robot_size//2, r1_py - robot_size//2, robot_size, robot_size), border_radius=4)
        
        # Robot 2
        self._draw_glow(self.screen, (r2_px, r2_py), self.COLOR_ROBOT2_GLOW, int(self.cell_size*0.45))
        pygame.draw.rect(self.screen, self.COLOR_ROBOT2, (r2_px - robot_size//2, r2_py - robot_size//2, robot_size, robot_size), border_radius=4)

    def _render_ui(self):
        # Gates Activated Text
        gates_text = f"Gates Activated: {sum(self.gate_activated)} / {self.NUM_GATES}"
        text_surface = self.font_main.render(gates_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (20, 10))

        # Steps Text
        steps_text = f"Steps: {self.steps} / {self.MAX_STEPS}"
        text_surface = self.font_small.render(steps_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(text_surface, text_rect)

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
            "gates_activated": sum(self.gate_activated),
            "robot1_pos": tuple(self.robot1_pos),
            "robot2_pos": tuple(self.robot2_pos),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # The main script needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Synchronization Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("----------------------\n")

    while running:
        # --- Event Handling ---
        action = [0, 0, 0] # Default action: no-op
        move_made = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    terminated = False
                    truncated = False
                    print("--- Environment Reset ---")
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # Register a move on keydown to allow single step actions
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    move_made = True

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            
            # Movement direction
            if keys[pygame.K_UP]: 
                action[0] = 1
            elif keys[pygame.K_DOWN]: 
                action[0] = 2
            elif keys[pygame.K_LEFT]: 
                action[0] = 3
            elif keys[pygame.K_RIGHT]: 
                action[0] = 4
            
            # Modifier keys
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            # If a move was made, step the environment
            if move_made:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}, Truncated: {truncated}")
        
        # --- Drawing ---
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(10) # Run at 10 FPS for manual play

    env.close()