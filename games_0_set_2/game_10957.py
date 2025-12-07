import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:16:08.839124
# Source Brief: brief_00957.md
# Brief Index: 957
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls water flow between two
    reservoirs to reach a target level.

    **Visuals:**
    The environment is rendered with a clean, geometric aesthetic. Water flow is
    visualized with animated particles, and all interactive elements provide
    clear visual feedback.

    **Gameplay:**
    - The agent controls 4 levers to manage water flow.
    - Lever 1: Fills Reservoir A from an external source.
    - Lever 2: Moves water from Reservoir A to Reservoir B.
    - Lever 3: Moves water from Reservoir B to Reservoir A.
    - Lever 4: Drains Reservoir B to an external sink.
    - Goal: Get both reservoirs to exactly 50 units within 30 seconds.
    - Failure: A reservoir overflows (> 100 units) or time runs out.

    **Action Space:**
    - `actions[0]` (Movement): 0=none, 1=up, 2=down, 3=left, 4=right
      - Up/Down: Increase/decrease the selected lever's setting.
      - Left/Right: Cycle through which lever is selected.
    - `actions[1]` (Space): Unused.
    - `actions[2]` (Shift): Unused.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage water flow between two reservoirs using a set of levers. "
        "Reach the target water level in both reservoirs before time runs out or they overflow."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a lever. Use ↑↓ arrow keys to adjust the selected lever's flow rate."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = FPS * 30  # 30 seconds

    # Colors
    COLOR_BG = (15, 23, 42)
    COLOR_CONTAINER = (100, 116, 139)
    COLOR_WATER = (59, 130, 246)
    COLOR_WATER_OVERFILL = (220, 38, 38)
    COLOR_TARGET = (16, 185, 129)
    COLOR_TEXT = (226, 232, 240)
    COLOR_LEVER_BG = (30, 41, 59)
    COLOR_LEVER_HANDLE = (245, 158, 11)
    COLOR_LEVER_SELECTED = (253, 224, 71)
    COLOR_PARTICLE = (147, 197, 253)

    # Game parameters
    MAX_CAPACITY = 100.0
    TARGET_LEVEL = 50.0
    TARGET_TOLERANCE = 1.5  # +/- this value to win
    MAX_FLOW_PER_SEC = 30.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.level_A = 0.0
        self.level_B = 0.0
        self.levers = np.array([0, 0, 0, 0])  # 4 levers, each with state 0, 1, or 2
        self.selected_lever = 0

        self.particles = []
        self.target_A_bonus_given = False
        self.target_B_bonus_given = False
        
        # --- UI Geometry ---
        self.res_A_rect = pygame.Rect(80, 80, 180, 280)
        self.res_B_rect = pygame.Rect(self.SCREEN_WIDTH - 80 - 180, 80, 180, 280)
        self.lever_rects = [
            pygame.Rect(self.SCREEN_WIDTH / 2 - 125, 100, 50, 15),
            pygame.Rect(self.SCREEN_WIDTH / 2 - 25, 180, 50, 15),
            pygame.Rect(self.SCREEN_WIDTH / 2 - 25, 230, 50, 15),
            pygame.Rect(self.SCREEN_WIDTH / 2 + 75, 300, 50, 15),
        ]

        # Initial reset to populate state
        # self.reset() # This is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Start with some water to make it interesting
        self.level_A = self.np_random.uniform(10, 30)
        self.level_B = self.np_random.uniform(10, 30)
        self.levers = np.array([0, 0, 0, 0])
        self.selected_lever = 0

        self.particles = []
        self.target_A_bonus_given = False
        self.target_B_bonus_given = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Action ---
        movement = action[0]
        self._handle_action(movement)

        # --- 2. Update Game State ---
        self._update_physics()
        self.steps += 1

        # --- 3. Calculate Reward & Check Termination ---
        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True


        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_action(self, movement):
        """Maps movement action to lever controls."""
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3:  # Left
            self.selected_lever = (self.selected_lever - 1 + 4) % 4
            # sfx: Lever select sound
        elif movement == 4:  # Right
            self.selected_lever = (self.selected_lever + 1) % 4
            # sfx: Lever select sound
        elif movement == 1:  # Up
            self.levers[self.selected_lever] = min(2, self.levers[self.selected_lever] + 1)
            # sfx: Lever click up sound
        elif movement == 2:  # Down
            self.levers[self.selected_lever] = max(0, self.levers[self.selected_lever] - 1)
            # sfx: Lever click down sound

    def _update_physics(self):
        """Updates water levels and particle effects."""
        flow_rates = (self.levers / 2.0) * self.MAX_FLOW_PER_SEC
        flow_per_step = flow_rates / self.FPS

        # Flow logic:
        # 1: Source -> A
        # 2: A -> B
        # 3: B -> A
        # 4: B -> Sink
        delta_A = flow_per_step[0] - flow_per_step[1] + flow_per_step[2]
        delta_B = flow_per_step[1] - flow_per_step[2] - flow_per_step[3]

        self.level_A += delta_A
        self.level_B += delta_B

        # Clamp levels to be non-negative
        self.level_A = max(0, self.level_A)
        self.level_B = max(0, self.level_B)

        # Update particles
        self._update_particles(flow_per_step)

    def _calculate_reward_and_termination(self):
        """Calculates reward and checks for win/loss/timeout conditions."""
        terminated = False
        reward = 0.0

        # Check for loss conditions
        if self.level_A > self.MAX_CAPACITY or self.level_B > self.MAX_CAPACITY:
            return -100.0, True  # Overfill penalty

        if self.steps >= self.MAX_STEPS:
            return 0.0, True  # Timeout

        # Check for win condition
        in_target_A = abs(self.level_A - self.TARGET_LEVEL) < self.TARGET_TOLERANCE
        in_target_B = abs(self.level_B - self.TARGET_LEVEL) < self.TARGET_TOLERANCE

        if in_target_A and in_target_B:
            return 100.0, True  # Win bonus

        # --- Step-based and event-based rewards ---
        reward += 0.1  # Survival reward

        # Bonus for achieving target for the first time
        if in_target_A and not self.target_A_bonus_given:
            reward += 5.0
            self.target_A_bonus_given = True
        if in_target_B and not self.target_B_bonus_given:
            reward += 5.0
            self.target_B_bonus_given = True
            
        # Small continuous reward for staying in the target zone
        if in_target_A: reward += 0.1
        if in_target_B: reward += 0.1

        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array in the correct format (H, W, C)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all primary game elements."""
        # Draw pipes first (background)
        self._draw_pipes()

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['radius'], self.COLOR_PARTICLE)

        # Draw reservoirs and water
        self._draw_reservoir(self.res_A_rect, self.level_A)
        self._draw_reservoir(self.res_B_rect, self.level_B)

        # Draw levers
        self._draw_levers()

    def _draw_reservoir(self, rect, level):
        """Draws a single reservoir and its water content."""
        # Draw container
        pygame.draw.rect(self.screen, self.COLOR_CONTAINER, rect, 2, border_radius=5)
        
        # Draw target line
        target_y = rect.bottom - (self.TARGET_LEVEL / self.MAX_CAPACITY) * rect.height
        pygame.gfxdraw.hline(self.screen, rect.left, rect.right, int(target_y), self.COLOR_TARGET)

        # Draw water
        if level > 0:
            water_level_px = (min(level, self.MAX_CAPACITY) / self.MAX_CAPACITY) * rect.height
            water_rect = pygame.Rect(
                rect.left + 1,
                rect.bottom - water_level_px,
                rect.width - 2,
                water_level_px,
            )
            water_color = self.COLOR_WATER_OVERFILL if level > self.MAX_CAPACITY else self.COLOR_WATER
            pygame.draw.rect(self.screen, water_color, water_rect, border_bottom_left_radius=4, border_bottom_right_radius=4)
            
            # Add a subtle highlight to the water surface
            highlight_rect = water_rect.copy()
            highlight_rect.height = 2
            pygame.draw.rect(self.screen, (255,255,255, 50), highlight_rect)


    def _draw_pipes(self):
        """Draws the static pipes connecting elements."""
        mid_x = self.SCREEN_WIDTH / 2
        # Pipe 1: Source -> A
        pygame.draw.line(self.screen, self.COLOR_CONTAINER, (mid_x - 100, 80), self.res_A_rect.midtop, 5)
        # Pipe 2: A -> B
        pygame.draw.line(self.screen, self.COLOR_CONTAINER, self.res_A_rect.midright, self.res_B_rect.midleft, 5)
        # Pipe 3: B -> A
        pygame.draw.line(self.screen, self.COLOR_CONTAINER, (self.res_B_rect.left, self.res_B_rect.centery + 20), (self.res_A_rect.right, self.res_A_rect.centery + 20), 5)
        # Pipe 4: B -> Sink
        pygame.draw.line(self.screen, self.COLOR_CONTAINER, self.res_B_rect.midbottom, (mid_x + 100, self.SCREEN_HEIGHT - 20), 5)

    def _draw_levers(self):
        """Draws the four control levers and their states."""
        for i, rect in enumerate(self.lever_rects):
            # Draw background
            pygame.draw.rect(self.screen, self.COLOR_LEVER_BG, rect, border_radius=4)
            
            # Draw highlight if selected
            if i == self.selected_lever:
                pygame.draw.rect(self.screen, self.COLOR_LEVER_SELECTED, rect, 2, border_radius=4)

            # Draw handle indicating position
            lever_state = self.levers[i] # 0, 1, or 2
            handle_pos_x = rect.left + 5 + (lever_state / 2.0) * (rect.width - 10)
            handle_rect = pygame.Rect(handle_pos_x - 5, rect.top - 5, 10, rect.height + 10)
            pygame.draw.rect(self.screen, self.COLOR_LEVER_HANDLE, handle_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_LEVER_SELECTED, handle_rect, 1, border_radius=3)

    def _update_particles(self, flow_per_step):
        """Creates and moves particles to visualize flow."""
        # sfx: Water flowing loop sound, volume based on total flow

        # Spawn new particles
        pipe_paths = {
            0: ((self.SCREEN_WIDTH / 2 - 100, 80), self.res_A_rect.midtop),
            1: (self.res_A_rect.midright, self.res_B_rect.midleft),
            2: ((self.res_B_rect.left, self.res_B_rect.centery + 20), (self.res_A_rect.right, self.res_A_rect.centery + 20)),
            3: (self.res_B_rect.midbottom, (self.SCREEN_WIDTH / 2 + 100, self.SCREEN_HEIGHT - 20))
        }

        for i, flow in enumerate(flow_per_step):
            if self.np_random.random() < flow * 2: # Probability of spawning a particle
                start, end = pipe_paths[i]
                vec = (end[0] - start[0], end[1] - start[1])
                dist = math.hypot(*vec)
                if dist == 0: continue
                vel = (vec[0] / dist * 3, vec[1] / dist * 3) # Speed of 3px/frame
                
                # Add some jitter to start position
                p_start = (start[0] + self.np_random.uniform(-2, 2), start[1] + self.np_random.uniform(-2, 2))

                self.particles.append({
                    "pos": list(p_start),
                    "vel": vel,
                    "life": dist / 3, # Lifetime in frames
                    "radius": int(self.np_random.uniform(2, 4))
                })

        # Move and kill old particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _render_ui(self):
        """Renders text information like score, timer, and levels."""
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"Time: {max(0, time_left):.1f}s"
        self._draw_text(timer_text, (self.SCREEN_WIDTH / 2, 30), self.font_small, self.COLOR_TEXT)

        # Score
        score_text = f"Score: {self.score:.1f}"
        self._draw_text(score_text, (self.SCREEN_WIDTH / 2, 60), self.font_small, self.COLOR_TEXT)
        
        # Reservoir A Level
        level_A_text = f"{self.level_A:.1f}"
        self._draw_text(level_A_text, self.res_A_rect.midtop, self.font_small, self.COLOR_TEXT, y_offset=-15)

        # Reservoir B Level
        level_B_text = f"{self.level_B:.1f}"
        self._draw_text(level_B_text, self.res_B_rect.midtop, self.font_small, self.COLOR_TEXT, y_offset=-15)

        # Game Over Text
        if self.game_over:
            in_target_A = abs(self.level_A - self.TARGET_LEVEL) < self.TARGET_TOLERANCE
            in_target_B = abs(self.level_B - self.TARGET_LEVEL) < self.TARGET_TOLERANCE
            
            if in_target_A and in_target_B and not self.steps >= self.MAX_STEPS:
                msg = "TARGET REACHED!"
                color = self.COLOR_TARGET
            elif self.level_A > self.MAX_CAPACITY or self.level_B > self.MAX_CAPACITY:
                msg = "OVERFLOW!"
                color = self.COLOR_WATER_OVERFILL
            else:
                msg = "TIME UP!"
                color = self.COLOR_LEVER_HANDLE
            
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50), self.font_large, color)


    def _draw_text(self, text, pos, font, color, y_offset=0):
        """Helper to draw centered text."""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(pos[0], pos[1] + y_offset))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level_A": self.level_A,
            "level_B": self.level_B,
            "levers": self.levers.tolist()
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
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
    # --- Manual Play Loop ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for display
    # This is not strictly necessary for headless mode, but useful for human play
    if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy":
        pygame.display.set_caption("Reservoir Control")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    else:
        screen = None

    terminated = False
    truncated = False
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                if not terminated and not truncated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the screen if a display is available
        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()