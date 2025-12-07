import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:49:58.530057
# Source Brief: brief_00105.md
# Brief Index: 105
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    CircuitMaster is a timing-based puzzle game where the agent must control four
    switches to match a target energy pattern.

    The core gameplay involves observing oscillating energy pulses and toggling
    switches to modulate their intensity. When the intensities of all four
    circuits match a randomly generated target pattern (e.g., high, low, low, high),
    the circuit is completed.

    The goal is to complete 5 circuits to win the game. There is a time limit
    for completing each individual circuit and an overall time limit for the episode.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A timing-based puzzle game where you toggle switches to match an oscillating energy pattern and complete circuits."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to toggle the corresponding switches."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    NUM_CIRCUITS = 4
    WIN_CONDITION = 5

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_CIRCUIT_BASE = (40, 55, 70)
    COLOR_PULSE_LOW = (100, 150, 255)
    COLOR_PULSE_HIGH = (255, 255, 150)
    COLOR_SWITCH_HOUSING = (60, 75, 90)
    COLOR_CHECKPOINT_INACTIVE = (60, 75, 90)
    COLOR_CHECKPOINT_ACTIVE = (100, 255, 100)
    COLOR_FINAL_SWITCH_INACTIVE = (200, 50, 50)
    COLOR_FINAL_SWITCH_ACTIVE = (100, 255, 100)
    COLOR_TEXT = (220, 230, 240)
    COLOR_TARGET_HIGH = (255, 255, 150)
    COLOR_TARGET_LOW = (100, 150, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        self.render_mode = render_mode
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.circuits_completed = 0
        self.circuit_steps = 0
        self.switches = []
        self.checkpoints = []
        self.target_pattern = []
        self.intensities = []
        self.pulse_phase = 0.0
        self.circuit_just_completed = False
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.circuits_completed = 0
        
        self._setup_new_circuit()
        
        return self._get_observation(), self._get_info()

    def _setup_new_circuit(self):
        """Initializes a new puzzle."""
        self.circuit_steps = 0
        self.switches = [False] * self.NUM_CIRCUITS # False: LOW, True: HIGH
        self.checkpoints = [False] * self.NUM_CIRCUITS
        self.target_pattern = [self.np_random.choice([True, False]) for _ in range(self.NUM_CIRCUITS)]
        self.intensities = [0.0] * self.NUM_CIRCUITS
        self.pulse_phase = self.np_random.uniform(0, 2 * math.pi)
        self.circuit_just_completed = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.circuit_steps += 1

        self._handle_actions(action)
        self._update_game_state()
        
        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward
        self.game_over = terminated

        if self.circuit_just_completed and not terminated:
            # SFX: Circuit Complete Sound
            self._setup_new_circuit()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        """Toggles switches based on the discrete movement action."""
        movement = action[0]
        
        if movement > 0: # 1-4 correspond to switches 0-3
            switch_idx = movement - 1
            if switch_idx < self.NUM_CIRCUITS:
                self.switches[switch_idx] = not self.switches[switch_idx]
                # SFX: Switch Toggle Click

    def _update_game_state(self):
        """Updates pulse intensity and checks for checkpoint activation."""
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * math.pi)
        base_intensity = (math.sin(self.pulse_phase) + 1) / 2.0  # Oscillates 0.0 to 1.0

        HIGH_THRESHOLD = 0.8
        LOW_THRESHOLD = 0.2

        for i in range(self.NUM_CIRCUITS):
            switch_modifier = 1.0 if self.switches[i] else 0.3
            self.intensities[i] = base_intensity * switch_modifier
            
            is_active = (self.intensities[i] > HIGH_THRESHOLD and self.target_pattern[i]) or \
                        (self.intensities[i] < LOW_THRESHOLD and not self.target_pattern[i])
            self.checkpoints[i] = is_active

    def _calculate_reward_and_termination(self):
        """Calculates rewards and checks for game-ending conditions."""
        reward = 0.0
        terminated = False
        
        # Small penalty for every step to encourage speed
        reward -= 0.005

        # Reward for each active checkpoint
        reward += 0.05 * sum(self.checkpoints)
        
        # Check for circuit completion
        if all(self.checkpoints):
            self.circuits_completed += 1
            self.circuit_just_completed = True
            reward += 1.0
            
            if self.circuits_completed >= self.WIN_CONDITION:
                # Victory condition
                reward += 10.0
                terminated = True
        
        # Check for timeouts
        if self.circuit_steps >= 200 and not self.circuit_just_completed:
            reward = -1.0
            terminated = True
        
        if self.steps >= 1000:
            terminated = True

        return reward, terminated

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
            "circuits_completed": self.circuits_completed,
        }

    def _render_game(self):
        """Renders all the core visual elements of the circuit board."""
        y_positions = [100, 160, 240, 300]
        
        # --- Final Switch ---
        final_switch_pos = (590, 200)
        final_switch_active = all(self.checkpoints)
        final_color = self.COLOR_FINAL_SWITCH_ACTIVE if final_switch_active else self.COLOR_FINAL_SWITCH_INACTIVE
        self._draw_glow_circle(self.screen, final_switch_pos, 20, final_color)
        pygame.draw.line(self.screen, self.COLOR_CIRCUIT_BASE, (540, 200), final_switch_pos, 4)

        for i in range(self.NUM_CIRCUITS):
            y = y_positions[i]
            
            # --- Path & Pulse ---
            start_pos = (50, y)
            switch_pos = (150, y)
            checkpoint_pos = (450, y)
            end_pos = (540, 200) # Connects to a central point before final switch

            # Base circuit path
            pygame.draw.line(self.screen, self.COLOR_CIRCUIT_BASE, start_pos, switch_pos, 4)
            pygame.draw.line(self.screen, self.COLOR_CIRCUIT_BASE, switch_pos, checkpoint_pos, 4)
            pygame.draw.line(self.screen, self.COLOR_CIRCUIT_BASE, checkpoint_pos, end_pos, 4)

            # Pulse visualization
            pulse_color = self._lerp_color(self.COLOR_PULSE_LOW, self.COLOR_PULSE_HIGH, self.intensities[i])
            pulse_width = int(2 + 6 * self.intensities[i])
            pygame.draw.line(self.screen, pulse_color, start_pos, switch_pos, pulse_width)
            pygame.draw.line(self.screen, pulse_color, switch_pos, checkpoint_pos, pulse_width)
            
            # --- Switch ---
            pygame.draw.circle(self.screen, self.COLOR_SWITCH_HOUSING, switch_pos, 15)
            switch_color = self.COLOR_PULSE_HIGH if self.switches[i] else self.COLOR_PULSE_LOW
            self._draw_glow_circle(self.screen, switch_pos, 8, switch_color)

            # --- Checkpoint ---
            pygame.draw.rect(self.screen, self.COLOR_CHECKPOINT_INACTIVE, (checkpoint_pos[0]-15, y-15, 30, 30), border_radius=4)
            if self.checkpoints[i]:
                # SFX: Checkpoint Active Hum
                self._draw_glow_rect(self.screen, (checkpoint_pos[0]-15, y-15, 30, 30), self.COLOR_CHECKPOINT_ACTIVE)

            # --- Target Pattern Indicator ---
            target_symbol = "+" if self.target_pattern[i] else "-"
            target_color = self.COLOR_TARGET_HIGH if self.target_pattern[i] else self.COLOR_TARGET_LOW
            symbol_surf = self.font_medium.render(target_symbol, True, target_color)
            self.screen.blit(symbol_surf, (checkpoint_pos[0] - symbol_surf.get_width()//2, y - symbol_surf.get_height()//2 - 2))

            # --- Intensity Bar ---
            bar_x = checkpoint_pos[0] + 25
            bar_h = 40
            pygame.draw.rect(self.screen, self.COLOR_SWITCH_HOUSING, (bar_x, y - bar_h//2, 8, bar_h))
            fill_h = max(1, int(self.intensities[i] * bar_h))
            pygame.draw.rect(self.screen, pulse_color, (bar_x, y + bar_h//2 - fill_h, 8, fill_h))

    def _render_ui(self):
        """Renders text information like score and instructions."""
        # Circuits Completed
        circuits_text = f"CIRCUITS: {self.circuits_completed} / {self.WIN_CONDITION}"
        self._draw_text(circuits_text, (20, 20), self.font_medium, self.COLOR_TEXT)

        # Controls
        controls_text = "ARROW KEYS: TOGGLE SWITCHES"
        self._draw_text(controls_text, (self.WIDTH - 20, 20), self.font_small, self.COLOR_TEXT, align="topright")
        
        # Circuit Timer
        time_left = max(0, 200 - self.circuit_steps)
        time_text = f"TIMER: {time_left}"
        self._draw_text(time_text, (20, self.HEIGHT - 20), self.font_small, self.COLOR_TEXT, align="bottomleft")

        # Final Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))
            
            if self.circuits_completed >= self.WIN_CONDITION:
                msg = "SYSTEM ONLINE"
                color = self.COLOR_CHECKPOINT_ACTIVE
            else:
                msg = "CONNECTION TIMEOUT"
                color = self.COLOR_FINAL_SWITCH_INACTIVE
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, color, align="center")


    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "bottomleft":
            text_rect.bottomleft = pos
        else: # topleft
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

    def _draw_glow_circle(self, surface, pos, radius, color):
        """Draws a circle with a soft glow effect."""
        pos = (int(pos[0]), int(pos[1]))
        for i in range(4):
            alpha = 150 - i * 35
            if alpha < 0: continue
            glow_color = (*color, alpha)
            r = int(radius + i * 2)
            try:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], r, glow_color)
            except TypeError: # older pygame.gfxdraw might not handle alpha
                pass
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def _draw_glow_rect(self, surface, rect, color):
        """Draws a rectangle with a soft glow effect."""
        r = pygame.Rect(rect)
        for i in range(4):
            alpha = 150 - i * 35
            if alpha < 0: continue
            glow_color = (*color, alpha)
            glow_rect = r.inflate(i*4, i*4)
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, glow_color, temp_surf.get_rect(), border_radius=r.height//2)
            surface.blit(temp_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, r, border_radius=4)


    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block will not run in the hosted environment but is useful for local testing.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Circuit Master")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Controls ---")
    print("Up/Down/Left/Right Arrow: Toggle corresponding switch (1/2/3/4)")
    print("R: Reset environment")
    print("Q: Quit")
    
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Circuits: {info['circuits_completed']}")
            # Wait for reset
            pass

        clock.tick(GameEnv.FPS)
        
    env.close()