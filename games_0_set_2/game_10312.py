import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:11:00.308970
# Source Brief: brief_00312.md
# Brief Index: 312
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a pipe pressure synchronization puzzle game.

    The player's goal is to balance the pressure across a network of 7 interconnected
    pipes. This is achieved by selecting individual pipes and adjusting their flow rates.
    A "synchronization" occurs when all pipe pressures are within a small margin of
    each other. The episode is won by achieving 10 synchronizations.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0] (Movement): 0=None, 1=Up, 2=Down. Used to cycle through pipes. Left/Right are no-ops.
    - action[1] (Space): 0=Released, 1=Held. Increases flow rate of the selected pipe.
    - action[2] (Shift): 0=Released, 1=Held. Decreases flow rate of the selected pipe.

    Observation Space: A 640x400 RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance the pressure across a network of interconnected pipes by adjusting their flow rates. "
        "Achieve synchronization to win."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ arrow keys to select a pipe. Hold space to increase flow and shift to decrease flow."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_PIPES = 7
    MAX_STEPS = 1000
    WIN_CONDITION_SYNCS = 10
    
    MIN_PRESSURE = 1.0
    MAX_PRESSURE = 100.0
    OVERFLOW_THRESHOLD = 100.0
    
    MIN_FLOW = 0.0
    MAX_FLOW = 20.0
    
    SYNC_TOLERANCE = 5.0
    
    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_NODE = (200, 200, 220)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_UI_ACCENT = (0, 180, 255)
    COLOR_SELECTED_GLOW = (0, 200, 255, 100) # RGBA for alpha
    
    # Pressure gradient colors
    COLOR_LOW_PRESSURE = (0, 255, 100)
    COLOR_MID_PRESSURE = (255, 255, 0)
    COLOR_HIGH_PRESSURE = (255, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        self.font_main = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.sync_counter = 0
        self.game_over = False
        self.pressures = np.zeros(self.NUM_PIPES, dtype=np.float32)
        self.flow_rates = np.zeros(self.NUM_PIPES, dtype=np.float32)
        self.selected_pipe_index = 0
        self.last_sync_check_successful = False
        self.flow_particles = [[] for _ in range(self.NUM_PIPES)]
        self.sync_effect_timer = 0
        
        # --- Network Layout ---
        self._define_network_layout()

        # Initialize state variables by calling reset
        # self.reset() # reset() is called by the environment runner

    def _define_network_layout(self):
        """Defines the static positions of nodes and pipe connections."""
        self.node_pos = {
            'A': np.array([100, 200]), 'B': np.array([250, 200]),
            'C': np.array([400, 200]), 'D': np.array([550, 200]),
            'E': np.array([250, 100]), 'F': np.array([400, 300]),
            'G': np.array([100, 300]), 'H': np.array([550, 100])
        }
        self.pipe_defs = [
            ('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'E'), 
            ('C', 'F'), ('A', 'G'), ('D', 'H')
        ]
        
        # Pre-calculate pipe vectors and lengths for rendering and physics
        self.pipe_vectors = [self.node_pos[end] - self.node_pos[start] for start, end in self.pipe_defs]
        self.pipe_lengths = [np.linalg.norm(v) for v in self.pipe_vectors]
        self.pipe_midpoints = [self.node_pos[start] + v / 2 for (start, end), v in zip(self.pipe_defs, self.pipe_vectors)]

        # Pre-calculate adjacency list (which pipes are connected to which)
        self.pipe_adjacencies = [[] for _ in range(self.NUM_PIPES)]
        for i in range(self.NUM_PIPES):
            nodes_i = set(self.pipe_defs[i])
            for j in range(i + 1, self.NUM_PIPES):
                nodes_j = set(self.pipe_defs[j])
                if not nodes_i.isdisjoint(nodes_j):
                    self.pipe_adjacencies[i].append(j)
                    self.pipe_adjacencies[j].append(i)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.sync_counter = 0
        self.game_over = False
        
        # Initialize pressures and flow rates to a moderately unstable state
        self.pressures = self.np_random.uniform(low=20.0, high=80.0, size=self.NUM_PIPES).astype(np.float32)
        self.flow_rates = self.np_random.uniform(low=5.0, high=15.0, size=self.NUM_PIPES).astype(np.float32)
        
        self.selected_pipe_index = 0
        self.last_sync_check_successful = False
        self.flow_particles = [[] for _ in range(self.NUM_PIPES)]
        self.sync_effect_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            obs, info = self.reset()
            return obs, 0, True, False, info

        self._handle_actions(action)
        self._update_physics()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        """Update game state based on the agent's action."""
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Pipe selection with up/down arrows
        if movement == 1:  # Up
            self.selected_pipe_index = (self.selected_pipe_index - 1 + self.NUM_PIPES) % self.NUM_PIPES
        elif movement == 2:  # Down
            self.selected_pipe_index = (self.selected_pipe_index + 1) % self.NUM_PIPES
        
        # Adjust flow rate of the selected pipe
        if space_held:
            # SFX: Increase flow sound (hissing gets louder)
            self.flow_rates[self.selected_pipe_index] += 0.25
        if shift_held:
            # SFX: Decrease flow sound (hissing gets quieter)
            self.flow_rates[self.selected_pipe_index] -= 0.25
            
        self.flow_rates = np.clip(self.flow_rates, self.MIN_FLOW, self.MAX_FLOW)

    def _update_physics(self):
        """Simulate one time step of pressure and flow dynamics."""
        next_pressures = self.pressures.copy()

        # 1. Pressure change from flow rate and internal resistance
        pressure_gain = self.flow_rates * 0.2
        pressure_loss = self.pressures * 0.02 # Natural dissipation
        next_pressures += pressure_gain - pressure_loss

        # 2. Pressure equalization between connected pipes
        for i in range(self.NUM_PIPES):
            for j in self.pipe_adjacencies[i]:
                if i < j: # Process each pair only once
                    pressure_diff = self.pressures[i] - self.pressures[j]
                    equalization_amount = pressure_diff * 0.05 # How fast they equalize
                    next_pressures[i] -= equalization_amount
                    next_pressures[j] += equalization_amount
        
        self.pressures = next_pressures

        # 3. Handle overflows
        for i in range(self.NUM_PIPES):
            if self.pressures[i] > self.OVERFLOW_THRESHOLD:
                # SFX: Alarm/burst sound
                overflow_amount = self.pressures[i] - self.OVERFLOW_THRESHOLD
                self.pressures[i] = self.OVERFLOW_THRESHOLD * 0.9 # Reduce pressure by 10%
                
                # Distribute overflow pressure to neighbors
                connected_pipes = self.pipe_adjacencies[i]
                if connected_pipes:
                    amount_per_pipe = overflow_amount / len(connected_pipes)
                    for j in connected_pipes:
                        self.pressures[j] += amount_per_pipe

        # 4. Clamp final pressure values
        self.pressures = np.clip(self.pressures, self.MIN_PRESSURE, self.MAX_PRESSURE * 1.5) # Allow temporary over-pressure

    def _calculate_reward(self):
        """Calculate the reward for the current state."""
        reward = 0.0
        
        # Check for synchronization
        avg_pressure = np.mean(self.pressures)
        is_synced = np.all(np.abs(self.pressures - avg_pressure) <= self.SYNC_TOLERANCE)
        
        if is_synced and not self.last_sync_check_successful:
            self.sync_counter += 1
            reward += 10.0  # Event-based reward for new sync
            self.last_sync_check_successful = True
            self.sync_effect_timer = 30 # 1 second at 30fps
            # SFX: Success chime
        elif not is_synced:
            self.last_sync_check_successful = False

        # Continuous reward for being close to the mean
        std_dev = np.std(self.pressures)
        reward -= std_dev * 0.01

        # Goal-oriented reward for winning
        if self.sync_counter >= self.WIN_CONDITION_SYNCS:
            reward += 100.0

        return reward

    def _check_termination(self):
        """Check if the episode should terminate due to winning."""
        if self.sync_counter >= self.WIN_CONDITION_SYNCS:
            self.game_over = True
            return True
        return False

    def _get_info(self):
        return {
            "score": self.sync_counter,
            "steps": self.steps,
            "pressures": self.pressures,
            "flow_rates": self.flow_rates,
        }

    def _get_observation(self):
        """Render the game state to an RGB array."""
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Update and render effects
        self._update_and_render_effects()
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Render the pipes, nodes, and flow particles."""
        # Render pipes
        for i in range(self.NUM_PIPES):
            start_pos = self.node_pos[self.pipe_defs[i][0]]
            end_pos = self.node_pos[self.pipe_defs[i][1]]
            
            # Determine pipe color based on pressure
            pressure_norm = np.clip((self.pressures[i] - self.MIN_PRESSURE) / (self.MAX_PRESSURE - self.MIN_PRESSURE), 0, 1)
            color = self._lerp_color_triple(pressure_norm, self.COLOR_LOW_PRESSURE, self.COLOR_MID_PRESSURE, self.COLOR_HIGH_PRESSURE)
            
            # Draw glowing effect for selected pipe
            if i == self.selected_pipe_index:
                pygame.draw.line(self.screen, self.COLOR_SELECTED_GLOW, start_pos, end_pos, 16)
                pygame.draw.line(self.screen, self.COLOR_SELECTED_GLOW, start_pos, end_pos, 24)

            # Draw main pipe line
            pygame.draw.line(self.screen, color, start_pos, end_pos, 8)
        
        # Render nodes on top of pipes
        for pos in self.node_pos.values():
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 12, self.COLOR_NODE)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 12, self.COLOR_NODE)

    def _update_and_render_effects(self):
        """Handle particle animations and sync effects."""
        # Sync effect
        if self.sync_effect_timer > 0:
            alpha = int(100 * (self.sync_effect_timer / 30))
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, alpha))
            self.screen.blit(overlay, (0, 0))
            self.sync_effect_timer -= 1
        
        # Flow particles
        for i in range(self.NUM_PIPES):
            # Spawn new particles based on flow rate
            if self.np_random.random() < self.flow_rates[i] / self.MAX_FLOW * 0.5:
                self.flow_particles[i].append({'progress': 0.0, 'offset': self.np_random.uniform(-2, 2)})

            # Update and draw existing particles
            remaining_particles = []
            start_pos = self.node_pos[self.pipe_defs[i][0]]
            vec = self.pipe_vectors[i]
            
            for p in self.flow_particles[i]:
                p['progress'] += self.flow_rates[i] * 0.005
                if p['progress'] < 1.0:
                    remaining_particles.append(p)
                    
                    # Calculate particle position
                    pos = start_pos + vec * p['progress']
                    perp_vec = np.array([-vec[1], vec[0]]) / (self.pipe_lengths[i] + 1e-6)
                    pos += perp_vec * p['offset']
                    
                    # Fade particle as it moves
                    alpha = int(200 * (1 - p['progress']))
                    color = (*self.COLOR_UI_ACCENT, alpha)
                    
                    # Draw anti-aliased, alpha-blended particle
                    temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(temp_surf, 2, 2, 2, color)
                    self.screen.blit(temp_surf, (int(pos[0] - 2), int(pos[1] - 2)), special_flags=pygame.BLEND_RGBA_ADD)
            
            self.flow_particles[i] = remaining_particles


    def _render_ui(self):
        """Render text information and overlays."""
        # --- Main UI Panel ---
        self._draw_text(f"SYNCHRONIZATIONS: {self.sync_counter} / {self.WIN_CONDITION_SYNCS}", (self.SCREEN_WIDTH / 2, 30), self.font_title, self.COLOR_UI_ACCENT)
        self._draw_text(f"STEPS: {self.steps} / {self.MAX_STEPS}", (self.SCREEN_WIDTH - 10, 20), self.font_ui, self.COLOR_TEXT, align="right")
        
        # --- Pipe Information ---
        for i in range(self.NUM_PIPES):
            mid_point = self.pipe_midpoints[i]
            is_selected = (i == self.selected_pipe_index)
            color = self.COLOR_UI_ACCENT if is_selected else self.COLOR_TEXT
            
            # Text background for readability
            text_bg_rect = pygame.Rect(0, 0, 100, 38)
            text_bg_rect.center = (mid_point[0], mid_point[1] - 35) if mid_point[1] > self.SCREEN_HEIGHT/2 else (mid_point[0], mid_point[1] + 35)
            
            s = pygame.Surface((100, 38), pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(s, text_bg_rect.topleft)

            # Display Pressure
            p_text = f"P: {self.pressures[i]:.1f}"
            self._draw_text(p_text, text_bg_rect.center - np.array([0, 8]), self.font_main, color)
            
            # Display Flow Rate
            f_text = f"F: {self.flow_rates[i]:.1f}"
            self._draw_text(f_text, text_bg_rect.center + np.array([0, 8]), self.font_main, color)
            
            # Display Pipe Index
            self._draw_text(f"Pipe {i}", text_bg_rect.center + np.array([0, -30]), self.font_main, color)
            
        # --- End Game Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.sync_counter >= self.WIN_CONDITION_SYNCS:
                msg = "SYSTEM SYNCHRONIZED"
                color = self.COLOR_LOW_PRESSURE
            else:
                msg = "MAX STEPS REACHED"
                color = self.COLOR_HIGH_PRESSURE
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), self.font_title, color)


    def _draw_text(self, text, pos, font, color, align="center"):
        """Helper function to draw text with a shadow for better visibility."""
        text_surface = font.render(text, True, color)
        text_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surface.get_rect()
        
        if align == "center":
            text_rect.center = (int(pos[0]), int(pos[1]))
        elif align == "left":
            text_rect.midleft = (int(pos[0]), int(pos[1]))
        elif align == "right":
            text_rect.midright = (int(pos[0]), int(pos[1]))
            
        self.screen.blit(text_shadow, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    @staticmethod
    def _lerp_color(val, start_color, end_color):
        """Linearly interpolate between two colors."""
        return tuple(int(s + (e - s) * val) for s, e in zip(start_color, end_color))

    def _lerp_color_triple(self, val, color1, color2, color3):
        """Interpolate between three colors (e.g., green -> yellow -> red)."""
        if val < 0.5:
            return self._lerp_color(val * 2, color1, color2)
        else:
            return self._lerp_color((val - 0.5) * 2, color2, color3)

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # --- Manual Play Example ---
    # To run this, you need to unset the dummy video driver
    # and have a display.
    # For example, run:
    # SDL_VIDEODRIVER=x11 python your_file.py
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pressure Pipe Synchronization")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Up/Down Arrows: Select Pipe")
    print("Spacebar: Increase Flow")
    print("Left Shift: Decrease Flow")
    print("R: Reset Environment")
    print("Q: Quit")

    while running:
        # Action defaults
        movement, space, shift = 0, 0, 0
        
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

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = np.array([movement, space, shift])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth feel

    env.close()