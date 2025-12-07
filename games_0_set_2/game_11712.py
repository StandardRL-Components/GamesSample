import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:24:27.026838
# Source Brief: brief_01712.md
# Brief Index: 1712
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for "Chromatic Circuits".

    The goal is to guide 5 colored lights along their fixed tracks to match a
    target color sequence. Moving a light through a node on its track cycles
    its color (Blue -> Red -> Green -> Blue). The game is timed.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Light Selection):
        - 0: No-op (no change in selection)
        - 1: Select Light 0
        - 2: Select Light 1
        - 3: Select Light 2
        - 4: Select Light 3. If the previous selection action was also 4, select Light 4.
    - `action[1]` (Move Clockwise):
        - 0: Released
        - 1: Held (moves selected light clockwise to the next node)
    - `action[2]` (Move Counter-Clockwise):
        - 0: Released
        - 1: Held (moves selected light counter-clockwise to the next node)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +1.0 for each light whose color matches the target sequence at each step.
    - -0.01 penalty per step to encourage efficiency.
    - +100 for winning (matching the full sequence).
    - -100 for losing (timer runs out).

    **Termination:**
    - The episode ends when the target sequence is matched.
    - The episode ends when the timer (2500 steps) runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide five colored lights along their tracks to match the target sequence. "
        "Moving a light cycles its color, but watch the timer!"
    )
    user_guide = (
        "Controls: Use number keys 1-5 to select a light. Press Space to move clockwise and "
        "Left Shift to move counter-clockwise."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.UI_FONT = pygame.font.SysFont("Arial", 24, bold=True)
        self.UI_FONT_SMALL = pygame.font.SysFont("Arial", 18)

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.NUM_LIGHTS = 5
        self.MAX_STEPS = 2500  # 50 seconds at 50 FPS
        self.ANIMATION_SPEED = 0.1 # Takes 10 frames to animate move
        self.LIGHT_RADIUS = 12
        self.NODE_RADIUS = 5
        self.SEQ_BOX_SIZE = 25

        # --- Color Palette ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_TRACK = (60, 60, 70)
        self.COLOR_NODE = (100, 100, 110)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_BLUE = (0, 150, 255)
        self.COLOR_RED = (255, 80, 80)
        self.COLOR_GREEN = (80, 255, 80)
        self.COLORS = [self.COLOR_BLUE, self.COLOR_RED, self.COLOR_GREEN]

        # --- Game Data Structures ---
        self._define_tracks()
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0
        self.lights = []
        self.target_sequence = []
        self.selected_light_idx = 0
        self.last_movement_action = -1

    def _define_tracks(self):
        """Hardcodes the node positions for the 5 tracks."""
        self.tracks = []
        # Track 0: Center Pentagon
        center_x, center_y, radius = 320, 220, 90
        self.tracks.append([
            (center_x + radius * math.cos(math.radians(angle)),
             center_y + radius * math.sin(math.radians(angle)))
            for angle in range(18, 360, 72)
        ])
        # Track 1: Top-Left Square
        pos_x, pos_y, size = 80, 80, 80
        self.tracks.append([
            (pos_x, pos_y), (pos_x + size, pos_y),
            (pos_x + size, pos_y + size), (pos_x, pos_y + size)
        ])
        # Track 2: Top-Right Triangle
        pos_x, pos_y, size = 520, 100, 100
        self.tracks.append([
            (pos_x, pos_y - size / 2 * math.sqrt(3) / 2),
            (pos_x - size / 2, pos_y + size / 2 * math.sqrt(3) / 2),
            (pos_x + size / 2, pos_y + size / 2 * math.sqrt(3) / 2)
        ])
        # Track 3: Bottom-Left Line
        self.tracks.append([
            (60, 350), (120, 350), (180, 350), (240, 350)
        ])
        # Track 4: Bottom-Right Circle
        center_x, center_y, radius = 540, 320, 60
        self.tracks.append([
            (center_x + radius * math.cos(math.radians(angle)),
             center_y + radius * math.sin(math.radians(angle)))
            for angle in range(0, 360, 60)
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.selected_light_idx = 0
        self.last_movement_action = -1

        self.target_sequence = self.np_random.integers(0, 3, size=self.NUM_LIGHTS).tolist()
        self._initialize_lights()

        return self._get_observation(), self._get_info()

    def _initialize_lights(self):
        self.lights = []
        occupied_nodes = set()
        for i in range(self.NUM_LIGHTS):
            track_idx = i
            while True:
                node_idx = self.np_random.integers(0, len(self.tracks[track_idx]))
                if (track_idx, node_idx) not in occupied_nodes:
                    occupied_nodes.add((track_idx, node_idx))
                    break
            
            pos = self.tracks[track_idx][node_idx]
            color_idx = self.np_random.integers(0, 3)
            
            self.lights.append({
                'pos': np.array(pos, dtype=float),
                'color_idx': color_idx,
                'track_idx': track_idx,
                'node_idx': node_idx,
                'anim_progress': 1.0,
                'anim_start_pos': np.array(pos, dtype=float),
                'anim_target_pos': np.array(pos, dtype=float),
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1

        self._handle_actions(action)
        self._update_animations()

        is_victory = self._check_victory()
        terminated = self.timer <= 0 or is_victory
        truncated = False # This environment does not truncate
        reward = self._calculate_reward(is_victory, terminated)
        self.score += reward

        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Handle selection
        if movement in [1, 2, 3, 4]:
            if movement == 4 and self.last_movement_action == 4:
                self.selected_light_idx = 4
            else:
                self.selected_light_idx = movement - 1
        self.last_movement_action = movement

        # 2. Handle movement command
        selected_light = self.lights[self.selected_light_idx]
        if selected_light['anim_progress'] >= 1.0 and (space_held or shift_held):
            direction = 1 if space_held else -1
            track = self.tracks[selected_light['track_idx']]
            num_nodes = len(track)
            
            target_node_idx = (selected_light['node_idx'] + direction) % num_nodes
            
            is_occupied = any(
                i != self.selected_light_idx and
                light['track_idx'] == selected_light['track_idx'] and
                light['node_idx'] == target_node_idx
                for i, light in enumerate(self.lights)
            )
            
            if not is_occupied:
                selected_light['anim_progress'] = 0.0
                selected_light['anim_start_pos'] = selected_light['pos'].copy()
                selected_light['anim_target_pos'] = np.array(track[target_node_idx], dtype=float)
                selected_light['node_idx'] = target_node_idx
                selected_light['color_idx'] = (selected_light['color_idx'] + 1) % 3

    def _update_animations(self):
        for light in self.lights:
            if light['anim_progress'] < 1.0:
                light['anim_progress'] = min(1.0, light['anim_progress'] + self.ANIMATION_SPEED)
                # Ease-out quad interpolation for smoother stop
                t = light['anim_progress']
                eased_t = 1 - (1 - t) * (1 - t)
                light['pos'] = light['anim_start_pos'] + (light['anim_target_pos'] - light['anim_start_pos']) * eased_t
                
                if light['anim_progress'] >= 1.0:
                    light['pos'] = light['anim_target_pos'].copy()

    def _check_victory(self):
        return all(l['color_idx'] == self.target_sequence[i] for i, l in enumerate(self.lights))

    def _calculate_reward(self, is_victory, is_terminated):
        if is_terminated:
            return 100.0 if is_victory else -100.0
        
        matches = sum(1.0 for i, l in enumerate(self.lights) if l['color_idx'] == self.target_sequence[i])
        action_cost = -0.01
        return matches + action_cost

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_tracks_and_nodes()
        self._render_lights()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_tracks_and_nodes(self):
        for track in self.tracks:
            if len(track) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_TRACK, True, track, 2)
            for node_pos in track:
                pygame.gfxdraw.filled_circle(self.screen, int(node_pos[0]), int(node_pos[1]), self.NODE_RADIUS, self.COLOR_NODE)
                pygame.gfxdraw.aacircle(self.screen, int(node_pos[0]), int(node_pos[1]), self.NODE_RADIUS, self.COLOR_NODE)

    def _render_lights(self):
        for i, light in enumerate(self.lights):
            pos = (int(light['pos'][0]), int(light['pos'][1]))
            color = self.COLORS[light['color_idx']]
            
            # Glow effect
            glow_radius = int(self.LIGHT_RADIUS * 2.0)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.LIGHT_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.LIGHT_RADIUS, color)

            if i == self.selected_light_idx:
                pulse_progress = (pygame.time.get_ticks() % 1000) / 1000.0
                pulse_alpha = 100 + 155 * abs(math.sin(pulse_progress * math.pi * 2))
                pulse_radius = self.LIGHT_RADIUS + 5 + 3 * abs(math.sin(pulse_progress * math.pi * 2))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse_radius), (*self.COLOR_UI, int(pulse_alpha)))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse_radius)+1, (*self.COLOR_UI, int(pulse_alpha/2)))

    def _render_ui(self):
        self._render_sequences()
        self._render_timer()
        self.render_text(f"Score: {self.score:.2f}", (20, 20), self.UI_FONT, self.COLOR_UI)

    def _render_sequences(self):
        y_offset_target = 35
        self.render_text("Target:", (self.SCREEN_WIDTH / 2 - 120, y_offset_target - 5), self.UI_FONT_SMALL, self.COLOR_UI)
        for i, color_idx in enumerate(self.target_sequence):
            color = self.COLORS[color_idx]
            rect = pygame.Rect(self.SCREEN_WIDTH / 2 - 60 + i * (self.SEQ_BOX_SIZE + 5), y_offset_target - self.SEQ_BOX_SIZE/2, self.SEQ_BOX_SIZE, self.SEQ_BOX_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_UI, rect, 1, border_radius=4)

        y_offset_current = 70
        current_sequence = [light['color_idx'] for light in self.lights]
        self.render_text("Current:", (self.SCREEN_WIDTH / 2 - 120, y_offset_current - 5), self.UI_FONT_SMALL, self.COLOR_UI)
        for i, color_idx in enumerate(current_sequence):
            color = self.COLORS[color_idx]
            rect = pygame.Rect(self.SCREEN_WIDTH / 2 - 60 + i * (self.SEQ_BOX_SIZE + 5), y_offset_current - self.SEQ_BOX_SIZE/2, self.SEQ_BOX_SIZE, self.SEQ_BOX_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if color_idx == self.target_sequence[i]:
                pygame.draw.rect(self.screen, self.COLOR_GREEN, rect, 2, border_radius=4)
            else:
                pygame.draw.rect(self.screen, self.COLOR_TRACK, rect, 1, border_radius=4)

    def _render_timer(self):
        seconds_left = max(0, self.timer / 50.0)
        timer_color = self.COLOR_UI if seconds_left > 10 else self.COLOR_RED
        self.render_text(f"Time: {seconds_left:.1f}", (self.SCREEN_WIDTH - 150, 20), self.UI_FONT, timer_color)

    def render_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires the SDL_VIDEODRIVER to be set to a display driver
    # e.g., os.environ["SDL_VIDEODRIVER"] = "x11" or remove the dummy setting.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        # For manual play, we need a display.
        # This is a simple way to hint that the environment should be set up for display.
        # In a real-world scenario, you might pass a render_mode to __init__
        # and handle display setup there.
        print("Switching to a display-enabled SDL video driver for manual play.")
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Keys 1-5: Select Light 0-4
    # Space: Move Clockwise
    # Left Shift: Move Counter-Clockwise
    
    action = [0, 0, 0] # No-op, no move, no move
    
    # Pygame window for human play
    pygame.display.set_caption("Chromatic Circuits - Manual Test")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n--- Manual Control ---")
    print("Keys 1-5: Select a light")
    print("Space:     Move selected light clockwise")
    print("L-Shift:   Move selected light counter-clockwise")
    print("Q:         Quit")
    
    while not done:
        # Reset actions for this frame
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                # Light selection
                if event.key == pygame.K_1: action[0] = 1
                if event.key == pygame.K_2: action[0] = 2
                if event.key == pygame.K_3: action[0] = 3
                if event.key == pygame.K_4: action[0] = 4
                if event.key == pygame.K_5: 
                    # Simulate double-tap for light 5
                    action[0] = 4
                    env.last_movement_action = 4

        # Hold-down keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(50) # Run at 50 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()