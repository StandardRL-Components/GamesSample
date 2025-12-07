import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates three interconnected sliders 
    to morph a 3D wireframe shape to match a target configuration.
    
    The environment is designed with a focus on visual quality and intuitive gameplay,
    providing a polished puzzle experience suitable for both human players and RL agents.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
                 Up/Down modifies the selected slider. Left/Right are no-ops.
    - action[1]: Space button (0=released, 1=pressed)
                 Pressing cycles to the next slider.
    - action[2]: Shift button (0=released, 1=pressed)
                 Pressing cycles to the previous slider.
    
    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the current game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate interconnected sliders to morph a 3D wireframe shape and match it to the target configuration."
    )
    user_guide = (
        "Use the ↑ and ↓ arrow keys to adjust the selected slider. Press space to cycle to the next slider, and shift to cycle to the previous one."
    )
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame and Display ---
        self.W, self.H = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_slider = pygame.font.SysFont("monospace", 14)

        # --- Visual & Style Constants ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (35, 40, 45)
        self.COLOR_PLAYER_SHAPE = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200, 32) # RGBA
        self.COLOR_TARGET_SHAPE = (100, 100, 110, 128) # RGBA
        self.COLOR_SLIDER_BG = (50, 55, 60)
        self.COLOR_SLIDER_HANDLE = (180, 185, 190)
        self.COLOR_SLIDER_SELECTED = (0, 150, 255)
        self.COLOR_TEXT = (220, 220, 220)
        
        # --- Game Mechanics Constants ---
        self.MAX_STEPS = 1000
        self.VICTORY_THRESHOLD = 95.0 # %
        self.SLIDER_COUNT = 3
        self.SLIDER_MIN, self.SLIDER_MAX = 0, 100
        
        # --- 3D Shape & Projection ---
        self.base_vertices = self._create_base_cube(size=80)
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
            (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        self.projection_angle = math.pi / 6
        self.projection_scale = 1.3
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.similarity = 0.0
        self.last_similarity = 0.0
        self.slider_values = np.zeros(self.SLIDER_COUNT, dtype=np.int32)
        self.target_slider_values = np.zeros(self.SLIDER_COUNT, dtype=np.int32)
        self.player_vertices = np.copy(self.base_vertices)
        self.target_vertices = np.copy(self.base_vertices)
        self.selected_slider = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.game_over = False

        # Calculate max possible distance for similarity normalization
        shape_at_min = self._morph_shape(np.array([self.SLIDER_MIN] * self.SLIDER_COUNT))
        shape_at_max = self._morph_shape(np.array([self.SLIDER_MAX] * self.SLIDER_COUNT))
        self.max_distance = np.linalg.norm(shape_at_max - shape_at_min)
        if self.max_distance == 0: self.max_distance = 1.0 # Avoid division by zero

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.selected_slider = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Set initial and target slider values
        self.slider_values = np.full(self.SLIDER_COUNT, (self.SLIDER_MIN + self.SLIDER_MAX) // 2, dtype=np.int32)
        
        # Ensure target is sufficiently different from start
        while True:
            self.target_slider_values = self.np_random.integers(
                self.SLIDER_MIN, self.SLIDER_MAX + 1, size=self.SLIDER_COUNT
            )
            dist = np.linalg.norm(self.target_slider_values - self.slider_values)
            if dist > self.SLIDER_MAX * 0.5: # Ensure target is not too easy
                break

        # Calculate initial shapes and similarity
        self.player_vertices = self._morph_shape(self.slider_values)
        self.target_vertices = self._morph_shape(self.target_slider_values)
        self.similarity = self._calculate_similarity()
        self.last_similarity = self.similarity
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack and Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle slider selection (on press, not hold)
        if space_held and not self.prev_space_held:
            self.selected_slider = (self.selected_slider + 1) % self.SLIDER_COUNT
        if shift_held and not self.prev_shift_held:
            self.selected_slider = (self.selected_slider - 1 + self.SLIDER_COUNT) % self.SLIDER_COUNT

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # Handle slider value adjustment
        slider_changed = False
        if movement == 1: # Up
            if self.slider_values[self.selected_slider] < self.SLIDER_MAX:
                self.slider_values[self.selected_slider] += 1
                slider_changed = True
        elif movement == 2: # Down
            if self.slider_values[self.selected_slider] > self.SLIDER_MIN:
                self.slider_values[self.selected_slider] -= 1
                slider_changed = True
        
        # --- Update Game State ---
        self.steps += 1
        self.player_vertices = self._morph_shape(self.slider_values)
        self.similarity = self._calculate_similarity()
        
        # --- Calculate Reward ---
        reward = (self.similarity - self.last_similarity) * 0.1
        self.last_similarity = self.similarity
        
        # --- Check Termination Conditions ---
        victory = self.similarity >= self.VICTORY_THRESHOLD
        timeout = self.steps >= self.MAX_STEPS
        terminated = victory or timeout
        
        if terminated:
            self.game_over = True
            if victory:
                reward = 100.0 # Goal-oriented reward
            else: # Timeout
                reward = -100.0 # Penalty for failure

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_wireframe(self.target_vertices, self.COLOR_TARGET_SHAPE, width=1, is_target=True)
        self._draw_wireframe(self.player_vertices, self.COLOR_PLAYER_SHAPE, width=2, glow_color=self.COLOR_PLAYER_GLOW, glow_width=7)

    def _render_ui(self):
        # Display Similarity and Steps
        sim_text = f"Similarity: {self.similarity:.2f}%"
        step_text = f"Step: {self.steps}/{self.MAX_STEPS}"
        sim_surf = self.font_ui.render(sim_text, True, self.COLOR_TEXT)
        step_surf = self.font_ui.render(step_text, True, self.COLOR_TEXT)
        self.screen.blit(sim_surf, (20, 15))
        self.screen.blit(step_surf, (self.W - step_surf.get_width() - 20, 15))

        # Draw Sliders
        self._draw_sliders()

    def _draw_grid(self, grid_size=40):
        for x in range(0, self.W, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.H))
        for y in range(0, self.H, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.W, y))

    def _draw_wireframe(self, vertices, color, width, glow_color=None, glow_width=0, is_target=False):
        projected = [self._project(v[0], v[1], v[2]) for v in vertices]
        
        # Use a separate surface for transparency effects
        shape_surface = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        for i, j in self.edges:
            p1 = (int(projected[i][0]), int(projected[i][1]))
            p2 = (int(projected[j][0]), int(projected[j][1]))
            
            if glow_color and glow_width > 0:
                pygame.draw.line(shape_surface, glow_color, p1, p2, glow_width)
            
            if is_target:
                pygame.gfxdraw.line(shape_surface, p1[0], p1[1], p2[0], p2[1], color)
            else:
                pygame.draw.line(shape_surface, color, p1, p2, width)

        self.screen.blit(shape_surface, (0, 0))

    def _draw_sliders(self):
        slider_width, slider_height = 150, 10
        start_x = (self.W - (self.SLIDER_COUNT * slider_width + (self.SLIDER_COUNT - 1) * 20)) / 2
        y_pos = self.H - 40

        for i in range(self.SLIDER_COUNT):
            x_pos = start_x + i * (slider_width + 20)
            
            # Slider track
            rect_bg = pygame.Rect(x_pos, y_pos, slider_width, slider_height)
            pygame.draw.rect(self.screen, self.COLOR_SLIDER_BG, rect_bg, border_radius=5)
            
            # Handle
            handle_x = x_pos + (self.slider_values[i] / self.SLIDER_MAX) * slider_width
            handle_color = self.COLOR_SLIDER_SELECTED if i == self.selected_slider else self.COLOR_SLIDER_HANDLE
            pygame.draw.circle(self.screen, handle_color, (int(handle_x), int(y_pos + slider_height / 2)), 8)
            
            # Value text
            value_text = f"{self.slider_values[i]}"
            text_surf = self.font_slider.render(value_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(x_pos + slider_width / 2, y_pos - 15))
            self.screen.blit(text_surf, text_rect)

    # --- Private Helper Methods: Game Logic ---

    def _get_info(self):
        return {
            "similarity": self.similarity,
            "steps": self.steps,
            "slider_values": self.slider_values.tolist(),
            "target_values": self.target_slider_values.tolist(),
        }

    def _create_base_cube(self, size):
        half = size / 2
        return np.array([
            [-half, -half, -half], [half, -half, -half], [half, half, -half], [-half, half, -half],
            [-half, -half, half], [half, -half, half], [half, half, half], [-half, half, half]
        ])

    def _project(self, x, y, z):
        screen_x = self.W / 2 + self.projection_scale * (x - y) * math.cos(self.projection_angle)
        screen_y = self.H / 2 + self.projection_scale * ((x + y) * math.sin(self.projection_angle) - z)
        return screen_x, screen_y

    def _morph_shape(self, slider_vals):
        s = slider_vals / self.SLIDER_MAX
        s1, s2, s3 = s[0], s[1], s[2]
        morphed_vertices = np.zeros_like(self.base_vertices)

        for i, (x, y, z) in enumerate(self.base_vertices):
            # Scale based on individual sliders
            scale_x = 1.0 + 0.5 * s1
            scale_y = 1.0 + 0.5 * s2
            scale_z = 1.0 + 0.5 * s3
            
            # Twist and shear based on slider interactions
            twist_x = 40 * s3 * math.sin(math.pi * s2)
            twist_y = 40 * s1 * math.cos(math.pi * s3)
            twist_z = 40 * s2 * math.sin(math.pi * s1)
            
            # Apply transformations
            new_x = x * scale_x + twist_x
            new_y = y * scale_y + twist_y
            new_z = z * scale_z + twist_z
            
            morphed_vertices[i] = [new_x, new_y, new_z]
            
        return morphed_vertices

    def _calculate_similarity(self):
        distance = np.linalg.norm(self.player_vertices - self.target_vertices)
        normalized_distance = min(1.0, distance / self.max_distance)
        similarity_score = (1.0 - normalized_distance) * 100.0
        return max(0.0, similarity_score)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Skipping manual play in headless mode.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.W, env.H))
        pygame.display.set_caption("Shape Morph Puzzle")
        clock = pygame.time.Clock()
        
        running = True
        terminated = False
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False

            if not terminated:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
                
                # Use get_just_pressed for single-press actions
                # This requires a bit of state management outside the env
                # For simplicity, we use the env's internal logic
                if keys[pygame.K_SPACE]: space = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
                action = [movement, space, shift]
                obs, reward, terminated, _, info = env.step(action)
                
                print(f"Step: {info['steps']}, Similarity: {info['similarity']:.2f}, Reward: {reward:.2f}")

            # Draw the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            if terminated:
                # Display end-game message
                end_font = pygame.font.SysFont("monospace", 40, bold=True)
                msg = "VICTORY!" if info['similarity'] >= env.VICTORY_THRESHOLD else "TIME OUT"
                color = (100, 255, 100) if info['similarity'] >= env.VICTORY_THRESHOLD else (255, 100, 100)
                end_surf = end_font.render(msg, True, color)
                end_rect = end_surf.get_rect(center=(env.W/2, env.H/2 - 50))
                
                reset_font = pygame.font.SysFont("monospace", 20)
                reset_surf = reset_font.render("Press 'R' to restart", True, (200, 200, 200))
                reset_rect = reset_surf.get_rect(center=(env.W/2, env.H/2))

                screen.blit(end_surf, end_rect)
                screen.blit(reset_surf, reset_rect)

            pygame.display.flip()
            clock.tick(30) # Limit frame rate for human play

        env.close()