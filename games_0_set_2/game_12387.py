import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:47:44.012252
# Source Brief: brief_02387.md
# Brief Index: 2387
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a player controls a cursor to collect numbered squares.
    The goal is to reach a target score within a time limit. The game prioritizes
    visual polish and a satisfying gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Collect numbered squares to reach the target score before time runs out. "
        "Higher numbers are rarer and worth more points."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move the cursor. "
        "Press space to collect the number under the cursor."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    MAX_STEPS = 1800 # 60 seconds at 30 FPS
    TARGET_SCORE = 1000
    INITIAL_NUMBERS = 8
    
    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_CURSOR_GLOW = (255, 200, 0, 50)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (0, 0, 0, 150)
    
    # --- Gameplay ---
    PLAYER_LERP_RATE = 0.4  # For smooth visual movement

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_number = pygame.font.SysFont("Arial", 16, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.numbers = []
        self.particles = []
        self.prev_space_held = False
        
        # --- Initial Reset ---
        # Note: self.reset() is called here, but a full reset is needed
        # to ensure all attributes are defined before validation.
        self._initialize_state_attributes()

    def _initialize_state_attributes(self):
        """Initializes all state attributes to default values."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_visual_pos = self._grid_to_pixel(self.player_pos)
        self.numbers = []
        self.particles = []
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_state_attributes()

        for _ in range(self.INITIAL_NUMBERS):
            self._spawn_number()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Unpacking ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # --- Game Logic: Player Movement ---
        self._handle_movement(movement)
        
        # --- Game Logic: Collection ---
        collection_reward = self._handle_collection(space_held)
        reward += collection_reward

        # --- Game Logic: Updates ---
        self._update_particles()
        self._update_visuals()
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 100  # Victory bonus
            terminated = True
            # Sound: Play victory fanfare
        elif self.steps >= self.MAX_STEPS:
            reward -= 100  # Time-out penalty
            terminated = True
            # Sound: Play failure sound
        
        self.game_over = terminated
        self.prev_space_held = space_held

        # The 'truncated' flag is not used in this environment, so it's always False.
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement_action):
        dx, dy = 0, 0
        if movement_action == 1: dy = -1  # Up
        elif movement_action == 2: dy = 1   # Down
        elif movement_action == 3: dx = -1  # Left
        elif movement_action == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy
            
            # Boundary check
            self.player_pos[0] = max(0, min(self.GRID_WIDTH - 1, new_x))
            self.player_pos[1] = max(0, min(self.GRID_HEIGHT - 1, new_y))

    def _handle_collection(self, space_held):
        reward = 0
        # Use rising edge of space press to prevent multiple collections
        if space_held and not self.prev_space_held:
            collected_index = -1
            for i, num in enumerate(self.numbers):
                if num['pos'] == self.player_pos:
                    collected_index = i
                    break
            
            if collected_index != -1:
                # Sound: Play collection sound
                collected_num = self.numbers.pop(collected_index)
                value = collected_num['value']
                self.score += value

                # Calculate reward based on brief
                reward += 0.1  # Base reward for any collection
                if value > 9: reward += 1.0
                if value > 99: reward += 5.0
                
                # Visual feedback
                self._create_collection_effect(collected_num['pos'], value)
                
                # Replenish number
                self._spawn_number()
        return reward

    def _update_visuals(self):
        """Smoothly interpolate visual positions towards logical positions."""
        target_pixel_pos = self._grid_to_pixel(self.player_pos)
        self.player_visual_pos[0] += (target_pixel_pos[0] - self.player_visual_pos[0]) * self.PLAYER_LERP_RATE
        self.player_visual_pos[1] += (target_pixel_pos[1] - self.player_visual_pos[1]) * self.PLAYER_LERP_RATE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_particles()
        self._draw_numbers()
        self._draw_player()

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topleft=(10, 10))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, score_rect.inflate(10, 5))
        self.screen.blit(score_text, score_rect)

        # Time display
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_main.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, time_rect.inflate(10, 5))
        self.screen.blit(time_text, time_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper and Rendering Methods ---

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_player(self):
        pos_x, pos_y = int(self.player_visual_pos[0]), int(self.player_visual_pos[1])
        size = int(self.CELL_SIZE * 0.7)
        
        # Glow effect
        glow_radius = int(size * 0.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_CURSOR_GLOW)
        self.screen.blit(glow_surf, (pos_x - glow_radius, pos_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main cursor
        player_rect = pygame.Rect(pos_x - size // 2, pos_y - size // 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, player_rect, border_radius=4)

    def _draw_numbers(self):
        size = int(self.CELL_SIZE * 0.8)
        for num in self.numbers:
            px, py = self._grid_to_pixel(num['pos'])
            color = self._get_color_for_value(num['value'])
            
            num_rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            pygame.draw.rect(self.screen, color, num_rect, border_radius=3)
            
            text_surf = self.font_number.render(str(num['value']), True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=num_rect.center)
            self.screen.blit(text_surf, text_rect)

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.box(self.screen, pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size), color)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_collection_effect(self, grid_pos, value):
        px, py = self._grid_to_pixel(grid_pos)
        color = self._get_color_for_value(value)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': self.np_random.integers(4, 9),
                'life': self.np_random.integers(20, 41),
                'max_life': 40
            })

    def _spawn_number(self):
        occupied_cells = {tuple(n['pos']) for n in self.numbers}
        occupied_cells.add(tuple(self.player_pos))
        
        while True:
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if tuple(pos) not in occupied_cells:
                break
        
        # Generate value based on weighted probability
        rand_val = self.np_random.random()
        if rand_val < 0.70:  # 70% chance
            value = self.np_random.integers(1, 10)
        elif rand_val < 0.95:  # 25% chance
            value = self.np_random.integers(10, 100)
        else:  # 5% chance
            value = self.np_random.integers(100, 251)

        self.numbers.append({'pos': pos, 'value': value})

    def _get_color_for_value(self, value):
        if value < 10: # Blue range
            lerp = (value - 1) / 8.0
            return (int(100 + 50 * lerp), int(150 + 50 * lerp), 255)
        elif value < 100: # Green range
            lerp = (value - 10) / 89.0
            return (int(100 + 50 * lerp), 255, int(150 - 50 * lerp))
        else: # Red range
            lerp = min(1.0, (value - 100) / 150.0)
            return (255, int(150 - 100 * lerp), int(100 - 50 * lerp))

    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return [px, py]

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will not be executed by the autograder.
    
    # Un-set the dummy video driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Number Collector")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop for manual play
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}")
    env.close()