import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a grid-based puzzle game.

    The agent controls a cursor on a grid of switches. Activating a switch
    toggles its state and the state of its four orthogonal neighbors. The goal
    is to activate a special endpoint switch. A "short circuit" occurs if any
    switch becomes surrounded by four active neighbors, resulting in a loss.
    The game's difficulty increases by expanding the grid size with each level cleared.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Activate switch (0=released, 1=pressed)
    - actions[2]: Shift (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +1 for moving one step closer to the endpoint (Manhattan distance).
    - -1 for moving one step further from the endpoint.
    - +100 for clearing a level by activating the endpoint.
    - -50 for causing a short circuit.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Toggle switches on a grid to activate the endpoint. Avoid creating a 'short circuit' "
        "by surrounding any switch with four active neighbors."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to activate the switch under the cursor."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Gymnasium Spaces ===
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # === Pygame Setup ===
        self.render_mode = render_mode
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # === Colors & Visuals ===
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_GRID = (48, 48, 74)
        self.COLOR_INACTIVE = (46, 204, 113)
        self.COLOR_ACTIVE = (52, 152, 219)
        self.COLOR_ENDPOINT = (241, 196, 15)
        self.COLOR_CURSOR = (236, 240, 241)
        self.COLOR_SHORT_CIRCUIT = (231, 76, 60)
        self.COLOR_TEXT = (220, 220, 240)

        # === Game Parameters ===
        self.max_episode_steps = 1000

        # === Game State (initialized in reset) ===
        self.level = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid_size = (0, 0)
        self.grid = None
        self.cursor_pos = (0, 0)
        self.endpoint_pos = (0, 0)
        self.prev_space_held = False
        self.animation_effects = []
        self.short_circuit_flash_timer = 0
        self.cell_size = 0
        self.start_x = 0
        self.start_y = 0


    def _setup_level(self):
        """Initializes the grid and game elements for the current level."""
        rows = 5 + (self.level - 1) * 2
        cols = 5 + (self.level - 1) * 2
        self.grid_size = (rows, cols)

        self.grid = np.zeros(self.grid_size, dtype=int)
        
        # Ensure endpoint and cursor start at different, valid locations
        self.endpoint_pos = (self.np_random.integers(0, rows), self.np_random.integers(0, cols))
        self.cursor_pos = self.endpoint_pos
        while self.cursor_pos == self.endpoint_pos:
            self.cursor_pos = (self.np_random.integers(0, rows), self.np_random.integers(0, cols))

        self.animation_effects = []
        self.short_circuit_flash_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.prev_space_held = False

        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        
        movement, space_held, _ = action
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # --- 1. Handle Movement and Calculate Movement Reward ---
        old_dist = self._manhattan_distance(self.cursor_pos, self.endpoint_pos)
        
        if movement == 1:  # Up
            self.cursor_pos = ((self.cursor_pos[0] - 1) % self.grid_size[0], self.cursor_pos[1])
        elif movement == 2:  # Down
            self.cursor_pos = ((self.cursor_pos[0] + 1) % self.grid_size[0], self.cursor_pos[1])
        elif movement == 3:  # Left
            self.cursor_pos = (self.cursor_pos[0], (self.cursor_pos[1] - 1) % self.grid_size[1])
        elif movement == 4:  # Right
            self.cursor_pos = (self.cursor_pos[0], (self.cursor_pos[1] + 1) % self.grid_size[1])

        new_dist = self._manhattan_distance(self.cursor_pos, self.endpoint_pos)
        movement_reward = old_dist - new_dist
        reward += movement_reward
        self.score += movement_reward

        # --- 2. Handle Switch Activation ---
        if space_pressed:
            self._activate_switch(self.cursor_pos)

            if self.grid[self.endpoint_pos] == 1:
                # --- VICTORY ---
                reward += 100
                self.score += 100
                self.level += 1
                self._setup_level()
            elif self._check_for_short_circuit():
                # --- LOSS (Short Circuit) ---
                reward -= 50
                self.score -= 50
                self.game_over = True
                terminated = True
                self.short_circuit_flash_timer = 30 # 1 second at 30fps

        # --- 3. Check for Termination ---
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _activate_switch(self, pos):
        r, c = pos
        switches_to_toggle = [(r, c)]
        
        # Add orthogonal neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                switches_to_toggle.append((nr, nc))
        
        for sr, sc in switches_to_toggle:
            self.grid[sr, sc] = 1 - self.grid[sr, sc] # Toggle 0 to 1 or 1 to 0
            self.animation_effects.append({
                "pos": (sr, sc),
                "radius": 0,
                "max_radius": 1.0,
                "lifetime": 15,
                "max_lifetime": 15,
                "color": self.COLOR_ACTIVE if self.grid[sr, sc] == 1 else self.COLOR_INACTIVE
            })

    def _check_for_short_circuit(self):
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                active_neighbors = 0
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                        if self.grid[nr, nc] == 1:
                            active_neighbors += 1
                if active_neighbors == 4:
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_effects()
        self._render_ui()

        if self.short_circuit_flash_timer > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            alpha = int(128 * (self.short_circuit_flash_timer / 30))
            flash_surface.fill((*self.COLOR_SHORT_CIRCUIT, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.short_circuit_flash_timer -= 1

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "cursor_pos": self.cursor_pos,
            "endpoint_pos": self.endpoint_pos,
        }

    def _render_game(self):
        # Calculate grid geometry to center it with a margin
        margin = 0.15
        usable_width = self.screen_width * (1 - margin)
        usable_height = self.screen_height * (1 - margin)
        
        cell_size_w = usable_width / self.grid_size[1]
        cell_size_h = usable_height / self.grid_size[0]
        self.cell_size = int(min(cell_size_w, cell_size_h))

        grid_pixel_width = self.cell_size * self.grid_size[1]
        grid_pixel_height = self.cell_size * self.grid_size[0]
        
        self.start_x = (self.screen_width - grid_pixel_width) // 2
        self.start_y = (self.screen_height - grid_pixel_height) // 2

        # Draw switches
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                rect = pygame.Rect(
                    self.start_x + c * self.cell_size,
                    self.start_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                color = self.COLOR_ACTIVE if self.grid[r, c] == 1 else self.COLOR_INACTIVE
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect) # Border/Grid line
                pygame.draw.rect(self.screen, color, rect.inflate(-4, -4)) # Inner switch

                # Draw endpoint star
                if (r, c) == self.endpoint_pos:
                    self._draw_star(rect.center, self.cell_size * 0.3)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.start_x + cursor_c * self.cell_size,
            self.start_y + cursor_r * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3) # Thick border

    def _draw_star(self, center, size):
        points = []
        for i in range(10):
            angle = math.radians(i * 36)
            radius = size if i % 2 == 0 else size * 0.4
            x = center[0] + radius * math.cos(angle - math.pi / 2)
            y = center[1] + radius * math.sin(angle - math.pi / 2)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENDPOINT)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENDPOINT)

    def _update_and_render_effects(self):
        active_effects = []
        for effect in self.animation_effects:
            effect["lifetime"] -= 1
            if effect["lifetime"] > 0:
                effect["radius"] += effect["max_radius"] / effect["max_lifetime"]
                
                r, c = effect["pos"]
                center_x = int(self.start_x + (c + 0.5) * self.cell_size)
                center_y = int(self.start_y + (r + 0.5) * self.cell_size)
                
                radius = int(effect["radius"] * self.cell_size * 0.5)
                alpha = int(200 * (effect["lifetime"] / effect["max_lifetime"]))
                
                if radius > 0:
                    # Use a temporary surface for alpha blending
                    temp_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    color_with_alpha = (*effect["color"], alpha)
                    pygame.gfxdraw.filled_circle(temp_surface, radius, radius, radius, color_with_alpha)
                    pygame.gfxdraw.aacircle(temp_surface, radius, radius, radius, color_with_alpha)
                    self.screen.blit(temp_surface, (center_x - radius, center_y - radius))
                
                active_effects.append(effect)
        self.animation_effects = active_effects

    def _render_ui(self):
        level_text = self.font_large.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, 10))

        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.screen_width - 20, 10))
        self.screen.blit(score_text, score_rect)

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.max_episode_steps}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(bottomright=(self.screen_width - 20, self.screen_height - 10))
        self.screen.blit(steps_text, steps_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    env.reset()
    
    running = True
    terminated = False
    
    # Use a display screen for manual play
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Grid Switch Puzzle")
    
    action = [0, 0, 0] # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Reset action on key up is not ideal for continuous play
        # action = [0, 0, 0]

        if terminated:
            # If the game is over, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
            
            # Render one last time to show final state
            surf = pygame.surfarray.make_surface(env._get_observation())
            surf = pygame.transform.rotate(surf, 90)
            surf = pygame.transform.flip(surf, True, False)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            continue

        keys = pygame.key.get_pressed()
        
        # --- Map keyboard to MultiDiscrete action ---
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to reset.")

        # --- Render to display ---
        # The observation is (H, W, C), but pygame blit needs a surface
        # We can just get the surface from the env's internal screen
        # The observation is already transposed correctly for surfarray
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()