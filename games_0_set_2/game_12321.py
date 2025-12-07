import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:53:27.161718
# Source Brief: brief_02321.md
# Brief Index: 2321
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for managing a cellular ecosystem.

    The player controls a cursor to interact with the environment. They can
    adjust nutrient flow by rotating portals and directly manipulate local
    populations using specialized tools. The goal is to maintain a balanced
    ecosystem with multiple species, preventing any from going extinct.

    **Visuals:**
    - Abstract, scientific aesthetic with glowing elements.
    - Nutrients are represented by a blueish heat-map.
    - Populations are visualized as small, colored, animated circles.
    - Nutrient flow through portals is shown with animated particles.

    **Gameplay:**
    - Move a cursor around the 32x20 grid.
    - When over a portal, use movement keys to rotate its output direction.
    - Use 'Space' to spend nutrients and grow the population under the cursor.
    - Use 'Shift' to cull the population under the cursor, releasing some nutrients.
    - Achieve stability by keeping all species alive. Milestones are awarded for
      long periods of stability, which also introduces new species, increasing
      the challenge.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage a delicate cellular ecosystem by directing nutrient flow and manipulating local populations. "
        "Maintain stability and introduce new species to increase complexity."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor and rotate portals. "
        "Press 'space' to grow a population and 'shift' to cull it."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20  # 640/32=20, 400/20=20
    MAX_STEPS = 5000
    STABILITY_THRESHOLD = 100 # Min total population for each species to be stable
    VICTORY_STREAK = 1500 # Steps of stability to win a round and add a new species

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_NUTRIENT = (100, 150, 255)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_PORTAL = (255, 0, 255)
    COLOR_UI_BG = (25, 30, 45, 180)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_STABILITY_BAR = (0, 255, 150)
    COLOR_STABILITY_BAR_BG = (60, 60, 80)

    SPECIES_DEFS = [
        {"name": "A", "color": (255, 80, 80), "growth_rate": 0.0015, "consumption": 0.02},
        {"name": "B", "color": (80, 120, 255), "growth_rate": 0.0012, "consumption": 0.015},
        {"name": "C", "color": (80, 255, 120), "growth_rate": 0.0018, "consumption": 0.025},
        {"name": "D", "color": (255, 150, 50), "growth_rate": 0.0010, "consumption": 0.01},
        {"name": "E", "color": (200, 100, 255), "growth_rate": 0.0020, "consumption": 0.03},
    ]

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.nutrients = None
        self.populations = None
        self.species = []
        self.portals = []
        self.particles = []
        self.stability_steps = 0
        self.last_portal_rotated_step = -10 # Cooldown for portal selection cycling

        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # this is for dev, not for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stability_steps = 0

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)

        # Grid-based state
        self.nutrients = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float32)
        
        # Species setup
        self.species = [s.copy() for s in self.SPECIES_DEFS[:3]]
        self.populations = np.zeros((len(self.species), self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float32)
        
        # Initial Portals
        self.portals = [
            {"pos": (5, 10), "dir": (1, 0)}, # Right
            {"pos": (26, 5), "dir": (0, 1)}, # Down
            {"pos": (26, 15), "dir": (0, -1)}, # Up
        ]
        
        # Initial nutrient and population placement
        self.nutrients[self.portals[0]["pos"]] = 2000.0 # High nutrient at source
        
        for i in range(len(self.species)):
            # Place initial populations near other portals
            px, py = self.portals[(i + 1) % len(self.portals)]["pos"]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = np.clip(px+dx, 0, self.GRID_WIDTH-1), np.clip(py+dy, 0, self.GRID_HEIGHT-1)
                    self.populations[i, nx, ny] = 50

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Input and Actions ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_portals()
        self._update_diffusion()
        self._update_populations()
        self._update_particles()
        
        # --- Check Stability and Termination ---
        is_stable = all(np.sum(self.populations[i]) > self.STABILITY_THRESHOLD for i in range(len(self.species)))
        
        if is_stable:
            self.stability_steps += 1
            reward += 0.1 # Small reward for being stable
            if self.stability_steps > 0 and self.stability_steps % 100 == 0:
                reward += 5 # Bonus for sustained stability
                # sfx: stability_milestone_sound
        else:
            self.stability_steps = 0
            
        # Check for victory condition (milestone)
        if self.stability_steps >= self.VICTORY_STREAK:
            reward += 100
            self.score += 100
            self.stability_steps = 0
            if len(self.species) < len(self.SPECIES_DEFS):
                self._add_new_species()
                # sfx: new_species_unlocked_sound
            
        terminated = self._check_termination()
        truncated = False # This env does not truncate based on time limit
        if terminated and not self.game_over:
            # Extinction is the primary loss condition
            is_extinction = any(np.sum(p) <= 0 for p in self.populations)
            if is_extinction:
                reward = -100
                # sfx: extinction_fail_sound
            self.game_over = True
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        cursor_speed = 8.0
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT - 1)

        cursor_gx, cursor_gy = int(self.cursor_pos[0] / self.CELL_SIZE), int(self.cursor_pos[1] / self.CELL_SIZE)

        # --- Portal Rotation ---
        rotated_this_step = False
        for portal in self.portals:
            if (cursor_gx, cursor_gy) == portal["pos"] and movement in [1, 2, 3, 4]:
                if not rotated_this_step: # Only rotate one portal per step
                    # sfx: portal_rotate_sound
                    if movement == 1: portal["dir"] = (0, -1) # Up
                    elif movement == 2: portal["dir"] = (0, 1) # Down
                    elif movement == 3: portal["dir"] = (-1, 0) # Left
                    elif movement == 4: portal["dir"] = (1, 0) # Right
                    rotated_this_step = True
        
        # --- Population Manipulation ---
        if space_held: # Grow population
            # sfx: grow_population_sound
            # Find dominant species under cursor
            if np.sum(self.populations[:, cursor_gx, cursor_gy]) > 0:
                species_idx = np.argmax(self.populations[:, cursor_gx, cursor_gy])
                cost = 5.0 # Nutrient cost to grow
                if self.nutrients[cursor_gx, cursor_gy] > cost:
                    self.nutrients[cursor_gx, cursor_gy] -= cost
                    self.populations[species_idx, cursor_gx, cursor_gy] += 1.0

        if shift_held: # Cull population
            # sfx: cull_population_sound
            if np.sum(self.populations[:, cursor_gx, cursor_gy]) > 0:
                species_idx = np.argmax(self.populations[:, cursor_gx, cursor_gy])
                if self.populations[species_idx, cursor_gx, cursor_gy] > 1.0:
                    self.populations[species_idx, cursor_gx, cursor_gy] -= 1.0
                    self.nutrients[cursor_gx, cursor_gy] += 1.0 # Reclaim some nutrients

    def _update_portals(self):
        transfer_rate = 50.0
        for portal in self.portals:
            px, py = portal["pos"]
            dx, dy = portal["dir"]
            nx, ny = np.clip(px + dx, 0, self.GRID_WIDTH - 1), np.clip(py + dy, 0, self.GRID_HEIGHT - 1)
            
            transfer_amount = min(self.nutrients[px, py], transfer_rate)
            if transfer_amount > 0:
                self.nutrients[px, py] -= transfer_amount
                self.nutrients[nx, ny] += transfer_amount
                # Create visual particles
                for _ in range(5):
                    start_pos = np.array([(px + 0.5) * self.CELL_SIZE, (py + 0.5) * self.CELL_SIZE])
                    end_pos = np.array([(nx + 0.5) * self.CELL_SIZE, (ny + 0.5) * self.CELL_SIZE])
                    self.particles.append({
                        "pos": start_pos + self.np_random.standard_normal(2) * 3,
                        "vel": (end_pos - start_pos) / 15.0 + self.np_random.standard_normal(2) * 0.2,
                        "life": 15 + self.np_random.integers(0, 6),
                        "color": (255, 255, 255)
                    })
    
    def _update_diffusion(self):
        # Simple 4-neighbor diffusion
        diffusion_rate = 0.05
        
        # Using manual convolution with 'wrap' padding to conserve nutrients
        nutrient_change = np.zeros_like(self.nutrients)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                center_val = self.nutrients[x, y]
                neighbors_sum = (self.nutrients[(x + 1) % self.GRID_WIDTH, y] +
                                 self.nutrients[(x - 1) % self.GRID_WIDTH, y] +
                                 self.nutrients[x, (y + 1) % self.GRID_HEIGHT] +
                                 self.nutrients[x, (y - 1) % self.GRID_HEIGHT])
                nutrient_change[x, y] = (neighbors_sum - 4 * center_val) * diffusion_rate
        
        self.nutrients += nutrient_change
        self.nutrients.clip(min=0, out=self.nutrients)

    def _update_populations(self):
        for i, s in enumerate(self.species):
            # Consumption
            consumed = self.populations[i] * s["consumption"]
            consumed = np.minimum(consumed, self.nutrients) # Can't consume more than available
            self.nutrients -= consumed
            
            # Growth and Decay
            natural_decay = self.populations[i] * 0.005 # All species decay slowly
            growth = self.populations[i] * consumed * s["growth_rate"]
            
            self.populations[i] += growth - natural_decay
            self.populations[i].clip(min=0, out=self.populations[i])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _add_new_species(self):
        if len(self.species) < len(self.SPECIES_DEFS):
            new_species_def = self.SPECIES_DEFS[len(self.species)].copy()
            self.species.append(new_species_def)
            
            # Add new population array
            new_pop_layer = np.zeros((1, self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.float32)
            self.populations = np.vstack([self.populations, new_pop_layer])
            
            # Seed the new population somewhere
            for _ in range(5): # Try a few random spots
                px, py = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
                if np.sum(self.populations[:, px, py]) == 0: # Find an empty spot
                    self.populations[-1, px, py] = 50
                    break
            else: # If no empty spot, place anywhere
                px, py = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
                self.populations[-1, px, py] = 50


    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if any(np.sum(p) <= 0 for p in self.populations):
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stability_steps": self.stability_steps,
            "species_count": len(self.species),
            "total_population": int(np.sum(self.populations)),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background_grid()
        self._render_nutrients()
        self._render_populations()
        self._render_portals()
        self._render_particles()
        self._render_cursor()

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_nutrients(self):
        max_nutrient = max(50.0, np.max(self.nutrients)) # Avoid division by zero and have a baseline
        nutrient_surface = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                alpha = int(np.clip(self.nutrients[x, y] / max_nutrient, 0, 1) * 150)
                if alpha > 0:
                    nutrient_surface.set_at((x, y), self.COLOR_NUTRIENT + (alpha,))
        
        scaled_surface = pygame.transform.scale(nutrient_surface, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen.blit(scaled_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_populations(self):
        for i, s in enumerate(self.species):
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    pop_count = self.populations[i, x, y]
                    if pop_count > 1:
                        # Draw multiple small circles to represent density
                        num_circles = min(8, int(math.log(pop_count, 2)))
                        for _ in range(num_circles):
                            offset_x = (self.np_random.random() - 0.5) * self.CELL_SIZE * 0.8
                            offset_y = (self.np_random.random() - 0.5) * self.CELL_SIZE * 0.8
                            center_x = int((x + 0.5) * self.CELL_SIZE + offset_x)
                            center_y = int((y + 0.5) * self.CELL_SIZE + offset_y)
                            
                            # Pulsing size effect
                            radius = 2 + math.sin(self.steps * 0.1 + x + y) * 0.5
                            
                            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius), s["color"])
                            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius), s["color"])

    def _render_portals(self):
        for portal in self.portals:
            px, py = portal["pos"]
            center_x, center_y = int((px + 0.5) * self.CELL_SIZE), int((py + 0.5) * self.CELL_SIZE)
            
            # Glowing effect
            for r in range(1, 5):
                alpha = 100 - r * 20
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.CELL_SIZE // 2 + r, self.COLOR_PORTAL + (alpha,))
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.CELL_SIZE // 2 - 2, self.COLOR_PORTAL)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.CELL_SIZE // 2 - 2, (255,200,255))
            
            # Arrow
            dx, dy = portal["dir"]
            arrow_start = (center_x + dx * 5, center_y + dy * 5)
            arrow_end = (center_x + dx * 9, center_y + dy * 9)
            pygame.draw.line(self.screen, (255, 255, 255), arrow_start, arrow_end, 2)
            
            # Arrowhead
            angle = math.atan2(-dy, dx)
            p1 = (arrow_end[0] - 4 * math.cos(angle - math.pi / 6), arrow_end[1] + 4 * math.sin(angle - math.pi / 6))
            p2 = (arrow_end[0] - 4 * math.cos(angle + math.pi / 6), arrow_end[1] + 4 * math.sin(angle + math.pi / 6))
            pygame.draw.polygon(self.screen, (255, 255, 255), [arrow_end, p1, p2])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(max(0, p["life"] / 20.0) * 255)
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, color)

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        size = 10
        # Glowing effect
        for r in range(1, 4):
            alpha = 100 - r * 25
            pygame.gfxdraw.aacircle(self.screen, x, y, size + r, self.COLOR_CURSOR + (alpha,))
        
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - size, y), (x + size, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - size), (x, y + size), 1)

    def _render_ui(self):
        # --- Top Panel ---
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, 70), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))
        
        # --- Score and Steps ---
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_medium.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 35))
        
        # --- Population Counts ---
        start_x = 200
        for i, s in enumerate(self.species):
            pop_total = int(np.sum(self.populations[i]))
            pop_text = self.font_medium.render(f"{pop_total}", True, s["color"])
            self.screen.blit(pop_text, (start_x + i * 80, 25))
            pygame.draw.rect(self.screen, s["color"], (start_x + i * 80, 15, 15, 10))

        # --- Stability Bar ---
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 10
        bar_x, bar_y = 10, self.SCREEN_HEIGHT - bar_height - 10
        
        fill_ratio = min(1.0, self.stability_steps / self.VICTORY_STREAK)
        fill_width = int(bar_width * fill_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_STABILITY_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_STABILITY_BAR, (bar_x, bar_y, fill_width, bar_height), border_radius=3)
        
        stability_text = self.font_small.render("SYSTEM STABILITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(stability_text, (bar_x, bar_y - 16))

        # --- Game Over Text ---
        if self.game_over:
            is_extinction = any(np.sum(p) <= 0 for p in self.populations)
            msg = "EXTINCTION EVENT" if is_extinction else "SIMULATION ENDED"
            color = (255, 50, 50) if is_extinction else self.COLOR_UI_TEXT
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            game_over_text = self.font_large.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_over_display = False
    
    # --- Manual Control Mapping ---
    # MultiDiscrete([5, 2, 2])
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    action = [0, 0, 0] 

    # We need a separate pygame window to display the rendered array
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        # In headless mode, we can't create a display.
        # This is for automated testing.
        # To play manually, unset SDL_VIDEODRIVER.
        print("Running in headless mode. No display will be shown.")
        # Simple loop to test the environment runs
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs, info = env.reset()
        env.close()
        exit()
        
    pygame.display.set_caption("Cellular Ecosystem Manager")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        action[0] = 0 # Reset movement action each frame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over_display = False
        
        if not game_over_display:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
                game_over_display = True
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    env.close()